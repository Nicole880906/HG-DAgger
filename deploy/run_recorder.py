#!/usr/bin/env python3
"""Record a deploy session to disk in the demo-collector's episode format.

One deploy session becomes one *run* directory.  Every frame is written,
whether the diffusion policy or the human was driving, and each frame carries a
``control_mode`` label saying which.  ``dagger/merge_interventions.py`` later
slices the human-driven stretches out into episode directories that
``data_processing/convert_drawing_6d_abs.py`` consumes with no changes -- so
corrections recorded here retrain the policy through exactly the same path the
original demonstrations took.

Why record the policy frames too, when HG-DAgger only trains on expert data:
the mode label is per frame, so keeping everything costs disk and nothing else,
and it means the decision of how much lead-in to include around each
intervention -- or whether a given takeover was a real correction or a slip of
the foot -- is made later, looking at the data, instead of being baked in
irreversibly at 30 Hz while the robot is moving.

Sampling rate
-------------
Frames are appended by a **30 Hz timer**, not by the control loop, which runs at
the checkpoint's action rate (5 Hz for drawing policies). The converter takes
every 6th frame to reach the 5 Hz the policy
trains at, so a run recorded at the control rate would come out at 1.7 Hz and
the corrections would be sampled three times coarser than the demonstrations
they are meant to join.  Keep the recorder at the collector's 30 Hz.

The JSON layout, key for key, is ``src/arclab_dvrk/src/data_collection/
writer.py``'s, with the ``control_mode`` / ``t`` fields added per frame.  The
converter ignores unknown fields, so a run directory is also readable by every
tool written for the demonstration data.
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# The control phases a frame can be recorded under. These are the deploy
# node's handover phases verbatim: a frame is labelled with who was driving, and
# during the two alignment windows the honest answer is "nobody -- the arm is
# being held while control changes hands".
POLICY = "policy"
ALIGN_TO_EXPERT = "align_to_expert"
EXPERT = "expert"
ALIGN_TO_POLICY = "align_to_policy"

PHASES = (POLICY, ALIGN_TO_EXPERT, EXPERT, ALIGN_TO_POLICY)

JOINT_NAMES = {
    "psm1_joint_state": [
        "psm1_yaw_joint", "psm1_pitch_end_joint", "psm1_main_insertion_joint",
        "psm1_tool_roll_joint", "psm1_tool_pitch_joint", "psm1_tool_yaw_joint",
        "psm1_tool_gripper1_joint", "psm1_tool_gripper2_joint",
    ],
    "psm2_joint_state": [
        "psm2_yaw_joint", "psm2_pitch_end_joint", "psm2_main_insertion_joint",
        "psm2_tool_roll_joint", "psm2_tool_pitch_joint", "psm2_tool_yaw_joint",
        "psm2_tool_gripper1_joint", "psm2_tool_gripper2_joint",
    ],
    "psm1_ee": ["PSM1PosX", "PSM1PosY", "PSM1PosZ",
                "PSM1QuatW", "PSM1QuatX", "PSM1QuatY", "PSM1QuatZ"],
    "psm2_ee": ["PSM2PosX", "PSM2PosY", "PSM2PosZ",
                "PSM2QuatW", "PSM2QuatX", "PSM2QuatY", "PSM2QuatZ"],
}


def build_states(
    cutter_pos: np.ndarray,
    cutter_quat_wxyz: np.ndarray,
    cutter_jaw: float,
    retract_pos: np.ndarray,
    retract_quat_wxyz: np.ndarray,
    retract_jaw: float,
    cutter_qpos: np.ndarray | None = None,
    retract_qpos: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assemble one frame's ``states`` block in the collector's schema.

    Arm naming follows the rest of the stack: **cutter is PSM2, retraction is
    PSM1**, and the quaternion is stored ``wxyz`` because that is what
    ``convert_drawing_6d_abs.py`` feeds to ``pytorch3d.quaternion_to_matrix``.
    Passing ``xyzw`` here would produce a silently wrong rotation in training
    data that still looks perfectly well-formed.
    """
    def arm_js(qpos: np.ndarray | None, jaw: float) -> dict[str, Any]:
        q = [] if qpos is None else np.asarray(qpos, dtype=np.float64).reshape(-1).tolist()
        return {
            "qpos": q,
            "qvel": [],
            "qeffort": [],
            "gripper": float(jaw),
            "gripper_effort": 0.0,
        }

    return {
        "psm_cutter_js": arm_js(cutter_qpos, cutter_jaw),
        "psm_retraction_js": arm_js(retract_qpos, retract_jaw),
        "psm_cutter_ee": {
            "psm_cutter_pos": np.asarray(cutter_pos, dtype=np.float64).reshape(3).tolist(),
            "psm_cutter_quat": np.asarray(cutter_quat_wxyz, dtype=np.float64).reshape(4).tolist(),
        },
        "psm_retraction_ee": {
            "psm_retraction_pos": np.asarray(retract_pos, dtype=np.float64).reshape(3).tolist(),
            "psm_retraction_quat": np.asarray(retract_quat_wxyz, dtype=np.float64).reshape(4).tolist(),
        },
    }


class RunRecorder:
    """Append-only writer for one deploy session.

    Thread-safe: the recording timer and the control loop run in the same rclpy
    executor, but ``mark_mode`` may be called from either, and the JSON flush
    walks the frame list.
    """

    def __init__(
        self,
        run_dir: str | Path,
        task: str = "drawing",
        image_size: tuple[int, int] = (640, 480),
        frequency: float = 30.0,
        jpeg_quality: int = 95,
        flush_every: int = 150,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.color_dir = self.run_dir / "colors"
        self.json_path = self.run_dir / "data.json"
        self.color_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.image_size = image_size
        self.frequency = float(frequency)
        self.jpeg_quality = int(jpeg_quality)
        self.flush_every = int(flush_every)

        self._lock = threading.Lock()
        self._frames: list[dict[str, Any]] = []
        self._mode = POLICY
        self._since_flush = 0
        # Per-mode frame tallies, so the end-of-run summary can state how much
        # of the session the human actually drove without re-reading the JSON.
        self.counts = {phase: 0 for phase in PHASES}

    # ------------------------------------------------------------------ mode
    def mark_mode(self, mode: str) -> None:
        if mode not in PHASES:
            raise ValueError(f"unknown control phase {mode!r}; expected one of {PHASES}")
        with self._lock:
            self._mode = mode

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def n_frames(self) -> int:
        with self._lock:
            return len(self._frames)

    # ----------------------------------------------------------------- frames
    def add_frame(
        self,
        image_bgr: np.ndarray,
        states: dict[str, Any],
        t: float,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Write one frame's image and queue its record.  Returns the index.

        ``image_bgr`` is BGR because that is what ``cv2.imwrite`` expects and
        what the collector wrote; the deploy node holds RGB and converts on the
        way in.  Getting this backwards swaps the red and blue channels of the
        training images, which trains fine and then fails on the real camera.
        """
        with self._lock:
            idx = len(self._frames)
            mode = self._mode
        name = f"left_image_{idx:06d}.jpg"
        ok = cv2.imwrite(
            str(self.color_dir / name),
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"failed to write {self.color_dir / name}")

        record: dict[str, Any] = {
            "idx": idx,
            "colors": {"left_image": f"colors/{name}"},
            "states": states,
            "control_mode": mode,
            "t": float(t),
        }
        if extra:
            record.update(extra)
        with self._lock:
            self._frames.append(record)
            self.counts[mode] += 1
            self._since_flush += 1
            due = self._since_flush >= self.flush_every
        if due:
            self.flush()
        return idx

    # ------------------------------------------------------------------ save
    def _payload(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._frames)
            counts = dict(self.counts)
        return {
            "info": {
                "date": datetime.date.today().strftime("%Y-%m-%d"),
                "author": "expert_intervention",
                "image": {"width": self.image_size[0], "height": self.image_size[1],
                          "fps": self.frequency},
                "joint_names": JOINT_NAMES,
            },
            "text": {"goal": self.task},
            "run": {
                "control_modes": list(PHASES),
                "frame_counts": counts,
                "n_frames": len(frames),
            },
            "data": frames,
        }

    def flush(self) -> None:
        """Rewrite data.json from the frames recorded so far.

        Called periodically during the run, not only at the end: a deploy
        session that dies on a ROS error or a Ctrl+C mid-intervention should
        still leave usable correction data behind.  The write goes to a
        temporary file and is renamed over the target, so an interrupted flush
        cannot leave a truncated JSON where a complete one used to be.
        """
        payload = self._payload()
        tmp = self.json_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp.replace(self.json_path)
        with self._lock:
            self._since_flush = 0

    def close(self) -> dict[str, int]:
        self.flush()
        with self._lock:
            return dict(self.counts)


def next_run_dir(root: str | Path, prefix: str = "run_") -> Path:
    """``<root>/<prefix>NNNN`` with the lowest unused number."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for path in root.glob(f"{prefix}*"):
        suffix = path.name[len(prefix):]
        if path.is_dir() and suffix.isdigit():
            used.append(int(suffix))
    return root / f"{prefix}{(max(used) + 1 if used else 0):04d}"
