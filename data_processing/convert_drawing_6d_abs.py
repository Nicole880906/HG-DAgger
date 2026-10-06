#!/usr/bin/env python3
"""Convert collected episodes to a Diffusion Policy Zarr: EE-pose state, absolute action.

Reads the episode layout that both producers in this project write --
``data_collection.sh`` for demonstrations and ``dagger/merge_interventions.py``
for extracted expert corrections -- so a DAgger round converts through exactly
this script, with no special casing for where an episode came from.

* ``agent_pos`` is the **end-effector pose** of each arm.  Per arm the layout is
  ``[pos_x, pos_y, pos_z, r6d_0, ..., r6d_5, gripper]`` (10 values): Cartesian
  gripper position (3), a continuous 6D rotation (Zhou et al. 2019, the
  representation diffusion_policy trains on), and the scalar gripper opening.
  Both arms concatenated, cutter (PSM2) first then retraction (PSM1), give 20D.
* ``action`` is the **absolute** ``agent_pos`` of the frame ``action_offset``
  raw frames later, clamped at the episode end -- not a delta.  The deploy node
  commands the predicted row directly; adding it to the current pose would
  double every motion.

Every 6th 30 Hz frame is sampled, giving 5 Hz.  That stride is why the deploy
node records interventions at 30 Hz rather than at its own control rate.

The 6D rotation matches ``pytorch3d`` exactly: the source quaternion (stored
``wxyz``) becomes a rotation matrix and the first two rows are flattened, i.e.
``pytorch3d.transforms.matrix_to_rotation_6d``.

    <output>.zarr/
      data/
        image            uint8   (T, H, W, 3), RGB
        agent_pos        float32 (T, D), current EE pose [pos(3), r6d(6), grip(1)] per arm
        action           float32 (T, D), absolute EE pose of frame t+offset
      meta/
        episode_ends     int64  (E,)
        frame_indices    int64  (T,)
        episode_indices  int32  (T,)

Example
-------
::

    python data_processing/convert_drawing_6d_abs.py \
        data/drawing_circle data/diffusion_policy/circle.zarr
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import zarr
from numcodecs import Blosc
from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix


ARM_GROUPS = {
    "cutter": "psm_cutter_js",
    "retraction": "psm_retraction_js",
}
EE_GROUPS = {
    "cutter": "psm_cutter_ee",
    "retraction": "psm_retraction_ee",
}
# Per-arm keys inside the ``*_ee`` group are named after the arm.
EE_POS_KEYS = {
    "cutter": "psm_cutter_pos",
    "retraction": "psm_retraction_pos",
}
EE_QUAT_KEYS = {
    "cutter": "psm_cutter_quat",
    "retraction": "psm_retraction_quat",
}
IMAGE_KEYS = {
    "left": "left_image",
    "right": "right_image",
}

# Per-arm state layout: pos(3) + rotation_6d(6) + gripper(1).
POSE_DIM_PER_ARM = 10

FRAME_STRIDE = 6
RECORD_RATE_HZ = 30.0
POLICY_RATE_HZ = RECORD_RATE_HZ / FRAME_STRIDE
WIDTH = 160
HEIGHT = 120
IMAGE = "left"
ARMS = "both"
MAX_EPISODES: int | None = None


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    path: Path
    raw_length: int
    sample_indices: np.ndarray

    @property
    def sample_length(self) -> int:
        return int(self.sample_indices.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing episode_*/data.json")
    parser.add_argument("output_zarr", type=Path, help="Destination Diffusion Policy Zarr")
    # DAgger rounds re-convert the same output path repeatedly, so this is a
    # flag rather than the module constant the original used.
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output Zarr (and any stale .partial)")
    return parser.parse_args()


def load_frames(episode_path: Path) -> list[dict]:
    json_path = episode_path / "data.json"
    try:
        with json_path.open() as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to read {json_path}: {exc}") from exc
    frames = payload.get("data")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{json_path} has no non-empty 'data' list")
    return frames


def selected_arm_names(arms: str) -> tuple[str, ...]:
    if arms == "both":
        return ("cutter", "retraction")
    return (arms,)


def quat_wxyz_to_rotation_6d(quat: np.ndarray, context: str) -> np.ndarray:
    """Convert a ``wxyz`` quaternion to the 6D rotation of ``matrix_to_rotation_6d``.

    Uses ``pytorch3d.transforms`` directly (quaternion real-part-first ``wxyz``,
    6D = first two rows of the rotation matrix flattened) so the output is
    drop-in compatible with diffusion_policy's RotationTransformer.
    """
    q = np.asarray(quat, dtype=np.float32)
    if q.shape != (4,):
        raise ValueError(f"expected 4 quaternion values at {context}, got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"degenerate quaternion at {context}: {quat}")
    quat_t = torch.from_numpy(q / norm)  # normalise; w, x, y, z
    rot6d = matrix_to_rotation_6d(quaternion_to_matrix(quat_t))
    return rot6d.numpy().reshape(6).astype(np.float32)


def pose_state_vector(frame: dict, arms: tuple[str, ...], context: str) -> np.ndarray:
    """End-effector pose state: [pos(3), rot6d(6), gripper(1)] per arm."""
    values: list[float] = []
    try:
        states = frame["states"]
        for arm in arms:
            ee = states[EE_GROUPS[arm]]
            pos = np.asarray(ee[EE_POS_KEYS[arm]], dtype=np.float32)
            if pos.shape != (3,):
                raise ValueError(f"expected 3 position values for {arm}, found {pos.shape}")
            quat = ee[EE_QUAT_KEYS[arm]]
            rot6d = quat_wxyz_to_rotation_6d(quat, f"{context} ({arm})")
            gripper = states[ARM_GROUPS[arm]]["gripper"]
            values.extend(pos.tolist())
            values.extend(rot6d.tolist())
            values.append(float(gripper))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid robot state at {context}: {exc}") from exc
    out = np.asarray(values, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"non-finite robot state at {context}")
    return out


def image_path(episode_path: Path, frame: dict, image_key: str, context: str) -> Path:
    try:
        relpath = frame["colors"][image_key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing colors/{image_key} at {context}") from exc
    path = episode_path / relpath
    if not path.is_file():
        raise FileNotFoundError(f"missing image at {context}: {path}")
    return path


def discover_episodes(
    input_dir: Path,
    frame_stride: int,
    action_offset: int,
    image_key: str,
    arms: tuple[str, ...],
    max_episodes: int | None,
) -> list[EpisodeInfo]:
    episode_paths = sorted(path for path in input_dir.glob("episode_*") if path.is_dir())
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"no episode_* directories under {input_dir}")

    result: list[EpisodeInfo] = []
    for episode_i, episode_path in enumerate(episode_paths):
        frames = load_frames(episode_path)
        indices = np.arange(0, len(frames), frame_stride, dtype=np.int64)
        for raw_i in indices:
            context = f"{episode_path.name} frame {int(raw_i)}"
            frame = frames[int(raw_i)]
            pose_state_vector(frame, arms, context)
            target_i = min(int(raw_i) + action_offset, len(frames) - 1)
            pose_state_vector(
                frames[target_i], arms,
                f"{episode_path.name} target frame {target_i}",
            )
            image_path(episode_path, frame, image_key, context)
        result.append(EpisodeInfo(
            episode_path.name, episode_path, len(frames), indices))
        if (episode_i + 1) % 50 == 0 or episode_i + 1 == len(episode_paths):
            print(f"validated {episode_i + 1}/{len(episode_paths)} episodes")
    return result


def decode_resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.shape != (height, width, 3) or image.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image {path}: {image.shape}, {image.dtype}")
    return image


def create_output_arrays(
    root: zarr.Group,
    total: int,
    n_episodes: int,
    height: int,
    width: int,
    state_dim: int,
) -> dict[str, zarr.Array]:
    data = root.require_group("data")
    meta = root.require_group("meta")
    image_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.NOSHUFFLE)
    numeric_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    arrays = {
        "image": data.create_dataset(
            "image", shape=(total, height, width, 3), chunks=(1, height, width, 3),
            dtype=np.uint8, compressor=image_compressor,
        ),
        "agent_pos": data.create_dataset(
            "agent_pos", shape=(total, state_dim), chunks=(min(1024, total), state_dim),
            dtype=np.float32, compressor=numeric_compressor,
        ),
        "action": data.create_dataset(
            "action", shape=(total, state_dim), chunks=(min(1024, total), state_dim),
            dtype=np.float32, compressor=numeric_compressor,
        ),
        "episode_ends": meta.create_dataset(
            "episode_ends", shape=(n_episodes,), chunks=(n_episodes,), dtype=np.int64,
            compressor=None,
        ),
        "frame_indices": meta.create_dataset(
            "frame_indices", shape=(total,), chunks=(min(4096, total),), dtype=np.int64,
            compressor=numeric_compressor,
        ),
        "episode_indices": meta.create_dataset(
            "episode_indices", shape=(total,), chunks=(min(4096, total),), dtype=np.int32,
            compressor=numeric_compressor,
        ),
    }
    return arrays


def convert(
    episodes: list[EpisodeInfo],
    output_zarr: Path,
    image_key: str,
    arms: tuple[str, ...],
    action_offset: int,
    width: int,
    height: int,
    workers: int,
    overwrite: bool,
) -> None:
    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"{output_zarr} exists; pass --overwrite to replace it")
        if output_zarr.is_dir():
            shutil.rmtree(output_zarr)
        else:
            output_zarr.unlink()

    partial = output_zarr.with_name(output_zarr.name + ".partial")
    if partial.exists():
        if not overwrite:
            raise FileExistsError(f"stale partial output {partial}; pass --overwrite to replace it")
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)

    total = sum(ep.sample_length for ep in episodes)
    state_dim = len(arms) * POSE_DIM_PER_ARM
    root = zarr.open_group(str(partial), mode="w")
    arrays = create_output_arrays(root, total, len(episodes), height, width, state_dim)
    root.attrs.update({
        "format": "original_diffusion_policy_replay_buffer",
        "source_dataset": str(episodes[0].path.parent.resolve()),
        "image_key": image_key,
        "image_color_order": "RGB",
        "image_height": height,
        "image_width": width,
        "arms": list(arms),
        "state_layout_per_arm": [
            "pos_x", "pos_y", "pos_z",
            "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
            "gripper",
        ],
        "rotation_representation": "rotation_6d",
        "rotation_6d_convention": "pytorch3d matrix_to_rotation_6d (first two rows); source quat wxyz",
        "action_representation": "absolute_next_state",
        "action_offset_raw_frames": action_offset,
        "record_rate_hz": RECORD_RATE_HZ,
        "sample_rate_hz": POLICY_RATE_HZ,
        "action_offset_seconds": action_offset / RECORD_RATE_HZ,
    })

    cursor = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for episode_i, episode in enumerate(episodes):
                frames = load_frames(episode.path)
                indices = episode.sample_indices
                n = episode.sample_length
                obs_states = np.empty((n, state_dim), dtype=np.float32)
                actions = np.empty((n, state_dim), dtype=np.float32)
                paths: list[Path] = []
                for row, raw_i_np in enumerate(indices):
                    raw_i = int(raw_i_np)
                    target_i = min(raw_i + action_offset, len(frames) - 1)
                    obs_states[row] = pose_state_vector(
                        frames[raw_i], arms, f"{episode.name} frame {raw_i}")
                    actions[row] = pose_state_vector(
                        frames[target_i], arms, f"{episode.name} target frame {target_i}")
                    paths.append(image_path(
                        episode.path, frames[raw_i], image_key, f"{episode.name} frame {raw_i}"))

                images = np.stack(
                    list(pool.map(lambda p: decode_resize_rgb(p, width, height), paths)), axis=0)
                end = cursor + n
                arrays["image"][cursor:end] = images
                arrays["agent_pos"][cursor:end] = obs_states
                arrays["action"][cursor:end] = actions
                arrays["frame_indices"][cursor:end] = indices
                arrays["episode_indices"][cursor:end] = episode_i
                arrays["episode_ends"][episode_i] = end
                cursor = end
                print(f"[{episode_i + 1:04d}/{len(episodes):04d}] {episode.name}: "
                      f"{episode.raw_length} raw -> {n} samples; total={cursor}")

        if cursor != total:
            raise RuntimeError(f"wrote {cursor} samples but allocated {total}")
        verify_output(partial, total, len(episodes), height, width, state_dim)
        partial.rename(output_zarr)
    except Exception:
        print(f"conversion failed; partial output retained at {partial}")
        raise


def verify_output(
    path: Path,
    expected_total: int,
    expected_episodes: int,
    height: int,
    width: int,
    state_dim: int,
) -> None:
    root = zarr.open_group(str(path), mode="r")
    required = {
        "data/image": ((expected_total, height, width, 3), np.dtype(np.uint8)),
        "data/agent_pos": ((expected_total, state_dim), np.dtype(np.float32)),
        "data/action": ((expected_total, state_dim), np.dtype(np.float32)),
        "meta/episode_ends": ((expected_episodes,), np.dtype(np.int64)),
    }
    for key, (shape, dtype) in required.items():
        arr = root[key]
        if arr.shape != shape or np.dtype(arr.dtype) != dtype:
            raise ValueError(f"bad {key}: shape={arr.shape}, dtype={arr.dtype}; expected {shape}, {dtype}")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends[-1] != expected_total or np.any(np.diff(ends) <= 0):
        raise ValueError(f"invalid episode boundaries: final={ends[-1]}, total={expected_total}")
    for key in ("data/agent_pos", "data/action"):
        arr = root[key]
        for start in range(0, expected_total, 4096):
            if not np.isfinite(arr[start:start + 4096]).all():
                raise ValueError(f"non-finite values in {key} near row {start}")
    sample_rows = sorted({0, expected_total // 2, expected_total - 1})
    for row in sample_rows:
        image = root["data/image"][row]
        if image.max() == image.min():
            raise ValueError(f"image row {row} is constant and likely corrupt")
    print("post-write verification passed")


def human_bytes(value: int) -> str:
    units: Iterable[str] = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    frame_stride = FRAME_STRIDE
    action_offset = frame_stride
    workers = min(8, os.cpu_count() or 1)

    input_dir = args.input_dir.resolve()
    output_zarr = args.output_zarr.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    arms = selected_arm_names(ARMS)
    image_key = IMAGE_KEYS[IMAGE]
    episodes = discover_episodes(
        input_dir, frame_stride, action_offset, image_key, arms, MAX_EPISODES)
    total = sum(ep.sample_length for ep in episodes)
    state_dim = len(arms) * POSE_DIM_PER_ARM
    raw_image_bytes = total * HEIGHT * WIDTH * 3

    print("\nconversion plan")
    print(f"  input:          {input_dir}")
    print(f"  output:         {output_zarr}")
    print(f"  episodes:       {len(episodes)}")
    print(f"  samples:        {total}")
    print(f"  frame stride:   {frame_stride} ({POLICY_RATE_HZ:g} Hz policy rate)")
    print(f"  action offset:  {action_offset} raw frames")
    print(f"  image:          {image_key}, RGB {HEIGHT}x{WIDTH}")
    print(f"  arms:           {', '.join(arms)}")
    print(f"  state/action:   {state_dim}D EE pose [pos(3)+rot6d(6)+grip(1)] per arm; absolute action")
    print(f"  raw image size: {human_bytes(raw_image_bytes)} before Zarr compression")

    convert(
        episodes=episodes,
        output_zarr=output_zarr,
        image_key=image_key,
        arms=arms,
        action_offset=action_offset,
        width=WIDTH,
        height=HEIGHT,
        workers=workers,
        overwrite=args.overwrite,
    )
    print(f"\nwrote Diffusion Policy replay buffer: {output_zarr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
