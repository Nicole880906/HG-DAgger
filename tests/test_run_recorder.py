"""The session recorder: does a deploy run produce data the converter can read?

The recorder writes the demonstration collector's schema so corrections merge
back through the same converter.  These tests pin the parts of that schema the
converter actually reads -- get one key or one quaternion order wrong and the
data still looks fine, trains fine, and produces a policy that points the tool
somewhere else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from run_recorder import (  # noqa: E402
    ALIGN_TO_EXPERT,
    ALIGN_TO_POLICY,
    EXPERT,
    PHASES,
    POLICY,
    RunRecorder,
    build_states,
    next_run_dir,
)

# The keys convert_drawing_6d_abs.py reads out of every frame.
EE_GROUPS = {"cutter": "psm_cutter_ee", "retraction": "psm_retraction_ee"}
EE_POS_KEYS = {"cutter": "psm_cutter_pos", "retraction": "psm_retraction_pos"}
EE_QUAT_KEYS = {"cutter": "psm_cutter_quat", "retraction": "psm_retraction_quat"}
JS_GROUPS = {"cutter": "psm_cutter_js", "retraction": "psm_retraction_js"}


def sample_states(**overrides):
    kwargs = dict(
        cutter_pos=np.array([0.01, 0.02, 0.03]),
        cutter_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        cutter_jaw=0.5,
        retract_pos=np.array([0.04, 0.05, 0.06]),
        retract_quat_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        retract_jaw=-0.2,
    )
    kwargs.update(overrides)
    return build_states(**kwargs)


def frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


class TestBuildStates:
    def test_has_every_group_the_converter_reads(self):
        states = sample_states()
        for arm in ("cutter", "retraction"):
            assert EE_GROUPS[arm] in states
            assert EE_POS_KEYS[arm] in states[EE_GROUPS[arm]]
            assert EE_QUAT_KEYS[arm] in states[EE_GROUPS[arm]]
            assert "gripper" in states[JS_GROUPS[arm]]

    def test_position_and_quaternion_have_the_expected_widths(self):
        states = sample_states()
        for arm in ("cutter", "retraction"):
            assert len(states[EE_GROUPS[arm]][EE_POS_KEYS[arm]]) == 3
            assert len(states[EE_GROUPS[arm]][EE_QUAT_KEYS[arm]]) == 4

    def test_quaternion_is_stored_real_part_first(self):
        """wxyz, because the converter feeds it to pytorch3d.quaternion_to_matrix.

        An xyzw quaternion here yields a valid-looking but wrong rotation, with
        no exception anywhere to notice it.
        """
        states = sample_states(cutter_quat_wxyz=np.array([0.9, 0.1, 0.2, 0.3]))
        assert states["psm_cutter_ee"]["psm_cutter_quat"][0] == pytest.approx(0.9)

    def test_cutter_is_psm2_and_retraction_is_psm1(self):
        """Arm order is the converter's concatenation order; swapping it trains
        each arm on the other's trajectory."""
        states = sample_states(cutter_pos=np.array([1.0, 2.0, 3.0]),
                               retract_pos=np.array([4.0, 5.0, 6.0]))
        assert states["psm_cutter_ee"]["psm_cutter_pos"] == [1.0, 2.0, 3.0]
        assert states["psm_retraction_ee"]["psm_retraction_pos"] == [4.0, 5.0, 6.0]

    def test_joint_positions_are_carried_when_present(self):
        states = sample_states(cutter_qpos=np.arange(6.0), retract_qpos=np.arange(6.0) * 2)
        assert states["psm_cutter_js"]["qpos"] == list(np.arange(6.0))
        assert states["psm_retraction_js"]["qpos"] == list(np.arange(6.0) * 2)

    def test_missing_joint_positions_degrade_to_an_empty_list(self):
        """measured_js may not have arrived yet; the EE path does not need it,
        so a frame must still be recordable."""
        states = sample_states()
        assert states["psm_cutter_js"]["qpos"] == []

    def test_everything_is_json_serialisable(self):
        """numpy floats survive a round trip only if converted; they are not
        JSON types, and the failure would come mid-run at the first flush."""
        json.dumps(sample_states(cutter_qpos=np.arange(6.0)))


class TestRecording:
    def test_frames_land_as_images_plus_json(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        for i in range(3):
            recorder.add_frame(frame(i), sample_states(), t=i / 30.0)
        recorder.close()

        payload = json.loads((tmp_path / "run_0000" / "data.json").read_text())
        assert len(payload["data"]) == 3
        for i, record in enumerate(payload["data"]):
            assert record["idx"] == i
            rel = record["colors"]["left_image"]
            assert rel == f"colors/left_image_{i:06d}.jpg"
            assert (tmp_path / "run_0000" / rel).is_file()

    def test_mode_label_applies_from_the_moment_it_is_set(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        recorder.add_frame(frame(0), sample_states(), t=0.0)
        recorder.mark_mode(EXPERT)
        recorder.add_frame(frame(1), sample_states(), t=1.0)
        recorder.add_frame(frame(2), sample_states(), t=2.0)
        recorder.mark_mode(POLICY)
        recorder.add_frame(frame(3), sample_states(), t=3.0)
        counts = recorder.close()

        modes = [r["control_mode"]
                 for r in json.loads((tmp_path / "run_0000" / "data.json").read_text())["data"]]
        assert modes == [POLICY, EXPERT, EXPERT, POLICY]
        assert counts[POLICY] == 2 and counts[EXPERT] == 2
        assert set(counts) == set(PHASES)

    def test_default_mode_is_policy(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        assert recorder.mode == POLICY

    def test_unknown_mode_is_rejected(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        with pytest.raises(ValueError):
            recorder.mark_mode("human")

    def test_both_alignment_windows_are_recordable_phases(self, tmp_path):
        """The handover holds the arm for five seconds each way and those frames
        are recorded too -- they are exactly what you read when auditing a
        handover, even though they are training-worthless."""
        recorder = RunRecorder(tmp_path / "run_0000")
        for phase in (POLICY, ALIGN_TO_EXPERT, EXPERT, ALIGN_TO_POLICY):
            recorder.mark_mode(phase)
            recorder.add_frame(frame(0), sample_states(), t=0.0)
        recorder.close()
        modes = [r["control_mode"]
                 for r in json.loads((tmp_path / "run_0000" / "data.json").read_text())["data"]]
        assert modes == [POLICY, ALIGN_TO_EXPERT, EXPERT, ALIGN_TO_POLICY]

    def test_image_bytes_round_trip(self, tmp_path):
        """JPEG is lossy, but the frame must at least come back the right shape
        and not be blank -- a silently empty image trains a blind policy."""
        import cv2

        recorder = RunRecorder(tmp_path / "run_0000")
        original = frame(7)
        recorder.add_frame(original, sample_states(), t=0.0)
        recorder.close()
        decoded = cv2.imread(str(tmp_path / "run_0000" / "colors" / "left_image_000000.jpg"))
        assert decoded is not None
        assert decoded.shape == original.shape
        assert decoded.max() > decoded.min()

    def test_extra_fields_are_merged_into_the_record(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        recorder.add_frame(frame(0), sample_states(), t=0.0, extra={"cycle": 42})
        recorder.close()
        record = json.loads((tmp_path / "run_0000" / "data.json").read_text())["data"][0]
        assert record["cycle"] == 42


class TestDurability:
    def test_json_is_readable_mid_run_without_closing(self, tmp_path):
        """A session killed during an intervention must still leave usable data."""
        recorder = RunRecorder(tmp_path / "run_0000", flush_every=2)
        recorder.mark_mode(EXPERT)
        for i in range(4):
            recorder.add_frame(frame(i), sample_states(), t=i / 30.0)
        # No close() -- simulate a hard stop.
        payload = json.loads((tmp_path / "run_0000" / "data.json").read_text())
        assert len(payload["data"]) == 4
        assert all(r["control_mode"] == EXPERT for r in payload["data"])

    def test_flush_leaves_no_temporary_file_behind(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        recorder.add_frame(frame(0), sample_states(), t=0.0)
        recorder.close()
        assert not (tmp_path / "run_0000" / "data.json.tmp").exists()

    def test_summary_counts_match_the_written_frames(self, tmp_path):
        recorder = RunRecorder(tmp_path / "run_0000")
        recorder.add_frame(frame(0), sample_states(), t=0.0)
        recorder.mark_mode(EXPERT)
        recorder.add_frame(frame(1), sample_states(), t=1.0)
        recorder.close()
        payload = json.loads((tmp_path / "run_0000" / "data.json").read_text())
        assert payload["run"]["frame_counts"][POLICY] == 1
        assert payload["run"]["frame_counts"][EXPERT] == 1
        assert payload["run"]["n_frames"] == 2
        assert recorder.n_frames == 2


class TestNextRunDir:
    def test_first_run_is_zero(self, tmp_path):
        assert next_run_dir(tmp_path).name == "run_0000"

    def test_numbering_continues_past_existing_runs(self, tmp_path):
        (tmp_path / "run_0000").mkdir()
        (tmp_path / "run_0004").mkdir()
        assert next_run_dir(tmp_path).name == "run_0005"

    def test_non_numeric_neighbours_are_ignored(self, tmp_path):
        (tmp_path / "run_notes").mkdir()
        (tmp_path / "run_0001").mkdir()
        assert next_run_dir(tmp_path).name == "run_0002"

    def test_missing_root_is_created(self, tmp_path):
        root = tmp_path / "deep" / "nested"
        assert next_run_dir(root).name == "run_0000"
        assert root.is_dir()
