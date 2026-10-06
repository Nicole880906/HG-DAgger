"""End to end: a recorded session becomes a trainable Zarr.

Runs the three real programs back to back on a synthetic session --
``RunRecorder`` -> ``dagger/merge_interventions.py`` ->
``data_processing/convert_drawing_6d_abs.py`` -- and checks the Zarr that comes
out.  Everything else in this suite tests one file; this is the test that the
schema the recorder writes is the schema the converter reads, which is the
seam most likely to drift and least likely to complain when it does.

Needs zarr, torch and pytorch3d, so it runs inside the training container:

    bash start_container.sh -- bash -lc \\
        '/opt/conda/envs/robodiff/bin/python -m pytest tests/ -q'

and is left out of collection elsewhere -- see tests/conftest.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import zarr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from run_recorder import EXPERT, POLICY, RunRecorder, build_states  # noqa: E402

# One arm's [pos(3), rot6d(6), grip(1)]; two arms concatenated is what the
# deploy node expects a checkpoint to predict.
POSE_DIM_PER_ARM = 10
OBS_DIM = 2 * POSE_DIM_PER_ARM
FRAME_STRIDE = 6


def unit_quat_wxyz(angle: float, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def synth_run(run_dir: Path, modes: list[str], seed: int = 0) -> Path:
    """A plausible session: both arms drift along a path, the pedal goes down."""
    rng = np.random.default_rng(seed)
    recorder = RunRecorder(run_dir, task="circle")
    for i, mode in enumerate(modes):
        recorder.mark_mode(mode)
        phase = 2 * np.pi * i / len(modes)
        recorder.add_frame(
            rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8),
            build_states(
                cutter_pos=np.array([0.05 * np.cos(phase), 0.05 * np.sin(phase), 0.1]),
                cutter_quat_wxyz=unit_quat_wxyz(phase / 4),
                cutter_jaw=0.3,
                retract_pos=np.array([0.02, 0.03, 0.09 + 0.001 * i]),
                retract_quat_wxyz=unit_quat_wxyz(-phase / 5, axis=(0.0, 1.0, 0.0)),
                retract_jaw=-0.1,
                cutter_qpos=np.arange(6.0),
                retract_qpos=np.arange(6.0),
            ),
            t=i / 30.0,
        )
    recorder.close()
    return run_dir


def run_tool(script: str, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *map(str, args)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"{script} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """One session -> extracted corrections -> Zarr, shared by the checks below."""
    tmp = tmp_path_factory.mktemp("pipeline")
    modes = ([POLICY] * 60) + ([EXPERT] * 60) + ([POLICY] * 60) + ([EXPERT] * 40) + ([POLICY] * 30)
    synth_run(tmp / "runs" / "run_0000", modes)

    dataset = tmp / "dagger"
    run_tool("dagger/merge_interventions.py", tmp / "runs" / "run_0000", "--out", dataset)

    out_zarr = tmp / "circle.zarr"
    run_tool("data_processing/convert_drawing_6d_abs.py", dataset, out_zarr)
    return dataset, zarr.open_group(str(out_zarr), mode="r"), modes


class TestMergeProducesEpisodes:
    def test_one_episode_per_intervention(self, converted):
        dataset, _, _ = converted
        episodes = sorted(p.name for p in dataset.glob("episode_*") if p.is_dir())
        assert episodes == ["episode_0000", "episode_0001"]


class TestConvertedZarr:
    def test_has_the_arrays_the_training_script_reads(self, converted):
        _, root, _ = converted
        for key in ("data/image", "data/agent_pos", "data/action", "meta/episode_ends"):
            assert key in root

    def test_state_and_action_are_20d(self, converted):
        """Both arms, 10D each. The deploy node refuses any other width, so a
        mismatch here would only surface at the robot."""
        _, root, _ = converted
        assert root["data/agent_pos"].shape[1] == OBS_DIM
        assert root["data/action"].shape[1] == OBS_DIM

    def test_images_are_the_policy_resolution(self, converted):
        _, root, _ = converted
        assert root["data/image"].shape[1:] == (120, 160, 3)
        assert root["data/image"].dtype == np.uint8

    def test_policy_rate_metadata_matches_the_frame_stride(self, converted):
        """Deployment must consume each learned waypoint at the Zarr sample rate."""
        _, root, _ = converted
        assert root.attrs["record_rate_hz"] == pytest.approx(30.0)
        assert root.attrs["sample_rate_hz"] == pytest.approx(30.0 / FRAME_STRIDE)
        assert root.attrs["action_offset_seconds"] == pytest.approx(1.0 / (30.0 / FRAME_STRIDE))

    def test_row_count_matches_the_stride(self, converted):
        """Every 6th frame of each extracted episode, and nothing else."""
        dataset, root, _ = converted
        import json

        expected = 0
        for episode in sorted(p for p in dataset.glob("episode_*") if p.is_dir()):
            n = len(json.loads((episode / "data.json").read_text())["data"])
            expected += len(range(0, n, FRAME_STRIDE))
        assert root["data/agent_pos"].shape[0] == expected

    def test_episode_boundaries_are_monotonic_and_complete(self, converted):
        _, root, _ = converted
        ends = np.asarray(root["meta/episode_ends"][:])
        assert ends.shape == (2,)
        assert np.all(np.diff(ends) > 0)
        assert ends[-1] == root["data/agent_pos"].shape[0]

    def test_everything_is_finite(self, converted):
        _, root, _ = converted
        assert np.isfinite(root["data/agent_pos"][:]).all()
        assert np.isfinite(root["data/action"][:]).all()

    def test_action_is_absolute_not_a_delta(self, converted):
        """``action[t]`` is the *state* one step ahead, so within an episode it
        equals ``agent_pos[t+1]``.  The deploy node commands the predicted row
        directly; if this were a delta, every motion would be doubled."""
        _, root, _ = converted
        agent = np.asarray(root["data/agent_pos"][:])
        action = np.asarray(root["data/action"][:])
        ends = np.asarray(root["meta/episode_ends"][:])
        start = 0
        for end in ends:
            # The final row's action is clamped at the episode end, so stop short.
            assert np.allclose(action[start:end - 1], agent[start + 1:end], atol=1e-5)
            start = end

    def test_rotation_block_is_a_valid_rotation(self, converted):
        """rot6d columns must Gram-Schmidt back to an orthonormal matrix -- the
        deploy node turns them straight into a commanded orientation."""
        sys.path.insert(0, str(REPO_ROOT / "deploy"))
        from geometry import rot6d_to_quat_xyzw
        from scipy.spatial.transform import Rotation as R

        _, root, _ = converted
        agent = np.asarray(root["data/agent_pos"][:])
        for row in (0, len(agent) // 2, len(agent) - 1):
            for base in (0, POSE_DIM_PER_ARM):
                quat = rot6d_to_quat_xyzw(agent[row, base + 3:base + 9])
                mat = R.from_quat(quat).as_matrix()
                assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-4)

    def test_gripper_values_survive_the_round_trip(self, converted):
        _, root, _ = converted
        agent = np.asarray(root["data/agent_pos"][:])
        assert np.allclose(agent[:, POSE_DIM_PER_ARM - 1], 0.3, atol=1e-5)
        assert np.allclose(agent[:, 2 * POSE_DIM_PER_ARM - 1], -0.1, atol=1e-5)

    def test_positions_land_in_the_right_arm_slots(self, converted):
        """Cutter (PSM2) occupies 0:10 and retraction (PSM1) 10:20, matching the
        slices the deploy node uses to address each controller."""
        _, root, _ = converted
        agent = np.asarray(root["data/agent_pos"][:])
        cutter_radius = np.linalg.norm(agent[:, 0:2], axis=1)
        assert np.allclose(cutter_radius, 0.05, atol=1e-4)
        assert np.allclose(agent[:, POSE_DIM_PER_ARM + 0], 0.02, atol=1e-5)
        assert np.allclose(agent[:, POSE_DIM_PER_ARM + 1], 0.03, atol=1e-5)
