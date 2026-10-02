from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

import replay  # noqa: E402


def write_zarr(path: Path, *, representation: str = "absolute_next_state") -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["action_representation"] = representation
    root.attrs["arms"] = ["cutter", "retraction"]
    root.create_dataset("meta/episode_ends", data=np.array([2, 3], dtype=np.int64))
    action = np.arange(60, dtype=np.float32).reshape(3, 20)
    root.create_dataset("data/action", data=action)
    root.create_dataset("data/agent_pos", data=action + 1)


def test_load_episode_uses_episode_boundaries(tmp_path):
    path = tmp_path / "episodes.zarr"
    write_zarr(path)

    action, state = replay.load_episode(path, 1)

    assert action.shape == (1, 20)
    assert action[0, 0] == 40
    assert state[0, 0] == 41


def test_load_episode_rejects_wrong_action_representation(tmp_path):
    path = tmp_path / "joint.zarr"
    write_zarr(path, representation="joint_absolute")

    with pytest.raises(SystemExit, match="absolute_next_state"):
        replay.load_episode(path, 0)


def test_trajectory_stats_reports_both_arms():
    action = np.zeros((2, 20), dtype=np.float64)
    action[1, replay.CUTTER.start] = 0.01
    action[1, replay.RETRACT.start] = 0.02

    lines = replay.trajectory_stats(action, action.copy(), rate=5.0)

    assert any("cutter (PSM2)" in line for line in lines)
    assert any("retraction (PSM1)" in line for line in lines)
