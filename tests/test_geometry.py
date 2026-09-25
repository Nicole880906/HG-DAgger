"""Rotation conventions and the motion clamp.

These run anywhere numpy and scipy are installed -- no ROS, no torch, no
checkpoint -- which is the point of keeping ``deploy/geometry.py`` separate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from geometry import (  # noqa: E402
    IDENTITY_ROT6D,
    CartesianGate,
    OrientationGate,
    angle_between_quats,
    matrix_to_rot6d,
    rot6d_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_rot6d,
    quat_xyzw_to_wxyz,
    rot6d_to_quat_xyzw,
)


def random_quats_xyzw(n: int, seed: int = 0) -> np.ndarray:
    return R.random(n, random_state=seed).as_quat()


def test_quat_rot6d_round_trip_preserves_rotation():
    """quat -> rot6d -> quat must be the same *rotation*, not the same 4 numbers.

    q and -q are the same rotation, so compare the matrices.
    """
    for quat in random_quats_xyzw(64):
        back = rot6d_to_quat_xyzw(quat_xyzw_to_rot6d(quat))
        assert np.allclose(R.from_quat(quat).as_matrix(),
                           R.from_quat(back).as_matrix(), atol=1e-5)


def test_rot6d_matches_pytorch3d_definition():
    """rot6d is the first two ROWS of the matrix, flattened.

    The converter produces it with pytorch3d and the deploy node consumes it
    with this code; if one of them ever used columns instead, the tool would be
    commanded into a transposed orientation.
    """
    for quat in random_quats_xyzw(16, seed=3):
        mat = R.from_quat(quat).as_matrix()
        assert np.allclose(quat_xyzw_to_rot6d(quat), mat[:2, :].reshape(6), atol=1e-6)


def test_rot6d_to_quat_orthonormalises_a_degenerate_input():
    """A diffusion sample's 6 numbers are not guaranteed orthonormal.

    Gram-Schmidt has to fix them up rather than emit a non-rotation, because the
    result is published straight to the arm.
    """
    sloppy = np.array([2.0, 0.0, 0.0, 0.7, 1.3, 0.0])
    quat = rot6d_to_quat_xyzw(sloppy)
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
    mat = R.from_quat(quat).as_matrix()
    assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-5)


def test_quaternion_order_helpers_are_inverses():
    for quat in random_quats_xyzw(16, seed=7):
        assert np.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(quat)), quat, atol=1e-12)


def test_wxyz_reorder_moves_the_real_part_to_the_front():
    xyzw = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(quat_xyzw_to_wxyz(xyzw), [0.9, 0.1, 0.2, 0.3])


class TestCartesianGate:
    def test_small_step_passes_through_unchanged(self):
        gate = CartesianGate(0.005)
        current = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        target = current + np.array([0.001, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(gate.clamp_pos(target, current), target, atol=1e-7)

    def test_big_step_is_clamped_to_the_limit_along_the_same_direction(self):
        gate = CartesianGate(0.005)
        current = np.zeros(3, dtype=np.float32)
        target = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        out = gate.clamp_pos(target, current)
        assert np.isclose(np.linalg.norm(out - current), 0.005, atol=1e-6)
        assert np.allclose(out / np.linalg.norm(out), [1.0, 0.0, 0.0], atol=1e-6)

    def test_clamp_is_on_the_vector_not_per_axis(self):
        """A diagonal command must not travel sqrt(3) * max_pos_step."""
        gate = CartesianGate(0.005)
        current = np.zeros(3, dtype=np.float32)
        target = np.full(3, 0.1, dtype=np.float32)
        out = gate.clamp_pos(target, current)
        assert np.isclose(np.linalg.norm(out), 0.005, atol=1e-6)

    def test_zero_delta_is_stable(self):
        gate = CartesianGate(0.005)
        current = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert np.allclose(gate.clamp_pos(current, current), current, atol=1e-7)

    @pytest.mark.parametrize("bad", [0.0, -0.001, float("nan"), float("inf")])
    def test_nonsense_limit_is_rejected_at_construction(self, bad):
        """Better to fail at startup than to silently disable the clamp."""
        with pytest.raises(ValueError):
            CartesianGate(bad)


class TestRot6dMatrix:
    def test_matrix_round_trip(self):
        for quat in random_quats_xyzw(32, seed=11):
            mat = R.from_quat(quat).as_matrix()
            assert np.allclose(rot6d_to_matrix(matrix_to_rot6d(mat)), mat, atol=1e-6)

    def test_identity_rot6d_is_not_zeros(self):
        """A relative action's rotation block is a DELTA rotation, and "no
        rotation" is [1,0,0,0,1,0]. Zeros are a degenerate non-rotation, and
        Gram-Schmidt would turn them into something arbitrary."""
        assert not np.allclose(IDENTITY_ROT6D, 0.0)
        assert np.allclose(rot6d_to_matrix(IDENTITY_ROT6D), np.eye(3), atol=1e-9)


class TestOrientationGate:
    def test_small_rotation_passes_through(self):
        gate = OrientationGate(np.deg2rad(2.0))
        current = np.eye(3)
        target = R.from_euler("z", 1.0, degrees=True).as_matrix()
        assert np.allclose(gate.clamp_matrix(target, current), target, atol=1e-9)

    def test_large_rotation_is_clamped_to_the_limit(self):
        gate = OrientationGate(np.deg2rad(2.0))
        current = np.eye(3)
        target = R.from_euler("z", 90.0, degrees=True).as_matrix()
        out = gate.clamp_matrix(target, current)
        swing = np.rad2deg(np.linalg.norm(R.from_matrix(out @ current.T).as_rotvec()))
        assert swing == pytest.approx(2.0, abs=1e-6)

    def test_the_clamped_step_keeps_the_original_axis(self):
        """Shortening the geodesic, not picking a different direction."""
        gate = OrientationGate(np.deg2rad(2.0))
        current = R.from_euler("x", 20, degrees=True).as_matrix()
        target = R.from_euler("y", 80, degrees=True).as_matrix()
        full = R.from_matrix(target @ current.T).as_rotvec()
        out = gate.clamp_matrix(target, current)
        step = R.from_matrix(out @ current.T).as_rotvec()
        cos = np.dot(full, step) / (np.linalg.norm(full) * np.linalg.norm(step))
        assert cos == pytest.approx(1.0, abs=1e-6)

    def test_output_is_always_a_valid_rotation(self):
        gate = OrientationGate(np.deg2rad(2.0))
        for quat in random_quats_xyzw(16, seed=5):
            out = gate.clamp_matrix(R.from_quat(quat).as_matrix(), np.eye(3))
            assert np.allclose(out @ out.T, np.eye(3), atol=1e-6)
            assert np.isclose(np.linalg.det(out), 1.0, atol=1e-6)

    def test_identical_orientations_are_stable(self):
        gate = OrientationGate(np.deg2rad(2.0))
        current = R.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        assert np.allclose(gate.clamp_matrix(current, current), current, atol=1e-9)

    def test_quaternion_interface_agrees_with_the_matrix_one(self):
        gate = OrientationGate(np.deg2rad(3.0))
        cur_q = R.from_euler("z", 10, degrees=True).as_quat()
        tgt_q = R.from_euler("z", 100, degrees=True).as_quat()
        out_q = gate.clamp_quat_xyzw(tgt_q, cur_q)
        out_m = gate.clamp_matrix(R.from_quat(tgt_q).as_matrix(),
                                  R.from_quat(cur_q).as_matrix())
        assert np.allclose(R.from_quat(out_q).as_matrix(), out_m, atol=1e-6)

    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
    def test_nonsense_limit_is_rejected(self, bad):
        with pytest.raises(ValueError):
            OrientationGate(bad)


class TestAngleBetweenQuats:
    def test_identical_orientations_are_zero(self):
        q = R.random(random_state=2).as_quat()
        assert angle_between_quats(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_known_angle(self):
        a = R.from_euler("z", 0, degrees=True).as_quat()
        b = R.from_euler("z", 30, degrees=True).as_quat()
        assert np.rad2deg(angle_between_quats(a, b)) == pytest.approx(30.0, abs=1e-6)

    def test_is_symmetric(self):
        a, b = R.random(2, random_state=4).as_quat()
        assert angle_between_quats(a, b) == pytest.approx(angle_between_quats(b, a), abs=1e-9)
