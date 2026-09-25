#!/usr/bin/env python3
"""Rotation conversions and the per-command motion clamp.

Kept apart from ``deploy_with_intervention.py`` so it imports with nothing but
numpy and scipy: the deploy node pulls in torch, hydra and the whole
diffusion_policy tree, which exist only in the training container, and this is
the part worth testing on any machine.  ``tests/test_geometry.py`` does.

Conventions, which three different files have to agree on:

* the **policy** emits a 6D rotation, the first two rows of the rotation matrix
  flattened (``pytorch3d.transforms.matrix_to_rotation_6d``);
* **ROS** and scipy use ``xyzw`` quaternions;
* the **recorded JSON** and the converter use ``wxyz`` quaternions, because the
  converter hands them to ``pytorch3d.quaternion_to_matrix``, which is real-part
  first.

Mixing the last two produces a rotation that is wrong but perfectly
well-formed -- no exception, no NaN, just a tool pointing somewhere else.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation -> 3x3 rotation matrix.

    Reproduces ``pytorch3d.transforms.rotation_6d_to_matrix`` in numpy:
    Gram-Schmidt on the two 3-vectors that form the first two rows, third row
    from their cross product.  Any six finite numbers come back as a valid
    rotation, which is the whole point of the representation -- the policy's
    output needs no normalisation layer and cannot be invalid.
    """
    d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = d[:3], d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> 6D rotation (first two ROWS, flattened).

    Rows, not columns: ``pytorch3d.transforms.matrix_to_rotation_6d`` is
    ``matrix[..., :2, :]``.  Using columns would train and deploy a consistently
    transposed orientation, which is not an error anywhere -- just a tool
    pointing somewhere else.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return m[:2, :].reshape(6).astype(np.float32)


IDENTITY_ROT6D = matrix_to_rot6d(np.eye(3))


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation -> ``xyzw`` quaternion.

    scipy and pytorch3d share the rotation-matrix convention, so this
    round-trips the converter exactly.
    """
    quat = R.from_matrix(rot6d_to_matrix(rot6d)).as_quat().astype(np.float32)
    return (quat / (np.linalg.norm(quat) + 1e-9)).astype(np.float32)


def quat_xyzw_to_rot6d(quat_xyzw) -> np.ndarray:
    """Measured ``xyzw`` quaternion -> 6D rotation (first two matrix rows)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-9)
    return R.from_quat(q).as_matrix()[:2, :].reshape(6).astype(np.float32)


def quat_xyzw_to_wxyz(quat_xyzw) -> np.ndarray:
    """ROS ``xyzw`` -> the ``wxyz`` the recorder and converter store."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(quat_wxyz) -> np.ndarray:
    """``wxyz`` -> ROS ``xyzw``."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class CartesianGate:
    """Clamp a commanded EE position to a bounded step from the measured one.

    The clamp is on the *vector*, not per axis, so a diagonal command is limited
    to ``max_pos_step`` of travel rather than ``sqrt(3)`` times it.

    This is the last thing between a diffusion sample and the arm.  The policy
    predicts an absolute pose, and an out-of-distribution observation can put
    that pose a long way from where the tool actually is; clamping turns what
    would be a lunge into a slow drift in the wrong direction, which is
    something a human at the console has time to react to.
    """

    def __init__(self, max_pos_step: float) -> None:
        if not np.isfinite(max_pos_step) or max_pos_step <= 0:
            raise ValueError(f"max_pos_step must be finite and positive, got {max_pos_step}")
        self.max_pos_step = float(max_pos_step)

    def clamp_pos(self, target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        target = np.asarray(target_pos, dtype=np.float32).reshape(3)
        current = np.asarray(current_pos, dtype=np.float32).reshape(3)
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist > self.max_pos_step and dist > 1e-9:
            delta = delta * (self.max_pos_step / dist)
        return (current + delta).astype(np.float32)


class OrientationGate:
    """Clamp a commanded orientation to a bounded angular step.

    The position counterpart to this, ``CartesianGate``, existed from the start;
    this did not, and its absence was a real hole.  ``_command_pose`` clamped the
    commanded position against the measured one and then passed the policy's
    quaternion straight through, so on the first command after an intervention
    the tool tip crept a few millimetres while the wrist was free to snap to any
    orientation the policy asked for.  A surgeon who has just repositioned the
    arm is exactly who that would surprise.

    The clamp is a geodesic step on SO(3): take the rotation that would carry
    the current orientation onto the target, and if it turns by more than
    ``max_angle``, keep its axis and shorten it.  That is the rotational
    equivalent of scaling a position delta, and it composes with the position
    clamp to bound the whole pose step.
    """

    def __init__(self, max_angle_rad: float) -> None:
        if not np.isfinite(max_angle_rad) or max_angle_rad <= 0:
            raise ValueError(
                f"max_angle_rad must be finite and positive, got {max_angle_rad}")
        self.max_angle_rad = float(max_angle_rad)

    def clamp_matrix(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Bounded step from ``current`` toward ``target``, both 3x3."""
        target = np.asarray(target, dtype=np.float64).reshape(3, 3)
        current = np.asarray(current, dtype=np.float64).reshape(3, 3)
        delta = target @ current.T
        rotvec = R.from_matrix(delta).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle <= self.max_angle_rad or angle < 1e-12:
            return target
        return R.from_rotvec(rotvec * (self.max_angle_rad / angle)).as_matrix() @ current

    def clamp_quat_xyzw(self, target_xyzw, current_xyzw) -> np.ndarray:
        """Same clamp, in and out as ``xyzw`` quaternions."""
        target = R.from_quat(np.asarray(target_xyzw, dtype=np.float64).reshape(4)).as_matrix()
        current = R.from_quat(np.asarray(current_xyzw, dtype=np.float64).reshape(4)).as_matrix()
        quat = R.from_matrix(self.clamp_matrix(target, current)).as_quat().astype(np.float32)
        return (quat / (np.linalg.norm(quat) + 1e-9)).astype(np.float32)


def angle_between_quats(a_xyzw, b_xyzw) -> float:
    """Geodesic angle in radians between two orientations.

    Used to report how far the MTM wrist is from the PSM tool during the
    alignment window.
    """
    a = R.from_quat(np.asarray(a_xyzw, dtype=np.float64).reshape(4))
    b = R.from_quat(np.asarray(b_xyzw, dtype=np.float64).reshape(4))
    return float(np.linalg.norm((a * b.inv()).as_rotvec()))
