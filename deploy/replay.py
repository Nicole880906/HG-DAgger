#!/usr/bin/env python3
"""Replay one absolute-EE episode from a Diffusion Policy Zarr on the dVRK.

The labels are replayed directly: no camera, policy, or inference is involved.
This checks the same base-frame pose, rotation-6D, jaw, and ``servo_cp`` path
used by ``deploy_with_intervention.py``.  A dry run is the default and never
initializes ROS or sends a command.

The Zarr must contain ``action_representation='absolute_next_state'`` and the
20D layout produced by ``convert_drawing_6d_abs.py``: cutter (PSM2) first, then
retraction (PSM1), each ``[position(3), rotation_6d(6), gripper(1)]``.

Examples
--------

    python deploy/replay.py --episode-index 0
    python deploy/replay.py --zarr data/diffusion_policy/task2.zarr --episode-index 0 --execute
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geometry import CartesianGate, OrientationGate, rot6d_to_matrix  # noqa: E402


DEFAULT_ZARR = REPO_ROOT / "data" / "diffusion_policy" / "task2.zarr"
ACTION_REPRESENTATION = "absolute_next_state"
POSE_DIM_PER_ARM = 10
OBS_DIM = 20
POS = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9
CUTTER = slice(0, 10)
RETRACT = slice(10, 20)
SETTLE_S = 4.0
FEEDBACK_TIMEOUT_S = 15.0


def load_episode(path: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one ``(action, agent_pos)`` episode and enforce its replay contract."""
    zarr_path = path.expanduser().resolve()
    if not zarr_path.is_dir():
        raise SystemExit(f"Zarr directory not found: {zarr_path}")
    root = zarr.open_group(str(zarr_path), mode="r")
    representation = root.attrs.get("action_representation")
    if representation != ACTION_REPRESENTATION:
        raise SystemExit(
            f"expected action_representation={ACTION_REPRESENTATION!r} in {zarr_path}, "
            f"got {representation!r}; this replay only supports absolute EE actions"
        )
    if tuple(root.attrs.get("arms", ())) != ("cutter", "retraction"):
        raise SystemExit(
            f"expected cutter/retraction arm order in {zarr_path}, got {root.attrs.get('arms')!r}"
        )

    try:
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        actions = root["data/action"]
        states = root["data/agent_pos"]
    except KeyError as exc:
        raise SystemExit(f"missing replay array in {zarr_path}: {exc}") from exc
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
        raise SystemExit(f"invalid meta/episode_ends in {zarr_path}")
    if not 0 <= episode_index < len(ends):
        raise SystemExit(f"episode-index {episode_index} out of range [0, {len(ends)})")

    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    action = np.asarray(actions[start:end], dtype=np.float64)
    agent_pos = np.asarray(states[start:end], dtype=np.float64)
    expected = (end - start, OBS_DIM)
    if action.shape != expected or agent_pos.shape != expected:
        raise SystemExit(
            f"expected action and agent_pos shape {expected}, got {action.shape} and {agent_pos.shape}"
        )
    if not np.isfinite(action).all() or not np.isfinite(agent_pos).all():
        raise SystemExit("episode contains non-finite action or agent_pos values")
    return action, agent_pos


def trajectory_stats(action: np.ndarray, agent_pos: np.ndarray, rate: float) -> list[str]:
    """Return concise dry-run stats for visual inspection before robot motion."""
    lines = [f"episode: {len(action)} steps  (~{len(action) / rate:.1f}s @ {rate:.1f} Hz)"]
    for name, arm in (("cutter (PSM2)", CUTTER), ("retraction (PSM1)", RETRACT)):
        pos = action[:, arm][:, POS]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(1)
        jaw = action[:, arm][:, GRIP]
        lines.append(
            f"  {name}: position step max={step.max() * 1000:.2f} mm "
            f"mean={step.mean() * 1000:.2f} mm | jaw [{jaw.min():+.3f}, {jaw.max():+.3f}] rad"
        )
    for name, arm in (("cutter", CUTTER), ("retraction", RETRACT)):
        jump = np.linalg.norm(action[0, arm][POS] - agent_pos[0, arm][POS])
        lines.append(f"  {name} first target from recorded start: {jump * 1000:.2f} mm")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR, help="absolute-EE source Zarr")
    parser.add_argument("--episode-index", type=int, default=0, help="zero-based episode index")
    parser.add_argument("--rate", type=float, default=5.0, help="replay rate in Hz; dataset rate is 5 Hz")
    parser.add_argument("--max-pos-step", type=float, default=0.005,
                        help="maximum streamed position change in metres")
    parser.add_argument("--max-angle-step-deg", type=float, default=3.0,
                        help="maximum streamed orientation change in degrees")
    parser.add_argument("--max-initial-jump", type=float, default=0.04,
                        help="abort if a first target is farther than this from a live arm, in metres")
    parser.add_argument("--execute", action="store_true",
                        help="command the robot; default is a dry run with no ROS initialization")
    args = parser.parse_args(argv)
    for name in ("rate", "max_pos_step", "max_angle_step_deg", "max_initial_jump"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    action, agent_pos = load_episode(args.zarr, args.episode_index)
    print(f"Zarr: {args.zarr.expanduser().resolve()}")
    for line in trajectory_stats(action, agent_pos, args.rate):
        print(line)
    if not args.execute:
        print("dry run complete; pass --execute to command the robot")
        return

    # ROS imports stay here so data inspection works in the training container.
    import rclpy
    from rclpy.node import Node

    from controller import PSM1Controller, PSM2Controller

    rclpy.init()
    node = Node("replay_absolute_ee")
    cutter = PSM2Controller(node, ns="PSM2", frame_id="PSM2_base")
    retraction = PSM1Controller(node, ns="PSM1", frame_id="PSM1_base")
    position_gate = CartesianGate(max_pos_step=args.max_pos_step)
    orientation_gate = OrientationGate(max_angle_rad=np.deg2rad(args.max_angle_step_deg))
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    def ready(controller) -> bool:
        return controller.current_jaw() is not None and bool(np.any(controller.current_pos()))

    try:
        node.get_logger().info("waiting for measured_cp and jaw feedback on both PSMs")
        deadline = time.monotonic() + FEEDBACK_TIMEOUT_S
        while rclpy.ok() and not (ready(cutter) and ready(retraction)):
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for measured_cp + jaw/measured_js on both arms")
            time.sleep(0.1)

        arms = (("cutter", CUTTER, cutter), ("retraction", RETRACT, retraction))
        for name, arm, controller in arms:
            jump = float(np.linalg.norm(action[0, arm][POS] - np.asarray(controller.current_pos())))
            if jump > args.max_initial_jump:
                raise RuntimeError(
                    f"{name} first target is {jump * 1000:.1f} mm away, above the "
                    f"{args.max_initial_jump * 1000:.1f} mm limit; reposition first"
                )

        dt = 1.0 / args.rate

        def command(arm_action: np.ndarray, controller, move_initial: bool) -> None:
            target_pos = np.asarray(arm_action[POS], dtype=np.float64)
            target_rot = rot6d_to_matrix(arm_action[ROT6D])
            target_jaw = float(arm_action[GRIP])
            current_pos = np.asarray(controller.current_pos(), dtype=np.float64)
            current_rot = R.from_quat(np.asarray(controller.current_orientation(), dtype=np.float64)).as_matrix()
            if move_initial:
                pos, rot = target_pos, target_rot
            else:
                pos = position_gate.clamp_pos(target_pos, current_pos)
                rot = orientation_gate.clamp_matrix(target_rot, current_rot)
            quat = R.from_matrix(rot).as_quat()
            quat = tuple(float(value) for value in quat / (np.linalg.norm(quat) + 1e-9))
            if move_initial:
                if controller is cutter:
                    controller.send_pose_psm2(pos, quat=quat)
                else:
                    controller.send_pose_psm1(pos, quat=quat)
            else:
                controller.send_servo_pose(pos, quat=quat)
            controller.set_jaw_goal(target_jaw)
            controller.hold_jaw_goal(dt)

        node.get_logger().warn(
            f"REPLAY START: {len(action)} steps at {args.rate:.1f} Hz; "
            f"initial move then {SETTLE_S:.1f}s settle"
        )
        for _, arm, controller in arms:
            command(action[0, arm], controller, move_initial=True)
        time.sleep(SETTLE_S)

        next_tick = time.monotonic()
        for index in range(1, len(action)):
            if not rclpy.ok():
                break
            for _, arm, controller in arms:
                command(action[index, arm], controller, move_initial=False)
            next_tick += dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        node.get_logger().info("replay complete")
    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        raise SystemExit(1) from exc
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
