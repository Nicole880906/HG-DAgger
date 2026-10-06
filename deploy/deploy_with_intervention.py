#!/usr/bin/env python3
"""Run a 20D absolute-EE Diffusion Policy on the dVRK, with pedal handover.

The policy draws the shape.  When the operator sees it wandering out of
distribution they hold the **COAG footpedal**, this node stops commanding the
arms within one control cycle, and dVRK's own MTM->PSM teleoperation drives
instead.  Releasing the pedal hands the arms back to the policy.  Every frame of
the session is recorded with a ``policy`` / ``expert`` label, so the corrections
can be folded back into the training set (``dagger/merge_interventions.py``).

Who drives what
---------------
Handover is by **strict exclusion**, not arbitration.  Two publishers streaming
setpoints at one PSM is not a blend, it is a race, and the arm ends up tracking
whichever message happened to land last.  So exactly one of the two is active at
any instant:

    pedal up   -> this node publishes servo_cp + jaw/servo_jp; dVRK's teleop
                  component is disengaged and publishes nothing
    pedal down -> this node publishes NOTHING at all; dVRK's teleop component
                  owns the arms

This node never talks to the MTMs and never implements teleoperation itself.  It
only gets out of the way.  That means your dVRK console must already have a
teleop pair configured (MTML-PSM1 / MTMR-PSM2 or whichever pairing you use) and
the usual operator-present interlock satisfied -- COAG alone will not move
anything if the console does not think an operator is at the master.  Verify the
pedal drives the arms through the console *before* running this with
``--execute``.

Dead-man semantics
------------------
The pedal is held, not toggled.  A foot coming off the pedal can only ever
return control to the policy, never strand it with the human; and because the
node refuses to start until it has actually seen the pedal topic
(``--pedal-timeout``), a policy can never be commanding the arms in a session
where the takeover path was silently broken.

Resuming
--------
On handback the queued action chunk is dropped and the policy replans from what
it can currently see.  It resumes into ``servo_cp`` streaming, position-clamped
to ``--max-pos-step`` per command against the *measured* pose, and only after
``--resume-delay`` seconds.  The one-shot ``move_cp`` approach used for the very
first engagement is deliberately not reused after an intervention: it is a
blocking multi-second trajectory that the arm interpolates internally and that
nothing here can clamp, which is the last thing you want pointed at a scene a
human just rearranged.

Example
-------
::

    # dry run first: no commands published, everything else exercised
    python deploy/deploy_with_intervention.py --checkpoint outputs/circle/best.ckpt

    # for real
    python deploy/deploy_with_intervention.py --checkpoint outputs/circle/best.ckpt \\
        --execute --task circle
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# The vendored diffusion_policy goes first on sys.path: loading a checkpoint
# calls hydra.utils.instantiate(cfg.policy), which imports the policy class by
# name, so a stale editable install pointing at another workspace would
# otherwise silently win.
for _path in (SCRIPT_DIR, REPO_ROOT / "src" / "diffusion"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import handover  # noqa: E402
from controller import PSM1Controller, PSM2Controller  # noqa: E402
from geometry import (  # noqa: E402
    CartesianGate,
    OrientationGate,
    angle_between_quats,
    quat_xyzw_to_rot6d,
    quat_xyzw_to_wxyz,
    rot6d_to_matrix,
)
from pedal import COAG_TOPIC, PedalMonitor, wait_for_pedal  # noqa: E402
from run_recorder import RunRecorder, build_states, next_run_dir  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

IMAGE_TOPIC = "/stereo/left/rectified_downscaled_image"

# Per-arm 10D EE layout and the 20D concatenation order, identical to the
# converter's: cutter (PSM2) first, then retraction (PSM1).
POSE_DIM_PER_ARM = 10
OBS_DIM = 20
POS = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9
CUTTER = slice(0, 10)
RETRACT = slice(10, 20)

# How to read each predicted row. Absolute is what
# ``data_processing/convert_drawing_6d_abs.py`` produces and what a checkpoint
# trained on it emits; relative is kept because a delta-action dataset only
# needs a different converter, not a different deploy path.
ABSOLUTE = "absolute"
RELATIVE = "relative"

SETTLE_S = 4.0


# ------------------------------------------------------------ policy source

def load_checkpoint_policy(args: argparse.Namespace):
    """Build the diffusion-policy inference engine for a checkpoint.

    torch, dill and the diffusion_policy tree are imported **here**, not at
    module scope, so that ``--help`` and argument validation work on an
    interpreter that has none of them -- which is every interpreter on this
    machine until the dVRK desktop's ROS environment gets torch installed.
    """
    import dill
    import torch

    from deploy_lib import SurgFlowDVRKDeploy, checkpoint_dims

    class GoalFreeDeploy(SurgFlowDVRKDeploy):
        """Inference engine that never conditions on a start/end goal.

        The base class always injects ``start_end_points``, which a goal-less
        checkpoint's normalizer has no parameters for -- it would raise inside
        the normalizer, mid-loop, with the arms live.
        """

        action_mode = ABSOLUTE

        def build_obs_dict(self):
            if not self.image_history or not self.agent_pos_history:
                raise RuntimeError("observation history is empty")
            images = list(self.image_history)
            agents = list(self.agent_pos_history)
            while len(images) < self.n_obs_steps:
                images.insert(0, images[0].copy())
                agents.insert(0, agents[0].copy())
            images = np.stack(images[-self.n_obs_steps:], axis=0)
            agents = np.stack(agents[-self.n_obs_steps:], axis=0)
            return {
                "image": torch.from_numpy(images).unsqueeze(0).to(self._torch_device),
                "agent_pos": torch.from_numpy(agents).unsqueeze(0).to(self._torch_device),
            }

    path = Path(args.checkpoint).expanduser().resolve()
    obs_dim, act_dim = checkpoint_dims(path)
    if act_dim != OBS_DIM:
        raise SystemExit(
            f"expected a {OBS_DIM}D absolute-EE checkpoint, got act_dim={act_dim} "
            f"(obs_dim={obs_dim})."
        )
    payload = torch.load(path.open("rb"), map_location="cpu", pickle_module=dill)
    if "start_end_points" in payload["cfg"].task.shape_meta.obs:
        raise SystemExit(
            "This checkpoint was trained WITH a start/end goal (shape_meta.obs has "
            "start_end_points); this node is the goal-free path."
        )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = GoalFreeDeploy(
        checkpoint_path=path, device=device, inference_steps=args.inference_steps)
    policy.load()
    if args.action_mode is not None:
        policy.action_mode = args.action_mode
    return policy


# -------------------------------------------------------------------- node

class InterventionDeployNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("deploy_with_intervention")
        self.args = args

        self.policy = load_checkpoint_policy(args)
        trained_rate = float(self.policy.control_rate_hz)
        if args.rate is None:
            args.rate = trained_rate
        elif not np.isclose(args.rate, trained_rate, rtol=0.0, atol=1e-6):
            raise SystemExit(
                f"--rate={args.rate:g} Hz does not match this checkpoint's "
                f"{trained_rate:g} Hz action spacing. Reconvert/retrain for that "
                "rate, or omit --rate to use the checkpoint value."
            )
        self.get_logger().info(
            f"Control rate: {args.rate:g} Hz (checkpoint action spacing: {trained_rate:g} Hz)")
        self.action_mode = self.policy.action_mode
        if self.policy.obs_dim != OBS_DIM or self.policy.act_dim != OBS_DIM:
            raise SystemExit(
                f"expected a {OBS_DIM}D policy, got obs_dim={self.policy.obs_dim} "
                f"act_dim={self.policy.act_dim}"
            )
        describe = getattr(self.policy, "describe", None)
        self.get_logger().info(
            describe() if describe else
            f"Loaded checkpoint policy: n_obs={self.policy.n_obs_steps} "
            f"n_act={self.policy.n_action_steps}"
        )
        self.get_logger().warn(
            f"Action mode: {self.action_mode.upper()}. "
            + ("Each predicted row is the pose to move TO."
               if self.action_mode == ABSOLUTE else
               "Each predicted row is a CHANGE from the measured pose; the "
               "rotation block is a delta rotation composed in the base frame.")
        )

        self.gate = CartesianGate(max_pos_step=float(args.max_pos_step))
        self.orientation_gate = OrientationGate(
            max_angle_rad=np.deg2rad(float(args.max_angle_step_deg)))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frames_rgb: deque[np.ndarray] = deque(
            maxlen=max(1, int(self.policy.n_obs_steps)))
        self._qpos = {"PSM1": None, "PSM2": None}

        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.create_subscription(
            JointState, "/PSM1/measured_js", lambda m: self._on_js("PSM1", m), 10)
        self.create_subscription(
            JointState, "/PSM2/measured_js", lambda m: self._on_js("PSM2", m), 10)

        # cutter -> PSM2, retraction -> PSM1, matching the converter's order.
        self.cutter = PSM2Controller(self, ns="PSM2", frame_id="PSM2_base")
        self.retract = PSM1Controller(self, ns="PSM1", frame_id="PSM1_base")

        # MTM orientation, read ONLY to report alignment during the handover
        # window. This node never publishes to an MTM: aligning a master that a
        # surgeon already has their hand on is dVRK's job and its interlock's,
        # not ours.
        self._mtm_quat: dict[str, tuple | None] = {}
        self.mtm_for = {"cutter": args.mtm_psm2, "retract": args.mtm_psm1}
        for arm_name, mtm in self.mtm_for.items():
            if not mtm:
                continue
            self._mtm_quat[mtm] = None
            self.create_subscription(
                PoseStamped, f"/{mtm}/measured_cp",
                lambda msg, name=mtm: self._on_mtm(name, msg), 1)

        # A checkpoint predicts both arms, so this node commands both. Each one
        # is streamed servo_cp every cycle, which means it will resist being
        # pushed and return to the commanded pose if it drifts -- both arms must
        # therefore be homed and enabled before this runs.
        self.commanded_arms = [("cutter", CUTTER, self.cutter),
                               ("retract", RETRACT, self.retract)]
        self.get_logger().warn(
            "Commanding: " + ", ".join(
                f"{name} ({'PSM2' if name == 'cutter' else 'PSM1'})"
                for name, _, _ in self.commanded_arms)
            + "  --  both arms must be homed and enabled"
        )

        self.pedal = PedalMonitor(self, topic=args.pedal_topic, name=args.pedal_name)
        self.handover = handover.HandoverMachine(align_seconds=float(args.align_seconds))

        self.action_queue: deque[np.ndarray] = deque()
        self.motion_busy = False
        self.initial_move_done = not self._want_initial_move()
        self.cycle = 0
        self._phase_started_at: float | None = None
        self._t0 = time.monotonic()
        self.cycle_log: list[dict] = []
        self.phase_spans: list[dict] = []

        self.recorder: RunRecorder | None = None
        if not args.no_record:
            run_dir = Path(args.run_dir) if args.run_dir else next_run_dir(
                REPO_ROOT / "data" / "intervention_runs" / args.task)
            self.recorder = RunRecorder(run_dir, task=args.task,
                                        frequency=float(args.record_rate))
            self.get_logger().info(f"Recording this session to {run_dir}")

        self._report_teleop_pairs()

        self.create_timer(1.0 / float(args.rate), self._control_step)
        if self.recorder is not None:
            self.create_timer(1.0 / float(args.record_rate), self._record_step)

        if not self.initial_move_done:
            self.get_logger().warn(
                f"Opening with a one-shot move_cp to the first target, then "
                f"{SETTLE_S:.0f}s settle. The pedal is NOT polled during it."
            )
        else:
            self.get_logger().info(
                "No opening move_cp: starting straight into clamped servo_cp "
                "streaming from the current pose."
            )

        if args.execute:
            self.get_logger().warn(
                f"Motion ENABLED. Hold {args.pedal_name.upper()} to take over. "
                f"Both directions pass through a {args.align_seconds:.0f}s alignment "
                f"window in which NOTHING commands the arms."
            )
        else:
            self.get_logger().warn(
                "DRY RUN: no servo_cp or jaw commands will be published. "
                "Pass --execute to move the arms."
            )

    def _want_initial_move(self) -> bool:
        """Whether to open with a one-shot move_cp to the first target."""
        if self.args.initial_move is not None:
            return bool(self.args.initial_move)
        return True

    # ------------------------------------------------------------ pre-flight
    def _report_teleop_pairs(self) -> None:
        """Say out loud whether a dVRK teleop component looks present.

        A warning, never a hard failure: topic discovery is asynchronous and a
        pair that has not advertised yet is not a pair that is missing.  But a
        session where the pedal does nothing because no teleop is configured
        looks, from the operator's chair, exactly like a session where the
        handover code is broken -- so it is worth one line at startup.
        """
        names = [n for n, _ in self.get_topic_names_and_types()]
        pairs = sorted({
            n.split("/")[1] for n in names
            if len(n.split("/")) > 2 and "_PSM" in n.split("/")[1]
            and n.split("/")[1].startswith("MTM")
        })
        if pairs:
            self.get_logger().info(f"dVRK teleop pairs advertising: {', '.join(pairs)}")
        else:
            self.get_logger().warn(
                "No MTM*_PSM* teleop topics found. The pedal will stop the policy, "
                "but nothing will drive the arms while it is held unless your "
                "console has a teleop pair configured and the operator-present "
                "interlock is satisfied. Verify on the console first."
            )

    # ---------------------------------------------------------------- ROS I/O
    def _on_image(self, msg: Image) -> None:
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.frames_rgb.append(rgb.copy())

    def _on_mtm(self, name: str, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        with self.lock:
            self._mtm_quat[name] = (float(q.x), float(q.y), float(q.z), float(q.w))

    def alignment_errors_deg(self) -> dict[str, float]:
        """Angle between each MTM wrist and its PSM tool, in degrees.

        Advisory only. A missing MTM topic leaves the arm out of the dict; the
        alignment window still holds the arms for its full duration, because the
        window's job is to keep this node silent across the transition and that
        does not depend on being able to measure anything.
        """
        out: dict[str, float] = {}
        for arm_name, ctrl in (("cutter", self.cutter), ("retract", self.retract)):
            mtm = self.mtm_for.get(arm_name)
            if not mtm:
                continue
            with self.lock:
                mtm_quat = self._mtm_quat.get(mtm)
            if mtm_quat is None:
                continue
            psm_quat = ctrl.current_orientation()
            if not any(psm_quat):
                continue
            out[arm_name] = float(np.rad2deg(angle_between_quats(mtm_quat, psm_quat)))
        return out

    def _on_js(self, arm: str, msg: JointState) -> None:
        with self.lock:
            self._qpos[arm] = np.asarray(msg.position, dtype=np.float64)

    def _arm_state(self, ctrl) -> np.ndarray | None:
        """10D EE state for one arm, or None if feedback is missing."""
        jaw = ctrl.current_jaw()
        if jaw is None:
            return None
        pos = np.asarray(ctrl.current_pos(), dtype=np.float32)
        # Guard against the controller's pre-feedback defaults (pos exactly 0).
        if not np.any(pos):
            return None
        rot6d = quat_xyzw_to_rot6d(ctrl.current_orientation())
        return np.concatenate([pos, rot6d, [np.float32(jaw)]]).astype(np.float32)

    def _agent_pos(self) -> np.ndarray | None:
        cutter = self._arm_state(self.cutter)
        retract = self._arm_state(self.retract)
        if cutter is None or retract is None:
            return None
        return np.concatenate([cutter, retract]).astype(np.float32)

    def _missing_ready_reason(self) -> str | None:
        """Why the policy cannot run yet, or None."""
        with self.lock:
            if not self.frames_rgb:
                return f"waiting for {IMAGE_TOPIC}"
        if self._agent_pos() is None:
            return "waiting for measured_cp + jaw/measured_js on both arms"
        return None

    # -------------------------------------------------------------- recording
    def _record_step(self) -> None:
        """Append one 30 Hz frame, whoever is driving.

        Runs on its own timer rather than inside the control loop so that
        corrections are sampled at the same rate as the demonstrations they will
        be merged with -- see ``run_recorder`` for why that matters.

        Returns silently once recording has been switched off, because the timer
        keeps firing after a write error disables it below -- asserting here
        would raise on every tick from then on, which is exactly the failure the
        error handler exists to prevent.
        """
        if self.recorder is None:
            return
        with self.lock:
            frame = self.frames_rgb[-1].copy() if self.frames_rgb else None
            qpos1 = self._qpos["PSM1"]
            qpos2 = self._qpos["PSM2"]
        if frame is None:
            # Recording was asked for and nothing is arriving. Say so, loudly and
            # repeatedly: the failure mode otherwise is a whole intervention
            # session that appears to be collecting training data and leaves an
            # empty run directory behind.
            self._log_throttled(
                "no_image_to_record",
                f"RECORDING NOTHING: --run-dir was given but no image has arrived "
                f"on {IMAGE_TOPIC}. Frames need an image; the session log (--log) "
                f"is unaffected. Pass --no-record to silence this.",
                period_s=10.0,
                level="warn",
            )
            return
        cutter_jaw = self.cutter.current_jaw()
        retract_jaw = self.retract.current_jaw()
        if cutter_jaw is None or retract_jaw is None:
            return
        cutter_pos = np.asarray(self.cutter.current_pos(), dtype=np.float64)
        retract_pos = np.asarray(self.retract.current_pos(), dtype=np.float64)
        if not np.any(cutter_pos) or not np.any(retract_pos):
            return
        states = build_states(
            cutter_pos=cutter_pos,
            cutter_quat_wxyz=quat_xyzw_to_wxyz(self.cutter.current_orientation()),
            cutter_jaw=cutter_jaw,
            retract_pos=retract_pos,
            retract_quat_wxyz=quat_xyzw_to_wxyz(self.retract.current_orientation()),
            retract_jaw=retract_jaw,
            cutter_qpos=qpos2,
            retract_qpos=qpos1,
        )
        try:
            self.recorder.add_frame(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                states,
                t=time.monotonic() - self._t0,
                extra={"cycle": int(self.cycle)},
            )
        except Exception as exc:  # noqa: BLE001
            # Never let the disk take the robot down. Losing the tail of a run
            # is recoverable; a control loop that stops because a JPEG failed to
            # write, while the arms are live, is not.
            self.get_logger().error(f"recording disabled after a write error: {exc}")
            self.recorder = None

    # ------------------------------------------------------------- handover
    def _on_phase_change(self, state) -> None:
        """React to a phase transition. Never commands anything itself."""
        now = time.monotonic()
        if self._phase_started_at is not None:
            self.phase_spans.append({
                "phase": state.previous,
                "start_s": self._phase_started_at - self._t0,
                "duration_s": now - self._phase_started_at,
            })
        self._phase_started_at = now

        if state.phase == handover.ALIGN_TO_EXPERT:
            # Drop the plan now, not when the surgeon finishes. It was
            # conditioned on a scene that is about to change, and replaying it
            # on handback would undo the correction.
            self.action_queue.clear()
            # No one-shot move_cp for the rest of the session: after a human has
            # repositioned the arms, the policy resumes into clamped servo_cp
            # streaming only.
            self.initial_move_done = True
            errors = self.alignment_errors_deg()
            detail = (", ".join(f"{k} {v:.1f} deg" for k, v in errors.items())
                      if errors else "no MTM topic to measure")
            self.get_logger().warn(
                f"[cycle {self.cycle}] TAKEOVER #{self.handover.n_takeovers}: "
                f"{self.args.pedal_name.upper()} pressed. Commands suspended. "
                f"Aligning for {self.args.align_seconds:.0f}s -- MTM/PSM: {detail}"
            )
        elif state.phase == handover.EXPERT:
            errors = self.alignment_errors_deg()
            detail = (", ".join(f"{k} {v:.1f} deg" for k, v in errors.items())
                      if errors else "unmeasured")
            self.get_logger().warn(
                f"[cycle {self.cycle}] SURGEON HAS THE ARMS. "
                f"Alignment at handover: {detail}"
            )
        elif state.phase == handover.ALIGN_TO_POLICY:
            self.action_queue.clear()
            self.get_logger().warn(
                f"[cycle {self.cycle}] {self.args.pedal_name.upper()} released. "
                f"Arms holding for {self.args.align_seconds:.0f}s while the policy "
                f"re-conditions on what you left it."
            )
        elif state.phase == handover.POLICY:
            self.action_queue.clear()
            self.get_logger().warn(
                f"[cycle {self.cycle}] POLICY RESUMES, replanning from the current "
                f"pose. First command clamped to {self.args.max_pos_step * 1000:.1f} mm "
                f"and {self.args.max_angle_step_deg:.1f} deg."
            )

    # ------------------------------------------------------------ motion exec
    def _resolve_target(self, arm_action: np.ndarray, ctrl):
        """Turn one 10D predicted row into an absolute (position, rotation, jaw).

        This is where the action convention is applied, and the only place it is.
        An absolute row IS the target; a relative row is a change from the
        measured pose, with its rotation block composed in the base frame as
        ``R_delta @ R_current``.  Adding an absolute row to the current pose
        would double every motion; treating a relative row as absolute would
        drive the arm toward the base-frame origin. Both are quiet failures at
        the robot, so the mode is logged loudly at startup.
        """
        current_pos = np.asarray(ctrl.current_pos(), dtype=np.float64).reshape(3)
        current_R = R.from_quat(
            np.asarray(ctrl.current_orientation(), dtype=np.float64).reshape(4)).as_matrix()
        current_jaw = ctrl.current_jaw()
        current_jaw = 0.0 if current_jaw is None else float(current_jaw)

        row_rot = rot6d_to_matrix(arm_action[ROT6D])
        if self.action_mode == RELATIVE:
            target_pos = current_pos + np.asarray(arm_action[POS], dtype=np.float64)
            target_R = row_rot @ current_R
            target_jaw = current_jaw + float(arm_action[GRIP])
        else:
            target_pos = np.asarray(arm_action[POS], dtype=np.float64)
            target_R = row_rot
            target_jaw = float(arm_action[GRIP])
        return target_pos, target_R, target_jaw, current_pos, current_R

    def _command_pose(self, arm_action: np.ndarray, ctrl, name: str,
                      streaming: bool = True) -> None:
        """Send one 10D waypoint to a single arm, clamped in position AND angle.

        ``streaming=True`` streams a clamped ``servo_cp`` setpoint;
        ``streaming=False`` sends a one-shot ``move_cp`` trajectory, only ever
        used for the very first approach. Streaming ``move_cp`` would preempt
        its own trajectory every tick and the arm would never move.
        """
        target_pos, target_R, target_jaw, current_pos, current_R = self._resolve_target(
            arm_action, ctrl)

        if streaming:
            pos = self.gate.clamp_pos(target_pos, current_pos)
            rot = self.orientation_gate.clamp_matrix(target_R, current_R)
        else:
            pos, rot = target_pos, target_R

        pos_jump = float(np.linalg.norm(target_pos - current_pos))
        ang_jump = float(np.linalg.norm(R.from_matrix(target_R @ current_R.T).as_rotvec()))
        if streaming and pos_jump > float(self.args.warn_jump):
            self.get_logger().warn(
                f"[{name} cyc {self.cycle}] commanded pos jump {pos_jump:.4f} m clamped "
                f"to {self.args.max_pos_step:.4f} m"
            )
        if streaming and np.rad2deg(ang_jump) > float(self.args.warn_angle_deg):
            self.get_logger().warn(
                f"[{name} cyc {self.cycle}] commanded angle jump "
                f"{np.rad2deg(ang_jump):.1f} deg clamped to "
                f"{self.args.max_angle_step_deg:.1f} deg"
            )
        if not self.args.execute:
            return

        quat = R.from_matrix(rot).as_quat()
        quat = tuple(float(v) for v in quat / (np.linalg.norm(quat) + 1e-9))
        pos = tuple(float(v) for v in pos)
        if streaming:
            ctrl.send_servo_pose(pos, quat=quat)
        elif name == "cutter":
            ctrl.send_pose_psm2(pos, quat=quat)
        else:
            ctrl.send_pose_psm1(pos, quat=quat)
        ctrl.set_jaw_goal(target_jaw)
        ctrl.hold_jaw_goal(1.0 / float(self.args.rate))

    def _maybe_initial_move(self, waypoint: np.ndarray) -> bool:
        """One-shot ``move_cp`` to the policy's first target, or nothing.

        This blocks the control loop for ``SETTLE_S`` with ``motion_busy`` set,
        and during those seconds the pedal is not polled while a trajectory the
        arm interpolates internally is running -- a stamp on the pedal cannot
        stop it. That is a fair trade for a checkpoint, whose first target may
        be a long way from wherever the arms happen to be parked, so it is on by
        default; ``--no-initial-move`` starts straight into clamped streaming
        instead, which is the safer choice once the arms are already in place.
        """
        if self.initial_move_done:
            return True
        current = self._agent_pos()
        if current is None:
            return False
        jump = 0.0
        for _, arm_slice, ctrl in self.commanded_arms:
            target_pos, _, _, current_pos, _ = self._resolve_target(
                waypoint[arm_slice], ctrl)
            jump = max(jump, float(np.linalg.norm(target_pos - current_pos)))
        if jump > float(self.args.max_initial_jump):
            self.get_logger().error(
                f"Initial EE jump is {jump:.4f} m (> {self.args.max_initial_jump:.4f}). "
                f"Reposition the arms closer to the first target first."
            )
            raise SystemExit(1)
        self.get_logger().info(
            f"Initial move to first target (jump={jump:.4f} m); settle {SETTLE_S:.1f}s")
        self.motion_busy = True
        try:
            for name, arm_slice, ctrl in self.commanded_arms:
                self._command_pose(waypoint[arm_slice], ctrl, name, streaming=False)
            time.sleep(SETTLE_S)
        finally:
            self.motion_busy = False
        self.initial_move_done = True
        return True

    # ------------------------------------------------------------ control loop
    def _control_step(self) -> None:
        if self.motion_busy:
            return

        # Poll the pedal and advance the phase machine first, unconditionally:
        # a takeover must be honoured even on a cycle that would otherwise
        # return early for missing data.
        state = self.handover.update(self.pedal.poll().pressed)
        if state.changed:
            self._on_phase_change(state)
        if self.recorder is not None:
            self.recorder.mark_mode(state.phase)

        if not state.commands_allowed:
            # ALIGN_TO_EXPERT, EXPERT and ALIGN_TO_POLICY all publish nothing.
            # Observations keep flowing so the policy resumes on what the
            # surgeon actually did rather than on the scene it last saw.
            self._sync_deploy_history()
            if state.aligning:
                self._log_throttled(
                    "aligning",
                    f"{handover.PHASE_LABEL[state.phase]}: "
                    f"{state.align_remaining:.1f}s left"
                    + self._alignment_suffix(),
                    period_s=1.0,
                )
            self._log_cycle(action=None, current=self._agent_pos(), phase=state.phase)
            self.cycle += 1
            return

        missing = self._missing_ready_reason()
        if missing is not None:
            self._log_throttled("not_ready", missing)
            return

        current = self._sync_deploy_history()
        if current is None:
            return

        if not self.action_queue:
            try:
                result = self.policy.predict_raw()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"predict failed: {exc}")
                return
            chunk = np.asarray(result["action_raw"], dtype=np.float32)  # [T, 20]
            waypoints = chunk[:1] if self.args.receding_horizon else chunk
            for wp in waypoints:
                self.action_queue.append(wp.astype(np.float32))

        waypoint = self.action_queue.popleft()
        if not self._maybe_initial_move(waypoint):
            self.action_queue.appendleft(waypoint.astype(np.float32))
            return

        self.motion_busy = True
        try:
            for name, arm_slice, ctrl in self.commanded_arms:
                self._command_pose(waypoint[arm_slice], ctrl, name)
            if self.args.move_pause > 0.0:
                time.sleep(float(self.args.move_pause))
        finally:
            self.motion_busy = False

        self._log_cycle(action=waypoint, current=current, phase=state.phase)
        self.cycle += 1

    def _alignment_suffix(self) -> str:
        errors = self.alignment_errors_deg()
        if not errors:
            return ""
        return "  MTM/PSM: " + ", ".join(f"{k} {v:.1f} deg" for k, v in errors.items())

    def _sync_deploy_history(self) -> np.ndarray | None:
        """Feed the policy the newest observation; return it, or None.

        Called on EVERY cycle including the alignment windows and while the
        surgeon is driving, so that when the policy resumes it plans from what
        the human actually did rather than from the scene it last saw.
        """
        current = self._agent_pos()
        with self.lock:
            frame = self.frames_rgb[-1].copy() if self.frames_rgb else None
        if current is None:
            return None
        if frame is None:
            return None
        self.policy.append_observation(frame, current)
        return current

    # ------------------------------------------------------------------- logs
    def _log_cycle(self, action: np.ndarray | None, current: np.ndarray | None,
                   phase: str) -> None:
        self.cycle_log.append({
            "cycle": int(self.cycle),
            "t": float(time.monotonic() - self._t0),
            "phase": handover.PHASES.index(phase),
            "takeover": int(self.handover.n_takeovers),
            # NaN, not zeros: a row with no policy command must not read back as
            # a command to the base-frame origin.
            "action": (np.full(OBS_DIM, np.nan, dtype=np.float32)
                       if action is None else np.asarray(action, dtype=np.float32).copy()),
            "current": (np.full(OBS_DIM, np.nan, dtype=np.float32)
                        if current is None else np.asarray(current, dtype=np.float32).copy()),
        })

    def save_log(self) -> Path | None:
        if not self.cycle_log:
            return None
        path = Path(self.args.log).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cycles=np.asarray([e["cycle"] for e in self.cycle_log], dtype=np.int64),
            t=np.asarray([e["t"] for e in self.cycle_log], dtype=np.float64),
            phase=np.asarray([e["phase"] for e in self.cycle_log], dtype=np.int8),
            phase_names=np.asarray(handover.PHASES),
            takeover=np.asarray([e["takeover"] for e in self.cycle_log], dtype=np.int32),
            action=np.stack([e["action"] for e in self.cycle_log], axis=0),
            current=np.stack([e["current"] for e in self.cycle_log], axis=0),
            n_takeovers=np.int64(self.handover.n_takeovers),
            action_mode=np.asarray(self.action_mode),
            align_seconds=np.float64(self.args.align_seconds),
            max_pos_step=np.float64(self.args.max_pos_step),
            max_angle_step_deg=np.float64(self.args.max_angle_step_deg),
            span_phases=np.asarray([s["phase"] for s in self.phase_spans]),
            span_starts=np.asarray([s["start_s"] for s in self.phase_spans], dtype=np.float64),
            span_durations=np.asarray([s["duration_s"] for s in self.phase_spans],
                                      dtype=np.float64),
        )
        return path

    def _log_throttled(self, key: str, msg: str, period_s: float = 2.0,
                       level: str = "info") -> None:
        """Log at most once per ``period_s`` for a given ``key``.

        The severity is dispatched by an explicit branch rather than
        ``getattr(logger, level)``, because rclpy caches a logger's severity
        against the *call site* -- the file, line and function of the caller.
        A single line that emits both info and warn raises
        ``ValueError: Logger severity cannot be changed between calls`` on the
        second severity, out of a timer callback, killing the node mid-session.
        Two branches are two lines, so two call sites, so no clash.
        """
        now = time.monotonic()
        if not hasattr(self, "_last_log_times"):
            self._last_log_times = {}
        if now - self._last_log_times.get(key, 0.0) < period_s:
            return
        self._last_log_times[key] = now
        logger = self.get_logger()
        if level == "warn":
            logger.warn(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.info(msg)

    def summary(self) -> str:
        total = max(1, len(self.cycle_log))
        counts = {phase: 0 for phase in handover.PHASES}
        for entry in self.cycle_log:
            counts[handover.PHASES[entry["phase"]]] += 1
        lines = [
            "",
            "=" * 68,
            "SESSION SUMMARY",
            "=" * 68,
            f"  action mode      : {self.action_mode}",
            f"  control cycles   : {len(self.cycle_log)}",
            f"  takeovers        : {self.handover.n_takeovers}",
            f"  alignment window : {self.args.align_seconds:.1f}s each way",
            "  cycles per phase :",
        ]
        for phase in handover.PHASES:
            lines.append(f"    {handover.PHASE_LABEL[phase]:<22} {counts[phase]:>6} "
                         f"({100.0 * counts[phase] / total:5.1f}%)")
        if self.phase_spans:
            lines.append("  timeline:")
            for span in self.phase_spans:
                lines.append(
                    f"    t={span['start_s']:7.1f}s  "
                    f"{handover.PHASE_LABEL[span['phase']]:<22} "
                    f"{span['duration_s']:6.1f}s")
        if self.recorder is not None:
            recorded = self.recorder.counts
            total_frames = sum(recorded.values())
            if total_frames == 0:
                lines.append("  frames recorded  : NONE -- no image ever arrived on")
                lines.append(f"                     {IMAGE_TOPIC}")
                lines.append("                     There is no training data from this run.")
            else:
                lines.append("  frames recorded  : " + ", ".join(
                    f"{k}={v}" for k, v in recorded.items() if v) +
                    f" -> {self.recorder.run_dir}")
        else:
            lines.append("  frames recorded  : none (--no-record)")
        lines.append("=" * 68)
        return "\n".join(lines)


# -------------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--checkpoint", required=True, help="Diffusion Policy .ckpt")

    p.add_argument("--device", default=None,
                   help="torch device for a checkpoint (default: cuda if available)")
    p.add_argument("--inference-steps", type=int, default=None,
                   help="Override DDPM inference steps (try 16-32 for live use)")
    p.add_argument("--action-mode", choices=(ABSOLUTE, RELATIVE), default=None,
                   help="How to read each predicted row. Defaults to absolute, "
                        "which is what convert_drawing_6d_abs.py produces. Nothing "
                        "cross-checks this against the checkpoint -- pass "
                        "'relative' only for a checkpoint trained on deltas.")
    p.add_argument("--execute", action="store_true",
                   help="Actually publish commands. Without it this is a dry run: "
                        "every other path runs, including the pedal handover.")
    p.add_argument("--task", default="drawing",
                   help="Task name, used for the run directory and the recorded goal")

    g = p.add_argument_group("control")
    g.add_argument("--rate", type=float, default=None,
                   help="Control loop rate (Hz). Defaults to the checkpoint's training rate "
                        "(5 Hz for older drawing checkpoints).")
    g.add_argument("--max-pos-step", type=float, default=0.005,
                   help="Max EE position change (m) per streamed command")
    g.add_argument("--max-angle-step-deg", type=float, default=2.0,
                   help="Max EE orientation change (deg) per streamed command")
    g.add_argument("--max-initial-jump", type=float, default=0.02,
                   help="Abort if the first target is farther than this (m) from current EE")
    g.add_argument("--warn-jump", type=float, default=0.01,
                   help="Warn when a commanded position delta exceeds this (m)")
    g.add_argument("--warn-angle-deg", type=float, default=5.0,
                   help="Warn when a commanded angular delta exceeds this (deg)")
    g.add_argument("--move-pause", type=float, default=0.2,
                   help="Seconds to pause after each waypoint")
    g.add_argument("--receding-horizon", action="store_true",
                   help="Execute only the first of n_action_steps, replan every tick")
    m = g.add_mutually_exclusive_group()
    m.add_argument("--initial-move", dest="initial_move", action="store_true",
                   default=None,
                   help="Open with a one-shot move_cp to the first target. On by "
                        "default, because a checkpoint's first target may be far "
                        f"from the parked arms. It blocks the loop for "
                        f"{SETTLE_S:.0f}s with the pedal unpolled.")
    m.add_argument("--no-initial-move", dest="initial_move", action="store_false",
                   help="Start straight into clamped servo_cp streaming, with no "
                        "opening move_cp. Use it when the arms are already parked "
                        "at the start of a demonstration.")

    g = p.add_argument_group("handover")
    g.add_argument("--pedal-topic", default=COAG_TOPIC,
                   help="sensor_msgs/Joy topic of the takeover pedal")
    g.add_argument("--pedal-name", default="coag", help="Pedal name, for log messages")
    g.add_argument("--pedal-timeout", type=float, default=30.0,
                   help="Seconds to wait at startup for the pedal topic before giving up")
    g.add_argument("--allow-no-pedal", action="store_true",
                   help="Start even if the pedal topic never reported. Unsafe: there is "
                        "then no way to take the arms back. Dry runs only.")
    g.add_argument("--align-seconds", type=float, default=1.0,
                   help="Alignment window on BOTH transitions. Nothing commands the "
                        "arms during it.")
    g.add_argument("--mtm-psm1", default="MTML",
                   help="MTM paired with PSM1, read only, to report alignment. "
                        "Empty string disables.")
    g.add_argument("--mtm-psm2", default="MTMR",
                   help="MTM paired with PSM2, read only, to report alignment. "
                        "Empty string disables.")

    g = p.add_argument_group("recording")
    g.add_argument("--no-record", action="store_true", help="Do not record the session")
    g.add_argument("--run-dir", default=None,
                   help="Explicit run directory (default: data/intervention_runs/<task>/run_NNNN)")
    g.add_argument("--record-rate", type=float, default=30.0,
                   help="Recording rate (Hz). Keep at the demo collector's 30 Hz so the "
                        "converter's stride of 6 yields the same 5 Hz.")
    g.add_argument("--log", type=Path, default=SCRIPT_DIR / "logs" / "intervention_run.npz",
                   help="Where to write the per-cycle .npz log")

    args = p.parse_args(argv)
    for name in ("max_pos_step", "record_rate", "max_angle_step_deg"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.rate is not None and args.rate <= 0:
        p.error("--rate must be positive")
    if args.align_seconds < 0:
        p.error("--align-seconds must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    rclpy.init()
    node = InterventionDeployNode(args)

    if not wait_for_pedal(node.pedal, args.pedal_timeout):
        message = (
            f"No message on {args.pedal_topic} within {args.pedal_timeout:.0f}s. "
            f"Is the dVRK console running on this ROS_DOMAIN_ID? Tap the "
            f"{args.pedal_name} pedal once to make it publish."
        )
        if args.execute and not args.allow_no_pedal:
            node.get_logger().error(message)
            node.destroy_node()
            rclpy.shutdown()
            raise SystemExit(
                "Refusing to command the arms with no confirmed takeover pedal. "
                "Fix the pedal, or re-run without --execute."
            )
        node.get_logger().warn(message + " Continuing without a takeover path.")
    else:
        node.get_logger().info(
            f"{args.pedal_topic} is live (currently "
            f"{'PRESSED' if node.pedal.pressed else 'released'})."
        )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        # SIGTERM -- from `timeout`, a supervisor, or `kill` -- makes rclpy's
        # signal handler shut the context down underneath spin(), which then
        # raises RCLError rather than KeyboardInterrupt. Without this the run
        # ends in a traceback instead of the summary, and the one thing you
        # wanted from a session that had to be killed is the record of what it
        # did before you killed it.
        if type(exc).__name__ not in ("RCLError", "ExternalShutdownException"):
            raise
        print(f"\nshutting down: {type(exc).__name__}")
    finally:
        print(node.summary())
        log_path = node.save_log()
        if log_path is not None:
            print(f"per-cycle log: {log_path}")
        if node.recorder is not None:
            node.recorder.close()
            print(f"session recording: {node.recorder.run_dir}")
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001 - the context may already be gone
            pass
        # Already shut down by the signal handler on the SIGTERM path.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
