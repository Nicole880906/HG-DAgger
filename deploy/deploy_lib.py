"""SurgFlow image Diffusion Policy inference engine for dVRK deployment.

Mirrors the role of ``deploy_dp3_dvrk.py::DP3DVRKDeploy``: load a checkpoint,
maintain observation history, run ``predict_action``, and return action chunks.
The ROS nodes in ``deploy_joint_abs_goal.py`` / ``deploy_pose_abs_goal.py`` handle topics and motion.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import dill
import hydra
import numpy as np
import torch

CUTTER_JOINTS = slice(0, 6)
CUTTER_JAW = 6
RETRACT_JOINTS = slice(7, 13)
RETRACT_JAW = 13
ACTION_DIM = 14
TARGET_HW = (120, 160)  # (H, W)


def checkpoint_dims(checkpoint: str | Path) -> tuple[int, int]:
    """Peek a checkpoint's (obs_state_dim, action_dim) from its stored cfg.

    Reads only the ``cfg`` blob (no policy weights) so the dispatcher can decide
    which deployment path to use before instantiating the policy.
    """
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path.open("rb"), map_location="cpu", pickle_module=dill)
    sm = payload["cfg"].task.shape_meta
    return int(sm.obs.agent_pos.shape[0]), int(sm.action.shape[0])


def make_joint_state(names: list[str], positions: np.ndarray) -> dict[str, Any]:
    """Build a JointState-compatible dict (caller wraps in ROS message)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    return {
        "name": list(names),
        "position": pos.tolist(),
        "velocity": [0.0] * len(pos),
        "effort": [0.0] * len(pos),
    }


class SafetyGate:
    """Clamp joint-position targets to limits and bounded per-step motion."""

    def __init__(
        self,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        max_step: float | np.ndarray = 0.05,
    ) -> None:
        if lower is None:
            lower = np.full(ACTION_DIM, -np.pi, dtype=np.float64)
        if upper is None:
            upper = np.full(ACTION_DIM, np.pi, dtype=np.float64)
        self.lower = np.asarray(lower, dtype=np.float64).reshape(ACTION_DIM)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(ACTION_DIM)
        self.max_step = np.asarray(max_step, dtype=np.float64).reshape(-1)
        if self.max_step.size == 1:
            self.max_step = np.full(ACTION_DIM, float(self.max_step[0]), dtype=np.float64)

    def sanitize(self, target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        target = np.asarray(target, dtype=np.float64).reshape(ACTION_DIM)
        current = np.asarray(current, dtype=np.float64).reshape(ACTION_DIM)
        out_of_range = (target < self.lower) | (target > self.upper)
        delta = np.clip(target - current, -self.max_step, self.max_step)
        safe = np.clip(current + delta, self.lower, self.upper)
        big_jump = np.abs(target - current) > self.max_step
        info = {
            "out_of_range_dims": int(out_of_range.sum()),
            "big_jump_dims": int(big_jump.sum()),
            "max_raw_jump": float(np.max(np.abs(target - current))),
            "max_applied_jump": float(np.max(np.abs(safe - current))),
        }
        return safe.astype(np.float32), info


def load_policy(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path.open("rb"), map_location="cpu", pickle_module=dill)
    cfg = copy.deepcopy(payload["cfg"])
    policy_key = "ema_model" if cfg.training.use_ema else "model"
    if policy_key not in payload["state_dicts"]:
        raise KeyError(f"checkpoint has no {policy_key!r} state dict")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][policy_key])
    policy.to(device)
    policy.eval()
    return policy, policy_key, cfg


def preprocess_image_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Native RGB HWC uint8/float -> CHW float32 in [0, 1] at policy resolution."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3, got {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.5:
            image = image / 255.0
    h, w = TARGET_HW
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return np.moveaxis(image, -1, 0).astype(np.float32)


@dataclass
class SurgFlowDVRKDeploy:
    """Image-policy inference helper for closed-loop dVRK deployment.

    Every supported checkpoint predicts **absolute** targets -- the predicted
    row IS the next state, for both the 14D joint and 20D EE representations.
    There is no delta path: adding ``current`` to an absolute prediction would
    double the motion, which on hardware looks like the arms moving randomly.
    """

    checkpoint_path: Path
    device: str = "cuda:0"
    inference_steps: int | None = None
    policy: torch.nn.Module | None = None
    cfg: Any | None = None
    policy_key: str | None = None
    n_obs_steps: int = 2
    n_action_steps: int = 4
    obs_dim: int = ACTION_DIM
    act_dim: int = ACTION_DIM
    image_history: deque = field(default_factory=deque)
    agent_pos_history: deque = field(default_factory=deque)
    start_end_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()
        self._torch_device = torch.device(self.device)

    def load(self) -> None:
        if self.policy is not None:
            return
        policy, policy_key, cfg = load_policy(self.checkpoint_path, self._torch_device)
        if self.inference_steps is not None:
            policy.num_inference_steps = int(self.inference_steps)
        self.policy = policy
        self.policy_key = policy_key
        self.cfg = cfg
        self.n_obs_steps = int(policy.n_obs_steps)
        self.n_action_steps = int(policy.n_action_steps)
        try:
            sm = cfg.task.shape_meta
            self.obs_dim = int(sm.obs.agent_pos.shape[0])
            self.act_dim = int(sm.action.shape[0])
        except Exception:  # noqa: BLE001 — fall back to the joint default
            self.obs_dim = ACTION_DIM
            self.act_dim = ACTION_DIM
        self.reset_history()

    def reset_history(self) -> None:
        self.image_history = deque(maxlen=self.n_obs_steps)
        self.agent_pos_history = deque(maxlen=self.n_obs_steps)

    def set_goal(self, start_end_uv: np.ndarray) -> None:
        goal = np.asarray(start_end_uv, dtype=np.float32).reshape(4)
        if not np.isfinite(goal).all():
            raise ValueError("start/end goal must be finite")
        self.start_end_points = goal

    def append_observation(self, image_rgb: np.ndarray, agent_pos: np.ndarray) -> None:
        self.load()
        agent = np.asarray(agent_pos, dtype=np.float32).reshape(self.obs_dim)
        if not np.isfinite(agent).all():
            raise ValueError("agent_pos contains non-finite values")
        self.image_history.append(preprocess_image_rgb(image_rgb))
        self.agent_pos_history.append(agent)

    def _pad_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.start_end_points is None:
            raise RuntimeError("goal not set; call set_goal() first")
        if not self.image_history or not self.agent_pos_history:
            raise RuntimeError("observation history is empty")

        images = list(self.image_history)
        agents = list(self.agent_pos_history)
        while len(images) < self.n_obs_steps:
            images.insert(0, images[0].copy())
            agents.insert(0, agents[0].copy())
        images = np.stack(images[-self.n_obs_steps :], axis=0)
        agents = np.stack(agents[-self.n_obs_steps :], axis=0)
        goals = np.broadcast_to(self.start_end_points, (self.n_obs_steps, 4)).astype(np.float32)
        return images, agents, goals

    def build_obs_dict(self) -> dict[str, torch.Tensor]:
        images, agents, goals = self._pad_history()
        obs = {
            "image": torch.from_numpy(images).unsqueeze(0).to(self._torch_device),
            "agent_pos": torch.from_numpy(agents).unsqueeze(0).to(self._torch_device),
            "start_end_points": torch.from_numpy(goals).unsqueeze(0).to(self._torch_device),
        }
        return obs

    def predict_raw(self) -> dict[str, Any]:
        """Run one inference and return the raw action chunk, unmodified.

        Representation-agnostic: the caller interprets the chunk according to the
        checkpoint's action representation (joint delta, absolute EE pose, ...).
        Shape is ``[n_action_steps, act_dim]``.
        """
        self.load()
        assert self.policy is not None
        total_t0 = time.perf_counter()
        obs = self.build_obs_dict()
        build_ms = (time.perf_counter() - total_t0) * 1000.0
        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.policy.predict_action(obs)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        action_chunk = result["action"].detach().cpu().numpy()[0].astype(np.float32)
        action_pred = result.get("action_pred")
        if action_pred is not None:
            action_pred = action_pred.detach().cpu().numpy()[0].astype(np.float32)
        return {
            "action_raw": action_chunk,
            "action_pred": action_pred,
            "meta": {
                "checkpoint_path": str(self.checkpoint_path),
                "policy_key": self.policy_key,
                "n_obs_steps": self.n_obs_steps,
                "n_action_steps": self.n_action_steps,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "inference_steps": int(self.policy.num_inference_steps),
                "timing_ms": {
                    "build_obs": build_ms,
                    "predict_action": infer_ms,
                    "total": (time.perf_counter() - total_t0) * 1000.0,
                },
            },
        }

    def predict(
        self,
        current_agent_pos: np.ndarray | None = None,
        gate: SafetyGate | None = None,
    ) -> dict[str, Any]:
        """Run one policy inference and return delta actions plus joint targets."""
        self.load()
        assert self.policy is not None

        total_t0 = time.perf_counter()
        obs = self.build_obs_dict()
        build_ms = (time.perf_counter() - total_t0) * 1000.0

        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.policy.predict_action(obs)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        action_chunk = result["action"].detach().cpu().numpy()[0].astype(np.float32)
        action_pred = result.get("action_pred")
        if action_pred is not None:
            action_pred = action_pred.detach().cpu().numpy()[0].astype(np.float32)

        if current_agent_pos is None:
            current_agent_pos = self.agent_pos_history[-1]
        current = np.asarray(current_agent_pos, dtype=np.float32).reshape(ACTION_DIM)

        # The predicted action already IS the absolute joint target -- never add
        # `current` to it. The downstream gate and micro-stepping bound how fast
        # the arm gets there.
        joint_targets = np.asarray(action_chunk, dtype=np.float32)
        if joint_targets.ndim == 1:
            joint_targets = joint_targets[np.newaxis, :]
        sanitized_chunk = joint_targets.copy()
        sanitize_info: list[dict[str, Any]] = []
        if gate is not None:
            ref = current.copy()
            for i in range(sanitized_chunk.shape[0]):
                safe, info = gate.sanitize(sanitized_chunk[i], ref)
                sanitized_chunk[i] = safe
                sanitize_info.append(info)
                ref = safe.astype(np.float64)

        return {
            "action_raw": action_chunk,
            "action_pred": action_pred,
            "joint_targets": sanitized_chunk,
            "current_agent_pos": current,
            "meta": {
                "checkpoint_path": str(self.checkpoint_path),
                "policy_key": self.policy_key,
                "n_obs_steps": self.n_obs_steps,
                "n_action_steps": self.n_action_steps,
                "inference_steps": int(self.policy.num_inference_steps),
                "timing_ms": {
                    "build_obs": build_ms,
                    "predict_action": infer_ms,
                    "total": (time.perf_counter() - total_t0) * 1000.0,
                },
                "sanitize": sanitize_info,
            },
        }
