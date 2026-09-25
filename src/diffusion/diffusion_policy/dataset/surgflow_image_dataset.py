"""Original Diffusion Policy dataset adapter for a SurgFlow replay Zarr."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class SurgFlowImageDataset(BaseImageDataset):
    """Serve image/state/action windows without loading the 4.8 GB Zarr into RAM."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 3,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
    ):
        super().__init__()
        path = Path(zarr_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"SurgFlow Zarr not found: {path}")
        if horizon < 1 or n_obs_steps < 1 or n_obs_steps > horizon:
            raise ValueError("require 1 <= n_obs_steps <= horizon")

        replay_buffer = ReplayBuffer.create_from_path(str(path), mode="r")
        self.has_goal = "start_end_points" in set(replay_buffer.keys())
        self._validate_replay_buffer(replay_buffer)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.sampler = self._make_sampler(train_mask)

    def _validate_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        required = {"image", "agent_pos", "action"}
        missing = required - set(replay_buffer.keys())
        if missing:
            raise KeyError(f"replay buffer is missing data arrays: {sorted(missing)}")
        image = replay_buffer["image"]
        agent_pos = replay_buffer["agent_pos"]
        action = replay_buffer["action"]
        if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"image must be uint8 (T,H,W,3), got {image.shape} {image.dtype}")
        if agent_pos.ndim != 2 or action.ndim != 2:
            raise ValueError("agent_pos and action must both be rank-2 arrays")
        if agent_pos.shape[1] < 1 or action.shape[1] < 1:
            raise ValueError(
                f"agent_pos/action must have a positive feature dim, got "
                f"{agent_pos.shape[1]}D/{action.shape[1]}D"
            )
        # start_end_points is optional; validate only when the goal is present.
        if self.has_goal:
            start_end_points = replay_buffer["start_end_points"]
            if start_end_points.ndim != 2 or start_end_points.shape[1] != 4:
                raise ValueError(
                    "start_end_points must be (T,4) [start_u, start_v, end_u, end_v], "
                    f"got {start_end_points.shape}"
                )

    def _make_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        # Only observations condition the first n_obs_steps. Loading all
        # 16 image frames would multiply disk traffic by eight.
        key_first_k = {
            "image": self.n_obs_steps,
            "agent_pos": self.n_obs_steps,
        }
        if self.has_goal:
            key_first_k["start_end_points"] = self.n_obs_steps
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            key_first_k=key_first_k,
        )

    def get_validation_dataset(self) -> "SurgFlowImageDataset":
        val_set = copy.copy(self)
        val_set.sampler = val_set._make_sampler(self.val_mask)
        val_set.train_mask = self.val_mask.copy()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["agent_pos"]
        )
        if self.has_goal:
            normalizer["start_end_points"] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer["start_end_points"]
            )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        obs_slice = slice(0, self.n_obs_steps)
        obs = {
            "image": np.moveaxis(sample["image"][obs_slice], -1, 1).astype(np.float32) / 255.0,
            "agent_pos": sample["agent_pos"][obs_slice].astype(np.float32),
        }
        if self.has_goal:
            obs["start_end_points"] = sample["start_end_points"][obs_slice].astype(np.float32)
        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)

