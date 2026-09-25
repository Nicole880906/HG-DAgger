#!/usr/bin/env python3
"""Evaluate a trained SurgFlow image policy without commanding a robot.

This script loads the EMA policy from a Diffusion Policy checkpoint, evaluates
uniformly selected windows from the held-out episode split, and writes metrics,
plots, predictions, and a non-actuating command-candidate CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Uniformly sample at most this many held-out windows (default: all).",
    )
    parser.add_argument(
        "--window-stride", type=int, default=1,
        help="Evaluate every Nth held-out window before max-windows sampling.",
    )
    parser.add_argument(
        "--inference-steps", type=int, default=None,
        help="Override checkpoint diffusion inference steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-examples", type=int, default=6)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


def load_policy(checkpoint: Path, device: torch.device, inference_steps: int | None):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint.open("rb"), map_location="cpu", pickle_module=dill)
    # Checkpoints retain the dataset path from their training host.
    # Prefer the runtime environment so a migrated checkpoint remains portable.
    cfg = copy.deepcopy(payload["cfg"])
    dataset_override = os.environ.get("SURGFLOW_DP_DATASET")
    if dataset_override:
        cfg.task.dataset.zarr_path = str(Path(dataset_override).expanduser().resolve())
    policy_key = "ema_model" if cfg.training.use_ema else "model"
    if policy_key not in payload["state_dicts"]:
        raise KeyError(f"checkpoint has no {policy_key!r} state dict")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][policy_key])
    if inference_steps is not None:
        if inference_steps < 1:
            raise ValueError("--inference-steps must be positive")
        policy.num_inference_steps = inference_steps
    policy.to(device)
    policy.eval()
    return payload, cfg, policy, policy_key


def checkpoint_scalar(payload: dict[str, Any], key: str) -> Any:
    value = payload.get("pickles", {}).get(key)
    return dill.loads(value) if isinstance(value, bytes) else value


def select_indices(length: int, stride: int, max_windows: int | None) -> np.ndarray:
    if length < 1:
        raise ValueError("validation split contains no windows")
    if stride < 1:
        raise ValueError("--window-stride must be positive")
    indices = np.arange(0, length, stride, dtype=np.int64)
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("--max-windows must be positive")
        if len(indices) > max_windows:
            positions = np.linspace(0, len(indices) - 1, max_windows, dtype=np.int64)
            indices = indices[positions]
    return indices


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    abs_error = np.abs(error)
    squared_error = np.square(error)
    return {
        "count": int(error.shape[0]),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "max_abs_error": float(abs_error.max()),
        "per_dimension_mae": abs_error.mean(axis=(0, 1)).tolist(),
        "per_dimension_rmse": np.sqrt(squared_error.mean(axis=(0, 1))).tolist(),
        "per_timestep_mae": abs_error.mean(axis=(0, 2)).tolist(),
        "per_timestep_rmse": np.sqrt(squared_error.mean(axis=(0, 2))).tolist(),
    }


def read_dataset_metadata(dataset_path: Path, action_dim: int) -> tuple[dict[str, Any], list[str]]:
    root = zarr.open_group(str(dataset_path), mode="r")
    attrs = dict(root.attrs)
    arms = list(attrs.get("arms", []))
    layout = list(attrs.get("state_layout_per_arm", []))
    names = [f"{arm}.{field}" for arm in arms for field in layout]
    if len(names) != action_dim:
        names = [f"action_{i}" for i in range(action_dim)]
    return attrs, names


def write_metric_csv(path: Path, names: list[str], full: dict[str, Any], executed: dict[str, Any]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dimension", "full_mae", "full_rmse", "executed_mae", "executed_rmse"])
        for i, name in enumerate(names):
            writer.writerow([
                name,
                full["per_dimension_mae"][i], full["per_dimension_rmse"][i],
                executed["per_dimension_mae"][i], executed["per_dimension_rmse"][i],
            ])


def write_timestep_csv(path: Path, full: dict[str, Any], executed: dict[str, Any], start: int) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "prediction_timestep", "mae", "rmse"])
        for t, (mae, rmse) in enumerate(zip(full["per_timestep_mae"], full["per_timestep_rmse"])):
            writer.writerow(["full", t, mae, rmse])
        for t, (mae, rmse) in enumerate(zip(executed["per_timestep_mae"], executed["per_timestep_rmse"])):
            writer.writerow(["executed", start + t, mae, rmse])


def write_command_candidates(
    path: Path,
    selected_indices: np.ndarray,
    episode_indices: np.ndarray,
    prediction: np.ndarray,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Write predictions for inspection only; this function has no robot I/O."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "validation_window", "episode_index", "command_step",
            "outside_demonstrated_range", *names,
        ])
        outside = (prediction < lower) | (prediction > upper)
        for row, window_index in enumerate(selected_indices):
            for step in range(prediction.shape[1]):
                writer.writerow([
                    int(window_index), int(episode_indices[row]), step,
                    bool(outside[row, step].any()), *prediction[row, step].tolist(),
                ])


def save_plots(
    output_dir: Path,
    names: list[str],
    full_metrics: dict[str, Any],
    executed_metrics: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    selected_indices: np.ndarray,
    count: int,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, full_metrics["per_dimension_mae"], width=0.4, label="full horizon")
    ax.bar(x + 0.2, executed_metrics["per_dimension_mae"], width=0.4, label="executed horizon")
    ax.set_xticks(x, names, rotation=60, ha="right")
    ax.set_ylabel("MAE (native joint/gripper units)")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "action_mae_by_dimension.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    example_dir = output_dir / "examples"
    example_dir.mkdir(exist_ok=True)
    for row in range(min(count, len(predictions))):
        split = max(1, len(names) // 2)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for ax, dim_range in zip(axes, (range(0, split), range(split, len(names)))):
            for dim in dim_range:
                color = f"C{dim % 10}"
                ax.plot(targets[row, :, dim], color=color, linestyle="--", alpha=0.75)
                ax.plot(predictions[row, :, dim], color=color, label=names[dim])
            ax.legend(ncol=4, fontsize=7)
            ax.set_ylabel("joint delta")
        axes[-1].set_xlabel("prediction timestep")
        fig.suptitle(f"Held-out window {int(selected_indices[row])}: solid=prediction, dashed=target")
        fig.tight_layout()
        path = example_dir / f"window_{int(selected_indices[row]):06d}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path.relative_to(output_dir)))
    return written


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload, cfg, policy, policy_key = load_policy(
        args.checkpoint.expanduser().resolve(), device, args.inference_steps)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    selected = select_indices(len(validation), args.window_stride, args.max_windows)
    dataset_path = Path(OmegaConf.to_container(cfg.task.dataset, resolve=True)["zarr_path"])
    attrs, action_names = read_dataset_metadata(dataset_path, policy.action_dim)
    val_episodes = np.flatnonzero(validation.train_mask).astype(int)

    check_report = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": checkpoint_scalar(payload, "epoch"),
        "checkpoint_global_step": checkpoint_scalar(payload, "global_step"),
        "checkpoint_policy": policy_key,
        "dataset": str(dataset_path),
        "dataset_action_representation": attrs.get("action_representation", "unknown"),
        "validation_episodes": val_episodes.tolist(),
        "validation_windows_available": len(validation),
        "validation_windows_selected": len(selected),
        "device": str(device),
        "inference_steps": int(policy.num_inference_steps),
    }
    if args.check_only:
        print(json.dumps(check_report, indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --check-only is used")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    loader = DataLoader(
        Subset(validation, selected.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    start = int(policy.n_obs_steps - 1)
    end = int(start + policy.n_action_steps)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Held-out policy sampling"):
            obs = {key: value.to(device, non_blocking=True) for key, value in batch["obs"].items()}
            result = policy.predict_action(obs)
            predictions.append(result["action_pred"].cpu().numpy())
            targets.append(batch["action"].numpy())
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    executed_prediction = prediction[:, start:end]
    executed_target = target[:, start:end]

    full_metrics = error_metrics(prediction, target)
    executed_metrics = error_metrics(executed_prediction, executed_target)
    demonstrated_actions = np.asarray(dataset.replay_buffer["action"][:], dtype=np.float32)
    demonstrated_lower = demonstrated_actions.min(axis=0)
    demonstrated_upper = demonstrated_actions.max(axis=0)
    outside = (executed_prediction < demonstrated_lower) | (executed_prediction > demonstrated_upper)

    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:])
    buffer_starts = validation.sampler.indices[selected, 0]
    selected_episodes = np.searchsorted(episode_ends, buffer_starts, side="right")

    summary = {
        **check_report,
        "warning": (
            "Offline predictions only. Demonstrated ranges are diagnostics, not certified dVRK limits; "
            "no artifact from this evaluator is safe to send directly to a robot."
        ),
        "full_horizon": full_metrics,
        "executed_horizon": executed_metrics,
        "executed_prediction_outside_demonstrated_range_fraction": float(outside.mean()),
        "executed_windows_with_any_outside_demonstrated_range": int(outside.any(axis=(1, 2)).sum()),
        "action_names": action_names,
        "demonstrated_action_min": demonstrated_lower.tolist(),
        "demonstrated_action_max": demonstrated_upper.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_metric_csv(output_dir / "metrics_by_dimension.csv", action_names, full_metrics, executed_metrics)
    write_timestep_csv(output_dir / "metrics_by_timestep.csv", full_metrics, executed_metrics, start)
    write_command_candidates(
        output_dir / "predicted_action_windows_DO_NOT_EXECUTE.csv",
        selected, selected_episodes, executed_prediction, action_names,
        demonstrated_lower, demonstrated_upper,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        validation_window_indices=selected,
        episode_indices=selected_episodes,
        action_names=np.asarray(action_names),
        prediction=prediction,
        target=target,
        executed_prediction=executed_prediction,
        executed_target=executed_target,
    )
    try:
        summary["plots"] = save_plots(
            output_dir, action_names, full_metrics, executed_metrics,
            prediction, target, selected, args.plot_examples,
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except ImportError as exc:
        print(f"warning: plots skipped because matplotlib is unavailable: {exc}")

    print(json.dumps({
        "output_dir": str(output_dir),
        "windows": len(selected),
        "executed_mae": executed_metrics["mae"],
        "executed_rmse": executed_metrics["rmse"],
        "outside_demonstrated_range_fraction": float(outside.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
