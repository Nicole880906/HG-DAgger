#!/usr/bin/env bash
# Validate a replay buffer and launch Diffusion Policy training for the drawing task.
#
# Usage:
#   bash train_drawing_policy.sh --check          # dataset adapter check only
#   bash train_drawing_policy.sh --smoke          # two epochs, three batches (default)
#   bash train_drawing_policy.sh --train [Hydra overrides ...]
#
# Works with any Zarr produced by data_processing/convert_drawing_6d_abs.py,
# whether it holds demonstrations only or demonstrations plus the expert
# corrections merged in by dagger/merge_interventions.py -- the state and action
# dimensions are read off the Zarr and forwarded to Hydra as shape_meta
# overrides, so nothing here needs configuring per round.
#
# Override the dataset/output with DP_DATASET / DP_OUTPUT:
#
#   DP_DATASET=data/diffusion_policy/circle.zarr \
#   DP_OUTPUT=outputs/circle_round0 \
#   bash train_drawing_policy.sh --train
#
# The config names below (train_surgflow_image_workspace, surgflow_image) are
# the vendored diffusion_policy library's own and are task-neutral despite the
# name; leave them alone.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP_DIR="$ROOT_DIR/src/diffusion"
export DP_DATASET="${DP_DATASET:-$ROOT_DIR/data/diffusion_policy/drawing.zarr}"
export DP_OUTPUT="${DP_OUTPUT:-$ROOT_DIR/outputs}"
export PYTHONPATH="$DP_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Resolve the interpreter. In the combined ROS container `python` is Humble's
# 3.10, which has no diffusion_policy install -- so when robodiff is not the
# active env, point straight at its binary rather than failing deep inside
# train.py. Set PYTHON_BIN to override any of this.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ "${CONDA_PREFIX:-}" == */robodiff ]]; then
        PYTHON_BIN=python
    elif [[ -x "${CONDA_DIR:-/opt/conda}/envs/robodiff/bin/python" ]]; then
        PYTHON_BIN="${CONDA_DIR:-/opt/conda}/envs/robodiff/bin/python"
    else
        PYTHON_BIN=python
    fi
fi

mode="smoke"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --check) mode="check"; shift ;;
        --smoke) mode="smoke"; shift ;;
        --train) mode="train"; shift ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "error: first argument must be --check, --smoke, or --train" >&2
            exit 2
            ;;
    esac
fi

if [[ ! -d "$DP_DATASET" ]]; then
    echo "error: dataset not found: $DP_DATASET" >&2
    echo "hint: convert episodes first:" >&2
    echo "  python data_processing/convert_drawing_6d_abs.py <episodes> <out>.zarr" >&2
    exit 1
fi

# Resolve to absolute paths: training runs after `cd "$DP_DIR"`, so a relative
# dataset/output path would otherwise be interpreted from src/diffusion.
DP_DATASET="$(cd "$DP_DATASET" && pwd)"
export DP_DATASET
case "$DP_OUTPUT" in
    /*) ;;
    *) DP_OUTPUT="$ROOT_DIR/$DP_OUTPUT" ;;
esac
export DP_OUTPUT

# Hand the paths to Hydra under the names its config actually reads. The
# vendored surgflow configs resolve ${oc.env:SURGFLOW_DP_DATASET,...} and
# ${oc.env:SURGFLOW_DP_OUTPUT,...}, whose fallbacks are absolute paths from the
# machine they were written on. Exporting only DP_DATASET left those fallbacks
# in force, so --check passed (it builds the dataset itself, from DP_DATASET)
# and --train then died inside hydra.utils.instantiate with a FileNotFoundError
# for a Zarr belonging to another workspace.
export SURGFLOW_DP_DATASET="$DP_DATASET"
export SURGFLOW_DP_OUTPUT="$DP_OUTPUT"

# Auto-detect state/action dims, action representation, and goal presence.
read -r STATE_DIM ACTION_DIM ACTION_REPR HAS_GOAL POLICY_RATE_HZ < <(
    "$PYTHON_BIN" - <<'PY'
import math, os, zarr
root = zarr.open_group(os.environ["DP_DATASET"], mode="r")
state_dim = int(root["data/agent_pos"].shape[1])
action_dim = int(root["data/action"].shape[1])
action_repr = root.attrs.get("action_representation", "unknown")
has_goal = "start_end_points" in root["data"]
policy_rate_hz = float(root.attrs.get("sample_rate_hz", 5.0))
if not math.isfinite(policy_rate_hz) or policy_rate_hz <= 0:
    raise ValueError(f"sample_rate_hz must be positive, got {policy_rate_hz}")
print(state_dim, action_dim, action_repr, int(has_goal), policy_rate_hz)
PY
)

echo "dataset:     $DP_DATASET"
echo "outputs:     $DP_OUTPUT"
echo "python:      $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "state dim:   $STATE_DIM"
echo "action dim:  $ACTION_DIM"
echo "action repr: $ACTION_REPR"
echo "policy rate: $POLICY_RATE_HZ Hz"
echo "goal:        $([[ "$HAS_GOAL" == "1" ]] && echo "start_end_points" || echo "none")"

# shape_meta overrides so the policy matches this dataset's dims.
shape_overrides=(
    # Explicit, not just the env var above: the override is visible in the run's
    # .hydra/config.yaml, so which Zarr a checkpoint was trained on stays
    # recoverable from the output directory alone.
    "task.dataset_path=$DP_DATASET"
    "task.control_rate_hz=$POLICY_RATE_HZ"
    "task.shape_meta.obs.agent_pos.shape=[$STATE_DIM]"
    "task.shape_meta.action.shape=[$ACTION_DIM]"
)
# The goal is conditioned by default; drop it from shape_meta when the Zarr
# has no start_end_points so the policy's obs encoder matches the dataset.
if [[ "$HAS_GOAL" != "1" ]]; then
    shape_overrides+=("~task.shape_meta.obs.start_end_points")
fi

STATE_DIM="$STATE_DIM" ACTION_DIM="$ACTION_DIM" HAS_GOAL="$HAS_GOAL" "$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys

required = ["torch", "torchvision", "zarr", "hydra", "diffusers", "wandb", "einops", "threadpoolctl"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")
if missing:
    print("\nEnvironment is not ready:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("Activate the Diffusion Policy environment and rerun this command.", file=sys.stderr)
    raise SystemExit(1)

from diffusion_policy.dataset.surgflow_image_dataset import SurgFlowImageDataset

zarr_path = os.environ["DP_DATASET"]
state_dim = int(os.environ["STATE_DIM"])
action_dim = int(os.environ["ACTION_DIM"])
has_goal = os.environ.get("HAS_GOAL") == "1"

dataset = SurgFlowImageDataset(
    zarr_path=zarr_path,
    horizon=16,
    pad_before=1,
    pad_after=3,
    n_obs_steps=2,
    val_ratio=0.10,
    seed=42,
)
validation = dataset.get_validation_dataset()
sample = dataset[0]
assert tuple(sample["obs"]["image"].shape) == (2, 3, 120, 160), sample["obs"]["image"].shape
assert tuple(sample["obs"]["agent_pos"].shape) == (2, state_dim), sample["obs"]["agent_pos"].shape
if has_goal:
    assert tuple(sample["obs"]["start_end_points"].shape) == (2, 4), sample["obs"]["start_end_points"].shape
else:
    assert "start_end_points" not in sample["obs"], "unexpected goal in goal-less dataset"
assert tuple(sample["action"].shape) == (16, action_dim), sample["action"].shape
assert len(dataset) > 0 and len(validation) > 0
print("dataset adapter check passed")
print(f"  train windows:      {len(dataset)}")
print(f"  validation windows: {len(validation)}")
print(f"  image sample:       {tuple(sample['obs']['image'].shape)}")
print(f"  state sample:       {tuple(sample['obs']['agent_pos'].shape)}")
goal_shape = tuple(sample['obs']['start_end_points'].shape) if has_goal else "none"
print(f"  goal sample:        {goal_shape}")
print(f"  action sample:      {tuple(sample['action'].shape)}")
PY

if [[ "$mode" == "check" ]]; then
    exit 0
fi

mkdir -p "$DP_OUTPUT"
cd "$DP_DIR"

overrides=("${shape_overrides[@]}" "$@")
if [[ "$mode" == "smoke" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    overrides+=(
        "training.debug=true"
        "logging.mode=disabled"
        "hydra.run.dir=$DP_OUTPUT/smoke_$timestamp"
    )
    echo "starting two-epoch/three-batch smoke test (state=${STATE_DIM}D, action=${ACTION_REPR})"
else
    # Without this the run lands in the config's built-in output path -- another
    # machine's -- and DP_OUTPUT is silently ignored. Pinning it here is also
    # what puts the checkpoint where the deploy command expects to find it.
    overrides+=("hydra.run.dir=$DP_OUTPUT")
    echo "starting full training (state=${STATE_DIM}D, action=${ACTION_REPR})"
    echo "checkpoints:  $DP_OUTPUT/checkpoints/latest.ckpt"
fi

exec "$PYTHON_BIN" train.py \
    --config-name=train_surgflow_image_workspace \
    "${overrides[@]}"
