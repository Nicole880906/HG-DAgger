#!/bin/bash
# Launch the combined ROS + Diffusion Policy container for diffusion_ws.
#
#   bash start_container.sh --build      # (re)build the image, then exit
#   bash start_container.sh              # interactive shell
#   bash start_container.sh -- CMD ...   # run a single command
#
# One container covers the whole offline pipeline. Inside it:
#
#   conda activate robodiff                       # for everything below
#   python data_processing/convert_drawing_6d_abs.py EPISODES OUT.zarr
#   DP_DATASET=OUT.zarr bash train_drawing_policy.sh --train
#
# `python3` is deliberately the ROS interpreter until you activate robodiff, so
# data collection behaves exactly as it does in dp3_ws.
#
# The repository is bind-mounted at /docker-ros/ws, the same path dp3_ws uses,
# matching the editable diffusion-policy install baked into the image. The
# entrypoint (inherited from dp3_surgflow) sources ROS, then creates a user
# matching the host UID/GID and re-execs, so generated files are not owned by
# root.
#
# Image and container are named robodiff_ros, distinct from the training-only
# `robodiff` image (diffusion.dockerfile) and from dp3_ws's `dp3_surgflow`.
#
# Closed-loop deploy runs on the dVRK control desktop against its own ROS 2
# install, not in this container. See "Where each step runs" in the README.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=robodiff_ros
BASE_IMAGE=dp3_surgflow

if [[ "${1:-}" == "--build" ]]; then
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        echo "error: base image '$BASE_IMAGE' not found." >&2
        echo "" >&2
        echo "It cannot be rebuilt from a dockerfile (expired ROS apt key; see" >&2
        echo "dp3_ws/SETUP.md). Transfer it from a machine that has it:" >&2
        echo "" >&2
        echo "  docker save $BASE_IMAGE | gzip > ${BASE_IMAGE}.tar.gz" >&2
        echo "  gunzip -c ${BASE_IMAGE}.tar.gz | docker load" >&2
        echo "" >&2
        echo "For a training-only machine with no dVRK, build the smaller" >&2
        echo "ROS-free image instead:" >&2
        echo "  docker build -f diffusion.dockerfile -t robodiff ." >&2
        exit 1
    fi
    exec docker build -f "$REPO_DIR/robodiff_ros.dockerfile" -t "$IMAGE" "$REPO_DIR"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: image '$IMAGE' not found. Build it first:" >&2
    echo "  bash start_container.sh --build" >&2
    exit 1
fi

[[ "${1:-}" == "--" ]] && shift

xhost +local:docker >/dev/null 2>&1 || true

TTY_FLAGS=(-i)
[[ -t 0 ]] && TTY_FLAGS+=(-t)

exec docker run "${TTY_FLAGS[@]}" --rm \
    --env "DISPLAY" \
    -e DOCKER_UID="$(id -u)" \
    -e DOCKER_GID="$(id -g)" \
    -e DOCKER_USER="$(id -un)" \
    --net=host \
    --ipc=host \
    --pid=host \
    --gpus all \
    --volume "$REPO_DIR":/docker-ros/ws \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --privileged \
    --name "$IMAGE" \
    "$IMAGE" "$@"
