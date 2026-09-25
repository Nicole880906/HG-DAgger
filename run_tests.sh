#!/usr/bin/env bash
# Run the test suite in both environments it needs.
#
#   bash run_tests.sh          # ROS interpreter, then the training container
#   bash run_tests.sh --ros    # only the ROS half
#   bash run_tests.sh --docker # only the container half
#
# No single interpreter can run everything: the ROS install has rclpy and
# OpenCV but no torch, and the training container has torch, zarr and pytorch3d
# but its conda Python cannot load ROS's C extensions. tests/conftest.py leaves
# out whatever is unrunnable and prints what it left out, so each half is green
# on its own -- run both to cover the suite.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
IMAGE="${IMAGE:-robodiff_ros}"

want_ros=1
want_docker=1
case "${1:-}" in
    --ros) want_docker=0 ;;
    --docker) want_ros=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    "") ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
esac

status=0

if [[ "$want_ros" == 1 ]]; then
    echo "=== ROS interpreter ($ROS_PYTHON) ==="
    if "$ROS_PYTHON" -c "import rclpy" >/dev/null 2>&1; then
        "$ROS_PYTHON" -m pytest "$REPO_DIR/tests" "${@:2}" || status=1
    else
        echo "skipped: $ROS_PYTHON cannot import rclpy (source /opt/ros/humble/setup.bash)"
        status=1
    fi
    echo
fi

if [[ "$want_docker" == 1 ]]; then
    echo "=== training container ($IMAGE) ==="
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        # PYTEST_DISABLE_PLUGIN_AUTOLOAD: ROS's site-packages are on PYTHONPATH
        # inside the image, and pytest would try to load its launch_testing
        # plugin into conda's Python and fail on a missing dependency.
        docker run --rm \
            -v "$REPO_DIR":/docker-ros/ws \
            -e DOCKER_UID="$(id -u)" -e DOCKER_GID="$(id -g)" -e DOCKER_USER="$(id -un)" \
            "$IMAGE" bash -lc '
                PY=/opt/conda/envs/robodiff/bin/python
                $PY -c "import pytest" >/dev/null 2>&1 || $PY -m pip install -q pytest
                cd /docker-ros/ws && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests
            ' || status=1
    else
        echo "skipped: docker image '$IMAGE' not found (bash start_container.sh --build)"
        status=1
    fi
fi

exit "$status"
