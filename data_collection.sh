#!/usr/bin/env bash
# Record drawing demonstrations on the dVRK.
#
#   bash data_collection.sh                  # -> data/drawing_circle/
#   bash data_collection.sh --task rectangle # -> data/drawing_rectangle/
#
# Teleoperate the arms from the console (or move them by hand) and draw the
# shape. Episode control is by jaw pinch, so your hands never leave the masters:
#
#   pinch the PSM1 jaw 3x within ~2 s  -> start recording
#   pinch the PSM2 jaw 3x within ~2 s  -> stop and save
#   Ctrl+C                             -> quit
#
# Frames are recorded at 30 Hz. Both arms are stored as joint angles and as
# end-effector pose; data_processing/convert_drawing_6d_abs.py reads the pose.
#
# Frames are saved unmodified. Any other argument is forwarded to the
# collector, and anything it does not recognise goes on to ROS.
#
# Requires ROS 2 Humble and the dVRK stack on this machine's ROS_DOMAIN_ID.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-111}"

TASK=circle
FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            [[ $# -ge 2 ]] || { echo "error: --task needs a value" >&2; exit 2; }
            TASK="$2"; shift 2 ;;
        --task=*)
            TASK="${1#--task=}"; shift ;;
        -h|--help)
            # Print the header comment block, whatever length it is now --
            # a hard-coded line range goes stale every time it is edited.
            awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"
            exit 0 ;;
        *)
            FORWARD+=("$1"); shift ;;
    esac
done

# --task is a shorthand for --task-dir. If the caller passed --task-dir
# explicitly it is in FORWARD and argparse's last-wins behaviour lets it
# override the default placed ahead of it.
TASK_DIR="data/drawing_${TASK}"

source /opt/ros/humble/setup.bash
for overlay in "$REPO_DIR/install/setup.bash" /docker-ros/ws/install/setup.bash; do
    if [[ -f "$overlay" ]]; then
        source "$overlay"
        break
    fi
done

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "task:        ${TASK}"
echo "episodes ->  ${REPO_DIR}/${TASK_DIR}"
echo
echo "  PSM1 jaw pinch 3x -> start episode"
echo "  PSM2 jaw pinch 3x -> end + save"
echo "  Ctrl+C            -> quit"
echo

exec python3 "$REPO_DIR/src/arclab_dvrk/src/data_collection/data_collection_json_gripper.py" \
    --task-dir "$TASK_DIR" "${FORWARD[@]+"${FORWARD[@]}"}"
