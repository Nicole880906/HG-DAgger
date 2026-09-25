#!/bin/bash
# Record all topics needed by DVRKDataCollector (keyboard variant).
# Usage:
#   ./record_bag.sh                  # saves to ./bags/YYYY-MM-DD_HH-MM-SS
#   ./record_bag.sh my_output_dir    # saves to my_output_dir

OUTPUT_DIR="${1:-./bags/$(date +%Y-%m-%d_%H-%M-%S)}"

echo "Recording to: $OUTPUT_DIR"
echo "Press Ctrl+C to stop."

ros2 bag record \
    --output "$OUTPUT_DIR" \
    /stereo/left/rectified_downscaled_image \
    /stereo/right/rectified_downscaled_image \
    /PSM1/measured_js \
    /PSM2/measured_js \
    /PSM1/measured_cp \
    /PSM2/measured_cp \
    /PSM1/jaw/measured_js \
    /PSM2/jaw/measured_js
