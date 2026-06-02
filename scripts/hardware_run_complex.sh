#!/usr/bin/env bash
# Run the complex ROS-based line follower on the AlphaBot2 Raspberry Pi.
#
# Usage:
#   bash scripts/hardware_run_complex.sh
#
# Optional overrides (environment variables):
#   ROBOT_HOST=deec@10.16.140.69
#   REMOTE_DIR=alphabot2_ws/src/
#   CAMERA_BACKEND=opencv
set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-deec@10.16.140.69}"
REMOTE_DIR="${REMOTE_DIR:-alphabot2_ws/src/}"
CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"

echo "[run-complex] robot: $ROBOT_HOST"
echo "[run-complex] remote dir: $REMOTE_DIR"
echo "[run-complex] camera backend preference: $CAMERA_BACKEND"

echo "[run-complex] starting line_follow_complex.py on robot"
# Keep sudo so GPIO/NeoPixel-dependent code can access privileged devices.
# Source ROS first because line_follow_complex.py imports rclpy and ROS messages.
ssh -t "$ROBOT_HOST" "cd $REMOTE_DIR && sudo -E env PATH=\$PATH CAMERA_BACKEND=$CAMERA_BACKEND bash -lc 'source /opt/ros/humble/setup.bash && python3 line_follow_complex.py'"
