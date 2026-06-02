#!/usr/bin/env bash
# Run the complex ROS-based line follower on the AlphaBot2 Raspberry Pi.
#
# Defaults:
# - Maze size: rows=7, cols=7
# - Start pose: row=0, col=0, heading=1 (East)
# - Goal: row=6, col=0 (fixture_7x7_rooms)
#
# Usage:
#   bash scripts/hardware_run_complex.sh
#
# Optional overrides (environment variables):
#   REMOTE_DIR=alphabot2_ws/src/
#   POLICY_FILE=policy.npz
#   ROWS=7 COLS=7 START_ROW=0 START_COL=0 START_HEADING=1
#   GOAL_ROW=6 GOAL_COL=0
#   MAX_STEPS=200
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_HOST="${ROBOT_HOST:-deec@10.16.140.69}"
REMOTE_DIR="${REMOTE_DIR:-alphabot2_ws/src/}"
POLICY_FILE="${POLICY_FILE:-policy.npz}"

ROWS="${ROWS:-7}"
COLS="${COLS:-7}"
START_ROW="${START_ROW:-0}"
START_COL="${START_COL:-0}"
START_HEADING="${START_HEADING:-1}"
GOAL_ROW="${GOAL_ROW:-6}"
GOAL_COL="${GOAL_COL:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"
LOCAL_COMPLEX_RUNNER="src/maze_mdp/maze_mdp/hardware/line_follow_complex.py"

# ROS 2 discovery domain: default to robot IP last octet unless explicitly set.
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  ROBOT_IP="${ROBOT_HOST##*@}"
  ROS_DOMAIN_ID="${ROBOT_IP##*.}"
fi
ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

echo "[run-complex] robot: $ROBOT_HOST"
echo "[run-complex] defaults: rows=$ROWS cols=$COLS row=$START_ROW col=$START_COL heading=$START_HEADING"
echo "[run-complex] goal: row=$GOAL_ROW col=$GOAL_COL"
echo "[run-complex] camera backend preference: $CAMERA_BACKEND"
echo "[run-complex] ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"

if [[ ! -f "$LOCAL_COMPLEX_RUNNER" ]]; then
  echo "ERROR: local runner not found: $LOCAL_COMPLEX_RUNNER" >&2
  exit 1
fi

echo "[run-complex] ensuring remote directory exists"
ssh "$ROBOT_HOST" "mkdir -p $REMOTE_DIR"

echo "[run-complex] syncing latest line_follow_complex.py to robot"
scp "$LOCAL_COMPLEX_RUNNER" "$ROBOT_HOST:$REMOTE_DIR"

REMOTE_POLICY_FILE="$POLICY_FILE"
if [[ -f "$POLICY_FILE" ]]; then
  echo "[run-complex] syncing local policy file to robot: $POLICY_FILE"
  scp "$POLICY_FILE" "$ROBOT_HOST:$REMOTE_DIR"
  REMOTE_POLICY_FILE="$(basename "$POLICY_FILE")"
fi

echo "[run-complex] starting line_follow_complex.py on robot"
# Keep sudo so GPIO/NeoPixel-dependent code can access privileged devices.
# Source ROS first because line_follow_complex.py imports rclpy and ROS messages.
ssh -t "$ROBOT_HOST" "cd $REMOTE_DIR && sudo -E env PATH=\$PATH CAMERA_BACKEND=$CAMERA_BACKEND ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY bash -lc 'source /opt/ros/humble/setup.bash && python3 line_follow_complex.py \
  --policy $REMOTE_POLICY_FILE \
  --rows $ROWS --cols $COLS \
  --row $START_ROW --col $START_COL --heading $START_HEADING \
  --goal-row $GOAL_ROW --goal-col $GOAL_COL \
  --max-steps $MAX_STEPS'"
