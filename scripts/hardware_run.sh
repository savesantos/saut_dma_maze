#!/usr/bin/env bash
# Run policy execution on the AlphaBot2 Raspberry Pi.
#
# Defaults:
# - Maze size: rows=7, cols=7
# - Start pose: row=0, col=0, heading=1 (East)
# - Goal: row=6, col=0 (fixture_7x7_rooms)
#
# Usage:
#   bash scripts/hardware_run.sh
#
# Optional overrides (environment variables):
#   REMOTE_DIR=alphabot2_ws/src/
#   POLICY_FILE=policy.npz
#   ROWS=7 COLS=7 START_ROW=0 START_COL=0 START_HEADING=1
#   GOAL_ROW=6 GOAL_COL=0
#   MAX_STEPS=200
#   EXTRA_ARGS='--camera-align-debug /tmp/align_frames'
set -euo pipefail

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

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[run] robot: $ROBOT_HOST"
echo "[run] defaults: rows=$ROWS cols=$COLS row=$START_ROW col=$START_COL heading=$START_HEADING"

echo "[run] command will execute on the robot in $REMOTE_DIR"
# sudo is required by rpi_ws281x for /dev/mem access (NeoPixel ring).
# The script degrades gracefully without LEDs if sudo is unavailable.
ssh -t "$ROBOT_HOST" "cd $REMOTE_DIR && sudo python3 line_follow_policy.py \
  --policy $POLICY_FILE \
  --rows $ROWS --cols $COLS \
  --row $START_ROW --col $START_COL --heading $START_HEADING \
  --goal-row $GOAL_ROW --goal-col $GOAL_COL \
  --max-steps $MAX_STEPS \
  $EXTRA_ARGS"
