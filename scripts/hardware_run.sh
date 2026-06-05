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
#   LINE_THRESHOLD=850 INTERSECTION_ACTIVE_COUNT=4 INTERSECTION_BRAKE_S=0.15
#   PID_LOG_EVERY=25   # 0 disables PID telemetry logs
#   TURN_TIME=0.3      # seconds for in-place rotation; reduce if >90° overshoot (e.g. 0.15)
#   TURN_PWM=12        # motor PWM during turn; lower = slower, more control
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
CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"
LINE_THRESHOLD="${LINE_THRESHOLD:-850}"
INTERSECTION_ACTIVE_COUNT="${INTERSECTION_ACTIVE_COUNT:-4}"
INTERSECTION_BRAKE_S="${INTERSECTION_BRAKE_S:-0.15}"
PID_LOG_EVERY="${PID_LOG_EVERY:-25}"
TURN_TIME="${TURN_TIME:-0.3}"
TURN_PWM="${TURN_PWM:-12}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[run] robot: $ROBOT_HOST"
echo "[run] defaults: rows=$ROWS cols=$COLS row=$START_ROW col=$START_COL heading=$START_HEADING"
echo "[run] camera backend preference: $CAMERA_BACKEND"
echo "[run] intersection stop: threshold=$LINE_THRESHOLD active_count=$INTERSECTION_ACTIVE_COUNT brake_s=$INTERSECTION_BRAKE_S"
echo "[run] PID telemetry log every: $PID_LOG_EVERY cycles"
echo "[run] turn: pwm=$TURN_PWM time=${TURN_TIME}s"

echo "[run] command will execute on the robot in $REMOTE_DIR"
# rpi_ws281x needs /dev/mem access (root) for the NeoPixel ring.
# We use 'sudo env PATH=...' so sudo picks up the same python3 (and
# site-packages) as the deec user, not the bare system interpreter.
# The script degrades gracefully without LEDs if sudo is unavailable.
ssh -t "$ROBOT_HOST" "cd $REMOTE_DIR && sudo env PATH=\$PATH CAMERA_BACKEND=$CAMERA_BACKEND \$(which python3) line_follow_policy.py \
  --policy $POLICY_FILE \
  --rows $ROWS --cols $COLS \
  --row $START_ROW --col $START_COL --heading $START_HEADING \
  --goal-row $GOAL_ROW --goal-col $GOAL_COL \
  --max-steps $MAX_STEPS \
  --line-threshold $LINE_THRESHOLD \
  --intersection-active-count $INTERSECTION_ACTIVE_COUNT \
  --intersection-brake-s $INTERSECTION_BRAKE_S \
  --pid-log-every $PID_LOG_EVERY \
  --turn-time $TURN_TIME \
  --turn-pwm $TURN_PWM \
  $EXTRA_ARGS"
