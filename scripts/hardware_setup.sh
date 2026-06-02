#!/usr/bin/env bash
# Prepare the AlphaBot2 Raspberry Pi for policy execution on hardware.
#
# Copies the robot-side runner + camera helper + one policy bundle to the Pi.
# Defaults target the 7x7 rooms maze artifact layout.
#
# Usage:
#   bash scripts/hardware_setup.sh
#   POLICY_PATH=data/training/qlearning/fixture_7x7_rooms/<run_id>/policy.npz \
#     bash scripts/hardware_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_HOST="${ROBOT_HOST:-deec@10.16.140.69}"
REMOTE_DIR="${REMOTE_DIR:-alphabot2_ws/src/}"
MAZE_NAME="${MAZE_NAME:-fixture_7x7_rooms}"
ALGO="${ALGO:-qlearning}"
POLICY_PATH="${POLICY_PATH:-}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ -z "$POLICY_PATH" ]]; then
    latest_run="$(ls -td "data/training/${ALGO}/${MAZE_NAME}/"* 2>/dev/null | head -1 || true)"
    if [[ -z "$latest_run" ]]; then
        echo "ERROR: no runs found under data/training/${ALGO}/${MAZE_NAME}/" >&2
        echo "Set POLICY_PATH to a policy.npz file explicitly." >&2
        exit 1
    fi
    POLICY_PATH="${latest_run}/policy.npz"
fi

if [[ ! -f "$POLICY_PATH" ]]; then
    echo "ERROR: policy file not found: $POLICY_PATH" >&2
    exit 1
fi

RUNNER_PATH="src/maze_mdp/maze_mdp/hardware/line_follow_policy.py"
ALIGNER_PATH="src/maze_mdp/maze_mdp/hardware/camera_align.py"

if [[ ! -f "$RUNNER_PATH" || ! -f "$ALIGNER_PATH" ]]; then
    echo "ERROR: hardware runner files not found under src/maze_mdp/maze_mdp/hardware/." >&2
    exit 1
fi

echo "[setup] robot: $ROBOT_HOST"
echo "[setup] remote dir: $REMOTE_DIR"
echo "[setup] policy: $POLICY_PATH"

ssh "$ROBOT_HOST" "mkdir -p $REMOTE_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
    echo "[setup] installing python deps on robot (sudo may prompt for password)..."
    ssh "$ROBOT_HOST" "sudo apt update && sudo apt install -y python3-picamera2 python3-libcamera python3-numpy"
fi

echo "[setup] copying runner + policy to robot..."
scp "$RUNNER_PATH" "$ROBOT_HOST:$REMOTE_DIR/line_follow_policy.py"
scp "$ALIGNER_PATH" "$ROBOT_HOST:$REMOTE_DIR/camera_align.py"
scp "$POLICY_PATH" "$ROBOT_HOST:$REMOTE_DIR/policy.npz"

echo "[done] hardware assets are on $ROBOT_HOST:$REMOTE_DIR"
