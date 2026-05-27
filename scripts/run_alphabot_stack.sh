#!/usr/bin/env bash
# Run the AlphaBot-side runtime stack:
# - alphabot2 base bringup
# - alphabot2 motion_driver
# - maze_mdp ir_driver_hardware
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/alphabot2_ws}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_alphabot_stack.sh [options]

Options:
  --workspace <path>   AlphaBot workspace path (default: ~/alphabot2_ws)
  --domain-id <int>    ROS_DOMAIN_ID value to export
  -h, --help           Show this help

Environment alternatives:
  WORKSPACE, ROS_DOMAIN_ID_VALUE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --domain-id)
      ROS_DOMAIN_ID_VALUE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$WORKSPACE" ]]; then
  echo "ERROR: workspace not found: $WORKSPACE" >&2
  exit 1
fi

# ROS 2 setup scripts reference unset vars under `set -u`; relax briefly.
set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u

if [[ -n "$ROS_DOMAIN_ID_VALUE" ]]; then
  export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
fi

cleanup() {
  if [[ -n "${PID_LAUNCH:-}" ]]; then kill "$PID_LAUNCH" 2>/dev/null || true; fi
  if [[ -n "${PID_MOTION:-}" ]]; then kill "$PID_MOTION" 2>/dev/null || true; fi
  if [[ -n "${PID_IR:-}" ]]; then kill "$PID_IR" 2>/dev/null || true; fi
}

trap cleanup EXIT INT TERM

ros2 launch alphabot2 alphabot2_launch.py &
PID_LAUNCH=$!

sleep 2

ros2 run alphabot2 motion_driver &
PID_MOTION=$!

sleep 2

ros2 run maze_mdp ir_driver_hardware &
PID_IR=$!

echo "AlphaBot stack started. PIDs: launch=${PID_LAUNCH}, motion=${PID_MOTION}, ir=${PID_IR}"
echo "Press Ctrl-C to stop all processes."

wait
