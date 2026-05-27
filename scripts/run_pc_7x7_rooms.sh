#!/usr/bin/env bash
# Run maze policy execution from the lab PC for fixture_7x7_rooms starting
# at row=0 col=0 heading=1. If policy is missing after git clone, train it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALGO="${ALGO:-vi}"
SEED="${SEED:-0}"
MAZE_NAME="${MAZE_NAME:-fixture_7x7_rooms}"
START_ROW="${START_ROW:-0}"
START_COL="${START_COL:-0}"
START_HEADING="${START_HEADING:-1}"
IR_DRIVER_BACKEND="${IR_DRIVER_BACKEND:-external}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_pc_7x7_rooms.sh [options]

Options:
  --algo <vi|sarsa|qlearning>     Policy algorithm (default: vi)
  --seed <int>                    Policy seed (default: 0)
  --maze <name>                   Maze fixture (default: fixture_7x7_rooms)
  --start-row <int>               Start row (default: 0)
  --start-col <int>               Start col (default: 0)
  --start-heading <0|1|2|3>       Start heading (default: 1)
  --ir-driver-backend <mode>      auto|hardware|external (default: external)
  --domain-id <int>               Export ROS_DOMAIN_ID in this shell
  -h, --help                      Show this help

Environment alternatives:
  ALGO, SEED, MAZE_NAME, START_ROW, START_COL, START_HEADING,
  IR_DRIVER_BACKEND, ROS_DOMAIN_ID_VALUE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --maze)
      MAZE_NAME="$2"
      shift 2
      ;;
    --start-row)
      START_ROW="$2"
      shift 2
      ;;
    --start-col)
      START_COL="$2"
      shift 2
      ;;
    --start-heading)
      START_HEADING="$2"
      shift 2
      ;;
    --ir-driver-backend)
      IR_DRIVER_BACKEND="$2"
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

if [[ "$ALGO" != "vi" && "$ALGO" != "sarsa" && "$ALGO" != "qlearning" ]]; then
  echo "ERROR: --algo must be one of: vi, sarsa, qlearning" >&2
  exit 1
fi

cd "$ROOT_DIR"

# ROS 2 setup scripts reference unset vars under `set -u`; relax briefly.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

if [[ -n "$ROS_DOMAIN_ID_VALUE" ]]; then
  export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
fi

policy_glob="data/training/${ALGO}/${MAZE_NAME}/*-seed${SEED}/policy.npz"
POLICY_PATH="$(ls -td ${policy_glob} 2>/dev/null | head -n 1 || true)"

if [[ -z "$POLICY_PATH" ]]; then
  echo "No trained policy found for ${ALGO}/${MAZE_NAME}/seed${SEED}. Training now..."
  ros2 run maze_mdp train --algo "$ALGO" --maze "$MAZE_NAME" --seed "$SEED" --out data
  POLICY_PATH="$(ls -td ${policy_glob} 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$POLICY_PATH" || ! -f "$POLICY_PATH" ]]; then
  echo "ERROR: policy file not found after training attempt." >&2
  exit 1
fi

MARKER_MAP_PATH="src/maze_bringup/config/markers/${MAZE_NAME}.yaml"
if [[ ! -f "$MARKER_MAP_PATH" ]]; then
  # Fallback so launch has a valid marker map path in repos where
  # fixture_7x7_rooms marker YAML is not present yet.
  MARKER_MAP_PATH="src/maze_bringup/config/markers/fixture_7x7_loop.yaml"
  if [[ ! -f "$MARKER_MAP_PATH" ]]; then
    echo "ERROR: marker map not found for ${MAZE_NAME} and no fallback available." >&2
    exit 1
  fi
  echo "WARNING: using marker-map fallback: ${MARKER_MAP_PATH}"
fi

echo "Launching with policy: $POLICY_PATH"
echo "Start pose: row=${START_ROW}, col=${START_COL}, heading=${START_HEADING}"

ros2 launch maze_bringup alphabot_maze.launch.py \
  maze_name:="$MAZE_NAME" \
  policy_path:="$POLICY_PATH" \
  start_row:="$START_ROW" \
  start_col:="$START_COL" \
  start_heading:="$START_HEADING" \
  ir_driver_backend:="$IR_DRIVER_BACKEND" \
  marker_map:="$ROOT_DIR/$MARKER_MAP_PATH"
