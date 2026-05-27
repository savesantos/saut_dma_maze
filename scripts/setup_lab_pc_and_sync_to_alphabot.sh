#!/usr/bin/env bash
# Build on the lab PC, ensure a 7x7-rooms policy exists, and sync required
# runtime artifacts to the AlphaBot workspace.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROBOT_HOST="${ROBOT_HOST:-}"
ROBOT_WS="${ROBOT_WS:-~/alphabot2_ws}"
ALGO="${ALGO:-vi}"
SEED="${SEED:-0}"
MAZE_NAME="${MAZE_NAME:-fixture_7x7_rooms}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_ROSDEP="${SKIP_ROSDEP:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_lab_pc_and_sync_to_alphabot.sh --robot-host <user@ip> [options]

Options:
  --robot-host <user@ip>   SSH target (required), example: deec@10.16.140.63
  --robot-ws <path>        Remote workspace path (default: ~/alphabot2_ws)
  --algo <vi|sarsa|qlearning>
                           Training algorithm for auto-generated policy (default: vi)
  --seed <int>             Policy seed (default: 0)
  --maze <name>            Maze fixture name (default: fixture_7x7_rooms)
  --skip-build             Skip local colcon build
  --run-rosdep             Run rosdep install on the PC before build
  -h, --help               Show this help

Environment alternatives:
  ROBOT_HOST, ROBOT_WS, ALGO, SEED, MAZE_NAME, SKIP_BUILD, SKIP_ROSDEP
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-host)
      ROBOT_HOST="$2"
      shift 2
      ;;
    --robot-ws)
      ROBOT_WS="$2"
      shift 2
      ;;
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
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --run-rosdep)
      SKIP_ROSDEP=0
      shift
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

if [[ -z "$ROBOT_HOST" ]]; then
  echo "ERROR: --robot-host is required." >&2
  usage
  exit 1
fi

if [[ "$ALGO" != "vi" && "$ALGO" != "sarsa" && "$ALGO" != "qlearning" ]]; then
  echo "ERROR: --algo must be one of: vi, sarsa, qlearning" >&2
  exit 1
fi

cd "$ROOT_DIR"

# ROS 2 setup scripts reference unset vars (e.g. AMENT_TRACE_SETUP_FILES);
# briefly relax nounset so `source` does not abort under `set -u`.
set +u
source /opt/ros/humble/setup.bash
set -u

if [[ "$SKIP_ROSDEP" -eq 0 ]]; then
  rosdep install -i --from-path src --rosdistro humble -y
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  colcon build --symlink-install
fi

set +u
source install/setup.bash
set -u

policy_glob="data/training/${ALGO}/${MAZE_NAME}/*-seed${SEED}/policy.npz"
latest_policy="$(ls -td ${policy_glob} 2>/dev/null | head -n 1 || true)"

if [[ -z "$latest_policy" ]]; then
  echo "No policy found for ${ALGO}/${MAZE_NAME}/seed${SEED}. Training now..."
  ros2 run maze_mdp train --algo "$ALGO" --maze "$MAZE_NAME" --seed "$SEED" --out data
  latest_policy="$(ls -td ${policy_glob} 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$latest_policy" || ! -f "$latest_policy" ]]; then
  echo "ERROR: policy file was not generated." >&2
  exit 1
fi

echo "Using policy: $latest_policy"

remote_policy_dir="${ROBOT_WS}/shared/policies/${MAZE_NAME}/${ALGO}"
remote_config_dir="${ROBOT_WS}/shared/bringup_config"
remote_scripts_dir="${ROBOT_WS}/scripts"
remote_src_dir="${ROBOT_WS}/src"

# Note: do NOT single-quote these paths on the remote side — ROBOT_WS may
# contain a leading '~' that must be expanded by the remote login shell.
ssh "$ROBOT_HOST" "mkdir -p ${remote_policy_dir} ${remote_config_dir}/mazes ${remote_config_dir}/markers ${remote_config_dir}/params ${remote_scripts_dir} ${remote_src_dir}"

rsync -av "$latest_policy" "${ROBOT_HOST}:${remote_policy_dir}/policy-seed${SEED}.npz"
rsync -av "src/maze_bringup/config/mazes/${MAZE_NAME}.yaml" "${ROBOT_HOST}:${remote_config_dir}/mazes/${MAZE_NAME}.yaml"
rsync -av "src/maze_bringup/config/params.yaml" "${ROBOT_HOST}:${remote_config_dir}/params/params.yaml"

# Sync robot-side ROS packages so `ros2 run maze_mdp ir_driver_hardware` works.
# We only need maze_mdp (ir_driver_hardware lives here) and its interface dep
# maze_msgs. The PC-only packages (maze_bringup, alphabot2_gazebo) are skipped.
rsync -av --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "src/maze_msgs/" "${ROBOT_HOST}:${remote_src_dir}/maze_msgs/"
rsync -av --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "src/maze_mdp/" "${ROBOT_HOST}:${remote_src_dir}/maze_mdp/"

# Sync robot-side helper scripts so they can be run directly on AlphaBot.
rsync -av "scripts/setup_alphabot.sh" "${ROBOT_HOST}:${remote_scripts_dir}/setup_alphabot.sh"
rsync -av "scripts/run_alphabot_stack.sh" "${ROBOT_HOST}:${remote_scripts_dir}/run_alphabot_stack.sh"
ssh "$ROBOT_HOST" "chmod +x ${remote_scripts_dir}/setup_alphabot.sh ${remote_scripts_dir}/run_alphabot_stack.sh"

# The rooms marker map is optional in this repo. Sync it only if present.
if [[ -f "src/maze_bringup/config/markers/${MAZE_NAME}.yaml" ]]; then
  rsync -av "src/maze_bringup/config/markers/${MAZE_NAME}.yaml" "${ROBOT_HOST}:${remote_config_dir}/markers/${MAZE_NAME}.yaml"
fi

cat <<EOF
Sync complete.

Remote policy path:
  ${remote_policy_dir}/policy-seed${SEED}.npz

Next on the robot (REQUIRED — builds the just-synced maze_mdp / maze_msgs):
  bash ${remote_scripts_dir}/setup_alphabot.sh --workspace "${ROBOT_WS}" --skip-git-pull
  bash ${remote_scripts_dir}/run_alphabot_stack.sh --workspace "${ROBOT_WS}" --domain-id <robot_ip_last_octet>
EOF
