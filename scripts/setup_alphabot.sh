#!/usr/bin/env bash
# Prepare the AlphaBot-side workspace (alphabot2_ws), update from git,
# install dependencies, and build.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/alphabot2_ws}"
SKIP_GIT_PULL="${SKIP_GIT_PULL:-0}"
SKIP_ROSDEP="${SKIP_ROSDEP:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_alphabot.sh [options]

Options:
  --workspace <path>   AlphaBot workspace path (default: ~/alphabot2_ws)
  --skip-git-pull      Do not run git pull
  --run-rosdep         Run rosdep install before build
  -h, --help           Show this help

Environment alternatives:
  WORKSPACE, SKIP_GIT_PULL, SKIP_ROSDEP
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --skip-git-pull)
      SKIP_GIT_PULL=1
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

if [[ ! -d "$WORKSPACE" ]]; then
  echo "ERROR: workspace not found: $WORKSPACE" >&2
  exit 1
fi

cd "$WORKSPACE"
# ROS 2 setup scripts reference unset vars under `set -u`; relax briefly.
set +u
source /opt/ros/humble/setup.bash
set -u

if [[ -d .git && "$SKIP_GIT_PULL" -eq 0 ]]; then
  git pull --ff-only
fi

if [[ "$SKIP_ROSDEP" -eq 0 ]]; then
  rosdep install -i --from-path src --rosdistro humble -y
fi

colcon build --symlink-install
set +u
source install/setup.bash
set -u

echo "AlphaBot workspace is ready: $WORKSPACE"
