#!/usr/bin/env bash
# Prepare the AlphaBot-side workspace (alphabot2_ws), update from git,
# install dependencies, and build.
#
# The AlphaBot2 runs on a Raspberry Pi. Heavy C++ packages like cv_bridge or
# image_transport take several minutes and can OOM if built in parallel.
# By default we therefore:
#   * only (re)build the packages strictly needed on the robot
#     (maze_mdp + maze_msgs and their deps via --packages-up-to);
#   * use --parallel-workers 1 and console_direct+ so the build is visible
#     and the Pi does not swap itself to death.
# Use --full to rebuild every package in the workspace.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/alphabot2_ws}"
SKIP_GIT_PULL="${SKIP_GIT_PULL:-0}"
SKIP_ROSDEP="${SKIP_ROSDEP:-1}"
FULL_BUILD="${FULL_BUILD:-0}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_alphabot.sh [options]

Options:
  --workspace <path>      AlphaBot workspace path (default: ~/alphabot2_ws)
  --skip-git-pull         Do not run git pull
  --run-rosdep            Run rosdep install before build
  --full                  Build the whole workspace (default: only maze_mdp + deps)
  --parallel-workers <n>  colcon parallel workers (default: 1, gentle on the Pi)
  -h, --help              Show this help

Environment alternatives:
  WORKSPACE, SKIP_GIT_PULL, SKIP_ROSDEP, FULL_BUILD, PARALLEL_WORKERS
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
    --full)
      FULL_BUILD=1
      shift
      ;;
    --parallel-workers)
      PARALLEL_WORKERS="$2"
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

# Expand a leading '~' that may have survived quoting on the command line
# (e.g. --workspace "~/alphabot2_ws" — bash does NOT expand ~ inside quotes).
case "$WORKSPACE" in
  '~'|'~/'*) WORKSPACE="${HOME}${WORKSPACE#\~}" ;;
esac

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

# Build options common to both modes.
common_args=(
  --symlink-install
  --parallel-workers "$PARALLEL_WORKERS"
  --event-handlers console_direct+
  --cmake-args -DCMAKE_BUILD_TYPE=Release
)

if [[ "$FULL_BUILD" -eq 1 ]]; then
  echo "[setup_alphabot] Full workspace build (--parallel-workers ${PARALLEL_WORKERS})"
  colcon build "${common_args[@]}"
else
  # Only build what the robot actually runs: ir_driver_hardware (maze_mdp)
  # plus its interface dependency maze_msgs. --packages-up-to pulls in any
  # missing build deps automatically.
  pkgs=()
  [[ -d src/maze_msgs ]] && pkgs+=(maze_msgs)
  [[ -d src/maze_mdp  ]] && pkgs+=(maze_mdp)
  if [[ ${#pkgs[@]} -eq 0 ]]; then
    echo "[setup_alphabot] No maze_* packages under src/. Falling back to full build."
    colcon build "${common_args[@]}"
  else
    echo "[setup_alphabot] Building only: ${pkgs[*]} (use --full for the whole ws)"
    colcon build "${common_args[@]}" --packages-up-to "${pkgs[@]}"
  fi
fi

set +u
source install/setup.bash
set -u

echo "AlphaBot workspace is ready: $WORKSPACE"
