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
PYTHON_ONLY="${PYTHON_ONLY:-0}"
USE_APT_VENDOR="${USE_APT_VENDOR:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"

# Vendor packages we prefer to install from apt instead of compiling from source.
# Compiling them from source on the Raspberry Pi is what was eating all the RAM
# and stalling the build for many minutes. These are all available in the
# official ros-humble-* apt repo, so we use those binaries and tell colcon to
# ignore the in-tree sources via COLCON_IGNORE files.
APT_VENDOR_SRC_DIRS=(
  cv_bridge
  image_transport
  image_common
  vision_opencv
  camera_calibration_parsers
  v4l2_camera
)
APT_VENDOR_PACKAGES=(
  ros-humble-cv-bridge
  ros-humble-image-transport
  ros-humble-image-common
  ros-humble-vision-opencv
  ros-humble-camera-calibration-parsers
  ros-humble-v4l2-camera
)

usage() {
  cat <<'EOF'
Usage: scripts/setup_alphabot.sh [options]

Options:
  --workspace <path>      AlphaBot workspace path (default: ~/alphabot2_ws)
  --skip-git-pull         Do not run git pull
  --run-rosdep            Run rosdep install before build
  --python-only           Do NOT run colcon. Just refresh maze_mdp's pure-Python
                          sources in the existing install tree. Use this for
                          everyday iteration once maze_msgs has been built once.
  --full                  Build every non-ignored package in the workspace
                          (default: only maze_msgs + maze_mdp).
  --no-apt-vendor         Do NOT install heavy ROS vendor packages from apt and
                          do NOT mark their in-tree sources with COLCON_IGNORE.
                          Use this only if you really want to compile cv_bridge
                          / image_transport / v4l2_camera from source.
  --parallel-workers <n>  colcon parallel workers (default: 1, gentle on the Pi)
  -h, --help              Show this help

Environment alternatives:
  WORKSPACE, SKIP_GIT_PULL, SKIP_ROSDEP, FULL_BUILD, PYTHON_ONLY,
  USE_APT_VENDOR, PARALLEL_WORKERS
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
    --python-only)
      PYTHON_ONLY=1
      shift
      ;;
    --no-apt-vendor)
      USE_APT_VENDOR=0
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

# Install heavy ROS vendor packages from apt, and mark the in-tree copies so
# colcon skips them entirely. Idempotent: safe to re-run.
if [[ "$USE_APT_VENDOR" -eq 1 && "$PYTHON_ONLY" -eq 0 ]]; then
  missing_pkgs=()
  for pkg in "${APT_VENDOR_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing_pkgs+=("$pkg")
    fi
  done
  if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
    echo "[setup_alphabot] Installing vendor packages from apt (avoids compiling on the Pi):"
    printf '  %s\n' "${missing_pkgs[@]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_pkgs[@]}"
  else
    echo "[setup_alphabot] Vendor apt packages already installed."
  fi

  for d in "${APT_VENDOR_SRC_DIRS[@]}"; do
    if [[ -d "src/$d" && ! -f "src/$d/COLCON_IGNORE" ]]; then
      touch "src/$d/COLCON_IGNORE"
      echo "[setup_alphabot] Marked src/$d with COLCON_IGNORE (using apt build instead)."
    fi
    # Drop stale build/install artefacts so colcon doesn't trip over them.
    if [[ -d "build/$d" || -d "install/$d" ]]; then
      rm -rf "build/$d" "install/$d"
      echo "[setup_alphabot] Removed stale build/install for $d."
    fi
  done
fi


if [[ "$SKIP_ROSDEP" -eq 0 ]]; then
  rosdep install -i --from-path src --rosdistro humble -y
fi

if [[ "$PYTHON_ONLY" -eq 1 ]]; then
  # Fast path: maze_mdp is a pure-Python (ament_python) package, so we can
  # just refresh its sources in the existing install tree without invoking
  # colcon / cmake / g++ at all. This is the everyday iteration path on the
  # robot. It REQUIRES that maze_msgs has already been built at least once
  # (which produces the architecture-specific typesupport .so files).
  src_dir="$WORKSPACE/src/maze_mdp/maze_mdp"
  # Look up the install dir from the existing build. ament_python installs
  # the python package under install/maze_mdp/lib/pythonX.Y/site-packages/.
  candidate_dirs=( "$WORKSPACE"/install/maze_mdp/lib/python*/site-packages/maze_mdp )
  install_dir="${candidate_dirs[0]}"
  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: $src_dir not found. Sync maze_mdp first." >&2
    exit 1
  fi
  if [[ ! -d "$install_dir" ]]; then
    echo "ERROR: $install_dir not found." >&2
    echo "  maze_mdp has never been built here. Run this script once without" >&2
    echo "  --python-only so colcon creates the install tree, then use" >&2
    echo "  --python-only for subsequent iterations." >&2
    exit 1
  fi
  if [[ ! -d "$WORKSPACE/install/maze_msgs" ]]; then
    echo "ERROR: $WORKSPACE/install/maze_msgs not found." >&2
    echo "  maze_msgs needs an arch-native build on the robot at least once." >&2
    echo "  Run: bash $0 --skip-git-pull   (without --python-only)" >&2
    exit 1
  fi
  echo "[setup_alphabot] --python-only: refreshing maze_mdp sources"
  echo "  src     : $src_dir"
  echo "  install : $install_dir"
  rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$src_dir"/ "$install_dir"/
  set +u
  source install/setup.bash
  set -u
  echo "AlphaBot workspace is ready (python-only refresh): $WORKSPACE"
  exit 0
fi

# Build options common to both modes.
# MAKEFLAGS=-j1 forces single-threaded make/ninja inside each package — without
# this, even --parallel-workers 1 spawns N compiler jobs per package and OOMs
# a 1 GB Pi during cv_bridge / image_transport.
export MAKEFLAGS="-j1"
common_args=(
  --symlink-install
  --parallel-workers "$PARALLEL_WORKERS"
  --event-handlers console_direct+
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_JOB_POOLS=compile=1
)

if [[ "$FULL_BUILD" -eq 1 ]]; then
  echo "[setup_alphabot] Full workspace build (--parallel-workers ${PARALLEL_WORKERS})"
  colcon build "${common_args[@]}"
else
  # Only build what the robot actually runs: ir_driver_hardware (maze_mdp)
  # plus its interface dependency maze_msgs.
  #
  # NOTE: we deliberately use --packages-select (not --packages-up-to) here.
  # maze_mdp's package.xml lists cv_bridge / sensor_msgs etc. as <depend>, but
  # ir_driver_hardware only needs them at runtime, and the system already
  # provides them under /opt/ros/humble. Building cv_bridge from source on the
  # Pi compiles OpenCV bindings in C++ and reliably OOMs a 1 GB Pi.
  pkgs=()
  [[ -d src/maze_msgs ]] && pkgs+=(maze_msgs)
  [[ -d src/maze_mdp  ]] && pkgs+=(maze_mdp)
  if [[ ${#pkgs[@]} -eq 0 ]]; then
    echo "[setup_alphabot] No maze_* packages under src/. Falling back to full build."
    colcon build "${common_args[@]}"
  else
    echo "[setup_alphabot] Building only (no deps from src/): ${pkgs[*]}"
    echo "[setup_alphabot] Use --full to rebuild every package in the workspace."
    colcon build "${common_args[@]}" --packages-select "${pkgs[@]}"
  fi
fi

set +u
source install/setup.bash
set -u

echo "AlphaBot workspace is ready: $WORKSPACE"
