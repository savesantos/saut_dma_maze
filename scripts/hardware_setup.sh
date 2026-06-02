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
CAMERA_REQUIRED="${CAMERA_REQUIRED:-1}"

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
    # Upload and execute a temporary remote script to avoid TTY/heredoc
    # interactions that can echo script contents and confuse prompts.
    tmp_setup_script="$(mktemp)"
    cat >"$tmp_setup_script" <<'REMOTE_SETUP'
set -euo pipefail

apt update

for pkg in python3-numpy; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
        echo "[setup] installing apt package: $pkg"
        apt install -y "$pkg"
    else
        echo "[setup] WARNING: apt package not available on this image: $pkg"
    fi
done

# Try distro packages first for camera stack.
for pkg in python3-picamera2 python3-libcamera; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
        echo "[setup] installing apt package: $pkg"
        apt install -y "$pkg"
    else
        echo "[setup] WARNING: apt package not available on this image: $pkg"
    fi
done

# Ubuntu images often miss picamera2/libcamera python bindings in apt.
# Fallback to pip for picamera2 and libcamera if still absent.
if ! python3 - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("picamera2") is not None else 1)
PY
then
    echo "[setup] attempting pip fallback for picamera2..."
    apt install -y python3-pip python3-wheel || true
    python3 -m pip install --break-system-packages --upgrade pip setuptools wheel || true
    python3 -m pip install --break-system-packages picamera2 || true
fi

if ! python3 - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("libcamera") is not None else 1)
PY
then
    echo "[setup] attempting pip fallback for libcamera..."
    python3 -m pip install --break-system-packages libcamera || true
fi

python3 - <<'PY'
import importlib.util

missing = [
    name for name in ("picamera2", "libcamera", "numpy")
    if importlib.util.find_spec(name) is None
]

if missing:
    print("[setup] WARNING: missing Python modules after apt install: {}"
          .format(", ".join(missing)))
    print("[setup] camera alignment may be unavailable until these modules are installed.")
else:
    print("[setup] Python module check passed: picamera2, libcamera, numpy")

# Exit non-zero so caller can fail fast when camera is required.
if missing:
    raise SystemExit(2)
PY
REMOTE_SETUP

    remote_setup_path="/tmp/hardware_setup_deps_$$.sh"
    scp "$tmp_setup_script" "$ROBOT_HOST:$remote_setup_path"
    rm -f "$tmp_setup_script"
    if ! ssh -tt "$ROBOT_HOST" "chmod +x '$remote_setup_path' && sudo bash '$remote_setup_path'; rc=\$?; rm -f '$remote_setup_path'; exit \$rc"; then
        if [[ "$CAMERA_REQUIRED" == "1" ]]; then
            echo "ERROR: required camera dependencies are still missing on the robot." >&2
            echo "Set CAMERA_REQUIRED=0 to continue without camera alignment." >&2
            exit 1
        fi
        echo "[setup] WARNING: camera dependency install incomplete; continuing because CAMERA_REQUIRED=0"
    fi
fi

echo "[setup] copying runner + policy to robot..."
scp "$RUNNER_PATH" "$ROBOT_HOST:$REMOTE_DIR/line_follow_policy.py"
scp "$ALIGNER_PATH" "$ROBOT_HOST:$REMOTE_DIR/camera_align.py"
scp "$POLICY_PATH" "$ROBOT_HOST:$REMOTE_DIR/policy.npz"

echo "[done] hardware assets are on $ROBOT_HOST:$REMOTE_DIR"
