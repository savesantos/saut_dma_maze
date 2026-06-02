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
# Optional: local directory containing AlphaBot2.py and TRSensors.py.
# If empty the script will search common locations on the robot.
ALPHABOT2_LIB_DIR="${ALPHABOT2_LIB_DIR:-}"

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

for pkg in python3-numpy python3-opencv v4l-utils; do
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

python3 - <<'PY'
import importlib.util

has_numpy = importlib.util.find_spec("numpy") is not None
has_picamera = importlib.util.find_spec("picamera2") is not None
has_libcamera = importlib.util.find_spec("libcamera") is not None
has_cv2 = importlib.util.find_spec("cv2") is not None

camera_ok = (has_picamera and has_libcamera) or has_cv2
missing = []
if not has_numpy:
    missing.append("numpy")
if not camera_ok:
    missing.append("camera_backend(picamera2+libcamera or cv2)")

if missing:
    print("[setup] WARNING: missing required Python modules after apt install: {}"
          .format(", ".join(missing)))
    print("[setup] camera alignment may be unavailable until these modules are installed.")
else:
    backend = "picamera2/libcamera" if (has_picamera and has_libcamera) else "opencv(cv2)"
    print("[setup] Python module check passed: numpy + {}".format(backend))

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
scp \
    "$RUNNER_PATH" \
    "$ALIGNER_PATH" \
    "$POLICY_PATH" \
    "$ROBOT_HOST:$REMOTE_DIR"

# ---------------------------------------------------------------- AlphaBot2 library
# AlphaBot2.py and TRSensors.py must be co-located with the runner so
# that `python3 line_follow_policy.py` can import them without any
# special PYTHONPATH. Try: (1) project-local copy, (2) ALPHABOT2_LIB_DIR,
# (3) robot search, (4) fail with instructions.
ab2_lib_source=""

# (1) Check project-local copy (src/maze_mdp/maze_mdp/hardware/)
project_lib_dir="src/maze_mdp/maze_mdp/hardware"
if [[ -f "$project_lib_dir/AlphaBot2.py" && -f "$project_lib_dir/TRSensors.py" ]]; then
    ab2_lib_source="$project_lib_dir"
    echo "[setup] found AlphaBot2 library in project: $ab2_lib_source"
fi

# (2) Check override directory
if [[ -z "$ab2_lib_source" && -n "$ALPHABOT2_LIB_DIR" ]]; then
    if [[ -f "$ALPHABOT2_LIB_DIR/AlphaBot2.py" && -f "$ALPHABOT2_LIB_DIR/TRSensors.py" ]]; then
        ab2_lib_source="$ALPHABOT2_LIB_DIR"
        echo "[setup] found AlphaBot2 library in ALPHABOT2_LIB_DIR: $ab2_lib_source"
    fi
fi

# (3) Search common locations on the robot
if [[ -z "$ab2_lib_source" ]]; then
    echo "[setup] searching for AlphaBot2.py + TRSensors.py on the robot..."
    # Ordered list of candidate directories on the Pi.
    for search_dir in \
        "/home/deec" \
        "/home/deec/AlphaBot2" \
        "/home/deec/AlphaBot2-Demo/RaspberryPi/AlphaBot2/python" \
        "/home/deec/AlphaBot2-Demo" \
        "/home/pi" \
        "/home/pi/AlphaBot2" \
        "/home/pi/AlphaBot2-Demo/RaspberryPi/AlphaBot2/python" \
        "/home/pi/AlphaBot2-Demo" \
        "/opt/AlphaBot2"
    do
        result="$(ssh "$ROBOT_HOST" \
            "test -f '${search_dir}/AlphaBot2.py' && test -f '${search_dir}/TRSensors.py' && echo yes || echo no" \
            2>/dev/null || echo no)"
        if [[ "$result" == "yes" ]]; then
            ab2_lib_source="remote:$search_dir"
            echo "[setup] found library on robot at $search_dir"
            break
        fi
    done
fi

# (4) Copy the library
if [[ -n "$ab2_lib_source" ]]; then
    if [[ "$ab2_lib_source" == remote:* ]]; then
        remote_path="${ab2_lib_source#remote:}"
        ssh "$ROBOT_HOST" \
            "cp '${remote_path}/AlphaBot2.py' '${remote_path}/TRSensors.py' '$REMOTE_DIR'"
    else
        scp \
            "$ab2_lib_source/AlphaBot2.py" \
            "$ab2_lib_source/TRSensors.py" \
            "$ROBOT_HOST:$REMOTE_DIR"
    fi
else
    echo "ERROR: AlphaBot2.py / TRSensors.py not found." >&2
    echo "Options:" >&2
    echo "  1. Project-local: git commit src/maze_mdp/maze_mdp/hardware/{AlphaBot2,TRSensors}.py" >&2
    echo "  2. Override: ALPHABOT2_LIB_DIR=/path/to/libs bash scripts/hardware_setup.sh" >&2
    echo "  3. On robot: wget https://files.waveshare.com/upload/3/39/AlphaBot2-Demo.zip" >&2
    echo "            unzip AlphaBot2-Demo.zip" >&2
    exit 1
fi

echo "[done] hardware assets are on $ROBOT_HOST:$REMOTE_DIR"
