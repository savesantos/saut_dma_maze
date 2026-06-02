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

# Wait for any running unattended-upgrades / apt process to release the lock.
# unattended-upgr commonly holds /var/lib/dpkg/lock-frontend on first boot.
echo "[setup] waiting for dpkg lock..."
if command -v systemctl >/dev/null 2>&1; then
    systemctl stop unattended-upgrades 2>/dev/null || true
fi
timeout=120
elapsed=0
while ! flock -n /var/lib/dpkg/lock-frontend -c true 2>/dev/null; do
    if [[ $elapsed -ge $timeout ]]; then
        echo "[setup] ERROR: dpkg lock not released after ${timeout}s" >&2
        exit 1
    fi
    echo "[setup] dpkg lock held, retrying in 5s..."
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "[setup] dpkg lock acquired."

# Build list of packages that are NOT already installed.
# Core packages needed (always try to install)
core_pkgs=(python3-numpy python3-rpi.gpio)
# Camera backends: try both picamera2/libcamera (preferred) and opencv (fallback)
camera_pkgs=(python3-picamera2 python3-libcamera python3-opencv v4l-utils)

to_install=()
for pkg in "${core_pkgs[@]}"; do
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        echo "[setup] already installed, skipping: $pkg"
    else
        to_install+=("$pkg")
    fi
done

# For camera packages, try to install but don't fail if unavailable
camera_to_install=()
for pkg in "${camera_pkgs[@]}"; do
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        echo "[setup] already installed, skipping: $pkg"
    else
        camera_to_install+=("$pkg")
    fi
done

if [[ ${#to_install[@]} -gt 0 || ${#camera_to_install[@]} -gt 0 ]]; then
    echo "[setup] updating apt cache..."
    apt update || echo "[setup] WARNING: apt update had issues, continuing anyway"
    
    # Install core packages first (these must succeed)
    if [[ ${#to_install[@]} -gt 0 ]]; then
        echo "[setup] installing core packages: ${to_install[*]}"
        for pkg in "${to_install[@]}"; do
            apt install -y "$pkg" || echo "[setup] WARNING: failed to install $pkg"
        done
    fi
    
    # Install camera packages (these can fail gracefully)
    if [[ ${#camera_to_install[@]} -gt 0 ]]; then
        echo "[setup] attempting to install camera/image packages: ${camera_to_install[*]}"
        for pkg in "${camera_to_install[@]}"; do
            if apt-cache show "$pkg" >/dev/null 2>&1; then
                echo "[setup] installing: $pkg"
                apt install -y "$pkg" || echo "[setup] WARNING: failed to install $pkg (continuing)"
            else
                echo "[setup] WARNING: package not available in apt cache: $pkg"
            fi
        done
    fi
else
    echo "[setup] all packages already present, skipping apt update."
fi

# Install pip packages needed by the hardware runner.
echo "[setup] installing pip packages..."
pip3 install --upgrade pip setuptools wheel >/dev/null 2>&1 || echo "[setup] WARNING: pip upgrade had issues"
pip3 install rpi-ws281x adafruit-circuitpython-neopixel >/dev/null 2>&1 || echo "[setup] WARNING: NeoPixel packages unavailable (LEDs may not work)"

# Check Python module availability with detailed diagnostics
python3 - <<'PY'
import importlib.util
import sys

has_numpy = importlib.util.find_spec("numpy") is not None
has_picamera = importlib.util.find_spec("picamera2") is not None
has_libcamera = importlib.util.find_spec("libcamera") is not None
has_cv2 = importlib.util.find_spec("cv2") is not None
has_rpi_gpio = importlib.util.find_spec("RPi.GPIO") is not None

# Diagnostics
print("[setup] === Module availability ===")
print("[setup] numpy: {}".format("✓" if has_numpy else "✗"))
print("[setup] RPi.GPIO: {}".format("✓" if has_rpi_gpio else "✗"))
print("[setup] picamera2: {}".format("✓" if has_picamera else "✗"))
print("[setup] libcamera: {}".format("✓" if has_libcamera else "✗"))
print("[setup] opencv(cv2): {}".format("✓" if has_cv2 else "✗"))

# Determine camera backend
camera_backend = None
if has_picamera and has_libcamera:
    camera_backend = "picamera2/libcamera (native)"
elif has_cv2:
    camera_backend = "opencv(cv2) (fallback)"

print("[setup] === Hardware requirements ===")
critical_ok = has_numpy and has_rpi_gpio
print("[setup] numpy + RPi.GPIO (CRITICAL): {}".format("✓" if critical_ok else "✗"))
print("[setup] camera backend: {}".format(camera_backend if camera_backend else "NONE"))

if not critical_ok:
    print("[setup] ERROR: critical modules missing (numpy, RPi.GPIO)")
    sys.exit(1)

if not camera_backend:
    print("[setup] WARNING: no camera backend available; alignment will be unavailable")
    sys.exit(2)

print("[setup] Module check passed")
PY
REMOTE_SETUP

    remote_setup_path="/tmp/hardware_setup_deps_$$.sh"
    scp "$tmp_setup_script" "$ROBOT_HOST:$remote_setup_path"
    rm -f "$tmp_setup_script"
    if ssh -tt "$ROBOT_HOST" "chmod +x '$remote_setup_path' && sudo bash '$remote_setup_path'; rc=\$?; rm -f '$remote_setup_path'; exit \$rc"; then
        :
    else
        setup_rc=$?
        if [[ $setup_rc -eq 1 ]]; then
            echo "ERROR: critical dependencies failed on the robot (numpy, RPi.GPIO)." >&2
            exit 1
        fi
        if [[ $setup_rc -eq 2 ]]; then
            if [[ "$CAMERA_REQUIRED" == "1" ]]; then
                echo "ERROR: required camera backend missing on the robot." >&2
                echo "Set CAMERA_REQUIRED=0 to continue without camera alignment." >&2
                exit 1
            fi
            echo "[setup] WARNING: camera backend unavailable; policy will run without camera alignment"
        else
            echo "ERROR: dependency setup failed on robot (exit $setup_rc)." >&2
            exit "$setup_rc"
        fi
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
