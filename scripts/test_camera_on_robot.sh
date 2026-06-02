#!/usr/bin/env bash
# Test camera availability and backends on the AlphaBot2 Raspberry Pi.
#
# Usage:
#   bash scripts/test_camera_on_robot.sh
#   ROBOT_HOST=deec@10.16.140.69 bash scripts/test_camera_on_robot.sh
set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-deec@10.16.140.69}"
REMOTE_DIR="${REMOTE_DIR:-alphabot2_ws/src/}"

echo "[test] connecting to $ROBOT_HOST..."
echo "[test] testing camera backends..."

ssh "$ROBOT_HOST" "python3 - <<'PYEOF'
import sys
import importlib.util

print('[test] === Python module check ===')

specs = {
    'numpy': importlib.util.find_spec('numpy'),
    'RPi.GPIO': importlib.util.find_spec('RPi.GPIO'),
    'picamera2': importlib.util.find_spec('picamera2'),
    'libcamera': importlib.util.find_spec('libcamera'),
    'cv2': importlib.util.find_spec('cv2'),
}

for name, spec in specs.items():
    status = '✓' if spec is not None else '✗'
    print('[test] {:<15} {}'.format(name, status))

# Test camera device availability
print('[test] === Camera device check ===')
import os
if os.path.exists('/dev/video0'):
    print('[test] /dev/video0 exists')
    # Try to get device info
    try:
        import subprocess
        result = subprocess.run(['v4l2-ctl', '-d', '/dev/video0', '--info'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print('[test] v4l2-ctl output:')
            for line in result.stdout.split('\n')[:5]:
                if line.strip():
                    print('[test]   ' + line)
    except Exception as e:
        print('[test] v4l2-ctl not available or failed: {}'.format(e))
else:
    print('[test] /dev/video0 NOT found')

# Test actual camera initialization
print('[test] === Camera backend test ===')
has_picamera = importlib.util.find_spec('picamera2') is not None
has_libcamera = importlib.util.find_spec('libcamera') is not None
has_cv2 = importlib.util.find_spec('cv2') is not None

if has_picamera and has_libcamera:
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={'size': (320, 240), 'format': 'RGB888'})
        cam.configure(config)
        cam.start()
        import time
        time.sleep(0.3)
        frame = cam.capture_array()
        print('[test] picamera2: SUCCESS (captured {}x{} frame)'.format(
            frame.shape[1], frame.shape[0]))
        cam.stop()
    except Exception as e:
        print('[test] picamera2: FAILED - {}'.format(e))
elif has_cv2:
    try:
        import cv2
        import os
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                print('[test] opencv: SUCCESS (captured {}x{} frame)'.format(
                    frame.shape[1], frame.shape[0]))
            else:
                print('[test] opencv: OPENED but read() failed')
            cam.release()
        else:
            print('[test] opencv: FAILED to open VideoCapture')
    except Exception as e:
        print('[test] opencv: FAILED - {}'.format(e))
else:
    print('[test] NO camera backend available')

PYEOF
"

echo "[test] done"
