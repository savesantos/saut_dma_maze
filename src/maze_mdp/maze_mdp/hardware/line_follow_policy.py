#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Robot-side line follower driven by a precomputed MDP/RL policy.

Runs directly on the AlphaBot2's Raspberry Pi (no ROS). The PID line
follower and motion primitives are taken verbatim from the
course-provided ``Line_Follow.py`` example that is known to work on
hardware; the only change is that the sequence of high-level actions
(``forward`` / ``turn_left`` / ``turn_right``) is read from a learned
policy ``.npz`` file instead of a hardcoded list.

Usage on the Pi (after ``scp`` of this file + ``policy.npz``)::

    python3 line_follow_policy.py \
        --policy policy.npz \
        --rows 7 --cols 7 \
        --row 0 --col 0 --heading 1 \
        --goal-row 6 --goal-col 0

State encoding mirrors ``maze_mdp.mdp``:
``s = (row * n_cols + col) * 4 + heading`` with headings
``N=0, E=1, S=2, W=3`` and actions ``FORWARD=0, TURN_LEFT=1,
TURN_RIGHT=2``. The script maintains its own pose estimate by
applying the deterministic effect of each executed action; there is
no localization on the robot.
"""

import argparse
import time

import numpy as np

import RPi.GPIO as GPIO
from AlphaBot2 import AlphaBot2
from rpi_ws281x import Adafruit_NeoPixel, Color
from TRSensors import TRSensor

from camera_align import AlignOutcome, CameraAligner  # noqa: E402


# ---------------------------------------------------------------- MDP constants
# Mirrors maze_mdp.mdp.Action / Heading (cannot import here: the Pi
# does not have the ROS workspace installed).
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2
# heading -> (dr, dc) for N, E, S, W
HEADING_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))


def parse_args():
    """Parse CLI args, including the initial (row, col, heading)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True,
                   help='Path to policy.npz produced by maze_mdp '
                        '(contains "pi" and/or "Q").')
    p.add_argument('--rows', type=int, required=True, help='Maze rows.')
    p.add_argument('--cols', type=int, required=True, help='Maze cols.')
    p.add_argument('--row', type=int, required=True,
                   help='Initial robot row (0-indexed, top = 0).')
    p.add_argument('--col', type=int, required=True,
                   help='Initial robot column (0-indexed, left = 0).')
    p.add_argument('--heading', type=int, required=True, choices=[0, 1, 2, 3],
                   help='Initial heading: 0=N, 1=E, 2=S, 3=W.')
    p.add_argument('--goal-row', type=int, required=True, help='Goal row.')
    p.add_argument('--goal-col', type=int, required=True, help='Goal column.')
    p.add_argument('--max-steps', type=int, default=200,
                   help='Safety cap on number of executed actions.')
    p.add_argument('--no-camera-align', action='store_true',
                   help='Disable camera alignment after turns and fall '
                        'back to the open-loop forward burst.')
    p.add_argument('--camera-align-debug', default=None,
                   help='Directory to save raw frames + estimates for '
                        'offline tuning of the camera alignment.')
    p.add_argument('--image-center-col', type=float, default=None,
                   help='Pixel column of a centered line in the camera '
                        '(default: image width / 2).')
    p.add_argument('--theta-offset', type=float, default=0.0,
                   help='Camera tilt offset (rad) subtracted from the '
                        'measured line angle. Calibrate once on the bot.')
    return p.parse_args()


def load_policy(path):
    """Load ``pi`` array from a maze_mdp ``policy.npz`` bundle."""
    data = np.load(path, allow_pickle=True)
    if 'pi' in data.files:
        return np.asarray(data['pi'], dtype=np.int64)
    if 'Q' in data.files:
        return np.asarray(data['Q']).argmax(axis=1).astype(np.int64)
    raise RuntimeError("policy file has no 'pi' or 'Q' array: {}".format(path))


# ---------------------------------------------------------------- GPIO / LEDs
Button = 7

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Button, GPIO.IN, GPIO.PUD_UP)

LED_COUNT = 4
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 5
LED_BRIGHTNESS = 255
LED_INVERT = False


# ---------------------------------------------------------------- PID state
maximum = 35
integral = 0
last_proportional = 0


# ---------------------------------------------------------------- motion primitives
# Verbatim from the working Line_Follow2 reference.
def go_forward():
    Ab.setPWMA(15)
    Ab.setPWMB(15)
    Ab.forward()
    time.sleep(0.5)

    Ab.stop()
    time.sleep(1)
    Ab.setPWMA(0)
    Ab.setPWMB(0)
    Ab.forward()


def turn_right():
    Ab.setPWMA(15)
    Ab.setPWMB(15)
    Ab.forward()
    time.sleep(0.3)

    Ab.stop()

    time.sleep(0.05)

    Ab.setPWMA(15)
    Ab.setPWMB(15)
    Ab.right()
    time.sleep(0.3)

    Ab.stop()

    time.sleep(0.1)
    Ab.setPWMA(0)
    Ab.setPWMB(0)


def turn_left():
    Ab.setPWMA(15)
    Ab.setPWMB(15)
    Ab.forward()
    time.sleep(0.3)

    Ab.stop()

    time.sleep(0.05)

    Ab.setPWMA(15)
    Ab.setPWMB(15)
    Ab.left()
    time.sleep(0.3)

    Ab.stop()

    time.sleep(0.1)
    Ab.setPWMA(0)
    Ab.setPWMB(0)


# ---------------------------------------------------------------- main
args = parse_args()
pi = load_policy(args.policy)
expected_states = args.rows * args.cols * 4
if pi.size != expected_states:
    raise RuntimeError(
        'policy size {} does not match rows*cols*4 = {}'
        .format(pi.size, expected_states))

# Robot pose estimate; updated after each executed primitive.
robot_row = args.row
robot_col = args.col
robot_heading = args.heading
goal_cell = (args.goal_row, args.goal_col)
steps_taken = 0
done = False


def state_index(r, c, h):
    return (r * args.cols + c) * 4 + h


# Neopixel ring (same colors as the reference script).
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                          LED_INVERT, LED_BRIGHTNESS)
strip.begin()
for i in range(4):
    strip.setPixelColor(i, Color(100, 100, 100))
strip.show()

TR = TRSensor()
Ab = AlphaBot2()
Ab.stop()
print("Line follow + policy execution")
print("start cell=({},{},{}) goal=({},{})".format(
    robot_row, robot_col, robot_heading, goal_cell[0], goal_cell[1]))
time.sleep(0.5)

# Camera aligner (default on; disable with --no-camera-align).
if args.no_camera_align:
    aligner = None
    print('[main] camera alignment disabled by CLI flag.')
else:
    aligner = CameraAligner(
        Ab, TR,
        image_center_col=args.image_center_col,
        theta_offset=args.theta_offset,
        debug_dir=args.camera_align_debug,
    )
    if not aligner.start():
        print('[main] camera unavailable; falling back to open-loop turns.')
        aligner = None


def post_turn_motion():
    """Replace the open-loop forward burst after a turn.

    Returns True when the aligner reports the robot has cleared the
    intersection and re-acquired the line, so the caller can update its
    pose estimate; returns False if we fell back to open-loop.
    """
    if aligner is None:
        Ab.setPWMA(15)
        Ab.setPWMB(15)
        Ab.forward()
        time.sleep(0.5)
        Ab.stop()
        return False
    outcome = aligner.align_and_creep()
    print('[main] post-turn align outcome: {}'.format(outcome.value))
    return outcome == AlignOutcome.OK

# Calibration sweep, verbatim from the reference.
for i in range(0, 100):
    if i < 25 or i >= 75:
        Ab.right()
        Ab.setPWMA(20)
        Ab.setPWMB(20)
    else:
        Ab.left()
        Ab.setPWMA(20)
        Ab.setPWMB(20)
    TR.calibrate()
Ab.stop()
print(TR.calibratedMin)
print(TR.calibratedMax)

time.sleep(3)
Ab.forward()

# Sensor 0 (leftmost IR channel) is physically damaged on this robot;
# ignore it everywhere. The line-position estimate below is the same
# weighted-average algorithm as ``TRSensor.readLine`` (see
# AlphaBot2-Demo/.../TRSensors.py) but the loop skips index 0, so the
# value range is 1000..4000 with a centered line at LINE_CENTER = 2500.
# Intersections likewise only require channels 1..4 to be saturated.
LINE_CENTER = 2500
_LAST_POSITION = LINE_CENTER
_USED_INDICES = (1, 2, 3, 4)
_POS_MIN = _USED_INDICES[0] * 1000          # 1000
_POS_MAX = _USED_INDICES[-1] * 1000         # 4000


def read_line_ignoring_s0(white_line=0):
    """Mirror ``TRSensor.readLine`` but skip the broken channel 0."""
    global _LAST_POSITION
    sensor_values = TR.readCalibrated()
    avg = 0
    total = 0
    on_line = 0
    for i in _USED_INDICES:
        value = sensor_values[i]
        if white_line:
            value = 1000 - value
        # keep track of whether we see the line at all
        if value > 200:
            on_line = 1
        # only average in values above the noise threshold
        if value > 50:
            avg += value * (i * 1000)
            total += value

    if not on_line:
        # If it last read left of center, snap to the left extreme;
        # otherwise to the right extreme. Matches the original logic.
        if _LAST_POSITION < (_POS_MIN + _POS_MAX) / 2:
            _LAST_POSITION = _POS_MIN
        else:
            _LAST_POSITION = _POS_MAX
    else:
        _LAST_POSITION = avg / total

    return _LAST_POSITION, sensor_values


while True:
    try:
        position, Sensors = read_line_ignoring_s0()

        intersection = (Sensors[1] > 900 and Sensors[2] > 900
                        and Sensors[3] > 900 and Sensors[4] > 900)

        if intersection:
            Ab.setPWMA(0)
            Ab.setPWMB(0)

            if done or steps_taken >= args.max_steps:
                Ab.stop()
                for i in range(4):
                    strip.setPixelColor(i, Color(0, 0, 100))
                strip.show()
                print('Stopping (done={}, steps={}).'.format(done, steps_taken))
                break

            if (robot_row, robot_col) == goal_cell:
                done = True
                Ab.stop()
                for i in range(4):
                    strip.setPixelColor(i, Color(0, 0, 100))
                strip.show()
                print('Goal reached at ({},{}).'.format(robot_row, robot_col))
                break

            s = state_index(robot_row, robot_col, robot_heading)
            action = int(pi[s])
            steps_taken += 1
            print('step={} cell=({},{},{}) action={}'.format(
                steps_taken, robot_row, robot_col, robot_heading, action))

            if action == FORWARD:
                go_forward()
                dr, dc = HEADING_DELTA[robot_heading]
                robot_row += dr
                robot_col += dc
            elif action == TURN_RIGHT:
                turn_right()
                robot_heading = (robot_heading + 1) % 4
                # After a turn the robot is still sitting on the cross.
                # Use the camera aligner (if available) to rotate-in-place
                # to parallel with the new line, then creep until the IR
                # array re-acquires it. Falls back to the legacy open-loop
                # forward burst if the camera is disabled / failed.
                next_s = state_index(robot_row, robot_col, robot_heading)
                if int(pi[next_s]) == FORWARD:
                    steps_taken += 1
                    post_turn_motion()
                    dr, dc = HEADING_DELTA[robot_heading]
                    robot_row += dr
                    robot_col += dc
            elif action == TURN_LEFT:
                turn_left()
                robot_heading = (robot_heading - 1) % 4
                next_s = state_index(robot_row, robot_col, robot_heading)
                if int(pi[next_s]) == FORWARD:
                    steps_taken += 1
                    post_turn_motion()
                    dr, dc = HEADING_DELTA[robot_heading]
                    robot_row += dr
                    robot_col += dc
            else:
                print('Unknown action {}, stopping.'.format(action))
                Ab.stop()
                break

            for i in range(4):
                strip.setPixelColor(i, Color(100, 0, 0))
            strip.show()
        else:
            # PID line-following, verbatim from the reference but with the
            # neutral center shifted to LINE_CENTER (sensor 0 ignored).
            proportional = position - LINE_CENTER
            derivative = proportional - last_proportional
            integral += proportional
            last_proportional = proportional

            power_difference = (proportional / 30
                                + integral / 10000
                                + derivative * 2)

            if power_difference > maximum:
                power_difference = maximum
            if power_difference < -maximum:
                power_difference = -maximum
            print(position, power_difference)
            if power_difference < 0:
                Ab.setPWMA(0.5 * (maximum + power_difference))
                Ab.setPWMB(0.5 * maximum)
            else:
                Ab.setPWMA(0.5 * maximum)
                Ab.setPWMB(0.5 * (maximum - power_difference))
                for i in range(4):
                    strip.setPixelColor(i, Color(0, 100, 0))
                strip.show()

    except KeyboardInterrupt:
        Ab.stop()
        break

if aligner is not None:
    aligner.stop()
