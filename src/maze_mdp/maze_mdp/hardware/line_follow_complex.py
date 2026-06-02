import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
import argparse

try:
    from AlphaBot2 import AlphaBot2
except ImportError:
    AlphaBot2 = None

# MDP constants (mirrors maze_mdp.py)
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2
# heading -> (dr, dc) for N, E, S, W
HEADING_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))

HAS_DISPLAY = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
HEADLESS = not HAS_DISPLAY

# Configuration (adapted from process_images.py)
ROI_TOP_FRACTION = 0.5
MIN_POINTS = 10
OFFSET_DEADBAND = 20
MIN_LINE_WIDTH_PX = 5
MIN_TAPE_WIDTH_PX = 100
MAX_TAPE_WIDTH_PX = 260
LINE_INLIER_DIST_PX = 16
BORDER_MARGIN_PX = 20
PRINT_WIDTHS = False

# Input scaling
RESIZE_MAX_WIDTH = 320
ENABLE_VIZ = True

# Frame throttling
THROTTLE_MODE = 'count'  # 'count' or 'time'
PROCESS_EVERY_N_FRAMES = 6
MAX_PROCESS_FPS = 5.0

# Line follow control
FORWARD_SPEED = 20
TURN_SPEED = 15
MAX_OFFSET = 160  # Maximum offset (half of typical 320px width)

# MDP configuration (defaults; can be overridden via CLI args)
MAZE_ROWS_DEFAULT = 7
MAZE_COLS_DEFAULT = 7
INITIAL_ROW_DEFAULT = 0
INITIAL_COL_DEFAULT = 0
INITIAL_HEADING_DEFAULT = 1  # 0=N, 1=E, 2=S, 3=W
GOAL_ROW_DEFAULT = 6
GOAL_COL_DEFAULT = 0
MAX_STEPS_DEFAULT = 200
POLICY_FILE_DEFAULT = 'policy.npz'
INTERSECTION_THRESHOLD = 900  # Sensor value threshold for intersection detection
RAW_IMAGE_TOPICS = ('/alphabot2/image_raw', '/image_raw', '/image/raw')
COMPRESSED_IMAGE_TOPICS = (
    '/alphabot2/image/compressed',
    '/image/compressed',
    '/alphabot2/image_raw/compressed',
    '/image_raw/compressed',
    '/image/raw/compressed',
)

# Continuous spin configuration (from complex_camera.py)
SPIN_SPEED = 6
LINE_LOST_THRESHOLD = 3  # Frames before declaring line lost during turn
LINE_REACQUIRE_THRESHOLD = 1  # Frames to confirm line reacquired


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=20
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    h, w = binary.shape
    if BORDER_MARGIN_PX > 0:
        binary[:, :BORDER_MARGIN_PX] = 0
        binary[:, w - BORDER_MARGIN_PX:] = 0

    return binary


def find_line_midpoints(binary: np.ndarray, roi_top: int) -> List[Tuple[int, int]]:
    roi = binary[roi_top:, :]
    h_roi = roi.shape[0]
    midpoints: List[Tuple[int, int]] = []

    if PRINT_WIDTHS:
        print(f"\n{'y':>5}  {'x_left':>7}  {'x_right':>8}  {'width':>6}  {'status':>10}")
        print("-" * 45)

    for y_rel in range(h_roi):
        row = roi[y_rel]
        x_tape = np.where(row == 255)[0]

        if len(x_tape) < MIN_LINE_WIDTH_PX:
            continue

        x_left = int(x_tape[0])
        x_right = int(x_tape[-1])
        width = x_right - x_left
        x_mid = (x_left + x_right) // 2
        y_abs = y_rel + roi_top

        if (MAX_TAPE_WIDTH_PX is not None and width > MAX_TAPE_WIDTH_PX) or \
           (MIN_TAPE_WIDTH_PX is not None and width < MIN_TAPE_WIDTH_PX):
            status = "REJECTED"
        else:
            status = "ok"
            midpoints.append((x_mid, y_abs))

        if PRINT_WIDTHS:
            print(f"{y_abs:>5}  {x_left:>7}  {x_right:>8}  {width:>6}  {status:>10}")

    return midpoints


def fit_line(points: List[Tuple[int, int]]):
    pts = np.array(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    dy = np.diff(ys)
    dx = np.diff(xs)
    valid = np.abs(dy) > 0.5
    slopes = dx[valid] / dy[valid]

    if len(slopes) == 0:
        x0 = float(np.median(xs))
        y0 = float(np.median(ys))
        return 0.0, 1.0, x0, y0

    hist, edges = np.histogram(slopes, bins=100,
                               range=(slopes.min() - 0.1, slopes.max() + 0.1))
    best_bin = int(np.argmax(hist))
    modal_slope = float((edges[best_bin] + edges[best_bin + 1]) / 2.0)

    x0 = float(np.median(xs))
    y0 = float(np.median(ys))

    norm = float(np.sqrt(modal_slope ** 2 + 1.0))
    vx = modal_slope / norm
    vy = 1.0 / norm

    dists = np.abs(vy * (xs - x0) - vx * (ys - y0))
    inlier_mask = dists <= LINE_INLIER_DIST_PX
    inlier_pts = pts[inlier_mask]

    if inlier_pts.shape[0] >= max(3, len(pts) // 2):
        line = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])

    return vx, vy, x0, y0


def compute_offset_and_angle(vx: float, vy: float, x0: float, y0: float,
                             frame_width: int, eval_y: int) -> Tuple[int, float]:
    if abs(vy) < 1e-6:
        x_at_eval = x0
    else:
        t = (eval_y - y0) / vy
        x_at_eval = x0 + vx * t
    offset = int(x_at_eval) - frame_width // 2
    angle = float(np.degrees(np.arctan2(vx, vy)))
    return offset, angle


def compute_motor_commands(offset: int, frame_width: int) -> Tuple[int, int, str]:
    """
    Compute motor PWM values and direction based on line offset.
    Returns: (pwm_a, pwm_b, direction)
    """
    center = frame_width // 2
    
    # Normalized offset (-1 to 1)
    norm_offset = offset / MAX_OFFSET
    norm_offset = np.clip(norm_offset, -1.0, 1.0)
    
    # Calculate speed adjustment based on offset
    # If offset is positive (line to the right), turn right
    # If offset is negative (line to the left), turn left
    
    if abs(offset) < OFFSET_DEADBAND:
        # Go straight
        return FORWARD_SPEED, FORWARD_SPEED, "forward"
    elif offset > 0:
        # Line is to the right, turn right
        right_speed = max(5, int(FORWARD_SPEED * (1.0 - abs(norm_offset))))
        left_speed = FORWARD_SPEED
        return left_speed, right_speed, "right"
    else:
        # Line is to the left, turn left
        left_speed = max(5, int(FORWARD_SPEED * (1.0 - abs(norm_offset))))
        right_speed = FORWARD_SPEED
        return right_speed, left_speed, "left"


def load_policy(path: str) -> Optional[np.ndarray]:
    """Load pi array from a maze_mdp policy.npz bundle.
    
    Looks for 'pi' (deterministic policy) first, then tries 'Q' (value function)
    and uses argmax to extract the policy.
    """
    if not os.path.exists(path):
        return None
    
    data = np.load(path, allow_pickle=True)
    if 'pi' in data.files:
        return np.asarray(data['pi'], dtype=np.int64)
    if 'Q' in data.files:
        return np.asarray(data['Q']).argmax(axis=1).astype(np.int64)
    return None


def state_index(row: int, col: int, heading: int, n_cols: int) -> int:
    """Compute state index for MDP: s = (row * n_cols + col) * 4 + heading"""
    return (row * n_cols + col) * 4 + heading


def draw_visualization(frame: np.ndarray,
                       binary: np.ndarray,
                       midpoints: List[Tuple[int, int]],
                       line_params,
                       offset,
                       angle,
                       roi_top: int,
                       motor_commands: Tuple[int, int, str] = None) -> np.ndarray:

    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (0, roi_top), (w, roi_top), (180, 180, 0), 1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 80, 0), 2)

    for (cx, cy) in midpoints[::3]:
        cv2.circle(vis, (cx, cy), 3, (0, 200, 255), -1)

    if line_params is not None:
        vx, vy, x0, y0 = line_params
        scale = 300
        p1 = (int(x0 - vx * scale), int(y0 - vy * scale))
        p2 = (int(x0 + vx * scale), int(y0 + vy * scale))
        cv2.line(vis, p1, p2, (0, 255, 80), 3)

        arrow_y = h - 20
        denom = vy if abs(vy) > 1e-6 else 1
        arrow_x = int(x0 + vx * ((arrow_y - y0) / denom))

        if offset is not None:
            if offset > OFFSET_DEADBAND:
                direction, col = "TURN RIGHT >>", (0, 100, 255)
            elif offset < -OFFSET_DEADBAND:
                direction, col = "<< TURN LEFT", (0, 100, 255)
            else:
                direction, col = "CENTERED", (0, 255, 80)

            info = (f"offset={offset:+d}px  |  angle={angle:+.1f}deg  |  {direction}")
            cv2.putText(vis, info, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4)
            cv2.putText(vis, info, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2)
            
            # Display motor commands if available
            if motor_commands is not None:
                pwm_a, pwm_b, motor_dir = motor_commands
                motor_info = f"MOTORS: A={pwm_a} B={pwm_b} ({motor_dir})"
                cv2.putText(vis, motor_info, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3)
                cv2.putText(vis, motor_info, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 0), 1)
    else:
        cv2.putText(vis, "NOT DETECTED",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 3)

    bin_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (cx, cy) in midpoints[::3]:
        cv2.circle(bin_color, (cx, cy), 3, (0, 255, 128), -1)

    # Stack visualization and binary panel horizontally
    try:
        combined = np.hstack([vis, bin_color])
    except Exception:
        combined = vis

    return combined


class LineFollowComplexNode(Node):
    def __init__(self, config: dict):
        super().__init__('line_follow_complex')
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.last_process_time = None
        self.window_name = 'Line Follow Complex'
        self._active_image_topic = None

        # Diagnostics state
        self.start_time = time.monotonic()
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_throttled = 0
        self.last_frame_time = None
        self.last_processed_time = None
        self.last_line_detected = None
        self.last_intersection_state = None
        self.last_motor_command = None
        self.last_motor_log_time = 0.0
        self.last_policy_wait_log_time = 0.0
        self.last_graph_debug_time = 0.0
        self.last_qos_debug_time = 0.0

        # Load config
        self.maze_rows = config['maze_rows']
        self.maze_cols = config['maze_cols']
        self.goal_row = config['goal_row']
        self.goal_col = config['goal_col']
        self.max_steps = config['max_steps']
        self.policy_file = config['policy_file']

        # MDP state tracking
        self.robot_row = config['initial_row']
        self.robot_col = config['initial_col']
        self.robot_heading = config['initial_heading']
        self.steps_taken = 0
        self.done = False
        self.at_intersection = False
        self.previous_at_intersection = False
        
        # Continuous spin state (for camera-based turn recovery)
        self.spinning_direction = None
        self.line_lost_during_turn = False
        self.consecutive_line_lost_frames = 0
        self.consecutive_line_detect_frames = 0
        
        # Load policy
        self.pi = None
        self.load_policy()

        if AlphaBot2 is not None:
            try:
                self.Ab = AlphaBot2()
                self.Ab.stop()
            except Exception as e:
                self.get_logger().error(f'AlphaBot2 init failed: {e}')
                self.Ab = None
        else:
            self.Ab = None

        qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._raw_image_subscriptions = []
        self._compressed_image_subscriptions = []

        for topic in RAW_IMAGE_TOPICS:
            self._raw_image_subscriptions.append(
                self.create_subscription(Image, topic, self.image_callback, qos_profile_sensor_data)
            )
            self._raw_image_subscriptions.append(
                self.create_subscription(Image, topic, self.image_callback, qos_reliable)
            )

        for topic in COMPRESSED_IMAGE_TOPICS:
            self._compressed_image_subscriptions.append(
                self.create_subscription(CompressedImage, topic, self.compressed_image_callback, qos_profile_sensor_data)
            )
            self._compressed_image_subscriptions.append(
                self.create_subscription(CompressedImage, topic, self.compressed_image_callback, qos_reliable)
            )
        self.get_logger().info(
            f'Subscribed to raw image topics: {", ".join(RAW_IMAGE_TOPICS)}'
        )
        self.get_logger().info(
            f'Subscribed to compressed image topics: {", ".join(COMPRESSED_IMAGE_TOPICS)}'
        )
        self.get_logger().info(
            f'Created camera subscriptions: raw={len(self._raw_image_subscriptions)} '
            f'compressed={len(self._compressed_image_subscriptions)} (best-effort + reliable)'
        )
        self.get_logger().info(
            'Config: '
            f'policy={self.policy_file} maze={self.maze_rows}x{self.maze_cols} '
            f'start=({self.robot_row},{self.robot_col},{self.robot_heading}) '
            f'goal=({self.goal_row},{self.goal_col}) max_steps={self.max_steps}'
        )
        self.get_logger().info(
            'Frame processing: '
            f'throttle_mode={THROTTLE_MODE} process_every_n={PROCESS_EVERY_N_FRAMES} '
            f'max_fps={MAX_PROCESS_FPS}'
        )
        self.get_logger().info(
            'ROS env: '
            f'ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "unset")} '
            f'ROS_LOCALHOST_ONLY={os.environ.get("ROS_LOCALHOST_ONLY", "unset")}'
        )

        self.publisher = self.create_publisher(
            Image,
            '/alphabot2/image_processed',
            10
        )

        if ENABLE_VIZ and not HEADLESS:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Optional visualization publisher (BGR image)
        self.viz_pub = self.create_publisher(
            Image,
            '/alphabot2/image_viz',
            10
        )

        self.diag_timer = self.create_timer(2.0, self._diagnostics_tick)

    def _diagnostics_tick(self):
        now = time.monotonic()
        age_frame = (now - self.last_frame_time) if self.last_frame_time is not None else None
        age_processed = (now - self.last_processed_time) if self.last_processed_time is not None else None
        line_state = 'unknown' if self.last_line_detected is None else ('detected' if self.last_line_detected else 'missing')
        intersection_state = 'unknown' if self.at_intersection is None else str(self.at_intersection)
        motion_state = (
            f'spin={self.spinning_direction}'
            if self.spinning_direction is not None
            else 'spin=None'
        )
        age_frame_str = f'{age_frame:.2f}s' if age_frame is not None else 'n/a'
        age_processed_str = f'{age_processed:.2f}s' if age_processed is not None else 'n/a'
        self.get_logger().info(
            f'Heartbeat: frames rx/proc/drop={self.frames_received}/{self.frames_processed}/{self.frames_throttled} '
            f'last_frame_age={age_frame_str}'
        )
        self.get_logger().info(
            f'Status: last_proc_age={age_processed_str}'
        )
        self.get_logger().info(
            f'state=({self.robot_row},{self.robot_col},{self.robot_heading}) '
            f'steps={self.steps_taken}/{self.max_steps} done={self.done} '
            f'intersection={intersection_state} line={line_state} {motion_state}'
        )

        # If callbacks are never firing, inspect ROS graph publisher visibility.
        if self.frames_received == 0 and (now - self.last_graph_debug_time) >= 6.0:
            self.last_graph_debug_time = now
            topic_counts = []
            for topic in RAW_IMAGE_TOPICS + COMPRESSED_IMAGE_TOPICS:
                try:
                    count = len(self.get_publishers_info_by_topic(topic))
                except Exception:
                    count = -1
                topic_counts.append(f'{topic}:{count}')
            self.get_logger().info('Camera graph publishers: ' + ', '.join(topic_counts))

        if self.frames_received == 0 and (now - self.last_qos_debug_time) >= 10.0:
            self.last_qos_debug_time = now
            self._log_camera_offer_qos()

    def _log_camera_offer_qos(self):
        for topic in RAW_IMAGE_TOPICS + COMPRESSED_IMAGE_TOPICS:
            try:
                infos = self.get_publishers_info_by_topic(topic)
            except Exception as exc:
                self.get_logger().warning(f'Could not inspect QoS offers for {topic}: {exc}')
                continue

            if not infos:
                continue

            for idx, info in enumerate(infos):
                qos = info.qos_profile
                self.get_logger().info(
                    f'QoS offer topic={topic} pub#{idx} '
                    f'reliability={qos.reliability} durability={qos.durability} '
                    f'history={qos.history} depth={qos.depth}'
                )

    def _log_policy_wait_reason(self, reason: str):
        now = time.monotonic()
        if now - self.last_policy_wait_log_time >= 1.0:
            self.last_policy_wait_log_time = now
            self.get_logger().info(f'Policy waiting: {reason}')

    def load_policy(self):
        """Load policy from policy.npz file"""
        # Search for policy file: first check explicit path, then relative to script dir
        policy_paths = [
            self.policy_file,
            os.path.join(os.path.dirname(__file__), self.policy_file),
        ]
        
        for policy_file in policy_paths:
            if os.path.exists(policy_file):
                data = np.load(policy_file, allow_pickle=True)
                if 'pi' in data.files:
                    self.pi = np.asarray(data['pi'], dtype=np.int64)
                elif 'Q' in data.files:
                    self.pi = np.asarray(data['Q']).argmax(axis=1).astype(np.int64)
                else:
                    continue
                
                expected_states = self.maze_rows * self.maze_cols * 4
                if self.pi.size != expected_states:
                    self.get_logger().error(
                        f'policy size {self.pi.size} does not match rows*cols*4 = {expected_states}'
                    )
                    self.pi = None
                else:
                    self.get_logger().info(f'Loaded policy from {policy_file} with {self.pi.size} states')
                return
        
        self.get_logger().warning(f'Failed to load policy from any of: {policy_paths}')

    def execute_action(self, action):
        """Execute an action from the policy.
        For turns, use camera-based burst spin instead of open-loop.
        """
        if self.Ab is None:
            self.get_logger().warning('execute_action called but AlphaBot2 control is unavailable')
            return

        action_map = {FORWARD: 'FORWARD', TURN_LEFT: 'TURN_LEFT', TURN_RIGHT: 'TURN_RIGHT'}
        self.get_logger().info(f'Executing action: {action_map.get(action, str(action))}')

        if action == FORWARD:
            self.Ab.forward()
            self.Ab.setPWMA(FORWARD_SPEED)
            self.Ab.setPWMB(FORWARD_SPEED)
        elif action == TURN_LEFT:
            # Use burst spin with camera feedback
            self.request_spin('left')
        elif action == TURN_RIGHT:
            # Use burst spin with camera feedback
            self.request_spin('right')
        else:
            self.Ab.stop()

    def request_spin(self, direction: str):
        """Request continuous spin for turns (camera-based line recovery)"""
        if self.Ab is None:
            self.get_logger().warning('Robot control unavailable; cannot spin')
            return

        if direction == self.spinning_direction:
            return

        self.spinning_direction = direction
        self.line_lost_during_turn = False
        self.consecutive_line_lost_frames = 0
        self.consecutive_line_detect_frames = 0
        self.get_logger().info(f'Starting spin {direction}')
        
        if direction == 'left':
            self.Ab.left()
        else:
            self.Ab.right()
        self.Ab.setPWMA(SPIN_SPEED)
        self.Ab.setPWMB(SPIN_SPEED)

    def stop_spin(self):
        """Stop spinning when line is reacquired"""
        if self.Ab is None:
            self.spinning_direction = None
            self.line_lost_during_turn = False
            self.consecutive_line_lost_frames = 0
            self.consecutive_line_detect_frames = 0
            return

        if self.spinning_direction is not None:
            self.Ab.stop()
            self.get_logger().info('Stopping spin because line was detected')
        self.spinning_direction = None
        self.line_lost_during_turn = False
        self.consecutive_line_lost_frames = 0
        self.consecutive_line_detect_frames = 0

    def update_pose(self, action):
        """Update robot pose estimate after executing an action"""
        if action == FORWARD:
            dr, dc = HEADING_DELTA[self.robot_heading]
            self.robot_row += dr
            self.robot_col += dc
        elif action == TURN_LEFT:
            self.robot_heading = (self.robot_heading - 1) % 4
        elif action == TURN_RIGHT:
            self.robot_heading = (self.robot_heading + 1) % 4

    def set_motor_command(self, pwm_a: int, pwm_b: int, direction: str):
        """Set motor speeds and direction"""
        if self.Ab is None:
            self.get_logger().warning('set_motor_command skipped because AlphaBot2 control is unavailable')
            return

        cmd = (pwm_a, pwm_b, direction)
        now = time.monotonic()
        if cmd != self.last_motor_command or (now - self.last_motor_log_time) >= 1.0:
            self.last_motor_command = cmd
            self.last_motor_log_time = now
            self.get_logger().info(f'Motor command: dir={direction} pwm_a={pwm_a} pwm_b={pwm_b}')

        if direction == "forward":
            self.Ab.forward()
        elif direction == "left":
            self.Ab.left()
        elif direction == "right":
            self.Ab.right()
        else:
            self.Ab.stop()
            return

        self.Ab.setPWMA(pwm_a)
        self.Ab.setPWMB(pwm_b)

    def _should_process_frame(self) -> bool:
        self.frame_counter += 1
        if THROTTLE_MODE == 'count':
            if self.frame_counter % PROCESS_EVERY_N_FRAMES != 0:
                self.frames_throttled += 1
                return False
        else:
            now = self.get_clock().now().to_sec()
            if self.last_process_time is not None and \
               (now - self.last_process_time) < (1.0 / MAX_PROCESS_FPS):
                self.frames_throttled += 1
                return False
            self.last_process_time = now
        return True

    def _process_frame(self, header, frame: np.ndarray):
        self.frames_processed += 1
        self.last_processed_time = time.monotonic()

        if frame.shape[1] > RESIZE_MAX_WIDTH:
            scale = RESIZE_MAX_WIDTH / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (RESIZE_MAX_WIDTH, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA
            )

        h, w = frame.shape[:2]
        roi_top = int(h * ROI_TOP_FRACTION)

        binary = preprocess(frame)
        midpoints = find_line_midpoints(binary, roi_top)

        offset = None
        angle = None
        line_params = None
        current_action = None
        action_status = "No policy loaded"

        line_detected = len(midpoints) >= MIN_POINTS
        if self.last_line_detected is None or self.last_line_detected != line_detected:
            self.last_line_detected = line_detected
            self.get_logger().info(
                f'Line detection changed: detected={line_detected} midpoints={len(midpoints)}'
            )

        if line_detected:
            self.consecutive_line_detect_frames += 1
            self.consecutive_line_lost_frames = 0
            vx, vy, x0, y0 = fit_line(midpoints)
            line_params = (vx, vy, x0, y0)
            eval_y = int(h * 0.85)
            offset, angle = compute_offset_and_angle(vx, vy, x0, y0, w, eval_y)
            
            # If spinning from a turn, check if line is stable before stopping
            if self.spinning_direction is not None:
                if self.line_lost_during_turn and self.consecutive_line_detect_frames >= LINE_REACQUIRE_THRESHOLD:
                    spin_direction = self.spinning_direction
                    self.stop_spin()
                    self.update_pose(TURN_LEFT if spin_direction == 'left' else TURN_RIGHT)
                    self.get_logger().info(
                        f'Turn complete after line reacquire; updated pose to '
                        f'({self.robot_row},{self.robot_col},{self.robot_heading})'
                    )
                else:
                    self.get_logger().debug('Detected line but waiting for stable reacquire')
            else:
                # Default line-following behavior when not spinning or executing policy
                motor_commands = compute_motor_commands(offset, w)
                if not (self.pi is not None and self.at_intersection and not self.previous_at_intersection):
                    # Only apply line-following if we're not in the middle of a policy action
                    self.set_motor_command(motor_commands[0], motor_commands[1], motor_commands[2])
        else:
            self.consecutive_line_lost_frames += 1
            self.consecutive_line_detect_frames = 0
            
            # While spinning from a turn, track if line is lost
            if self.spinning_direction is not None:
                if self.consecutive_line_lost_frames >= LINE_LOST_THRESHOLD:
                    self.line_lost_during_turn = True
                    self.get_logger().debug('Line considered lost after 3 missed frames')
                self.request_spin(self.spinning_direction)
            elif self.Ab is not None:
                # Stop if line is lost and not spinning
                self.get_logger().info('Line lost while not spinning; stopping motors')
                self.Ab.stop()
        
        # Check for intersection (simplified: check if line is centered/wide)
        # In a real system, this would use IR sensors or detect wide white area
        self.at_intersection = len(midpoints) == 0 or line_detected and (offset is not None and abs(offset) < 30)
        if self.last_intersection_state is None or self.last_intersection_state != self.at_intersection:
            self.last_intersection_state = self.at_intersection
            self.get_logger().info(
                f'Intersection state changed: at_intersection={self.at_intersection} '
                f'offset={offset} midpoints={len(midpoints)}'
            )
        
        # Execute policy action only at intersections and when not spinning
        if self.pi is not None and self.at_intersection and not self.previous_at_intersection:
            if self.spinning_direction is None:  # Only act if spin is complete
                if not self.done and self.steps_taken < self.max_steps:
                    s = state_index(self.robot_row, self.robot_col, self.robot_heading, self.maze_cols)
                    if s < 0 or s >= self.pi.size:
                        self.get_logger().error(
                            f'Computed state index out of bounds: s={s} policy_size={self.pi.size} '
                            f'state=({self.robot_row},{self.robot_col},{self.robot_heading})'
                        )
                        if self.Ab is not None:
                            self.Ab.stop()
                        return
                    action = int(self.pi[s])
                    self.execute_action(action)
                    self.steps_taken += 1
                    
                    # Update pose immediately for forward, defer for turns
                    if action == FORWARD:
                        self.update_pose(action)
                    
                    # Map action to string for display
                    action_map = {FORWARD: 'FORWARD', TURN_LEFT: 'LEFT', TURN_RIGHT: 'RIGHT'}
                    action_str = action_map.get(action, '?')
                    
                    current_action = action
                    action_status = f"Step {self.steps_taken}: {action_str} at ({self.robot_row},{self.robot_col},{self.robot_heading})"
                    self.get_logger().info(action_status)
                    
                    # Check if goal reached
                    if (self.robot_row, self.robot_col) == (self.goal_row, self.goal_col):
                        self.done = True
                        if self.Ab is not None:
                            self.Ab.stop()
                        self.get_logger().info(f'Goal reached at ({self.robot_row},{self.robot_col})!')
                else:
                    action_status = "Policy complete" if self.done else "Max steps reached"
                    if self.Ab is not None:
                        self.Ab.stop()
            else:
                action_status = f"Spinning {self.spinning_direction}..."
                self._log_policy_wait_reason(f'spinning_direction={self.spinning_direction}')
        elif self.pi is None:
            action_status = "No policy loaded"
            self._log_policy_wait_reason('policy is not loaded')
        else:
            action_status = f"State: ({self.robot_row},{self.robot_col},{self.robot_heading})"
            if not self.at_intersection:
                self._log_policy_wait_reason('not at intersection yet')
            elif self.previous_at_intersection:
                self._log_policy_wait_reason('still in same intersection window; waiting for edge')
        
        self.previous_at_intersection = self.at_intersection

        # Publish processed edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        processed_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
        processed_msg.header = header
        self.publisher.publish(processed_msg)

        if ENABLE_VIZ and not HEADLESS:
            viz = draw_visualization(frame, binary, midpoints, line_params, offset, angle, roi_top)
            
            # Add MDP state and action info to visualization
            state_text = f"Cell: ({self.robot_row}, {self.robot_col}) Heading: {self.robot_heading}"
            cv2.putText(viz, state_text, (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(viz, state_text, (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 255, 100), 1)
            
            cv2.putText(viz, action_status, (12, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(viz, action_status, (12, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 200, 255), 1)
            
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
                viz_msg.header = header
                self.viz_pub.publish(viz_msg)
            except Exception:
                pass

            cv2.imshow(self.window_name, viz)
            cv2.waitKey(1)

    def image_callback(self, msg: Image):
        if self._active_image_topic is None:
            self._active_image_topic = msg.header.frame_id or 'raw'
            self.get_logger().info(
                f'Receiving raw camera frames; line follower is active (frame_id={self._active_image_topic})'
            )

        self.frames_received += 1
        self.last_frame_time = time.monotonic()

        if not self._should_process_frame():
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert raw image: {e}')
            return

        self._process_frame(msg.header, frame)

    def compressed_image_callback(self, msg: CompressedImage):
        if self._active_image_topic is None:
            self._active_image_topic = 'compressed'
            self.get_logger().info('Receiving compressed camera frames; line follower is active')

        self.frames_received += 1
        self.last_frame_time = time.monotonic()

        if not self._should_process_frame():
            return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert compressed image: {e}')
            return

        self._process_frame(msg.header, frame)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Line follow complex node with policy execution on AlphaBot2'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default=POLICY_FILE_DEFAULT,
        help=f'Path to policy.npz file (default: {POLICY_FILE_DEFAULT})'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=MAZE_ROWS_DEFAULT,
        help=f'Maze rows (default: {MAZE_ROWS_DEFAULT})'
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=MAZE_COLS_DEFAULT,
        help=f'Maze columns (default: {MAZE_COLS_DEFAULT})'
    )
    parser.add_argument(
        '--row',
        type=int,
        default=INITIAL_ROW_DEFAULT,
        help=f'Initial robot row (default: {INITIAL_ROW_DEFAULT})'
    )
    parser.add_argument(
        '--col',
        type=int,
        default=INITIAL_COL_DEFAULT,
        help=f'Initial robot column (default: {INITIAL_COL_DEFAULT})'
    )
    parser.add_argument(
        '--heading',
        type=int,
        default=INITIAL_HEADING_DEFAULT,
        help=f'Initial heading: 0=N, 1=E, 2=S, 3=W (default: {INITIAL_HEADING_DEFAULT})'
    )
    parser.add_argument(
        '--goal-row',
        type=int,
        default=GOAL_ROW_DEFAULT,
        help=f'Goal row (default: {GOAL_ROW_DEFAULT})'
    )
    parser.add_argument(
        '--goal-col',
        type=int,
        default=GOAL_COL_DEFAULT,
        help=f'Goal column (default: {GOAL_COL_DEFAULT})'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=MAX_STEPS_DEFAULT,
        help=f'Maximum steps (default: {MAX_STEPS_DEFAULT})'
    )
    
    parsed_args = parser.parse_args(args)
    
    config = {
        'policy_file': parsed_args.policy,
        'maze_rows': parsed_args.rows,
        'maze_cols': parsed_args.cols,
        'initial_row': parsed_args.row,
        'initial_col': parsed_args.col,
        'initial_heading': parsed_args.heading,
        'goal_row': parsed_args.goal_row,
        'goal_col': parsed_args.goal_col,
        'max_steps': parsed_args.max_steps,
    }
    
    rclpy.init(args=None)  # Don't pass sys.argv to rclpy, only our custom args
    node = LineFollowComplexNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            # Shutdown may already be in progress after Ctrl+C.
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
