"""End-to-end Gazebo Classic launch: maze world + AlphaBot2 + controller stack.

Brings up:
- Gazebo Classic with a pre-generated SDF world for the chosen maze fixture.
- The AlphaBot2 URDF spawned at the start cell with the requested heading.
- ``maze_publisher`` (latched ``/maze`` from the fixture YAML).
- ``ir_driver_gazebo`` (analytic IR/marker driver consuming
  ``/virtual_odometry``).
- ``action_executor`` (closed-loop discrete action driver).
- ``cell_tracker`` (cell estimate from action results).
- ``policy_runner`` in ``mode='action'`` with ``exit_on_goal:=True``.

When ``policy_runner`` exits (goal reached) the whole launch shuts down.

Args (all optional, sensible defaults):
- ``maze_name``      (default ``fixture_3x3``)
- ``policy_path``    (default empty -- you must pass a ``.npz`` policy)
- ``start_row``      (default ``0``)
- ``start_col``      (default ``0``)
- ``start_heading``  (default ``1`` = East)
- ``cell_size``      (default ``0.20`` m)
- ``headless``       (default ``true``; set to ``false`` for the gzclient GUI)
"""

import math
import os
from typing import List

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


# Heading -> yaw (rad). Matches maze_to_sdf world frame:
# x = col*cs, y = -row*cs; yaw=0 -> East.
_HEADING_TO_YAW = {0: math.pi / 2,   # N
                   1: 0.0,           # E
                   2: -math.pi / 2,  # S
                   3: math.pi}       # W


def _pin_gazebo_camera() -> None:
    """Strip persisted camera state from ``~/.gazebo/gui.ini``.

    gzclient saves its window geometry and last user-camera pose to
    ``~/.gazebo/gui.ini`` on exit, then reapplies them on the next start
    *after* loading the world's ``<gui><camera>`` block. The net effect is
    that the top-down camera baked into our generated ``.world`` files is
    silently overridden by whatever orbit pose the user last left the
    client in.

    To make the SDF camera authoritative again we remove the persisted
    ``[geometry]`` and ``[camera]`` sections (leaving all other sections
    untouched). gzclient regenerates them with neutral defaults that do
    not override the world pose.
    """
    ini_path = os.path.expanduser('~/.gazebo/gui.ini')
    if not os.path.exists(ini_path):
        return
    try:
        with open(ini_path, 'r') as f:
            lines = f.readlines()
    except OSError:
        return
    cleaned: List[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1].strip().lower()
            skip = section in ('geometry', 'camera')
            if skip:
                continue
        if skip:
            continue
        cleaned.append(line)
    try:
        with open(ini_path, 'w') as f:
            f.writelines(cleaned)
    except OSError:
        return


def _spawn_and_nodes(context: LaunchContext, *args, **kwargs):
    """Compute spawn pose from start cell, then emit Gazebo + node actions."""
    bringup_share = get_package_share_directory('maze_bringup')
    gazebo_share = get_package_share_directory('alphabot2_gazebo')

    maze_name = LaunchConfiguration('maze_name').perform(context)
    cell_size = float(LaunchConfiguration('cell_size').perform(context))
    start_row = int(LaunchConfiguration('start_row').perform(context))
    start_col = int(LaunchConfiguration('start_col').perform(context))
    start_heading = int(LaunchConfiguration('start_heading').perform(context))
    headless = LaunchConfiguration('headless').perform(context)
    policy_path = LaunchConfiguration('policy_path').perform(context)
    if not policy_path:
        raise RuntimeError(
            'gazebo_maze.launch.py: policy_path:=<file.npz> is required')

    maze_path = os.path.join(
        bringup_share, 'config', 'mazes', f'{maze_name}.yaml')
    world_path = os.path.join(
        gazebo_share, 'worlds', f'{maze_name}.world')
    if not os.path.exists(world_path):
        raise RuntimeError(
            f'World file not found: {world_path}. Regenerate it via\n'
            f'  python3 -m maze_mdp.analysis.maze_to_sdf '
            f'src/maze_bringup/config/mazes/{maze_name}.yaml '
            f'src/alphabot2_gazebo/worlds/{maze_name}.world')

    spawn_x = start_col * cell_size
    spawn_y = -start_row * cell_size
    spawn_yaw = _HEADING_TO_YAW.get(start_heading, 0.0)

    # Pin Gazebo's user camera to the top-down view defined in the world
    # file. gzclient persists ``[geometry]`` and ``[camera]`` in
    # ``~/.gazebo/gui.ini`` and applies them on the next start, which
    # silently overrides the world's ``<gui><camera>`` block. Strip those
    # two sections so the SDF wins. We only touch gui.ini when the GUI is
    # actually being launched (``headless == 'false'``) and we keep all
    # other user-tuned sections intact.
    if str(headless).lower() == 'false':
        _pin_gazebo_camera()

    # Read maze dimensions for cell_tracker.
    with open(maze_path, 'r') as f:
        maze_spec = yaml.safe_load(f)
    layout = maze_spec['layout']
    rows = len(layout)
    cols = len(layout[0]) if rows else 0

    spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'spawn_robot.launch.py')),
        launch_arguments={
            'world': world_path,
            'headless': headless,
            'x': f'{spawn_x:.4f}',
            'y': f'{spawn_y:.4f}',
            'yaw': f'{spawn_yaw:.4f}',
        }.items(),
    )

    maze_publisher = Node(
        package='maze_mdp',
        executable='maze_publisher',
        name='maze_publisher',
        output='screen',
        parameters=[{'maze_path': maze_path}],
    )

    ir_driver = Node(
        package='maze_mdp',
        executable='ir_driver_gazebo',
        name='ir_driver_gazebo',
        output='screen',
        parameters=[{
            'maze_path': maze_path,
            'cell_size': cell_size,
            # Generous marker proximity: policy emits DRIVE_UNTIL_MARKER as
            # soon as the agent is one cell from the goal, so we need to
            # see the marker from at least cell_size away. Disable facing
            # check (set tol to pi) so that imperfect orientation after a
            # turn doesn't blind the robot to the marker.
            'marker_proximity_m': cell_size * 2.0,
            'marker_facing_tol': 3.14,
        }],
    )

    action_executor = Node(
        package='maze_mdp',
        executable='action_executor',
        name='action_executor',
        output='screen',
        parameters=[{
            'forward_speed': 0.10,
            'turn_speed': 0.60,
            'control_rate_hz': 20.0,
            'action_timeout_s': 12.0,
            # Line-follow PID, derived from the small-angle line-follower
            # ODE so the same gains transfer to the AlphaBot2 with only a
            # touch-up of ki if needed.
            #
            # Plant (small heading errors):
            #   d(lat)/dt    = v * theta
            #   d(theta)/dt  = omega = -(kp/W)*lat - (kd*v/W)*theta
            # where W = line_capture_width = 0.04 m and v = forward_speed.
            # That is a 2nd-order LTI system with:
            #   omega_n = sqrt(v * kp / W)
            #   zeta    = (kd/2) * sqrt(v / (W * kp))
            #
            # Target a well-damped response (zeta ~= 0.7) so the robot
            # rejects post-turn heading bias and small motor asymmetry
            # without zig-zag and without monotonic drift off the line:
            #   kp = 1.2 -> omega_n ~= 1.73 rad/s (settles in ~1.6 s)
            #   kd = 1.0 -> zeta ~= 0.72
            # The previous (kp=1.0, kd=0.10) was zeta ~= 0.08, essentially
            # undamped: any heading bias walked the robot off the line
            # before P could pull it back, and at small |pose| the
            # restoring force was too weak (1 cm offset -> only 0.25 rad/s
            # -> 40 cm turning radius, two cells per correction cycle).
            #
            # ki stays at 0: D-action plus the line-loss hold below
            # rejects bias without integral windup. On real hardware add
            # ki ~= 0.05-0.1 only if a measurable steady-state drift
            # remains after kp/kd are tuned by this same procedure.
            'line_p_gain': 1.2,
            'line_i_gain': 0.0,
            'line_d_gain': 1.0,
            'line_d_filter_tau': 0.04,
            'line_i_clamp': 0.5,
            'line_omega_clamp': 1.8,
            # Sim-specific turn calibration: gazebo_ros_diff_drive reaches
            # ~80% of commanded omega in steady state, so a commanded-yaw
            # integral of pi/2 only rotates ~73 degrees. Bump the target.
            # On real hardware this should be re-measured (likely closer
            # to pi/2).
            'turn_target_yaw_rad': 1.96,
            'turn_max_yaw_rad': 2.80,
        }],
    )

    cell_tracker = Node(
        package='maze_mdp',
        executable='cell_tracker',
        name='cell_tracker',
        output='screen',
        parameters=[{
            'rows': rows,
            'cols': cols,
            'start_row': start_row,
            'start_col': start_col,
            'start_heading': start_heading,
        }],
    )

    policy_runner = Node(
        package='maze_mdp',
        executable='policy_runner',
        name='policy_runner',
        output='screen',
        parameters=[{
            'policy_path': policy_path,
            'mode': 'action',
            'exit_on_goal': True,
            'control_rate_hz': 5.0,
        }],
    )

    actions: List = [
        spawn_launch,
        maze_publisher,
        ir_driver,
        action_executor,
        cell_tracker,
        # Delay the policy so Gazebo has fully spawned the robot and
        # /virtual_odometry/line_pose are flowing; otherwise the executor
        # times out on LINE_LOST while the robot is invisible to ROS,
        # corrupting the cell tracker's idea of where the robot is.
        TimerAction(period=8.0, actions=[policy_runner]),
        # Tear the whole launch down when the policy reaches the goal.
        RegisterEventHandler(
            OnProcessExit(
                target_action=policy_runner,
                on_exit=[EmitEvent(
                    event=Shutdown(reason='policy_runner finished'))],
            )
        ),
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    """Build the launch description with all CLI args."""
    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_3x3'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('start_row', default_value='0'),
        DeclareLaunchArgument('start_col', default_value='0'),
        DeclareLaunchArgument('start_heading', default_value='1'),  # E
        DeclareLaunchArgument('cell_size', default_value='0.20'),
        DeclareLaunchArgument('headless', default_value='true'),
        OpaqueFunction(function=_spawn_and_nodes),
    ])
