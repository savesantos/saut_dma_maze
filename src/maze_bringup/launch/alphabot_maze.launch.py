"""End-to-end hardware launch: full controller stack on the AlphaBot2.

Brings up the same node graph as ``gazebo_maze.launch.py``, but with
the IR strip and the goal-marker pipeline backed by real hardware:

- ``maze_publisher``      (latched ``/maze`` from the fixture YAML)
- ``ir_driver_hardware``  (Waveshare TRSensors, 3 central channels)
- ``fiducial_localizer``  (camera ArUco/AprilTag -> ``/goal_marker_seen``
                           and discrete cell estimate)
- ``action_executor``     (closed-loop discrete action driver,
                           publishes ``/alphabot2/cmd_vel``)
- ``cell_tracker``        (cell estimate from action results)
- ``policy_runner``       (``mode='action'`` with ``exit_on_goal:=True``)

Does **not** launch the AlphaBot2 driver itself (the course-staff
``alphabot2`` package + ``motion_driver``). Those run on the
RaspberryPi over SSH per the lab guide; this launch runs on the laptop
with ``ROS_DOMAIN_ID`` set to the robot's IP last octet.

Launch arguments (all optional unless noted):
- ``maze_name``      (default ``fixture_3x3``)
- ``policy_path``    (required ``.npz`` policy file)
- ``start_row`` / ``start_col`` / ``start_heading``
- ``cell_size``      (default ``0.20`` m -- match the printed maze)
- ``params_file``    (optional YAML overlay for shared params)
- ``marker_map``     (default: per-maze YAML in config/markers/)
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
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Heading -> yaw (rad). Matches the world frame used by maze_to_sdf:
# x = col*cs, y = -row*cs; yaw=0 -> East.
_HEADING_TO_YAW = {0: math.pi / 2,   # N
                   1: 0.0,           # E
                   2: -math.pi / 2,  # S
                   3: math.pi}       # W


def _build_nodes(context: LaunchContext, *args, **kwargs):
    """Read the maze YAML, then emit the node graph."""
    bringup_share = get_package_share_directory('maze_bringup')

    maze_name = LaunchConfiguration('maze_name').perform(context)
    start_row = int(LaunchConfiguration('start_row').perform(context))
    start_col = int(LaunchConfiguration('start_col').perform(context))
    start_heading = int(LaunchConfiguration('start_heading').perform(context))
    policy_path = LaunchConfiguration('policy_path').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)
    marker_map = LaunchConfiguration('marker_map').perform(context)

    if not policy_path:
        raise RuntimeError(
            'alphabot_maze.launch.py: policy_path:=<file.npz> is required')

    maze_path = os.path.join(
        bringup_share, 'config', 'mazes', f'{maze_name}.yaml')
    if not os.path.exists(maze_path):
        raise RuntimeError(f'Maze fixture not found: {maze_path}')

    # Default marker map: config/markers/<maze_name>.yaml if not given.
    if not marker_map:
        marker_map = os.path.join(
            bringup_share, 'config', 'markers', f'{maze_name}.yaml')

    with open(maze_path, 'r') as f:
        maze_spec = yaml.safe_load(f)
    layout = maze_spec['layout']
    rows = len(layout)
    cols = len(layout[0]) if rows else 0
    # Yaw is informational here (start orientation); pose is owned by the
    # operator who places the robot at the start cell.
    _ = _HEADING_TO_YAW.get(start_heading, 0.0)

    common_params: List = []
    if params_file:
        common_params.append(params_file)

    maze_publisher = Node(
        package='maze_mdp',
        executable='maze_publisher',
        name='maze_publisher',
        output='screen',
        parameters=common_params + [{'maze_path': maze_path}],
    )

    ir_driver = Node(
        package='maze_mdp',
        executable='ir_driver_hardware',
        name='ir_driver_hardware',
        output='screen',
        parameters=common_params + [{
            # Calibration sweep on startup. Override via params_file
            # once the floor + lighting are characterised.
            'calibration_seconds': 6.0,
            # Sign convention is verified per-robot during first run.
            'sensor_sign': 1,
        }],
    )

    fiducial_localizer = Node(
        package='maze_mdp',
        executable='fiducial_localizer',
        name='fiducial_localizer',
        output='screen',
        parameters=common_params + [{'marker_map_path': marker_map}],
    )

    action_executor = Node(
        package='maze_mdp',
        executable='action_executor',
        name='action_executor',
        output='screen',
        parameters=common_params + [{
            # Hardware-specific tuning: motor model, PID gains and the
            # commanded-yaw integral for a 90-degree turn must be
            # re-measured on the actual AlphaBot2. The defaults below
            # mirror the Gazebo launch and are the starting point.
            'forward_speed': 0.10,
            'turn_speed': 0.60,
            'control_rate_hz': 20.0,
            'action_timeout_s': 12.0,
            'line_p_gain': 1.2,
            'line_i_gain': 0.0,
            'line_d_gain': 1.0,
            'line_d_filter_tau': 0.04,
            'line_i_clamp': 0.5,
            'line_omega_clamp': 1.8,
            # Real motors typically need a turn integral closer to
            # pi/2 than the Gazebo bias-corrected 1.96 rad. Tune on
            # the robot; this is the starting value.
            'turn_target_yaw_rad': math.pi / 2,
            'turn_max_yaw_rad': 2.80,
        }],
    )

    cell_tracker = Node(
        package='maze_mdp',
        executable='cell_tracker',
        name='cell_tracker',
        output='screen',
        parameters=common_params + [{
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
        parameters=common_params + [{
            'policy_path': policy_path,
            'mode': 'action',
            'exit_on_goal': True,
            'control_rate_hz': 5.0,
        }],
    )

    return [
        maze_publisher,
        fiducial_localizer,
        ir_driver,
        action_executor,
        cell_tracker,
        # Delay the policy so the IR calibration sweep completes and the
        # first /line_pose messages are flowing before the executor
        # demands them.
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


def generate_launch_description() -> LaunchDescription:
    """Build the launch description with all CLI args."""
    bringup_share = get_package_share_directory('maze_bringup')
    default_params = os.path.join(bringup_share, 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_3x3'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('start_row', default_value='0'),
        DeclareLaunchArgument('start_col', default_value='0'),
        DeclareLaunchArgument('start_heading', default_value='1'),  # E
        DeclareLaunchArgument('cell_size', default_value='0.20'),
        DeclareLaunchArgument(
            'params_file',
            default_value=(default_params
                           if os.path.exists(default_params) else '')),
        DeclareLaunchArgument('marker_map', default_value=''),
        OpaqueFunction(function=_build_nodes),
    ])
