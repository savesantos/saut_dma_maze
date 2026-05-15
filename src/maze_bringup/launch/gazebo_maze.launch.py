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
