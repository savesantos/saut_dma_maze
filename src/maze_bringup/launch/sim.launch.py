"""Sim-only launch: maze_publisher + maze_sim_node + policy_runner.

Drop-in replacement for ``alphabot_maze.launch.py`` that swaps the physical
robot + ``fiducial_localizer`` for the in-process grid micro-simulator. Use
this for offline policy validation, deterministic seeded rollouts, and
recording deployment rosbags without hardware.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('maze_bringup')
    default_params = os.path.join(bringup_share, 'config', 'params.yaml')

    maze_name = LaunchConfiguration('maze_name')
    policy_path = LaunchConfiguration('policy_path')
    params_file = LaunchConfiguration('params_file')
    seed = LaunchConfiguration('seed')
    start_row = LaunchConfiguration('start_row')
    start_col = LaunchConfiguration('start_col')
    start_heading = LaunchConfiguration('start_heading')

    maze_path = PathJoinSubstitution([
        bringup_share, 'config', 'mazes', [maze_name, '.yaml'],
    ])

    policy_runner = Node(
        package='maze_mdp',
        executable='policy_runner',
        name='policy_runner',
        output='screen',
        parameters=[params_file, {'policy_path': policy_path, 'exit_on_goal': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_5x5_corridor'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('seed', default_value='-1'),
        # Initial cell overrides for maze_sim_node. Negative => random reset.
        DeclareLaunchArgument('start_row', default_value='-1'),
        DeclareLaunchArgument('start_col', default_value='-1'),
        DeclareLaunchArgument('start_heading', default_value='1'),  # E

        Node(
            package='maze_mdp',
            executable='maze_publisher',
            name='maze_publisher',
            output='screen',
            parameters=[params_file, {'maze_path': maze_path}],
        ),
        Node(
            package='maze_mdp',
            executable='maze_sim_node',
            name='maze_sim_node',
            output='screen',
            parameters=[params_file, {
                'seed': seed,
                'start_row': start_row,
                'start_col': start_col,
                'start_heading': start_heading,
            }],
        ),
        policy_runner,

        # When policy_runner exits (goal reached or error), tear the whole
        # launch down so the shell prompt returns automatically.
        RegisterEventHandler(
            OnProcessExit(
                target_action=policy_runner,
                on_exit=[EmitEvent(event=Shutdown(reason='policy_runner finished'))],
            )
        ),
    ])
