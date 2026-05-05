"""Top-level launch for the AlphaBot2 maze stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('maze_bringup')
    default_params = os.path.join(bringup_share, 'config', 'params.yaml')

    maze_name = LaunchConfiguration('maze_name')
    policy_path = LaunchConfiguration('policy_path')
    params_file = LaunchConfiguration('params_file')
    marker_map = LaunchConfiguration('marker_map')

    maze_path = PathJoinSubstitution([
        bringup_share, 'config', 'mazes', [maze_name, '.yaml'],
    ])

    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_5x5_corridor'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument(
            'marker_map',
            default_value=PathJoinSubstitution([
                bringup_share, 'config', 'markers', [maze_name, '.yaml'],
            ]),
        ),

        Node(
            package='maze_mdp',
            executable='maze_publisher',
            name='maze_publisher',
            output='screen',
            parameters=[params_file, {'maze_path': maze_path}],
        ),
        Node(
            package='maze_mdp',
            executable='fiducial_localizer',
            name='fiducial_localizer',
            output='screen',
            parameters=[params_file, {'marker_map_path': marker_map}],
        ),
        Node(
            package='maze_mdp',
            executable='policy_runner',
            name='policy_runner',
            output='screen',
            parameters=[params_file, {'policy_path': policy_path}],
        ),
    ])
