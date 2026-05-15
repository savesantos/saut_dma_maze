"""Spawn the AlphaBot2 in a Gazebo world (default: empty white plane)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Bring up gzserver+gzclient with the AlphaBot2 spawned at the origin."""
    pkg_share = get_package_share_directory('alphabot2_gazebo')
    default_world = os.path.join(pkg_share, 'worlds', 'empty.world')
    urdf_path = os.path.join(pkg_share, 'urdf', 'alphabot2.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    world_arg = DeclareLaunchArgument(
        'world', default_value=default_world,
        description='Path to the Gazebo .world file to load.')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='If true, run gzserver only (no gzclient GUI).')
    x_arg = DeclareLaunchArgument('x', default_value='0.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.05')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')

    gazebo_pkg = get_package_share_directory('gazebo_ros')
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gzserver.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'false',
        }.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gzclient.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_alphabot2',
        output='screen',
        arguments=[
            '-entity', 'alphabot2',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
    )

    return LaunchDescription([
        world_arg, headless_arg, x_arg, y_arg, z_arg, yaw_arg,
        gzserver, gzclient, rsp, spawn,
    ])
