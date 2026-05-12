"""Sim + RViz2 launch: sim graph from sim.launch.py plus visualization.

Adds ``maze_viz_node`` (republishes our custom topics as standard RViz types
and broadcasts the ``map -> base_link`` TF) and an ``rviz2`` instance loading
the bundled config at ``config/rviz/maze.rviz``.

Use for live debugging and report screen-recordings. For headless training
sweeps, keep using ``sim.launch.py`` to avoid the RViz overhead.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('maze_bringup')
    default_params = os.path.join(bringup_share, 'config', 'params.yaml')
    default_rviz = os.path.join(bringup_share, 'config', 'rviz', 'maze.rviz')

    maze_name = LaunchConfiguration('maze_name')
    policy_path = LaunchConfiguration('policy_path')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')
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
        parameters=[params_file, {'policy_path': policy_path}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_5x5_corridor'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
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
        Node(
            package='maze_mdp',
            executable='maze_viz_node',
            name='maze_viz_node',
            output='screen',
            parameters=[params_file, {'policy_path': policy_path}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', rviz_config],
            # Force the Mesa software rasterizer (llvmpipe) for RViz only.
            # The default WSLg path uses the D3D12 Mesa driver, which
            # renders the OGRE scene fully black ("D3D12 (Intel ... )" in
            # glxinfo). llvmpipe is slower but renders correctly. Set
            # ``RVIZ_FORCE_SOFTWARE_GL=0`` in the environment to opt out
            # on a native Linux machine.
            additional_env={
                'LIBGL_ALWAYS_SOFTWARE':
                    os.environ.get('RVIZ_FORCE_SOFTWARE_GL', '1'),
                'GALLIUM_DRIVER': 'llvmpipe',
            },
        ),
        policy_runner,
        # NOTE: unlike sim.launch.py, we deliberately do NOT shut the launch
        # down when policy_runner exits. The whole point of sim_viz is to
        # inspect the maze, value heatmap, policy arrows and trail in RViz
        # after the run; let the user Ctrl-C when finished.
    ])
