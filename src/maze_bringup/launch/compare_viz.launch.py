"""
Side-by-side comparison launch: VI vs. SARSA vs. Q-Learning in one RViz.

Brings up:
    * ``maze_publisher``  — latched ``/maze``.
    * ``maze_viz_node``   — walls + goal + value heatmap + policy arrows.
                            The single-robot ``/robot_marker`` is unused
                            because no ``maze_sim_node`` is running.
    * ``compare_node``    — runs the supplied policies in parallel and
                            publishes ``/compare/markers``.
    * ``rviz2``           — bundled ``compare.rviz`` with one MarkerArray
                            display per policy colour.

Pass three policy ``.npz`` paths via ``vi_policy``, ``sarsa_policy``,
``qlearning_policy`` (absolute paths). Optional: ``start_row``/``start_col``/
``start_heading`` to fix the shared starting cell, ``seed`` to seed the
slip-noise RNGs.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

try:
    from maze_mdp.analysis.select_best_run import selected_policy_path
except Exception:  # noqa: BLE001 - keep launch importable without analysis deps
    selected_policy_path = None  # type: ignore[assignment]


def _resolve_policy(explicit: str, algo: str, maze_name: str) -> str:
    """Use the explicit launch arg if set; otherwise dereference selected.json."""
    if explicit:
        return explicit
    if selected_policy_path is None:
        raise RuntimeError(
            f'No {algo} policy provided and maze_mdp.analysis is unavailable.'
        )
    path = selected_policy_path(algo, maze_name)
    if path is None:
        raise RuntimeError(
            f'No selected policy for ({algo}, {maze_name}). '
            'Run `python -m maze_mdp.analysis.select_best_run` first '
            f'or pass {algo}_policy:=<path> explicitly.'
        )
    return str(path)


def _make_compare_node(context, *args, **kwargs):
    """Build ``compare_node`` after resolving the per-policy substitutions.

    Putting the three policy paths inside a Python list of
    ``LaunchConfiguration``s makes ros2 launch serialize them as one
    concatenated string when writing the YAML params file. Resolving the
    substitutions here yields a real Python list of strings, which
    serializes correctly as ``STRING_ARRAY``.
    """
    maze_name = LaunchConfiguration('maze_name').perform(context)
    return [Node(
        package='maze_mdp',
        executable='compare_node',
        name='compare_node',
        output='screen',
        parameters=[{
            'policy_paths': [
                _resolve_policy(
                    LaunchConfiguration('vi_policy').perform(context),
                    'vi', maze_name,
                ),
                _resolve_policy(
                    LaunchConfiguration('sarsa_policy').perform(context),
                    'sarsa', maze_name,
                ),
                _resolve_policy(
                    LaunchConfiguration('qlearning_policy').perform(context),
                    'qlearning', maze_name,
                ),
            ],
            'labels': ['Value Iteration', 'SARSA', 'Q-Learning'],
            'colors_rgb': [
                '0.10,0.40,1.00',
                '1.00,0.45,0.10',
                '0.20,0.85,0.30',
            ],
            'seed': int(LaunchConfiguration('seed').perform(context)),
            'start_row': int(LaunchConfiguration('start_row').perform(context)),
            'start_col': int(LaunchConfiguration('start_col').perform(context)),
            'start_heading': int(LaunchConfiguration('start_heading').perform(context)),
            'slip_prob': float(LaunchConfiguration('slip_prob').perform(context)),
            'tick_rate_hz': float(LaunchConfiguration('tick_rate_hz').perform(context)),
        }],
    )]


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory('maze_bringup')
    default_params = os.path.join(bringup_share, 'config', 'params.yaml')
    default_rviz = os.path.join(bringup_share, 'config', 'rviz', 'compare.rviz')

    maze_name = LaunchConfiguration('maze_name')
    params_file = LaunchConfiguration('params_file')
    rviz_config = LaunchConfiguration('rviz_config')

    maze_path = PathJoinSubstitution([
        bringup_share, 'config', 'mazes', [maze_name, '.yaml'],
    ])

    return LaunchDescription([
        DeclareLaunchArgument('maze_name', default_value='fixture_5x5_corridor'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('start_row', default_value='-1'),
        DeclareLaunchArgument('start_col', default_value='-1'),
        DeclareLaunchArgument('start_heading', default_value='1'),  # E
        DeclareLaunchArgument('vi_policy', default_value=''),
        DeclareLaunchArgument('sarsa_policy', default_value=''),
        DeclareLaunchArgument('qlearning_policy', default_value=''),
        DeclareLaunchArgument('slip_prob', default_value='0.1'),
        DeclareLaunchArgument('tick_rate_hz', default_value='4.0'),

        Node(
            package='maze_mdp',
            executable='maze_publisher',
            name='maze_publisher',
            output='screen',
            parameters=[params_file, {'maze_path': maze_path}],
        ),
        Node(
            package='maze_mdp',
            executable='maze_viz_node',
            name='maze_viz_node',
            output='screen',
            # No policy_path here on purpose: each agent has its own policy
            # in compare_node, and overlaying one of them as a global heatmap
            # would be misleading.
            parameters=[params_file],
        ),
        OpaqueFunction(function=_make_compare_node),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', rviz_config],
            additional_env={
                'LIBGL_ALWAYS_SOFTWARE':
                    os.environ.get('RVIZ_FORCE_SOFTWARE_GL', '1'),
                'GALLIUM_DRIVER': 'llvmpipe',
            },
        ),
    ])
