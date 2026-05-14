"""
Bring up :class:`ActionExecutorNode` plus :class:`IRDriverSim` for testing.

Used to validate the executor state machine end-to-end without any
hardware. Send a discrete action goal manually, e.g.::

    ros2 topic pub --once /action_goal maze_msgs/msg/DiscreteActionGoal \\
        "{goal_id: 1, action: 0}"

and watch ``/action_result`` and ``/alphabot2/cmd_vel``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    forward_duration_s = LaunchConfiguration('forward_duration_s')

    return LaunchDescription([
        DeclareLaunchArgument(
            'forward_duration_s', default_value='1.0',
            description='Seconds of pose=0 before /intersection on FORWARD.',
        ),
        Node(
            package='maze_mdp',
            executable='action_executor',
            name='action_executor',
            output='screen',
            parameters=[{
                'forward_speed': 0.10,
                'turn_speed': 0.60,
                'control_rate_hz': 20.0,
            }],
        ),
        Node(
            package='maze_mdp',
            executable='ir_driver_sim',
            name='ir_driver_sim',
            output='screen',
            parameters=[{
                'forward_duration_s': forward_duration_s,
            }],
        ),
    ])
