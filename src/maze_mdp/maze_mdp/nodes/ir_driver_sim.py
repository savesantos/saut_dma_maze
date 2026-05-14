"""
Synthetic IR-strip publisher for executor bring-up tests.

Publishes ``/line_pose``, ``/intersection``, ``/line_lost`` and
``/goal_marker_seen`` according to a scripted timeline keyed by the latest
``/action_goal``:

- ``FORWARD`` (action=0): emit pose=0 for ``forward_duration_s``, then
  ``/intersection``.
- ``TURN_LEFT`` / ``TURN_RIGHT`` (action=1/2): emit pose excursion +0.7
  then pose ~0.0 to mimic the perpendicular line sliding under the centre.
- ``DRIVE_UNTIL_MARKER`` (action=3): emit pose=0 for ``forward_duration_s``,
  then ``/goal_marker_seen=True``.

Used for integration tests without hardware or Gazebo.
"""

from __future__ import annotations

from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Float32

from maze_msgs.msg import DiscreteActionGoal


# An event is (delay_after_previous_s, kind, payload).
Event = Tuple[float, str, float]
Script = List[Event]


def _forward_script(forward_s: float) -> Script:
    return [
        (0.0, 'pose', 0.0),
        (forward_s, 'intersection', 0.0),
    ]


def _turn_script() -> Script:
    return [
        (0.1, 'pose', 0.7),
        (0.4, 'pose', 0.05),
    ]


def _approach_script(forward_s: float) -> Script:
    return [
        (0.0, 'pose', 0.0),
        (forward_s, 'marker', 0.0),
    ]


class IRDriverSim(Node):
    """Replay a deterministic script of IR events for integration tests."""

    def __init__(self) -> None:
        super().__init__('ir_driver_sim')
        self.declare_parameter('forward_duration_s', 1.0)
        self.declare_parameter('line_pose_topic', '/line_pose')
        self.declare_parameter('intersection_topic', '/intersection')
        self.declare_parameter('line_lost_topic', '/line_lost')
        self.declare_parameter('publish_rate_hz', 20.0)

        self._forward_s = float(
            self.get_parameter('forward_duration_s').value)

        self._pose_pub = self.create_publisher(
            Float32, str(self.get_parameter('line_pose_topic').value), 10)
        self._cross_pub = self.create_publisher(
            Empty, str(self.get_parameter('intersection_topic').value), 10)
        self._lost_pub = self.create_publisher(
            Empty, str(self.get_parameter('line_lost_topic').value), 10)
        self._marker_pub = self.create_publisher(
            Bool, '/goal_marker_seen', 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self._dt = 1.0 / max(rate, 1e-3)
        self._events: Script = []
        self._t = 0.0
        self._next_t: float | None = None
        self._idx = 0
        self._last_pose = 0.0
        self._timer = self.create_timer(self._dt, self._tick)
        self.create_subscription(
            DiscreteActionGoal, '/action_goal', self._on_goal, 10)
        self.get_logger().info('IRDriverSim ready; awaiting /action_goal')

    def _on_goal(self, msg: DiscreteActionGoal) -> None:
        if int(msg.action) == 0:
            self._events = _forward_script(self._forward_s)
        elif int(msg.action) == 3:
            self._events = _approach_script(self._forward_s)
        else:
            self._events = _turn_script()
        self._t = 0.0
        self._idx = 0
        self._next_t = self._events[0][0]
        self.get_logger().info(
            f'replaying {len(self._events)} events for action={msg.action}')

    def _tick(self) -> None:
        if not self._events:
            return
        self._t += self._dt
        while (self._idx < len(self._events)
                and self._next_t is not None
                and self._t >= self._next_t):
            _, kind, payload = self._events[self._idx]
            if kind == 'pose':
                self._last_pose = float(payload)
                self._pose_pub.publish(Float32(data=self._last_pose))
            elif kind == 'intersection':
                self._cross_pub.publish(Empty())
            elif kind == 'lost':
                self._lost_pub.publish(Empty())
            elif kind == 'marker':
                self._marker_pub.publish(Bool(data=True))
            self._idx += 1
            if self._idx < len(self._events):
                self._next_t += self._events[self._idx][0]
            else:
                self._next_t = None
        # Steady-state heartbeat so line-lost timer doesn't trip.
        if self._idx > 0:
            self._pose_pub.publish(Float32(data=self._last_pose))


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp ir_driver_sim``."""
    rclpy.init(args=args)
    node = IRDriverSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
