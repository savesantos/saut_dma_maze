"""
Node that publishes the static maze grid and exposes a ``SetGoal`` service.

Loads a maze layout (YAML) at startup and publishes :class:`maze_msgs/MazeGrid`
on a latched topic so downstream nodes do not need to know the file path.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from maze_msgs.msg import MazeGrid
from maze_msgs.srv import SetGoal

from maze_mdp.simulator import FREE, GOAL, WALL, Maze, fixture_3x3


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _load_maze_from_yaml(path: Path) -> Maze:
    with path.open('r') as f:
        spec = yaml.safe_load(f)
    layout = spec['layout']
    goal = tuple(spec['goal'])
    return Maze.from_layout(layout=layout, goal=(int(goal[0]), int(goal[1])))


class MazePublisher(Node):
    """Publish the current maze on ``/maze`` and serve goal updates."""

    def __init__(self) -> None:
        super().__init__('maze_publisher')
        self.declare_parameter('maze_path', '')
        self.declare_parameter('topic', '/maze')
        self.declare_parameter('service', '/set_goal')

        topic = self.get_parameter('topic').get_parameter_value().string_value
        service = self.get_parameter('service').get_parameter_value().string_value
        maze_path = self.get_parameter('maze_path').get_parameter_value().string_value

        if maze_path:
            self._maze = _load_maze_from_yaml(Path(maze_path))
            self.get_logger().info(f'Loaded maze from {maze_path}')
        else:
            self._maze = fixture_3x3()
            self.get_logger().warn('No maze_path provided; using built-in 3x3 fixture.')

        self._publisher = self.create_publisher(MazeGrid, topic, _latched_qos())
        self._service = self.create_service(SetGoal, service, self._handle_set_goal)
        self._publish()

    def _publish(self) -> None:
        msg = MazeGrid()
        msg.rows = int(self._maze.rows)
        msg.cols = int(self._maze.cols)
        msg.cells = self._maze.cells.flatten().astype('int8').tolist()
        self._publisher.publish(msg)

    def _handle_set_goal(self, request: SetGoal.Request, response: SetGoal.Response):
        r, c = int(request.row), int(request.col)
        if not (0 <= r < self._maze.rows and 0 <= c < self._maze.cols):
            response.success = False
            response.message = f'Goal ({r}, {c}) outside grid.'
            return response
        if self._maze.cells[r, c] == WALL:
            response.success = False
            response.message = f'Goal ({r}, {c}) is a wall.'
            return response

        cells = self._maze.cells.copy()
        # Demote the previous goal, if any.
        cells[cells == GOAL] = FREE
        cells[r, c] = GOAL
        self._maze = Maze(cells=cells, goal=(r, c))
        self._publish()
        response.success = True
        response.message = f'Goal set to ({r}, {c}).'
        return response


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp maze_publisher``."""
    rclpy.init(args=args)
    node = MazePublisher()
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
