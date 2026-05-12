"""
Side-by-side policy comparison node.

Loads N (typically 3: VI / SARSA / Q-Learning) precomputed policies and runs
them simultaneously inside this single process against independent stochastic
copies of the *same* maze, started from the *same* cell. Each agent renders
as a small colored "car" (body + cabin + bumper) plus a same-coloured trail
in RViz, on the topic ``/compare/markers``.

The node is purely visual: it bypasses ``maze_sim_node`` and ``policy_runner``
because they are single-agent and hard-coded to ``/alphabot2/cmd_vel`` /
``/robot_cell``. The maze and goal still come from the latched ``/maze``
topic published by ``maze_publisher``.

Subscriptions
    * ``/maze``       (latched ``maze_msgs/MazeGrid``)
    * ``/sim_reset``  (``std_msgs/Empty``)  — restart all agents from a
      freshly sampled (or configured) start cell.

Publications
    * ``/compare/markers`` (``visualization_msgs/MarkerArray``)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Empty
from visualization_msgs.msg import Marker, MarkerArray

from maze_msgs.msg import MazeGrid

from maze_mdp.mdp import Heading, MDPConfig
from maze_mdp.policy import greedy_policy, load_policy
from maze_mdp.simulator import GOAL, GridMaze, Maze

from maze_mdp.nodes.maze_viz_node import build_car_marker_array

_HEADING_YAW = {
    int(Heading.N): math.pi / 2.0,
    int(Heading.E): 0.0,
    int(Heading.S): -math.pi / 2.0,
    int(Heading.W): math.pi,
}

# Default per-agent palette (R, G, B). Indices 0..2 are the canonical
# VI / SARSA / Q-Learning colours; extra entries are fallbacks.
_DEFAULT_COLORS = [
    (0.10, 0.40, 1.00),   # blue       — Value Iteration
    (1.00, 0.45, 0.10),   # orange     — SARSA
    (0.20, 0.85, 0.30),   # green      — Q-Learning
    (0.85, 0.20, 0.85),   # magenta
    (0.95, 0.85, 0.10),   # yellow
]


def _latched_sub_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _darken(rgb: tuple[float, float, float], factor: float = 0.45) -> tuple[float, float, float]:
    return (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor)


class _Agent:
    """Bookkeeping for a single policy in the comparison run."""

    def __init__(
        self,
        label: str,
        pi: np.ndarray,
        color: tuple[float, float, float],
        seed: int,
    ) -> None:
        self.label = label
        self.pi = pi
        self.color = color
        self.dark = _darken(color)
        self.seed = seed
        self.env: GridMaze | None = None
        self.done = False
        self.steps = 0
        self.trail: list[tuple[float, float]] = []  # world (x, y) cell centres


class CompareNode(Node):
    """Run several precomputed policies in parallel and visualise them."""

    def __init__(self) -> None:
        super().__init__('compare_node')
        # Inputs.
        self.declare_parameter('maze_topic', '/maze')
        self.declare_parameter('reset_topic', '/sim_reset')
        # Outputs.
        self.declare_parameter('markers_topic', '/compare/markers')
        # Geometry.
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('cell_size_m', 0.20)
        # Simulation.
        self.declare_parameter('tick_rate_hz', 4.0)
        self.declare_parameter('slip_prob', 0.1)
        self.declare_parameter('seed', 0)
        # Initial pose. Negative row/col -> uniform random over free cells.
        self.declare_parameter('start_row', -1)
        self.declare_parameter('start_col', -1)
        self.declare_parameter('start_heading', int(Heading.E))
        # Per-agent configuration. ROS 2 string arrays — one entry per agent,
        # the three lists must have the same length. ``policy_paths`` is the
        # only required one; ``labels`` defaults to the file stem and
        # ``colors_rgb`` to the canonical VI/SARSA/QL palette.
        self.declare_parameter('policy_paths', [''])
        self.declare_parameter('labels', [''])
        self.declare_parameter(
            'colors_rgb',
            [''],
            # Each entry is a 'r,g,b' triple in 0..1.
        )
        self.declare_parameter('trail_max_points', 2000)
        # Auto-restart: when every agent is done (goal or truncated), wait
        # this many seconds and then sample a new shared start cell and
        # reset all agents. Set to <= 0 to disable and require an external
        # /sim_reset to restart.
        self.declare_parameter('auto_restart_delay_s', 1.5)

        gp = self.get_parameter
        self._frame = gp('frame_id').get_parameter_value().string_value
        self._cell = float(gp('cell_size_m').value)
        self._dt = 1.0 / max(float(gp('tick_rate_hz').value), 1e-3)
        self._slip = float(gp('slip_prob').value)
        self._seed = int(gp('seed').value)
        self._trail_max = int(gp('trail_max_points').value)
        self._auto_restart = float(gp('auto_restart_delay_s').value)
        self._restart_at: float | None = None

        sr = int(gp('start_row').value)
        sc = int(gp('start_col').value)
        sh = int(gp('start_heading').value)
        self._configured_start: tuple[int, int, int] | None = (
            (sr, sc, sh) if sr >= 0 and sc >= 0 else None
        )
        self._current_start: tuple[int, int, int] | None = self._configured_start

        paths = [p for p in gp('policy_paths').get_parameter_value().string_array_value if p]
        if not paths:
            raise RuntimeError(
                'compare_node requires a non-empty policy_paths list parameter.'
            )
        labels_param = list(gp('labels').get_parameter_value().string_array_value)
        labels = [
            (labels_param[i] if i < len(labels_param) and labels_param[i] else Path(p).stem)
            for i, p in enumerate(paths)
        ]
        colors_param = [
            s for s in gp('colors_rgb').get_parameter_value().string_array_value if s
        ]
        colors: list[tuple[float, float, float]] = []
        for i in range(len(paths)):
            if i < len(colors_param):
                try:
                    parts = [float(x) for x in colors_param[i].split(',')]
                    if len(parts) != 3:
                        raise ValueError
                    colors.append((parts[0], parts[1], parts[2]))
                    continue
                except ValueError:
                    self.get_logger().warn(
                        f'Could not parse colors_rgb[{i}]={colors_param[i]!r}; using default.'
                    )
            colors.append(_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])

        self._agents: list[_Agent] = []
        for path, label, color in zip(paths, labels, colors):
            bundle = load_policy(path)
            if 'pi' in bundle:
                pi = np.asarray(bundle['pi'], dtype=np.int64)
            elif 'Q' in bundle:
                pi = greedy_policy(np.asarray(bundle['Q']))
            else:
                raise RuntimeError(f'No pi or Q array found in {path}')
            self._agents.append(_Agent(label=label, pi=pi, color=color, seed=self._seed))
            self.get_logger().info(
                f'compare_node: loaded "{label}" ({pi.size} states) from {path}'
            )

        # I/O.
        self._markers_pub = self.create_publisher(
            MarkerArray,
            gp('markers_topic').get_parameter_value().string_value,
            10,
        )
        self.create_subscription(
            MazeGrid,
            gp('maze_topic').get_parameter_value().string_value,
            self._on_maze,
            _latched_sub_qos(),
        )
        self.create_subscription(
            Empty,
            gp('reset_topic').get_parameter_value().string_value,
            self._on_reset,
            10,
        )
        self._maze: Maze | None = None
        self._timer = self.create_timer(self._dt, self._tick)

    # --------------------------------------------------------------- callbacks
    def _on_maze(self, msg: MazeGrid) -> None:
        if self._maze is not None:
            return
        cells = np.asarray(msg.cells, dtype=np.int8).reshape(int(msg.rows), int(msg.cols))
        goals = np.argwhere(cells == GOAL)
        if goals.size == 0:
            self.get_logger().warn('MazeGrid has no goal cell; waiting.')
            return
        goal = (int(goals[0, 0]), int(goals[0, 1]))
        layout = cells.copy()
        layout[layout == GOAL] = 0
        layout[goal] = GOAL
        self._maze = Maze(cells=layout, goal=goal)
        self._sample_shared_start()
        self._reset_agents(reason='maze loaded')

    def _on_reset(self, _msg: Empty) -> None:
        if self._maze is None:
            self.get_logger().warn('Reset before maze loaded; ignoring.')
            return
        self._sample_shared_start()
        self._reset_agents(reason='external reset')

    # ----------------------------------------------------------------- start
    def _sample_shared_start(self) -> None:
        """Pick the start cell that all agents will use for this episode."""
        assert self._maze is not None
        if self._configured_start is not None:
            self._current_start = self._configured_start
            return
        rng = np.random.default_rng(self._seed)
        free = [cell for cell in self._maze.free_cells if cell != self._maze.goal]
        idx = int(rng.integers(len(free)))
        r, c = free[idx]
        h = int(rng.integers(4))
        self._current_start = (r, c, h)
        # Roll the seed forward so successive random resets diverge.
        self._seed = (self._seed + 1) & 0x7FFFFFFF

    def _reset_agents(self, *, reason: str) -> None:
        assert self._maze is not None and self._current_start is not None
        sr, sc, sh = self._current_start
        for i, agent in enumerate(self._agents):
            # Each agent gets its own RNG seeded identically, so the slip
            # noise sequence is reproducible per agent — agents only diverge
            # once their policies pick different actions.
            agent.env = GridMaze(
                maze=self._maze,
                config=MDPConfig(slip_prob=self._slip),
                rng=np.random.default_rng(self._seed + 991 * i),
                start=(sr, sc, sh),
            )
            agent.env.reset()
            agent.done = False
            agent.steps = 0
            x, y = self._cell_xy(sr, sc)
            agent.trail = [(x, y)]
        self._restart_at = None
        self.get_logger().info(
            f'compare_node: reset ({reason}) start={self._current_start}, '
            f'agents={[a.label for a in self._agents]}'
        )
        self._publish_markers()

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        if self._maze is None:
            return
        # Auto-restart: when every agent has finished, wait the configured
        # delay and then re-sample the shared start cell + reset all agents.
        if self._auto_restart > 0.0 and self._agents and all(a.done for a in self._agents):
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._restart_at is None:
                self._restart_at = now + self._auto_restart
            elif now >= self._restart_at:
                self._sample_shared_start()
                self._reset_agents(reason='auto-restart')
            return
        any_progress = False
        for agent in self._agents:
            if agent.env is None or agent.done:
                continue
            r, c, h = agent.env.mdp.decode(agent.env.state)
            s = (r * self._maze.cols + c) * 4 + h
            action = int(agent.pi[s])
            _, _, done, info = agent.env.step(action)
            agent.steps += 1
            nr, nc, _ = agent.env.mdp.decode(agent.env.state)
            if (nr, nc) != (r, c):
                x, y = self._cell_xy(nr, nc)
                agent.trail.append((x, y))
                if len(agent.trail) > self._trail_max:
                    agent.trail = agent.trail[-self._trail_max:]
            if done:
                agent.done = True
                kind = 'goal' if info.get('terminated') else 'truncated'
                self.get_logger().info(
                    f'compare_node: "{agent.label}" finished ({kind}) in {agent.steps} steps.'
                )
            any_progress = True
        if any_progress:
            self._publish_markers()

    # --------------------------------------------------------------- markers
    def _publish_markers(self) -> None:
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        # Reserve disjoint id ranges per agent so markers don't collide.
        # Per agent: 3 car cubes + 1 trail line strip + 1 label = 5.
        per_agent = 8
        for i, agent in enumerate(self._agents):
            if agent.env is None:
                continue
            base = i * per_agent
            r, c, h = agent.env.mdp.decode(agent.env.state)
            x, y = self._cell_xy(r, c)
            # Per-agent in-cell offset so cars sharing a cell don't fully
            # overlap. Offsets are arranged on a small triangle inside the
            # cell; z-stagger keeps the heatmap floor below the cars.
            ox, oy = self._agent_offset(i, len(self._agents))
            x += ox
            y += oy
            z_lift = i * (self._cell * 0.01)
            yaw = _HEADING_YAW.get(int(h), 0.0)
            color = ColorRGBA(r=agent.color[0], g=agent.color[1], b=agent.color[2], a=1.0)
            dark = ColorRGBA(r=agent.dark[0], g=agent.dark[1], b=agent.dark[2], a=1.0)
            front = ColorRGBA(r=1.0, g=0.95, b=0.30, a=1.0)
            car = build_car_marker_array(
                frame_id=self._frame,
                stamp=stamp,
                namespace=f'agent_{i}',
                base_id=base,
                x=x,
                y=y,
                yaw=yaw,
                cell_size=self._cell,
                body_color=color,
                cabin_color=dark,
                front_color=front,
            )
            for m in car.markers:
                m.pose.position.z += z_lift
            arr.markers.extend(car.markers)

            # Trail.
            line = Marker()
            line.header.frame_id = self._frame
            line.header.stamp = stamp
            line.ns = f'agent_{i}_trail'
            line.id = base + 5
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = self._cell * 0.08
            line.color = ColorRGBA(
                r=agent.color[0], g=agent.color[1], b=agent.color[2], a=0.85
            )
            line.pose.orientation.w = 1.0
            for tx, ty in agent.trail:
                p = Point()
                p.x = float(tx + ox)
                p.y = float(ty + oy)
                p.z = float(self._cell * 0.02 + z_lift)
                line.points.append(p)
            arr.markers.append(line)

            # Label hovering above the car.
            text = Marker()
            text.header.frame_id = self._frame
            text.header.stamp = stamp
            text.ns = f'agent_{i}_label'
            text.id = base + 6
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = self._cell * 0.55 + z_lift
            text.pose.orientation.w = 1.0
            # Stack labels vertically so they don't overlap when cars share
            # a cell. ``i * 0.32 * cell`` lifts agent 1 above agent 0, etc.
            text.pose.position.z += i * self._cell * 0.32
            text.scale.z = self._cell * 0.30
            suffix = '' if not agent.done else ' [done]'
            text.text = f'{agent.label}{suffix}'
            text.color = ColorRGBA(
                r=agent.color[0], g=agent.color[1], b=agent.color[2], a=1.0
            )
            arr.markers.append(text)

        if arr.markers:
            self._markers_pub.publish(arr)

    # ----------------------------------------------------------------- utils
    def _cell_xy(self, row: int, col: int) -> tuple[float, float]:
        return float(col) * self._cell, -float(row) * self._cell

    def _agent_offset(self, i: int, n: int) -> tuple[float, float]:
        """Small in-cell (dx, dy) offset so cars sharing a cell stay visible.

        Agents are placed on a circle of radius ``0.18 * cell`` around the
        cell centre, evenly spaced. With ``n == 1`` the offset collapses to
        the centre.
        """
        if n <= 1:
            return 0.0, 0.0
        radius = 0.18 * self._cell
        angle = 2.0 * math.pi * i / n
        return radius * math.cos(angle), radius * math.sin(angle)


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp compare_node``."""
    rclpy.init(args=args)
    node = CompareNode()
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
