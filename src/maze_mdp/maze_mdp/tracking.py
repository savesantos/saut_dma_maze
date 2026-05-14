"""
ROS-free dead-reckoning of the robot's discrete maze pose.

The :class:`CellTracker` maintains the current ``(row, col, heading)`` cell
pose by composing successful discrete actions reported by the executor.
Failed actions are assumed to leave the pose unchanged: the executor stops
the robot before the geometric effect of the action has taken place
(line lost, timeout, collision, aborted).

This module is intentionally ROS-free so it can be unit-tested with plain
pytest and reused by the Gazebo and hardware stacks identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from maze_mdp.mdp import Action, Heading, _HEADING_DELTA

# Matches maze_msgs/DiscreteActionGoal.DRIVE_UNTIL_MARKER. Treated as a
# forward step by the tracker: the executor only reports success after
# physically crossing into the goal cell.
_DRIVE_UNTIL_MARKER: int = 3


@dataclass(frozen=True)
class CellPose:
    """Discrete pose: cell coordinates plus cardinal heading."""

    row: int
    col: int
    heading: int

    def __post_init__(self) -> None:
        if self.heading not in (0, 1, 2, 3):
            raise ValueError(
                f'heading must be 0..3 (N/E/S/W), got {self.heading}'
            )


class CellTracker:
    """
    Dead-reckon discrete pose from a stream of action outcomes.

    Parameters
    ----------
    rows, cols:
        Maze dimensions in cells. Used to clamp forward motion at the grid
        boundary so an unexpected boundary hit does not put the tracker into
        an out-of-grid state. Walls inside the grid are *not* enforced here:
        the executor reports success only when an action physically completed,
        so a successful FORWARD into a walled neighbour is, by construction,
        impossible.
    initial_pose:
        Starting ``CellPose``. Required (the user must declare it).

    """

    def __init__(self, rows: int, cols: int, initial_pose: CellPose) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError('rows and cols must be positive')
        if not 0 <= initial_pose.row < rows:
            raise ValueError(
                f'initial row {initial_pose.row} out of [0, {rows})'
            )
        if not 0 <= initial_pose.col < cols:
            raise ValueError(
                f'initial col {initial_pose.col} out of [0, {cols})'
            )
        self._rows = int(rows)
        self._cols = int(cols)
        self._pose = initial_pose

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def pose(self) -> CellPose:
        return self._pose

    def reset(self, pose: CellPose) -> None:
        """Re-localize the tracker (e.g. on ``SetStartPose``)."""
        if not 0 <= pose.row < self._rows:
            raise ValueError(f'row {pose.row} out of [0, {self._rows})')
        if not 0 <= pose.col < self._cols:
            raise ValueError(f'col {pose.col} out of [0, {self._cols})')
        self._pose = pose

    def apply(self, action: int, success: bool) -> CellPose:
        """
        Compose ``action`` onto the current pose and return the new pose.

        Failed actions leave the pose unchanged. Successful FORWARD that
        would leave the grid is clamped (pose unchanged); this is a safety
        net — under correct executor behaviour it should not happen.
        """
        if not success:
            return self._pose

        r, c, h = self._pose.row, self._pose.col, self._pose.heading
        a = int(action)
        if a == int(Action.FORWARD) or a == _DRIVE_UNTIL_MARKER:
            dr, dc = _HEADING_DELTA[h]
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < self._rows and 0 <= nc < self._cols:
                self._pose = CellPose(nr, nc, h)
        elif a == int(Action.TURN_LEFT):
            self._pose = CellPose(r, c, (h - 1) % 4)
        elif a == int(Action.TURN_RIGHT):
            self._pose = CellPose(r, c, (h + 1) % 4)
        else:
            raise ValueError(f'unknown action {a}')
        return self._pose


__all__ = ['CellPose', 'CellTracker', 'Action', 'Heading']
