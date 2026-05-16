"""Unit tests for the procedural maze->SDF generator."""

from pathlib import Path

import pytest

from maze_mdp.analysis.maze_to_sdf import (
    MazeSpec,
    _line_segments,
    cell_center,
    maze_to_sdf,
)


def _open_3x3() -> MazeSpec:
    return MazeSpec(
        rows=3, cols=3,
        walkable=[[True] * 3 for _ in range(3)],
        goal=(2, 2),
    )


def _corridor_5x5() -> MazeSpec:
    layout = [
        '.....',
        '.###.',
        '.....',
        '.###.',
        '.....',
    ]
    walkable = [[ch == '.' for ch in row] for row in layout]
    return MazeSpec(rows=5, cols=5, walkable=walkable, goal=(4, 4))


def test_cell_center_convention():
    # x = col*cs, y = -row*cs.
    assert cell_center(0, 0, 0.2) == (0.0, 0.0)
    assert cell_center(1, 2, 0.2) == pytest.approx((0.4, -0.2))


def test_open_3x3_segment_count():
    # Each walkable cell hosts a full cross (1 horizontal + 1 vertical).
    # 9 walkable cells -> 18 segments.
    assert len(_line_segments(_open_3x3(), 0.2)) == 18


def test_walls_break_segments():
    maze = _corridor_5x5()
    segs = _line_segments(maze, 0.2)
    # Each walkable cell contributes exactly 2 segments (cross arms).
    # Walkable count: rows 0,2,4 have 5 each; rows 1,3 have 2 each
    # (cols 0 and 4) -> 5*3 + 2*2 = 19 walkable cells -> 38 segments.
    walkable_count = sum(sum(row) for row in maze.walkable)
    assert walkable_count == 19
    assert len(segs) == 2 * walkable_count == 38


def test_sdf_render_contains_expected_blocks():
    sdf = maze_to_sdf(_open_3x3(), cell_size=0.2)
    assert '<sdf version="1.6">' in sdf
    assert 'goal_marker' in sdf
    # 18 line links (one horizontal + one vertical per walkable cell).
    assert sdf.count('<link name="line_') == 18


def test_real_fixture_files_parse(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    for name in ('fixture_3x3.yaml',
                 'fixture_5x5_corridor.yaml',
                 'fixture_7x7_loop.yaml'):
        path = repo / 'maze_bringup' / 'config' / 'mazes' / name
        if not path.exists():
            # Skip if the fixture is not installed in this checkout.
            continue
        maze = MazeSpec.from_yaml(path)
        sdf = maze_to_sdf(maze)
        assert '<world name="maze">' in sdf
        assert sdf.count('<link name="line_') > 0
