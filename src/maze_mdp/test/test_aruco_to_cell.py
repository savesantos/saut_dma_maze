"""Unit tests for the ArUco -> cell mapper (ROS-free)."""

import math

from maze_mdp.perception.aruco_to_cell import (
    MarkerSpec,
    detection_to_cell,
    load_marker_map,
    yaw_to_heading,
)


def test_yaw_to_heading_cardinals():
    # Heading enum: N=0, E=1, S=2, W=3.
    assert yaw_to_heading(0.0) == 1            # East
    assert yaw_to_heading(math.pi / 2) == 0    # North
    assert yaw_to_heading(math.pi) == 3        # West
    assert yaw_to_heading(-math.pi / 2) == 2   # South


def test_detection_unknown_marker_returns_none():
    out = detection_to_cell(99, 0.0, {}, confidence=1.0)
    assert out is None


def test_detection_to_cell_basic():
    mmap = {7: MarkerSpec(7, row=2, col=3, reference_yaw_rad=None)}
    out = detection_to_cell(7, 0.0, mmap, confidence=0.9)
    assert out is not None
    assert (out.row, out.col, out.heading) == (2, 3, 1)
    assert out.confidence == 0.9


def test_load_marker_map_rejects_duplicates():
    entries = [{'id': 1, 'row': 0, 'col': 0}, {'id': 1, 'row': 1, 'col': 1}]
    try:
        load_marker_map(entries)
    except ValueError:
        return
    raise AssertionError('expected ValueError on duplicate marker id')
