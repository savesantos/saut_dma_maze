"""
Map ArUco / AprilTag detections to discrete maze cells.

Pure-Python helpers used by the ROS ``fiducial_localizer`` node. Kept ROS-free
so the geometry can be unit-tested without ``rclpy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Heading enum mirror (avoid pulling MDP imports for a simple int enum).
_N, _E, _S, _W = 0, 1, 2, 3


@dataclass(frozen=True)
class MarkerSpec:
    """One row of the marker map: marker id -> cell + (optional) reference yaw."""

    marker_id: int
    row: int
    col: int
    # Marker yaw (radians, in the maze frame) used to disambiguate heading.
    # If ``None``, heading is taken from the detection directly.
    reference_yaw_rad: float | None = None


@dataclass(frozen=True)
class CellPoseEstimate:
    """Discrete cell estimate produced by the localizer."""

    row: int
    col: int
    heading: int
    confidence: float


def yaw_to_heading(yaw_rad: float) -> int:
    """
    Quantise a continuous yaw (radians, 0 = +x = East) to a 4-cardinal heading.

    Convention matches :class:`maze_mdp.mdp.Heading`: N=0, E=1, S=2, W=3.
    """
    # Wrap to [0, 2π) and split into 4 90°-wide bins centred on each cardinal.
    two_pi = 2.0 * np.pi
    yaw = float(yaw_rad) % two_pi
    # East bin: [-π/4, π/4); shift by +π/4 then divide.
    idx = int(np.floor(((yaw + np.pi / 4.0) % two_pi) / (np.pi / 2.0)))
    # idx: 0=E, 1=N (CCW), 2=W, 3=S. Remap to N=0, E=1, S=2, W=3.
    remap = {0: _E, 1: _N, 2: _W, 3: _S}
    return remap[idx]


def detection_to_cell(
    marker_id: int,
    detected_yaw_rad: float,
    marker_map: dict[int, MarkerSpec],
    confidence: float = 1.0,
) -> CellPoseEstimate | None:
    """
    Convert one marker detection to a :class:`CellPoseEstimate`.

    Returns ``None`` if the marker id is not in the map. The robot's heading
    is taken to be the detected yaw quantised to N/E/S/W; the reference yaw
    in the marker spec is reserved for future angular-correction logic.
    """
    spec = marker_map.get(int(marker_id))
    if spec is None:
        return None
    heading = yaw_to_heading(detected_yaw_rad)
    return CellPoseEstimate(
        row=spec.row,
        col=spec.col,
        heading=heading,
        confidence=float(confidence),
    )


def load_marker_map(entries: list[dict]) -> dict[int, MarkerSpec]:
    """Parse a list of YAML-loaded dicts into a ``{id: MarkerSpec}`` mapping."""
    out: dict[int, MarkerSpec] = {}
    for e in entries:
        mid = int(e['id'])
        if mid in out:
            raise ValueError(f'duplicate marker id {mid} in marker map')
        out[mid] = MarkerSpec(
            marker_id=mid,
            row=int(e['row']),
            col=int(e['col']),
            reference_yaw_rad=(
                float(e['reference_yaw_rad'])
                if e.get('reference_yaw_rad') is not None
                else None
            ),
        )
    return out
