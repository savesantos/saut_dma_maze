"""
Procedural Gazebo SDF generator for the maze fixtures.

Reads a maze YAML (``layout`` of ``.``/``#`` strings + ``goal: [r, c]``) and
writes an SDF ``.world`` file that matches the *visual* topology of the maze:
a white floor with black tape lines connecting walkable cells and a small red
goal marker on the goal cell.

Usage::

    python3 -m maze_mdp.analysis.maze_to_sdf <maze_yaml> <out_world>
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class MazeSpec:
    """Parsed maze fixture."""

    rows: int
    cols: int
    walkable: List[List[bool]]
    goal: Tuple[int, int]

    @classmethod
    def from_yaml(cls, path: Path) -> "MazeSpec":
        data = yaml.safe_load(path.read_text())
        layout = data["layout"]
        walkable = [[ch == "." for ch in row] for row in layout]
        goal = tuple(data["goal"])
        return cls(rows=len(layout), cols=len(layout[0]), walkable=walkable, goal=goal)


def cell_center(row: int, col: int, cell_size: float) -> Tuple[float, float]:
    """World (x, y) of cell centre (row grows toward -y, col toward +x)."""
    return col * cell_size, -row * cell_size


def _line_segments(maze: MazeSpec, cell_size: float) -> List[Tuple[float, float, float, float]]:
    """
    Return the list of black-tape segments as ``(cx, cy, length, yaw)``.

    Each segment is the straight strip joining two adjacent walkable cells.
    """
    segs: List[Tuple[float, float, float, float]] = []
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.walkable[r][c]:
                continue
            # Horizontal segment to right neighbour.
            if c + 1 < maze.cols and maze.walkable[r][c + 1]:
                cx, cy = cell_center(r, c, cell_size)
                nx, _ = cell_center(r, c + 1, cell_size)
                segs.append(((cx + nx) / 2.0, cy, cell_size, 0.0))
            # Vertical segment to bottom neighbour (row+1 -> -y direction).
            if r + 1 < maze.rows and maze.walkable[r + 1][c]:
                cx, cy = cell_center(r, c, cell_size)
                _, ny = cell_center(r + 1, c, cell_size)
                segs.append((cx, (cy + ny) / 2.0, cell_size, math.pi / 2))
    return segs


def _segment_link_xml(idx: int, cx: float, cy: float, length: float, yaw: float,
                      line_width: float, line_height: float) -> str:
    return (
        f'    <link name="line_{idx}">\n'
        f'      <pose>{cx:.4f} {cy:.4f} 0.0005 0 0 {yaw:.4f}</pose>\n'
        f'      <visual name="v">\n'
        f'        <geometry><box><size>{length:.4f} {line_width:.4f} {line_height:.4f}</size></box></geometry>\n'
        f'        <material>\n'
        f'          <ambient>0 0 0 1</ambient>\n'
        f'          <diffuse>0 0 0 1</diffuse>\n'
        f'          <script>\n'
        f'            <uri>file://media/materials/scripts/gazebo.material</uri>\n'
        f'            <name>Gazebo/Black</name>\n'
        f'          </script>\n'
        f'        </material>\n'
        f'      </visual>\n'
        f'    </link>\n'
    )


def _goal_marker_xml(cx: float, cy: float, cell_size: float) -> str:
    side = cell_size * 0.5
    return (
        f'    <link name="goal_marker">\n'
        f'      <pose>{cx:.4f} {cy:.4f} 0.0010 0 0 0</pose>\n'
        f'      <visual name="v">\n'
        f'        <geometry><box><size>{side:.4f} {side:.4f} 0.0010</size></box></geometry>\n'
        f'        <material>\n'
        f'          <ambient>0.8 0.1 0.1 1</ambient>\n'
        f'          <diffuse>0.8 0.1 0.1 1</diffuse>\n'
        f'          <script>\n'
        f'            <uri>file://media/materials/scripts/gazebo.material</uri>\n'
        f'            <name>Gazebo/Red</name>\n'
        f'          </script>\n'
        f'      </material>\n'
        f'      </visual>\n'
        f'    </link>\n'
    )


def maze_to_sdf(maze: MazeSpec, cell_size: float = 0.20,
                line_width: float = 0.015, line_height: float = 0.001) -> str:
    """Render a complete SDF world string for ``maze``."""
    segs = _line_segments(maze, cell_size)
    seg_xml = "".join(_segment_link_xml(i, *seg, line_width, line_height)
                      for i, seg in enumerate(segs))

    gx, gy = cell_center(*maze.goal, cell_size)
    goal_xml = _goal_marker_xml(gx, gy, cell_size)

    # Floor sized to fit the maze with a margin.
    floor = max(maze.cols, maze.rows) * cell_size + 0.4

    # Camera: top-down view centred on the maze.  Pitch ~ +pi/2 looks straight
    # down; yaw aligns image X with world +x and image up with world +y.
    cx_mid = (maze.cols - 1) * cell_size / 2.0
    cy_mid = -(maze.rows - 1) * cell_size / 2.0
    cam_x = cx_mid
    cam_y = cy_mid
    cam_z = max(maze.cols, maze.rows) * cell_size * 1.6 + 0.3
    cam_pitch = 1.5707  # straight down
    cam_yaw = 1.5707    # rotate so +x in world points right on screen

    return (
        f'<?xml version="1.0" ?>\n'
        f'<sdf version="1.6">\n'
        f'  <world name="maze">\n'
        f'    <gui fullscreen="0">\n'
        f'      <camera name="user_camera">\n'
        f'        <pose>{cam_x:.4f} {cam_y:.4f} {cam_z:.4f} 0 {cam_pitch:.4f} {cam_yaw:.4f}</pose>\n'
        f'      </camera>\n'
        f'    </gui>\n'
        f'\n'
        f'    <scene>\n'
        f'      <ambient>0.6 0.6 0.6 1</ambient>\n'
        f'      <background>0.7 0.7 0.8 1</background>\n'
        f'      <shadows>false</shadows>\n'
        f'    </scene>\n'
        f'\n'
        f'    <light name="sun" type="directional">\n'
        f'      <cast_shadows>false</cast_shadows>\n'
        f'      <pose>0 0 10 0 0 0</pose>\n'
        f'      <diffuse>0.9 0.9 0.9 1</diffuse>\n'
        f'      <specular>0.2 0.2 0.2 1</specular>\n'
        f'      <direction>-0.3 0.4 -1</direction>\n'
        f'    </light>\n'
        f'\n'
        f'    <model name="ground_plane">\n'
        f'      <static>true</static>\n'
        f'      <link name="link">\n'
        f'        <collision name="c">\n'
        f'          <geometry><plane><normal>0 0 1</normal><size>{floor:.2f} {floor:.2f}</size></plane></geometry>\n'
        f'          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>\n'
        f'        </collision>\n'
        f'        <visual name="v">\n'
        f'          <cast_shadows>false</cast_shadows>\n'
        f'          <geometry><plane><normal>0 0 1</normal><size>{floor:.2f} {floor:.2f}</size></plane></geometry>\n'
        f'          <material>\n'
        f'            <ambient>1 1 1 1</ambient>\n'
        f'            <diffuse>1 1 1 1</diffuse>\n'
        f'            <script>\n'
        f'              <uri>file://media/materials/scripts/gazebo.material</uri>\n'
        f'              <name>Gazebo/White</name>\n'
        f'            </script>\n'
        f'          </material>\n'
        f'        </visual>\n'
        f'      </link>\n'
        f'    </model>\n'
        f'\n'
        f'    <model name="maze_lines">\n'
        f'      <static>true</static>\n'
        f'{seg_xml}'
        f'{goal_xml}'
        f'    </model>\n'
        f'\n'
        f'    <physics type="ode">\n'
        f'      <real_time_update_rate>1000</real_time_update_rate>\n'
        f'      <max_step_size>0.001</max_step_size>\n'
        f'    </physics>\n'
        f'  </world>\n'
        f'</sdf>\n'
    )


def main() -> None:
    """Entry point: ``python3 -m maze_mdp.analysis.maze_to_sdf <yaml> <world>``."""
    p = argparse.ArgumentParser(description="Generate a Gazebo SDF world for a maze fixture.")
    p.add_argument("maze_yaml", type=Path)
    p.add_argument("out_world", type=Path)
    p.add_argument("--cell-size", type=float, default=0.20,
                   help="Cell side length in metres (default: 0.20).")
    p.add_argument("--line-width", type=float, default=0.015,
                   help="Black-tape line width (default: 0.015 m).")
    args = p.parse_args()

    maze = MazeSpec.from_yaml(args.maze_yaml)
    sdf = maze_to_sdf(maze, cell_size=args.cell_size, line_width=args.line_width)
    args.out_world.parent.mkdir(parents=True, exist_ok=True)
    args.out_world.write_text(sdf)
    print(f"Wrote {args.out_world} ({maze.rows}x{maze.cols}, goal={maze.goal})")


if __name__ == "__main__":
    main()
