# Hardware Deployment Protocol

How to record reproducible runs of the AlphaBot2 in the physical maze for the
report's comparative analysis.

## 1. Pre-flight checklist

- Battery > 75 %.
- Camera lens clean; lighting in the lab approximately matches the calibration
  photos (no direct sunlight, no flicker).
- Maze laid out matching the layout in
  [src/maze_bringup/config/mazes/](../src/maze_bringup/config/mazes/).
- ArUco markers affixed at the cells listed in
  [src/maze_bringup/config/markers/](../src/maze_bringup/config/markers/).
- Trained policy file exists (`data/training/<algo>/<maze>/<run_id>/policy.npz`).

## 2. Robot bring-up (per `Lab_guide_2526.pdf`)

```bash
ssh deec@10.16.140.<id>
ros2 launch alphabot2 alphabot2_launch.py     # in shell A
ros2 run alphabot2 motion_driver              # in shell B
```

## 3. Laptop side

```bash
export ROS_DOMAIN_ID=<id>     # last octet of the AlphaBot2 IP
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch maze_bringup alphabot_maze.launch.py \
  maze_name:=hw_5x5 \
  policy_path:=$(pwd)/data/training/qlearning/hw_5x5/<run_id>/policy.npz \
  record_bag:=true
```

## 4. Recording protocol

Each evaluation run records:

- A rosbag with `/image/compressed`, `/robot_cell`, `/alphabot2/cmd_vel`,
  `/maze`, written under `data/deployment/<maze>/<run_id>/bag/`.
- `summary.json` with `success` (bool), `steps` (int), `wall_clock_s`,
  `algo`, `maze`, `seed_of_policy`, `notes`.
- `trajectory.csv` extracted post-hoc by
  `python3 -m maze_mdp.experiments.deployment` (TODO).

The matrix is **3 algos × 3 runs** on a single hardware maze, but the schema
allows scaling without code changes.

## 5. Failure handling

- Robot loses localization (no `/robot_cell` for > 2 s) → stop, mark
  `success: false`, log `failure_mode: lost_localization` in `summary.json`.
- Robot bumps a wall hard → press the e-stop, mark `failure_mode: collision`.
- Goal not reached within `max_wall_clock_s` (default 120 s) →
  `failure_mode: timeout`.
