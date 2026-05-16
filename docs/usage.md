# Usage Guide

End-to-end recipes for the four ways to run this project: pure-Python tests, training, RViz micro-simulator, Gazebo, and hardware.
All commands run from the workspace root (`/home/.../saut_dma_maze`).

## 0. One-time setup

```bash
# System packages
sudo apt install python3-colcon-common-extensions python3-rosdep
source /opt/ros/humble/setup.bash

# ROS dependencies (first time only)
sudo rosdep init && rosdep update          # global, once per machine
rosdep install -i --from-path src --rosdistro humble -y

# Python plotting dependencies (matplotlib, pandas, etc.)
pip install -r requirements.txt

# Build
colcon build --symlink-install
source install/setup.bash
```

Every new shell needs both sources:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

`--symlink-install` lets plain `.py` edits reload without a rebuild.
Rebuild only after editing `setup.py` entry points, `package.xml`, `CMakeLists.txt`, or `.msg` / `.srv` files.

## 1. Pure-Python tests (no ROS)

The algorithms in `src/maze_mdp/maze_mdp/` are ROS-free.
Fastest iteration loop:

```bash
cd src/maze_mdp && python3 -m pytest -q
```

98 tests covering the MDP, Value Iteration, SARSA, Q-Learning, the action FSM, the line-follow PID, and the IR/marker geometry estimator.
Use this loop while editing algorithm code; the colcon test runner is only needed before pushing.

## 2. Train policies

### 2.1 Full report sweep (3 algos × 3 mazes × 5 seeds)

```bash
ros2 run maze_mdp sweep --config \
  $(ros2 pkg prefix maze_bringup)/share/maze_bringup/config/sweeps/default.yaml
```

Writes 45 runs under `data/training/<algo>/<maze>/<run_id>/`, each with `policy.npz`, `params.yaml`, `metrics.csv` (RL only) and `summary.json`.
Run IDs are timestamped, so re-running keeps older runs intact.

### 2.2 Single run (debugging or a missing seed)

```bash
ros2 run maze_mdp train \
  --algo vi --maze fixture_5x5_corridor --seed 0 --out data
```

`--algo` ∈ `vi | sarsa | qlearning`.
`--maze` ∈ `fixture_3x3 | fixture_5x5_corridor | fixture_7x7_loop`.
Override `--episodes`, `--gamma`, `--vi-tol` as needed.

### 2.3 Pick best seed for the heatmap

```bash
python3 -m maze_mdp.analysis.select_best_run
```

Writes `data/training/<algo>/<maze>/selected.json` consumed by the policy-heatmap figure.

### 2.4 Generate every report figure

```bash
bash scripts/make_all_figures.sh
```

Outputs PNG + PDF to `data/figures/`: `convergence`, `steps_curve`, `policy_heatmap`, `comparison`.
Each script is also runnable standalone (`python3 -m maze_mdp.analysis.plot_convergence`, etc.).

## 3. Replay in the RViz micro-simulator

The micro-simulator integrates `cmd_vel` into discrete MDP steps — no physics — and renders the maze, policy and value function in RViz2.

```bash
ALGO=vi
MAZE=fixture_5x5_corridor
SEED=0
POLICY=$PWD/$(ls -td data/training/$ALGO/$MAZE/*-seed$SEED | head -1)/policy.npz

ros2 launch maze_bringup sim_viz.launch.py \
  maze_name:=$MAZE policy_path:=$POLICY seed:=42
```

`seed:=42` is the **simulator** RNG (slip noise + start cell), independent of the training seed `SEED`.

RViz shows: maze walls, goal, robot arrow, virtual-odometry trail, accumulated cell trail, and a value-function heatmap.
Enable the **PolicyArrows** display in the side panel for per-`(cell, heading)` policy arrows.

### Reset and pick a start cell

```bash
# Re-roll start cell on the live launch
ros2 topic pub --once /sim_reset std_msgs/msg/Empty '{}'

# Fixed start at launch time (heading: 0=N, 1=E, 2=S, 3=W)
ros2 launch maze_bringup sim_viz.launch.py \
  maze_name:=$MAZE policy_path:=$POLICY \
  start_row:=0 start_col:=0 start_heading:=1

# Jump to a one-off cell without changing the default
ros2 topic pub --once /sim_reset_to maze_msgs/msg/CellPose \
  '{row: 4, col: 0, heading: 0}'
```

### Compare all three algorithms side by side

```bash
VI=$PWD/$(ls -td data/training/vi/$MAZE/*-seed$SEED        | head -1)/policy.npz
SA=$PWD/$(ls -td data/training/sarsa/$MAZE/*-seed$SEED     | head -1)/policy.npz
QL=$PWD/$(ls -td data/training/qlearning/$MAZE/*-seed$SEED | head -1)/policy.npz

ros2 launch maze_bringup compare_viz.launch.py \
  maze_name:=$MAZE \
  vi_policy:=$VI sarsa_policy:=$SA qlearning_policy:=$QL \
  start_row:=0 start_col:=0 start_heading:=1 seed:=42
```

Three differently-coloured "cars" run on independent stochastic copies of the same maze from the same start cell.

### Headless variant (CI, scripted runs)

```bash
ros2 launch maze_bringup sim.launch.py \
  maze_name:=$MAZE policy_path:=$POLICY seed:=42
```

Same micro-sim, no RViz, exits on goal.

### WSLg troubleshooting

If the RViz 3D viewport is fully black under WSL2, the launch file forces the Mesa software rasterizer for `rviz2` only.
On native Linux with a working GPU, opt out before launching:

```bash
export RVIZ_FORCE_SOFTWARE_GL=0
```

## 4. Run in Gazebo (full physics + line-following stack)

The Gazebo launch spawns the AlphaBot2 in a maze world generated from the same YAML used by the micro-sim, runs the analytic `ir_driver_gazebo` (no ground-texture sampling — see [control.md](control.md)), and pipes the closed-loop FSM through `action_executor`.

```bash
ros2 launch maze_bringup gazebo_maze.launch.py \
  maze_name:=fixture_7x7_loop \
  policy_path:=$PWD/data/training/vi/fixture_7x7_loop/<run_id>/policy.npz \
  headless:=false
```

Set `headless:=true` to skip the Gazebo client GUI.
The launch tears down automatically when `policy_runner` reaches the goal.

Tuning knobs (`line_p_gain`, `turn_target_yaw_rad`, `pivot_creep_s`, …) are documented in [control.md](control.md) §6.

## 5. Deploy on the AlphaBot2

Full protocol in [deployment.md](deployment.md).
Bring-up sketch:

```bash
# On the robot (SSH)
ssh deec@10.16.140.<id>
ros2 launch alphabot2 alphabot2_launch.py   # shell A
ros2 run alphabot2 motion_driver            # shell B

# On the laptop
export ROS_DOMAIN_ID=<id>                   # last octet of the robot IP
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 launch maze_bringup alphabot_maze.launch.py \
  maze_name:=hw_5x5 \
  policy_path:=$PWD/data/training/qlearning/hw_5x5/<run_id>/policy.npz \
  record_bag:=true
```

`ROS_DOMAIN_ID` must match the robot's IP last octet (50–70); never hardcode it.
The same FSM + line-follower runs as in Gazebo; only the `ir_driver` source changes (TRSensors `readLine()` instead of analytic geometry) and the goal detector becomes `fiducial_localizer` (ArUco / AprilTag).

## 6. Inner development loop

```bash
# edit code under src/maze_mdp/maze_mdp/... — no rebuild needed

# fast: pure-Python tests
cd src/maze_mdp && python3 -m pytest -q

# integration: colcon test
cd ../.. && colcon test --packages-select maze_mdp \
         && colcon test-result --verbose

# only if you edited setup.py / package.xml / CMakeLists.txt / .msg / .srv
colcon build --symlink-install --packages-select maze_mdp
source install/setup.bash
```

Linters (`ament_flake8`, `ament_pep257`, `ament_copyright`) ship with each package and run inside `colcon test`.
Do not delete them.
