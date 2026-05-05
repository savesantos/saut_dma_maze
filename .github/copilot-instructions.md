# GitHub Copilot Instructions

ROS 2 Humble (Python 3.10, Ubuntu 22.04) colcon workspace for autonomous maze solving on the **AlphaBot2 with camera** using MDPs and model-free RL (Value Iteration, SARSA, Q-Learning). Course project DMA, IST *Autonomous Systems* 2025/26.

See [AGENTS.md](../AGENTS.md) for full operational rules. The points below are the highest-signal items for Copilot completions.

## Architecture rules (do not violate)

- **Algorithm modules stay ROS-free.** Files under `src/maze_mdp/maze_mdp/` named `mdp.py`, `simulator.py`, `value_iteration.py`, `sarsa.py`, `qlearning.py` must NOT import `rclpy`, `geometry_msgs`, `nav_msgs`, etc. They run with plain `python3` for the micro-simulator.
- **ROS code lives only in `src/maze_mdp/maze_mdp/nodes/`** as classes extending `rclpy.node.Node`. Import algorithms from there:
  ```python
  from maze_mdp.value_iteration import value_iteration
  ```
- **`maze_bringup` is launch + YAML only.** No Python nodes, no algorithm code.
- **`maze_msgs` is ament_cmake** (custom `.msg`/`.srv` only). Consumers must add `<depend>maze_msgs</depend>` in their `package.xml`.

## AlphaBot2 facts (do not infer otherwise)

- No wheel odometry, no IMU, no LiDAR. **Camera + fiducial markers (ArUco / AprilTag) only.**
- **Localization & mapping:** any ROS 2 method is allowed (per professor, 2026-05) — fiducials, `robot_localization`, `slam_toolbox`, `nav2` AMCL, etc. It is not the project's contribution; the MDP only needs a discrete cell estimate.
- Subscribes: `/alphabot2/cmd_vel` (`geometry_msgs/Twist`) — namespaced, **not** `/cmd_vel`.
- Publishes: `/image/compressed` (`sensor_msgs/CompressedImage`), `/virtual_odometry` (`nav_msgs/Odometry`).
- `ROS_DOMAIN_ID` is set per shell from the robot IP last octet (50–70). Never hardcode it.

## Build / test workflow

- Always `colcon build --symlink-install` (lets Python edits reload without rebuilding).
- Rebuild **only** after editing: `setup.py` `entry_points`, `package.xml`, `CMakeLists.txt`, `.msg`, `.srv`.
- Source both in every shell: `source /opt/ros/humble/setup.bash && source install/setup.bash`.
- Tests: `colcon test --packages-select <pkg> && colcon test-result --verbose`. Lint tests (`test_flake8.py`, `test_pep257.py`, `test_copyright.py`) ship with each package and must pass — do not delete them.
- Pure-Python iteration on algorithms: `cd src/maze_mdp && python3 -m pytest`.

## Code style

- Enforced by `ament_flake8` + `ament_pep257`. Module-level docstrings required.
- snake_case modules and functions, PascalCase classes.
- License: `Apache-2.0`, matching each `package.xml`.
- New executables MUST be registered in `setup.py`:
  ```python
  entry_points={'console_scripts': ['policy_runner = maze_mdp.nodes.policy_runner:main']},
  ```
- Prefer NumPy for MDP transition tables; avoid pure-Python loops in `value_iteration` hot paths.

## Evaluation

- Use recorded `rosbag` data, not live robot runs, for repeatability.
- Comparative analysis Value Iteration vs. SARSA vs. Q-Learning across multiple mazes and multiple runs per maze.
- **Persist every run** (course project — results go into the IEEE report). Layout:
  - `data/training/<algo>/<maze>/<run_id>/` → policy/Q-table (`.npz`), `params.yaml`, `metrics.csv` (episode, return, steps, epsilon, td_error), `summary.json` (seed, wall-clock, convergence iter).
  - `data/deployment/<maze>/<run_id>/` → `rosbag`, trajectory CSV, success flag, steps-to-goal.
  - Always record the RNG `seed`. `data/` is gitignored.
- **Essential plots only** (do not over-produce):
  1. Convergence curve per algorithm (return vs. episode, mean ± std over seeds).
  2. Final policy / value-function heatmap per maze.
  3. Comparative steps-to-goal & success rate, VI vs. SARSA vs. Q-Learning, across mazes.
- Plotting code: `src/maze_mdp/maze_mdp/analysis/` (ROS-free, matplotlib). Output PNG + PDF to `data/figures/`. No committed notebooks.

## Documentation

- All docs live under `docs/` (only `README.md`, `AGENTS.md`, `CLAUDE.md`, `LICENSE` allowed at repo root).
- ATX headers, one sentence per line, KaTeX for math (`$...$`, `$$...$$`).
