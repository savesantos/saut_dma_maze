# AGENTS.md

ROS 2 Humble (Python 3.10, Ubuntu 22.04) colcon workspace implementing MDP-based and model-free RL maze solving (Value Iteration, SARSA, Q-Learning) for the **AlphaBot2 with camera**. Course project DMA, IST *Autonomous Systems* 2025/26.

## Landmines & Boundaries

✅ Always:
- Edit Python sources under `src/maze_mdp/maze_mdp/` and re-run without rebuilding (we use `--symlink-install`).
- Add unit tests under `src/<pkg>/test/test_*.py` and run `colcon test`.
- Develop and validate algorithms in the pure-Python micro-simulator first; test against synthetic mazes.
- Replay recorded `rosbag` data instead of running live on the robot whenever possible.

⚠️ Ask first:
- Adding a new `console_scripts` entry, dependency in any `package.xml`, or new `.msg`/`.srv` (requires `colcon build` and possibly cross-package rebuilds).
- Editing `src/maze_msgs/CMakeLists.txt` or `src/maze_msgs/package.xml` (touches the rosidl interface graph; consumers must declare `<depend>maze_msgs</depend>`).
- Deleting or rewriting tutorial-prescribed file layout (`docs/tutorial.md` § 1.1).
- Running anything against the physical AlphaBot2 (must be coordinated with the team and `ROS_DOMAIN_ID`).

🚫 Never:
- Import `rclpy` or any ROS package from algorithm modules (`mdp.py`, `simulator.py`, `value_iteration.py`, `sarsa.py`, `qlearning.py`). They must stay runnable with plain `python3` so the micro-simulator has zero ROS coupling. Put ROS code under `maze_mdp/nodes/` only.
- Put any logic in `maze_bringup`. It contains launch files and YAML configs only — no Python nodes, no algorithms.
- Assume the AlphaBot2 has wheel odometry, IMU, or LiDAR. It does **not**. Only the camera (`/image/compressed`, `sensor_msgs/CompressedImage`) and `/virtual_odometry` (`nav_msgs/Odometry`) are available; cmd topic is `/alphabot2/cmd_vel` (namespaced, **not** `/cmd_vel`).
- Commit `build/`, `install/`, `log/`, or `*.egg-info/` (already in `.gitignore`).
- Skip `--symlink-install` on `colcon build` — losing it breaks the inner dev loop.
- Hardcode `ROS_DOMAIN_ID`; it must be set per shell from the AlphaBot's IP last octet (50–70).

## Commands

Run from the workspace root. **Source both** in every new shell:

```bash
source /opt/ros/humble/setup.bash      # ROS 2 underlay
source install/setup.bash              # this workspace overlay (after first build)
```

| Task                        | Command                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| First-time deps             | `rosdep install -i --from-path src --rosdistro humble -y`            |
| Build all                   | `colcon build --symlink-install`                                     |
| Build one package           | `colcon build --symlink-install --packages-select maze_mdp`          |
| Test one package            | `colcon test --packages-select maze_mdp && colcon test-result --verbose` |
| Lint only (pure Python)     | `cd src/maze_mdp && python3 -m pytest test/test_flake8.py test/test_pep257.py` |
| Run a node                  | `ros2 run maze_mdp <executable>`                                     |
| Launch full stack           | `ros2 launch maze_bringup alphabot_maze.launch.py`                   |
| Pure-Python algo iteration  | `cd src/maze_mdp && python3 -m pytest`                               |
| Talk to the robot           | `export ROS_DOMAIN_ID=<50..70>` (per shell)                          |

Rebuild is **only** needed after editing: `setup.py` entry points, `package.xml`, `CMakeLists.txt`, or any `.msg`/`.srv`. Plain `.py` edits do not require it.

## Testing

- Framework: `pytest` driven by `colcon test`. Lint tests (`test_flake8.py`, `test_pep257.py`, `test_copyright.py`) ship with each package and **must pass** — do not delete them.
- TDD expectation per workplan: write tests for `mdp.py` and Value Iteration before SARSA/Q-Learning so RL results have a deterministic reference.
- For algorithm-only iteration, prefer `python3 -m pytest` inside `src/maze_mdp/` (skips colcon overhead). Run the full `colcon test` before pushing.
- Evaluation runs should consume `rosbag` recordings rather than live robot data — multiple runs per maze, multiple mazes, statistical comparison Value Iteration vs. SARSA vs. Q-Learning.

## Data & Plots for the Report

This is a graded course project: **every training and deployment run must persist its data** so results are reproducible and figures can be regenerated for the IEEE report.

- Save artifacts under `data/` (gitignored, except small reference fixtures): `data/training/<algo>/<maze>/<run_id>/` and `data/deployment/<maze>/<run_id>/`.
- Training artifacts (per run): the learned policy / Q-table / value function (`.npz` or `.npy`), the maze + MDP config used, hyperparameters (`params.yaml`), per-episode metrics (`metrics.csv`: episode, return, steps, epsilon, td_error), and a final `summary.json` (wall-clock, seed, convergence iter).
- Deployment artifacts (per run): `rosbag` of the run, executed trajectory (cell sequence + timestamps), success flag, steps-to-goal, collisions/replans.
- Always set and log a `seed`; runs without a recorded seed are not acceptable for the comparative analysis.
- **Essential plots only** (do not over-produce — the report has 6 pages):
  1. Convergence curve per algorithm (return vs. episode, mean ± std over runs).
  2. Final policy / value-function heatmap on each maze.
  3. Comparative bar/box plot: steps-to-goal and success rate, Value Iteration vs. SARSA vs. Q-Learning, across mazes.
- Plotting code lives in `src/maze_mdp/maze_mdp/analysis/` (ROS-free, matplotlib only) and writes PNG + PDF to `data/figures/`. One script per plot; no notebooks committed.

## Code Style

Enforced by `ament_flake8` and `ament_pep257`. Follow the pattern below.

✅ Prefer — keep algorithms ROS-free and import them from a node:

```python
# src/maze_mdp/maze_mdp/value_iteration.py  (no ROS imports)
import numpy as np
def value_iteration(mdp, gamma=0.95, tol=1e-6):
    ...
```

```python
# src/maze_mdp/maze_mdp/nodes/policy_runner.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from maze_mdp.value_iteration import value_iteration
```

❌ Avoid — coupling algorithms to ROS forces a colcon build for every tweak and breaks the micro-simulator:

```python
# DON'T do this in mdp.py / simulator.py / value_iteration.py / sarsa.py / qlearning.py
import rclpy            # 🚫
from rclpy.node import Node
```

Other rules:
- Snake_case modules and functions; PascalCase classes; module-level docstrings (pep257).
- License header: keep the `Apache-2.0` declaration consistent with each `package.xml`.
- New executables must be registered under `entry_points['console_scripts']` in `setup.py` — adding a file alone is not enough.

## AlphaBot2 Integration Notes

- Subscribes: `/alphabot2/cmd_vel` (`geometry_msgs/Twist`).
- Publishes: `/image/compressed` (`sensor_msgs/CompressedImage`), `/virtual_odometry` (`nav_msgs/Odometry`).
- Goal identification + cell-level localization is via **fiducial markers** (ArUco / AprilTag) on the camera stream — there is no laser scan to fall back on.
- **Localization & mapping scope (professor, 2026-05):** any ROS 2 method is acceptable (fiducial markers, `robot_localization`, `slam_toolbox`, `nav2` AMCL, etc.). It is not a graded contribution of this project — pick whatever yields a reliable discrete cell estimate for the MDP and move on. Do not over-invest engineering effort here.
- Bring-up on hardware (per `Lab_guide_2526.pdf`): SSH `deec@10.16.140.<id>`, then `ros2 launch alphabot2 alphabot2_launch.py` and in another SSH `ros2 run alphabot2 motion_driver`. On the laptop set `ROS_DOMAIN_ID=<id>` before `ros2 topic list`.

## Documentation Standards

All documentation lives in `docs/` (`docs/tutorial.md` is the canonical onboarding guide; `README.md` is the only project-level doc allowed at the repo root alongside `AGENTS.md`, `CLAUDE.md`, `LICENSE` files in each package).

- ATX-style headers (`#`), one sentence per line.
- Relative links between docs (e.g., `[tutorial](docs/tutorial.md)`).
- KaTeX math (`$...$` inline, `$$...$$` block) — already used in `README.md`.
- Mermaid diagrams: neutral theme with accessible colors.

## Additional Guidance

- Package-by-package onboarding: see [docs/tutorial.md](docs/tutorial.md).
- Course materials and constraints: `Lab_guide_2526.pdf`, `Projects_2526.pdf`, `Apresentação 1 SAut.pdf` at repo root.
- Deliverables: 6-page IEEE report + code by **5 Jun 2026**; weekly progress presentations.
