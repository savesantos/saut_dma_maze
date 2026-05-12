# Visualization quickstart

End-to-end recipe to wipe the workspace, build, train all three algorithms
and replay a policy in RViz2. Run every command from the workspace root
(`/home/salva/saut_dma_maze`).

## 1. Clean rebuild

Wipes the colcon outputs and builds the three packages from scratch.

```bash
cd /home/salva/saut_dma_maze
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

After the first build, only re-source `install/setup.bash` in new shells —
plain `.py` edits do not need a rebuild thanks to `--symlink-install`.

## 2. Train all production policies

The canonical training set for the report is defined in
[src/maze_bringup/config/sweeps/default.yaml](../src/maze_bringup/config/sweeps/default.yaml):
3 algorithms (`vi`, `sarsa`, `qlearning`) × 3 mazes (`fixture_3x3`,
`fixture_5x5_corridor`, `fixture_7x7_loop`) × 5 seeds (`0..4`) with
`gamma=0.99`, `slip_prob=0.1` and per-maze episode budgets sized as
`k * |S| * |A| / (1 - gamma)` (see
[docs/mdp_design.md](mdp_design.md) §7). Reproduce it in one command:

```bash
ros2 run maze_mdp sweep --config \
  $(ros2 pkg prefix maze_bringup)/share/maze_bringup/config/sweeps/default.yaml
```

This writes 45 runs to `data/training/<algo>/<maze>/<run_id>/`, each with
`policy.npz`, `params.yaml`, `metrics.csv` (RL only) and `summary.json`.
Re-running is idempotent in the sense that each run gets a fresh timestamped
`run_id`, so older runs are kept.

### Single-run trainer (debugging / re-runs)

`ros2 run maze_mdp train` exposes the same backend with one
`(algo, maze, seed)` triple. Use it to reproduce a specific failing run or
to add a seed without re-running the full sweep:

```bash
ros2 run maze_mdp train --algo vi        --maze fixture_5x5_corridor --seed 0 --out data
ros2 run maze_mdp train --algo sarsa     --maze fixture_5x5_corridor --seed 0 --out data
ros2 run maze_mdp train --algo qlearning --maze fixture_5x5_corridor --seed 0 --out data
```

Available mazes: `fixture_3x3`, `fixture_5x5_corridor`, `fixture_7x7_loop`.
Override episode count per run with `--episodes N`, discount with
`--gamma 0.99`, VI tolerance with `--vi-tol 1e-6`.

### Custom sweep

Copy [default.yaml](../src/maze_bringup/config/sweeps/default.yaml), edit
seeds / hyperparameters, and point `--config` at your file. The sweep
respects `td_config_overrides[<maze>]` so you can keep small mazes cheap
while giving the 7×7 loop its 30k episodes.

## 3. Replay a policy in RViz2

`sim_viz.launch.py` brings up `maze_publisher`, `maze_sim_node`,
`maze_viz_node`, `policy_runner`, and `rviz2` with the bundled config.
`policy_path` must be an **absolute path** to a `.npz` produced in step 2.

```bash
ALGO=vi
MAZE=fixture_5x5_corridor
SEED=0
POLICY=$PWD/$(ls -td data/training/$ALGO/$MAZE/*-seed$SEED | head -1)/policy.npz

ros2 launch maze_bringup sim_viz.launch.py \
  maze_name:=$MAZE \
  policy_path:=$POLICY \
  seed:=42
```

Swap `ALGO=sarsa` or `ALGO=qlearning` to compare. Switch `MAZE` to any
fixture; the launch resolves the matching YAML in
`maze_bringup/config/mazes/`. `seed:=42` is the **simulator** RNG seed
(slip noise + start cell), independent of the training seed `SEED`.

RViz displays:

- **MazeGrid** — walls (black) and free cells (white).
- **Goal** — green sphere on the goal cell.
- **RobotCell** — blue arrow at the discrete MDP pose from `/robot_cell`.
- **VirtualOdometry** — red arrow trail of the continuous pose.
- **Trail** — accumulated cell-to-cell path.
- **ValueHeatmap** — purple→yellow tint per cell using `max_h V(s)`.
- **PolicyArrows** — disabled by default; tick it in the **Displays** panel
  to render one small arrow per `(cell, heading)` colour-coded by action.

Unlike `sim.launch.py`, this launch stays alive after `policy_runner`
reaches the goal so you can inspect the heatmap, trail and final pose.
Use Ctrl-C to exit when done.

### Restart the run

The viz launch keeps `maze_sim_node` and `policy_runner` alive after the
goal so you can re-run the same policy from a fresh start cell without
re-launching the stack. From any sourced shell:

```bash
ros2 topic pub --once /sim_reset std_msgs/msg/Empty '{}'
```

`maze_sim_node` re-samples a start cell (or uses the configured
`start_row` / `start_col` / `start_heading` if set), republishes
`/robot_cell`, and `policy_runner` resumes from `WAITING_CELL` →
`RUNNING`. The trail in RViz keeps growing across resets — clear it
with the **Reset** button in the RViz toolbar if you want a clean
canvas.

This works under `sim_viz.launch.py` and any custom launch that does
not set `policy_runner`'s `exit_on_goal:=true` parameter (the headless
`sim.launch.py` sets it to `True`, so it tears down on goal as before).

### Choose the initial cell

Pick a fixed start cell at launch time with `start_row`, `start_col`
and `start_heading` (heading: `0`=N, `1`=E, `2`=S, `3`=W). Defaults are
`-1 / -1 / 1`, which means *random free cell, heading East*:

```bash
ros2 launch maze_bringup sim_viz.launch.py \
  maze_name:=$MAZE \
  policy_path:=$POLICY \
  start_row:=0 start_col:=0 start_heading:=1
```

To switch start cell while the launch is already running, set the
parameters on the live node and request a reset:

```bash
ros2 param set /maze_sim_node start_row 2
ros2 param set /maze_sim_node start_col 0
ros2 param set /maze_sim_node start_heading 1
ros2 topic pub --once /sim_reset std_msgs/msg/Empty '{}'
```

To jump to a one-off cell *without* persisting it as the new default
start, publish a `CellPose` directly on `/sim_reset_to`:

```bash
ros2 topic pub --once /sim_reset_to maze_msgs/msg/CellPose \
  '{row: 4, col: 0, heading: 0}'
```

Out-of-bounds cells, walls, and the goal cell are rejected with a
warning in `maze_sim_node`'s log; nothing changes in that case.

### Troubleshooting: black RViz viewport on WSLg

Under WSL2 / WSLg, `glxinfo | grep "OpenGL renderer"` reports a `D3D12`
string (Mesa-on-DirectX). The OGRE renderer used by RViz Humble does not
render correctly on that path and the 3D viewport stays fully black even
though every display shows `Status: Ok` and the topics carry valid data.

[src/maze_bringup/launch/sim_viz.launch.py](../src/maze_bringup/launch/sim_viz.launch.py)
works around this by forcing the Mesa software rasterizer (`llvmpipe`)
for the `rviz2` process only:

```python
additional_env={
    'LIBGL_ALWAYS_SOFTWARE':
        os.environ.get('RVIZ_FORCE_SOFTWARE_GL', '1'),
    'GALLIUM_DRIVER': 'llvmpipe',
},
```

The rest of the graph (sim, viz bridge, policy runner) is unaffected.
Frame rate drops to a CPU-bound ~10–30 fps but the maze, walls, goal,
robot arrow, trail and value heatmap render correctly.

On a native Linux machine with a working GPU driver, opt out before
launching:

```bash
export RVIZ_FORCE_SOFTWARE_GL=0
ros2 launch maze_bringup sim_viz.launch.py ...
```

A related quirk: RViz Humble silently drops any display whose `Topic:`
block omits `History Policy: Keep Last`, leaving only `Grid` in the
Displays panel. The bundled
[config/rviz/maze.rviz](../src/maze_bringup/config/rviz/maze.rviz)
specifies the field on every display — keep it there if you hand-edit the
file.

## 4. Headless variants

Skip RViz for sweeps and CI. The original launch keeps the same CLI:

```bash
ros2 launch maze_bringup sim.launch.py \
  maze_name:=fixture_5x5_corridor \
  policy_path:=$POLICY \
  seed:=42
```

Pure-Python micro-simulator (no ROS at all):

```bash
cd src/maze_mdp && python3 -m pytest
```

## 5. Generate report figures

After the sweep populates `data/training/`, regenerate every committed
figure (PNG + PDF) into `data/figures/` with one script:

```bash
./scripts/make_all_figures.sh
```

Equivalent breakdown — run individually if you only want one plot:

```bash
python3 -m maze_mdp.analysis.plot_convergence      # return vs. episode (mean +- std)
python3 -m maze_mdp.analysis.plot_steps_curve      # episode length vs. episode
python3 -m maze_mdp.analysis.plot_policy_heatmap   # final policy / V on each maze
python3 -m maze_mdp.analysis.plot_comparison       # VI vs. SARSA vs. Q-Learning bars
```

These modules live under
[src/maze_mdp/maze_mdp/analysis/](../src/maze_mdp/maze_mdp/analysis/) and
are ROS-free (matplotlib only), so they run from any shell with the venv
sourced — no `install/setup.bash` needed.
