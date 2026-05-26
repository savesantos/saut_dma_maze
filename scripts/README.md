# Run scripts — step-by-step

Helpers for running the maze stack on the lab PC + AlphaBot2.
Run everything from the repository root unless stated otherwise.

## TL;DR

| Step | Where | Command |
| --- | --- | --- |
| 1. Build + sync policy to robot | lab PC | `bash scripts/setup_lab_pc_and_sync_to_alphabot.sh --robot-host deec@10.16.140.<id>` |
| 2. Build robot workspace | AlphaBot2 (SSH) | `bash ~/alphabot2_ws/scripts/setup_alphabot.sh` |
| 3. Bring up robot drivers | AlphaBot2 (SSH) | `bash ~/alphabot2_ws/scripts/run_alphabot_stack.sh --domain-id <id>` |
| 4. Launch policy from PC | lab PC | `bash scripts/run_pc_7x7_rooms.sh --domain-id <id>` |

`<id>` is the last octet of the AlphaBot's IP (50–70). It MUST match between PC and robot.

---

## 1. Lab PC — build, train (if missing), and sync to the robot

`scripts/setup_lab_pc_and_sync_to_alphabot.sh`

- Sources ROS 2 Humble.
- Runs `colcon build --symlink-install`.
- Looks for a trained policy under `data/training/<algo>/<maze>/*-seed<seed>/policy.npz`; trains one if missing.
- `rsync`s the policy, maze YAML, params, marker map and robot-side scripts to `<ROBOT_WS>/shared/` and `<ROBOT_WS>/scripts/`.

```bash
bash scripts/setup_lab_pc_and_sync_to_alphabot.sh \
  --robot-host deec@10.16.140.<id> \
  --algo vi --seed 0 --maze fixture_7x7_rooms
```

Default `--robot-ws` is `~/alphabot2_ws`. Add `--skip-build` if the workspace is already built. Add `--run-rosdep` on first run.

---

## 2. AlphaBot2 — build / update the robot workspace

SSH into the robot first:

```bash
ssh deec@10.16.140.<id>
bash ~/alphabot2_ws/scripts/setup_alphabot.sh
```

What it does: `git pull --ff-only` in `~/alphabot2_ws` (unless `--skip-git-pull`), optional `rosdep install` (`--run-rosdep`), then `colcon build --symlink-install`.

Override the workspace path with `--workspace /path/to/ws` if needed.

---

## 3. AlphaBot2 — bring up the robot drivers

Still on the robot (same SSH or a second session):

```bash
bash ~/alphabot2_ws/scripts/run_alphabot_stack.sh --domain-id <id>
```

This launches:

1. `ros2 launch alphabot2 alphabot2_launch.py` — base + camera (publishes `/image/compressed`, `/alphabot2/cmd_vel`).
2. `ros2 run alphabot2 motion_driver` — wheel command bridge.
3. `ros2 run maze_mdp ir_driver_hardware` — TRSensors → `/line_pose`, `/intersection`, `/line_lost`.

Keep this terminal open until the run is done. `Ctrl-C` kills all three processes.

The `--domain-id` must equal the IP's last octet so the laptop can see these topics.

---

## 4. Lab PC — launch the policy + perception stack

In a separate terminal on the lab PC:

```bash
bash scripts/run_pc_7x7_rooms.sh --domain-id <id>
```

This launches `maze_bringup alphabot_maze.launch.py` with `ir_driver_backend:=external` (the IR node is already running on the robot from step 3). The launch starts on the PC:

- `maze_publisher` — latched `/maze`.
- `fiducial_localizer` — ArUco/AprilTag on `/image/compressed` → `/goal_marker_seen` (final-approach trigger).
- `yaw_estimator` — phase-correlation yaw delta from `/image/compressed` → `/yaw_delta` (camera-based, motor-calibration agnostic; closes the turn loop without depending on per-robot motor health).
- `action_executor` — discrete-action FSM, line-follow PID → `/alphabot2/cmd_vel`.
- `cell_tracker` — discrete-cell estimate from action results.
- `policy_runner` — reads the trained `policy.npz`, publishes `/action_goal`, exits on goal.

Common overrides:

```bash
bash scripts/run_pc_7x7_rooms.sh \
  --algo qlearning --seed 0 \
  --maze fixture_7x7_rooms \
  --start-row 0 --start-col 0 --start-heading 1 \
  --domain-id <id>
```

When `policy_runner` reaches the goal, the launch shuts the whole graph down.

---

## Where the artefacts live

| Artefact | Path |
| --- | --- |
| Trained policies | `data/training/<algo>/<maze>/<timestamp>-seed<n>/policy.npz` |
| Maze fixtures | `src/maze_bringup/config/mazes/<name>.yaml` |
| Marker maps | `src/maze_bringup/config/markers/<name>.yaml` |
| Shared params | `src/maze_bringup/config/params.yaml` |
| Deployment recordings | `data/deployment/<maze>/<run_id>/` (when `record_bag:=true`) |

---

## Troubleshooting

- **`ros2 topic list` is empty on the lab PC**: `ROS_DOMAIN_ID` mismatch. Both ends must use the same `--domain-id`.
- **Turn never completes / robot keeps spinning**: `/yaw_delta` not flowing. Check `ros2 topic hz /yaw_delta` on the PC; if silent, verify the camera stream `ros2 topic hz /image/compressed`. The executor falls back to commanded-yaw integration after `yaw_measurement_stale_s = 0.3` s of silence.
- **Robot stalls on bumps / paper seams**: bump `forward_speed` in [src/maze_bringup/launch/alphabot_maze.launch.py](../src/maze_bringup/launch/alphabot_maze.launch.py) (default 0.18 m/s).
- **No `/line_pose`**: the IR driver did not start on the robot, or the TRSensor calibration sweep is still in progress (~6 s on startup). Watch the `ir_driver_hardware` log.
- **`policy.npz` not found**: pass `--algo`/`--maze`/`--seed` matching an existing run, or let the scripts train one (the first run takes longer).

---

## Other helpers

- `scripts/make_all_figures.sh` — regenerate all report figures from `data/training/`.
- `scripts/rerun_archive.sh` — replay an archived deployment rosbag.

---

## Test camera-yaw turn guidance in Gazebo (no robot)

The simulated AlphaBot2 URDF has a forward-facing camera with the same HFOV (62.2°) as the real Pi Camera v2, so the exact same `yaw_estimator` pipeline runs in Gazebo. This is the fastest way to sanity-check the camera-based turn closure before going to the lab.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Train a small policy if you don't have one yet:
ros2 run maze_mdp train --algo vi --maze fixture_3x3 --seed 0 --out data

POLICY=$(ls -td data/training/vi/fixture_3x3/*-seed0/policy.npz | head -n 1)

ros2 launch maze_bringup gazebo_maze.launch.py \
  maze_name:=fixture_3x3 \
  policy_path:=$PWD/$POLICY \
  headless:=false
```

Verify the camera-yaw signal is flowing in a second shell:

```bash
ros2 topic hz /yaw_delta            # ~30 Hz once Gazebo is up
ros2 topic echo /yaw_delta --once   # near 0 when still, spikes during turns
```

What changed for Gazebo: `turn_target_yaw_rad` is back to π/2 (geometric) — the previous 1.96 rad bias-correction (compensating for `gazebo_ros_diff_drive`'s 80 % omega tracking) is no longer needed because the executor now uses the measured camera yaw. If `/yaw_delta` stalls for > 0.3 s, the executor falls back to commanded-yaw integration automatically.

### Tuning `gain` for the camera mount

The bare `width / HFOV` pixel-to-radian conversion is exact only when the optical axis is horizontal. The simulated AlphaBot2 URDF (and likely the real one) mounts the camera pitched ~45° down so it sees the line and floor markers. That reduces horizontal pixel shift per radian of yaw, so we set `gain ≈ 1 / cos(pitch)`:

- Forward camera (pitch = 0°): `gain = 1.0`
- 45° down (default URDF): `gain ≈ 1.414` (already set in [gazebo_maze.launch.py](../src/maze_bringup/launch/gazebo_maze.launch.py))
- 30° down: `gain ≈ 1.155`

This is a **per-mount** property, identical across all robots with the same chassis — still motor-calibration agnostic.

If turns are still off, measure the actual ratio with the calibrator (Gazebo only — uses commanded yaw as ground truth):

```bash
# Spawn the world with the calibrator instead of the policy stack:
ros2 launch maze_bringup gazebo_maze.launch.py \
  maze_name:=fixture_3x3 policy_path:=$PWD/$POLICY headless:=false &
# Kill action_executor and policy_runner so they don't fight cmd_vel:
ros2 lifecycle set /action_executor shutdown 2>/dev/null || \
  pkill -f 'maze_mdp action_executor'
ros2 run maze_mdp yaw_calibrator
# Reads back: "recommended gain = commanded/measured = X.XXX"
```

Set that value as the `gain` parameter in the launch file and re-run.
