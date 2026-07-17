# nav-stack

A Viam navigation stack that wraps the ROS2 **Nav2** and **slam_toolbox** packages,
so any Viam base can map an environment, localize within it, and navigate to named
locations or arbitrary map points while avoiding obstacles.

This module (`viam-labs:nav-stack`) provides three models:

| Model | API | Purpose |
| --- | --- | --- |
| `viam-labs:nav-stack:slam` | `rdk:service:slam` | Mapping + localization via slam_toolbox. Standard SLAM API (live map, position) + map management. |
| `viam-labs:nav-stack:navigation` | `rdk:service:navigation` | Nav2 navigation via the standard Navigation API (modes, waypoints, paths, obstacles) + rich `DoCommand` (locations, zones, go-to-point, plan-to-label). Built-in slam_toolbox runtime. |
| `viam-labs:nav-stack:navigation-external` | `rdk:service:navigation` | Same as `navigation`, but driven by **any** `rdk:service:slam` instead of the bundled slam_toolbox. Runs its own sensor bridge. |
| `viam-labs:nav-stack:nav-camera` | `rdk:component:camera` | Renders the navigation service's Nav2 costmap + active plan(s), robot pose, footprint and goal as a live camera image. Works with either navigation model / any SLAM backend. |

> **⚠️ Breaking change:** `navigation` and `navigation-external` now use the **`rdk:service:navigation`** API (previously `rdk:service:generic`). Existing configs must change each nav service's `"api"` from `"rdk:service:generic"` to `"rdk:service:navigation"` — the model name is unchanged. The full command surface remains available via `DoCommand`.

## How it works

The module bundles/orchestrates ROS2 and bridges it to your Viam components:

- Reads each Viam lidar -> publishes `/scan_<i>` (per lidar) and a merged `/scan`
  for slam_toolbox.
- Publishes odometry (`/odom`) and the `odom -> base_link` TF from a Viam movement
  sensor; publishes static `base_link -> laser_<i>` TFs from each lidar mount.
- Runs slam_toolbox (mapping or localization) and Nav2 (planner + controller +
  layered costmaps + behavior tree).
- Subscribes Nav2's `/cmd_vel` and drives the Viam base — only while navigating,
  with a watchdog that stops the base if commands go stale.

```mermaid
flowchart LR
  lidar["Viam lidar(s)"] --> bridge
  odom["Viam movement sensor"] --> bridge
  bridge -->|"merged /scan, /odom, TF"| slamtb["slam_toolbox"]
  bridge -->|"/scan_N, /odom, TF"| nav2["Nav2"]
  slamtb -->|"/map, map->odom"| nav2
  nav2 -->|"/cmd_vel"| bridge
  bridge -->|"SetVelocity"| base["Viam base"]
```

## Prerequisites

- A Linux host (arm64 or x86_64) running `viam-server` on **Ubuntu 22.04, 24.04, or 26.04**.
  **Pi 5 recommendation:** Ubuntu **24.04 LTS** (Jazzy). Ubuntu 26.04 (Lyrical) may install `ros-base` but Nav2 / slam_toolbox apt packages are often missing on arm64 until ROS publishes them for that distro.
- On first deploy, `setup.sh` runs automatically (`first_run` in `meta.json`) and will:
  1. Verify the Ubuntu version and pick a matching ROS 2 distro (LTS default):
     - 22.04 → **Humble**
     - 24.04 → **Jazzy** (set `ROS_DISTRO=kilted` for Kilted Kaiju)
     - 26.04 → **Lyrical Luth**
  2. **Install** ROS 2, Nav2, and slam_toolbox via `apt` if they are missing (`AUTO_INSTALL_DEPS=1`, the default).
  3. Create the Python venv and install pip dependencies.
  4. Write `.ros_env` so `run.sh` can source ROS without a manual module env block.

Set `AUTO_INSTALL_DEPS=0` in the module `env` block to only check and fail if system packages are missing.

`ROS_ENV` in the module config is **optional** after `setup.sh` has run; override it when you want a non-default distro (e.g. Kilted on 24.04):

```json
"env": {
  "ROS_DISTRO": "kilted",
  "AUTO_INSTALL_DEPS": "1"
}
```

Manual install (if you prefer to provision the image yourself):

```bash
sudo apt-get install ros-$ROS_DISTRO-ros-base \
                     ros-$ROS_DISTRO-navigation2 \
                     ros-$ROS_DISTRO-nav2-bringup \
                     ros-$ROS_DISTRO-slam-toolbox
```

- A configured Viam **base**, one or more **lidars** (configured as `camera`
  components returning point clouds; a true 2D lidar is ideal, depth cameras work
  via projection), and a **movement sensor** providing velocity for odometry.

If `viam-server` runs as root and DDS shared-memory fails, point
`FASTRTPS_DEFAULT_PROFILES_FILE` at a UDP-only FastDDS profile in the module env.

## Configuration

### SLAM service

```json
{
  "name": "slam",
  "api": "rdk:service:slam",
  "model": "viam-labs:nav-stack:slam",
  "attributes": {
    "base": "my-base",
    "movement_sensor": "odometry",
    "lidars": [
      { "name": "front-lidar", "mount": { "x": 0.2, "y": 0.0, "theta": 0.0 } },
      { "name": "rear-lidar",  "mount": { "x": -0.2, "y": 0.0, "theta": 3.14159 } }
    ],
    "mode": "mapping",
    "maps_dir": "/root/.viam/nav-stack/maps",
    "active_map": "ground-floor"
  }
}
```

A single lidar can be given as `"lidar": "front-lidar"`.

**Tuning via Viam config (no YAML editing required):**

| Attribute | Service | Description |
| --- | --- | --- |
| `mode` | SLAM | `mapping` or `localizing` — selects slam_toolbox node and sets its mode |
| `global_localize_on_start` | SLAM | When `true` in `localizing` mode, run `global_localize` automatically after startup (default `true`) |
| `global_localize_on_start_delay_s` | SLAM | Delay before startup auto-localize (default `4.0`) |
| `global_localize_on_start_options` | SLAM | Optional args merged into startup `global_localize` command; defaults prefer robust boot localization (`full_map: true`, `map_source: live`) |
| `global_localize_on_start_refine` | SLAM | Run a second auto `global_localize` pass after startup (default `true`) |
| `global_localize_on_start_refine_delay_s` | SLAM | Delay before second refine pass (default `8.0`) |
| `global_localize_on_start_refine_max_passes` | SLAM | Max startup refine passes while quality is below target (default `3`) |
| `global_localize_on_start_target_score` | SLAM | Stop refining once score reaches this threshold (default `0.7`) |
| `global_localize_on_start_target_ray_mae_m` | SLAM | Stop refining once ray MAE is at or below this threshold (default `0.4`) |
| `global_localize_on_start_post_apply_refine` | SLAM | Run one delayed post-apply `global_localize` pass (manual-equivalent) after startup (default `true`) |
| `global_localize_on_start_post_apply_refine_delay_s` | SLAM | Delay before post-apply refine pass (default `8.0`) |
| `global_localize_on_start_post_apply_refine_options` | SLAM | Optional args for post-apply refine (default `{ \"map_source\": \"live\" }`) |
| `global_localize_on_start_refine_options` | SLAM | Optional args for refine passes; defaults to local refinement (`full_map: false`, `map_source: live`, `local_yaw_window_deg: 120`, `search_radius_m: 6`) |
| `map_when_still` | SLAM | When `true` (point-cloud lidars only), publish `/scan` once per full stop after dwell, then only if still still after the lidar capture (motion during read aborts). Livox frames densify while stopped. Matcher uses gyro yaw prior with `coarse_search_angle_offset` ≈ ±30°; loop closure stays near stock (`loop_match_minimum_chain_size` 10, `loop_search_maximum_distance` 5 m, fine response ≥ 0.45) to avoid false corridor snaps. Default `false` |
| `map_when_still_dwell_s` | SLAM | Seconds fully stopped before a scan may publish (default `1.0`) |
| `map_when_still_yaw_step_deg` | SLAM | Extra mid-pivot scans every N degrees after dwell (default `0` = full-stop only; set e.g. `15` only if you pause briefly while turning) |
| `map_when_still_max_drift_m` / `_deg` | SLAM | Abort dwell if pose creeps while “still” (defaults `0.03` m / `1.5°`) |
| `wall_yaw_correction` | SLAM | Soft-correct odom yaw from a long side wall in each pause scan (anti-banana). Default `true` when `map_when_still` + point-cloud lidars |
| `wall_yaw_min_length_m` / `wall_yaw_max_step_deg` / `wall_yaw_blend` | SLAM | Wall fit length gate (default `2.0` m), max yaw step per pause (default `2°`), and blend toward the wall (default `0.5`) |
| `mapping_revisit_check` | SLAM | Mapping-time revisit watchdog: periodically scan-match against the live map near the current pose and shift the odom TF when a strong match disagrees, so a revisited corridor links up instead of duplicating. Default `true` when `map_when_still` + point-cloud lidars |
| `mapping_revisit_interval_s` / `_search_radius_m` / `_wide_radius_m` | SLAM | Check interval (default `20` s) and tiered search radii: local first (default `5` m), wider on weak match (default `12` m) |
| `mapping_revisit_min_score` / `_max_ray_mae_m` / `_full_map_min_score` | SLAM | Match quality gates (defaults `0.6` / `0.8` m); full-map fallback needs the stricter `0.75` score since self-similar offices produce convincing wrong corridors |
| `mapping_revisit_min_shift_m` / `_min_shift_deg` / `_max_shift_m` | SLAM | Correct only when the match moved at least `1.0` m / `10°` from the current pose and no more than `10` m (larger = likely false match) |
| `mapping_revisit_slice_verify` | SLAM | Multi-height-slice veto for revisit corrections (3D lidar only). The 2D map holds one z-band silhouette and desk clutter is self-similar in it; this records sparse per-band grids (knee + head height by default) from trusted pause scans and rejects a proposed correction whose pose disagrees with any band that has reference data there. Default `true` |
| `mapping_revisit_slice_bands` / `_slice_min_hit_rate` / `_slice_resolution_m` | SLAM | Extra height bands as `[z_min, z_max]` pairs in meters (default `[[0.15, 0.45], [1.6, 2.4]]`), per-band hit-rate gate (default `0.4`), and grid cell size (default `0.15` m) |
| `mapping_revisit_keyframes` | SLAM | Store a pause keyframe (2D endpoints + height slices + map pose) on every accepted `map_when_still` `/scan` publish, and match against those views when occupancy revisit scores are weak — helps when you stop at different places/angles than the first visit. Default `true` |
| `mapping_revisit_keyframe_min_spacing_m` / `_deg` / `_max` / `_match_tol_m` / `_min_score` | SLAM | Keyframe dedupe spacing (default `0.5` m / `20°`), max stored frames (`250`), NN match tolerance (`0.3` m), and accept threshold (`0.55` hit-rate) |
| `movement_sensor_yaw_deg` | SLAM | Yaw (degrees) of the movement sensor's +x axis relative to robot forward. Wit silk-screen Y forward with reverse +Y accel usually needs `90`; geometric Y-forward with correct-signed +Y needs `-90`. Pick the sign that makes forward drive produce positive robot-X velocity (default `0`) |
| `map_pose_yaw_offset_deg` | SLAM | Added to `GetPosition` yaw only (App arrow vs map). Prefer lidar `mount.theta` — park facing a wall and check status `nearest_return_bearing_deg` / `suggested_mount_theta_deg`. Cosmetics (±45) do not fix ghost walls (default `0`) |
| `heading_sensor_yaw_deg` | SLAM | Same mount-yaw correction for the dedicated `heading_sensor` (default `0`) |
| lidar `mount.pitch`, `mount.roll` | SLAM | Mount tilt in radians (positive pitch = forward axis tilted down). Levels the cloud before z filtering — even a ~2° mast tilt pulls floor returns into the z band at 15–20 m and imprints phantom borders at max range (default `0`) |
| `base_velocity_convention` | SLAM | `viam` (default, Y-forward) or `ros` (X-forward); legacy `mir` accepted as alias for `viam` — maps Nav2 `/cmd_vel` to Viam base `SetVelocity` axes |
| `scan_max_age_s` | SLAM | Safety cutoff for the `/scan` publish path: if the lidar reports a cache age (`get_laser_scan` `age_s`) above this, skip publishing that cycle rather than feed SLAM/Nav2 a stale, misregistered scan (default `2.0`) |
| `slam_toolbox` | SLAM | Common slam_toolbox params (resolution, max_laser_range, etc.) |
| `slam_params` | SLAM | Advanced: any other slam_toolbox ROS param (merged last) |
| `robot_radius`, `max_vel_x`, … | Nav | Top-level Nav2 footprint / velocity limits |
| `min_cmd_vel_x`, `min_cmd_vel_theta` | Nav | Stiction floors (default `0.15` m/s, `0.3` rad/s) applied to both `go_to_*` and Nav2 `/cmd_vel_smoothed` before `SetVelocity`. Raise if the cart hums/thunks but barely moves. Legacy aliases: `simple_min_vel_x` / `simple_min_vel_theta` |
| `nav2` | Nav | Common Nav2 params (goal tolerance, costmap size, etc.) |
| `nav2_params` | Nav | Advanced: nested Nav2 param overrides (merged last) |

Example with slam_toolbox tuning:

```json
{
  "name": "slam",
  "model": "viam-labs:nav-stack:slam",
  "attributes": {
    "base": "my-base",
    "movement_sensor": "odometry",
    "lidars": [{ "name": "front-lidar" }],
    "mode": "localizing",
    "maps_dir": "/root/.viam/nav-stack/maps",
    "active_map": "ground-floor",
    "global_localize_on_start": true,
    "global_localize_on_start_options": {
      "map_source": "live",
      "full_map": true
    },
    "global_localize_on_start_refine": true,
    "global_localize_on_start_refine_delay_s": 8.0,
    "global_localize_on_start_refine_max_passes": 3,
    "global_localize_on_start_target_score": 0.7,
    "global_localize_on_start_target_ray_mae_m": 0.4,
    "global_localize_on_start_post_apply_refine": true,
    "global_localize_on_start_post_apply_refine_delay_s": 8.0,
    "global_localize_on_start_post_apply_refine_options": {
      "map_source": "live"
    },
    "global_localize_on_start_refine_options": {
      "local_yaw_window_deg": 120.0
    },
    "slam_toolbox": {
      "resolution": 0.05,
      "max_laser_range": 25.0,
      "minimum_travel_distance": 0.3,
      "map_update_interval": 1.0
    }
  }
}
```

Startup auto-localize evaluates candidate poses first (`apply: false`) and only
publishes the best pose at the end, so early weak passes do not lock in a bad seed.

**Scan freshness / capture-time stamping.** When the lidar (e.g. `viam-labs:mir-base`)
reports a per-scan cache age (`age_s`) in its `get_laser_scan` output, the bridge
stamps the published scan at its capture time (`read_start - age_s`) instead of read
time. This keeps obstacles and scan-match registered where the robot actually was
when the scan was captured — important on a moving/rotating robot where a cached
scan stamped "now" would smear geometry and drive localization off. Scans older than
`scan_max_age_s` are dropped for the SLAM path. Producers that don't report `age_s`
fall back to read-time stamping (unchanged behavior).

`mode` changes take effect on reconfigure (or via `start_mapping` / `start_localizing` DoCommands).

For a **bare IMU** movement sensor (Wit, etc. with accel + gyro, no wheel pose), `/odom` yaw is integrated from **gyro Z only**. Absolute `orientation` / AHRS yaw from `get_readings()` is not snapped into the odom pose. With `map_when_still`, published TF **XY stays frozen** (IMU accel must not be the slam prior) while gyro yaw still updates for the App arrow; slam_toolbox always consumes that odom→base TF as its match prior — there is no real `use_odometry: false` switch. Defaults set `coarse_search_angle_offset` ≈ ±30° around that gyro prior (not ±180° — that caused false room-orientation ghosts). **Duplicated corridors after driving a loop** are usually failed loop closure (gyro drift) — pause often facing clear walls and prefer smaller circuits. Do **not** loosen `loop_match_minimum_chain_size` / `loop_search_*` aggressively; that trades missed closures for false corridor snaps and warped ghost maps.

**Wall-line yaw correction (anti-banana).** Long straight walls drawn as curves usually mean gyro heading walked off while driving parallel to the wall. With `map_when_still` + point-cloud lidars, `wall_yaw_correction` defaults **on**: each accepted pause scan looks for a long side wall (≥ `wall_yaw_min_length_m`) and soft-corrects `/odom` yaw by at most `wall_yaw_max_step_deg` (blended by `wall_yaw_blend`) so the wall lines up with robot +X. Status field `wall_yaw` reports the last observation. Disable with `"wall_yaw_correction": false` if a cluttered side repeatedly misleads the fit.

For **Viam wheeled bases** (`rdk:builtin:wheeled`) and **MiR250** (`viam-labs:mir-base`), keep the default `"base_velocity_convention": "viam"` so forward Nav2 commands map to Viam `linear.y` (Viam wheeled / MiR expect forward on Y, not X). Use `"ros"` only for bases that drive on `linear.x`. Legacy `"mir"` is accepted and normalized to `"viam"`. Odometry from `viam-labs:mir-base:movement` stays in ROS convention and does not need swapping. Nav-stack stops Nav2 motion with `set_velocity(0)` (not `Base.stop()`), so MiR Manualcontrol and `go_to_location` keep working after a navigation cancel or goal completion.

The MiR250 is **differential drive** — use `"kinematics": "differential"` (the default). Configuring `omni` (or an `Omni` MPPI motion model) makes Nav2 command lateral velocities the robot cannot execute, which stalls progress near goals and triggers endless spin recoveries. Also avoid `"vx_min": 0`: with reverse disabled, a diff-drive robot must rotate fully around to correct small overshoots.

For **MiR** movement sensors (`viam-labs:mir-base:movement`), the bridge reads a single `get_readings()` per odom tick. It uses **`odom_position_x_m` / `odom_position_y_m` / `odom_yaw_deg`** when present (true `/odom` frame from mir-base ≥ the odom-fields update). Map-frame `position_x_m`/`position_y_m` and fused `yaw_deg` are **not** used for `/odom` — slam_toolbox needs a smooth odom frame. Until mir-base exposes the odom fields, orientation falls back to velocity integration; upgrade mir-base or patch it to publish `odom_*` keys from the parsed `/odom` message. Raise `mir_rosbridge_timeout_s` (≥5) and `odom_rate_hz` (≥15) if updates lag.

### Navigation service

`viam-labs:nav-stack:navigation` implements the standard `rdk:service:navigation` API (so any Navigation-API client — e.g. a webapp — can drive it) plus a rich `DoCommand` surface. It runs Nav2 against the built-in SLAM model's shared runtime (`slam_service` names a `viam-labs:nav-stack:slam` service).

```json
{
  "name": "nav",
  "api": "rdk:service:navigation",
  "model": "viam-labs:nav-stack:navigation",
  "attributes": {
    "slam_service": "slam",
    "base": "my-base",
    "kinematics": "differential",
    "robot_radius": 0.22,
    "max_vel_x": 0.4,
    "max_vel_theta": 1.0,
    "inflation_radius": 0.45,
    "nav2": {
      "xy_goal_tolerance": 0.25,
      "local_costmap_width": 4.0,
      "cost_scaling_factor": 3.0,
      "replan_frequency": 2.0,
      "progress_movement_time_allowance": 10.0,
      "navigate_recovery_retries": 4,
      "recovery_wait_duration": 2.0
    }
  }
}
```

`nav2.replan_frequency` (default **2 Hz**, up from Nav2's stock 1 Hz) rewrites the
navigate-to-pose behavior tree so global plans refresh more often.
`progress_movement_time_allowance` (default **10 s**, down from 30) and
`navigate_recovery_retries` / `recovery_wait_duration` exit reverse/spin recovery
loops sooner. After changing these, run `restart_nav2` (or reconfigure) so the
generated BT + params reload.

Set `"kinematics": "omni"` and a non-zero `max_vel_y` for omnidirectional bases. The files under [`params/`](params/) are **reference defaults**; runtime params are generated from your service attributes. See [Navigation API + DoCommand](#navigation-api--docommand) for the full method/command surface (shared with `navigation-external`).

### Navigation with an external SLAM service

Use `viam-labs:nav-stack:navigation-external` to drive Nav2 from **any** `rdk:service:slam` (for example a third-party RTAB-Map module), instead of the bundled `slam_toolbox`. Same `rdk:service:navigation` API + DoCommand surface as `navigation`; the difference is that this model runs its **own** sensor bridge and bridges the external SLAM's pose + occupancy grid into ROS:

- `slam_service` names an `rdk:service:slam` dependency. The adapter calls its standard `GetPosition()` (→ `map → odom` TF) and a `get_grid` DoCommand returning `{rows, cols, xMin, yMin, cellSize, data}` with int8 cells (`-1`/`0`/`100`) (→ `/map` OccupancyGrid). No point-cloud rasterization.
- Because the external SLAM does not publish `/scan` or `/odom`, you configure the sensor bridge here too — `lidars` (Viam `camera` components, projected to `/scan`) and `movement_sensor`. It reuses the same bridge + odometry fusion as the SLAM model.
- The movement sensor is read via the **typed** `MovementSensor` API (`GetProperties()` then `AngularVelocity` / `LinearAcceleration` / `Orientation` / `LinearVelocity`), not `GetReadings()`, so any movement sensor works. An IMU-only sensor (e.g. a Livox Mid-360's IMU) auto-selects yaw-from-gyro + translation-from-lidar-odometry; its dead-reckoned `Position` is ignored unless you set `trust_movement_sensor_pose: true`.

```json
{
  "name": "nav",
  "api": "rdk:service:navigation",
  "model": "viam-labs:nav-stack:navigation-external",
  "attributes": {
    "slam_service": "rtabmap",
    "base": "my-base",
    "kinematics": "differential",
    "lidars": [{ "name": "mid360", "scan_source": "point_cloud" }],
    "movement_sensor": "mid360-imu",
    "imu_odom_mode": "accel_only",
    "lidar_odom_enabled": true,
    "robot_radius": 0.22,
    "max_vel_x": 0.4,
    "inflation_radius": 0.45
  }
}
```

Optional attributes: `trust_movement_sensor_pose` (default `false`), `snap_heading` (default `false`), plus the same bridge/odometry tuning fields as the SLAM service and the same `nav2` block as `navigation`. The built-in `navigation` model is unchanged; use it when you map with `nav-stack:slam`.

### Navigation API + DoCommand

Both `navigation` and `navigation-external` implement the same `rdk:service:navigation` API + `DoCommand` surface.

**Coordinate overload:** the map is not georeferenced — `GeoPoint.latitude` carries map-frame **x** metres and `GeoPoint.longitude` carries **y** metres. `GetLocation`, waypoints, and `GetObstacles` all use this (matches the convention of compatible Navigation modules, so a webapp built against one drives this backend unchanged).

**Standard API:**
- `GetLocation` → robot pose as a `GeoPoint` (lat=x, lng=y).
- `GetMode` / `SetMode` — `MANUAL` (cancel + stop) or `WAYPOINT` (drive the queue in order). `EXPLORE` is unsupported.
- `AddWaypoint` / `GetWaypoints` / `RemoveWaypoint` — the waypoint queue.
- `GetPaths` — remaining waypoints as the path (true Nav2 `/plan` is a follow-up).
- `GetObstacles` — `no_go`/`slow_down` annotations + local keepout zones as `GeoGeometry` (AABB).
- `GetProperties` → `MAP_TYPE_NONE`.

**DoCommand** (extras beyond the standard API):
- Locations/zones CRUD, `go_to_point`/`go_to_location` (simple closed-loop), `navigate_to_point`/`navigate_to_location` (Nav2), `cancel`, `get_status`, `start_nav2`/`restart_nav2`.
- `plan_to_label` `{label}` — navigate to a `label` annotation.
- `replan` — refresh annotation masks + return the current polyline.
- `clear_waypoints` — empty the queue.
- `get_motors_enabled` / `set_motors_enabled {enabled}` — plan-preview gate: when disabled, Nav2 still plans + publishes `cmd_vel` but the base is **not** driven (robot stays put).
- `get_planner_config` / `set_planner_config {inflation_radius, allow_unknown}` — set regenerates params + restarts Nav2.

### Annotations (no-go / slow-down / labels)

The **SLAM service** hosts annotations (GeoJSON-shaped, per-map, local map-frame metres) via `DoCommand`, mirroring the RTAB-Map module's schema:

| Command | Args | Result |
|---|---|---|
| `get_annotations` | `{}` | `{annotations: FeatureCollection}` |
| `set_annotations` | `{annotations: FeatureCollection}` | `{ids: [...]}` (bulk replace) |
| `add_annotation` | `{annotation: Feature}` | `{id: "<uuid>"}` |
| `update_annotation` | `{annotation: Feature}` | `{updated: bool}` |
| `delete_annotation` | `{id}` | `{deleted: bool}` |

Kinds: `no_go` (Polygon → keepout), `slow_down` (Polygon + `max_speed_m_s` → speed cap), `label` (Point → named goal for `plan_to_label`). The navigation service reads annotations from its SLAM source (built-in: nav-stack:slam's store; external: the SLAM dep's `get_annotations`) and applies `no_go`/`slow_down` to the Nav2 costmap masks **lazily on each plan** (so the latest annotations always apply). `slow_down` `max_speed_m_s` is converted to a percentage of `max_vel_x`.

### Visualizing what nav is planning (nav-camera)

`viam-labs:nav-stack:nav-camera` is a read-only `rdk:component:camera` that renders, as an image you can watch in the Viam app's camera stream, what the navigation service is doing — no rviz required. It draws Nav2's **global costmap** (so you see the inflated cost surface the planner actually reasons over) with these overlays:

- **global plan** (`/plan`) in green — the route to the current goal;
- **plan history** — superseded plans for the current goal, greyed out and faded oldest→faintest, so you can watch how the route changed as the robot replanned (reset on each new goal);
- **local plan** (`/local_plan`) in orange — the controller's short-horizon path;
- **robot pose + footprint** (red arrow + blue polygon) from the `map → base_link` TF;
- **goal marker** (magenta) with a heading tick.

Occupancy colouring: unknown = dark grey, free = light, obstacle inflation = grey→orange gradient, lethal/inscribed = near-black. World "up" renders as image up (rviz-like).

It reads directly from the running navigation service's in-process bridge (found by the `navigation` attribute), so there is no extra ROS process and no round-trip. Because it consumes only Nav2's standard costmap/plan topics, it works with **any** SLAM backend and with either `navigation` or `navigation-external`.

```json
{
  "name": "nav-view",
  "api": "rdk:component:camera",
  "model": "viam-labs:nav-stack:nav-camera",
  "attributes": {
    "navigation": "nav"
  }
}
```

- `navigation` (**required**) — the name of the `navigation` / `navigation-external` service to visualize. It is also declared as a dependency so it starts first.
- Optional: `max_dim` (longest output edge in px, default `700`), `plan_history_len` (faded trail length, default `8`), `robot_radius_m` (footprint fallback + pose-arrow size, default `0.22`), and per-overlay toggles `show_global_plan` / `show_local_plan` / `show_pose` / `show_footprint` / `show_goal` / `show_history` (all default `true`).

**Windowing** — by default the camera renders the whole map. `window_mode` crops/zooms it:
- `"full"` (default) — the entire occupancy grid.
- `"follow"` — a `window_size_m`-metre **square that tracks the robot** (falls back to the goal, then the grid centre, if there's no pose yet). Best for large maps where the whole grid is too zoomed-out to see the plan. `window_size_m` defaults to `6.0`.
- `"region"` — a fixed map-frame bounding box from `window_min_x` / `window_min_y` / `window_max_x` / `window_max_y` (metres). Best for watching one fixed spot (e.g. a doorway). If any bound is missing it falls back to `"full"`.

```json
{ "navigation": "nav", "window_mode": "follow", "window_size_m": 5.0 }
```

Until Nav2 has published a costmap (bringup is asynchronous), the camera returns a placeholder frame.

`DoCommand`:
- `{"command": "legend"}` — the colour key as a printable string in `legend`, so you can read the map without guessing colours.
- `{"command": "stats"}` (or any other command) — a text summary: whether the bridge/costmap is present, plan point counts, current goal/pose. Handy for verifying without a video stream.

## Workflows

### 1. Make a map

1. Configure the SLAM service with `"mode": "mapping"` (or call `start_mapping`).
2. Drive the base around manually (Viam remote control / SDK). The module only
   takes over the base while navigating, so manual driving and mapping don't
   conflict.
3. Save when done: `do_command({"command": "save_map"})`.

### 2. Localize on a saved map

```python
await slam.do_command({"command": "start_localizing", "map": "ground-floor"})
await slam.do_command({"command": "set_initial_pose", "pose": {"x": 0, "y": 0, "theta": 0}})
# XY roughly right but heading unknown/wrong? slam_toolbox only self-corrects
# ~±30° of yaw — add refine to run a full-yaw seeded scan match and apply it:
await slam.do_command({"command": "set_initial_pose",
                       "pose": {"x": 0, "y": 0, "theta": 0}, "refine": True})
```

If the nav-stack map is aligned with the MiR onboard map, seed from the MiR pose instead
(continuous laser matching on the MiR side; nav-stack still needs one `/initialpose` seed):

```python
await slam.do_command({
    "command": "start_localizing",
    "map": "ground-floor",
    "use_mir_pose": True,
})
# or after localizing:
await slam.do_command({"command": "relocalize", "use_mir_pose": True})
```

**Preferred:** match live lidar against the saved nav-stack occupancy map (no MiR map pose):

```python
await slam.do_command({"command": "global_localize"})
# search the whole map when pose is unknown:
await slam.do_command({"command": "global_localize", "full_map": True})
# narrow search around a rough guess (meters):
await slam.do_command({
    "command": "global_localize",
    "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
    "search_radius_m": 6.0,
})
# preview-only (do not publish /initialpose yet):
await slam.do_command({"command": "global_localize", "apply": False})
```

Returned fields include `pose`, `score`, `candidates_evaluated`, `scan_points_used`,
`in_map_points`, `hit_rate`, `ray_score`, and `ray_mae_m`. If `in_map_points` is
low or `ray_mae_m` is high, the match is unreliable.
Speed/robustness knobs: `local_yaw_window_deg`, `coarse_position_step_m`,
`coarse_yaw_step_deg`, `max_scan_points`, `min_in_map_points`,
`min_in_map_ratio`, `hit_radius_cells`, `ray_refine_candidates`,
`ray_refine_beams`, `ray_step_m`, `ray_weight`.

By default, `global_localize` now auto-falls back to full-map search when local
search quality is weak. Tune or disable with `auto_full_map_fallback`,
`fallback_score_threshold`, and `fallback_hit_rate_threshold`.

When you are roughly in the right place but nav-stack drifted (~2 m), trigger scan-to-map
matching with wider covariance:

```python
await slam.do_command({"command": "relocalize"})
```

### 3. Create and use locations

```python
# Save the robot's current spot as "kitchen"
await nav.do_command({"command": "add_location", "name": "kitchen"})
# Or specify a pose (meters / radians, map frame)
await nav.do_command({"command": "add_location", "name": "dock",
                      "pose": {"x": 1.0, "y": 2.0, "theta": 0.0}})

await nav.do_command({"command": "navigate_to_location", "name": "kitchen"})
await nav.do_command({"command": "navigate_to_point", "x": 3.5, "y": -1.0})
await nav.do_command({"command": "get_status"})
# get_status includes last_cmd_vel plus cmd_vel_history (last ~20 distinct
# ROS/Viam SetVelocity samples, oldest→newest — survives cancel/stop zeros)
# Plain-English snapshot of what nav is commanding right now (returns immediately):
# await nav.do_command({"command": "describe_motion"})
# → {"summary": "Nav2 navigating (goal 'kitchen' is about 2.5 m ahead and to the
#    right): driving forward at moderate speed while turning hard right for
#    about 3.0 s — closing distance toward the goal, steering toward the goal",
#    "goal_relative": "...", "toward_goal": "...", ...}
# Probe the Nav2 SetVelocity path without navigating:
# await nav.do_command({"command": "test_drive", "vx": 0.5, "angular_z_deg_s": 57.3, "duration_s": 2})
await nav.do_command({"command": "cancel"})

# Force Nav2 to stop and relaunch with freshly generated params (param
# changes normally apply automatically on reconfigure; this is the manual
# override). get_status includes "controller_frequency_loaded" to verify.
await nav.do_command({"command": "restart_nav2"})
```

Locations CRUD: `add_location`, `get_location`, `list_locations`,
`update_location`, `delete_location` (alias `remove_location`),
`delete_all_locations`.

### 4. Define virtual zones

Physical obstacles are avoided automatically. Virtual zones are user-defined:

```python
# A no-go region
await nav.do_command({"command": "add_zone", "name": "fragile-display",
                      "type": "keepout",
                      "geometry": {"type": "circle", "center": [4.0, 1.5], "radius": 0.8}})
# A slow-down region (30% of max speed)
await nav.do_command({"command": "add_zone", "name": "busy-aisle",
                      "type": "speed_limit", "speed_pct": 30,
                      "geometry": {"type": "polygon",
                                   "points": [[0,0],[2,0],[2,3],[0,3]]}})
```

Zones CRUD: `add_zone`, `get_zone`, `list_zones`, `update_zone`, `delete_zone`,
`delete_all_zones`. Geometry types: `circle`, `box` (optionally `rotation`),
`polygon`. Locations and zones are stored per-map.

### Map management (SLAM service)

`list_maps`, `get_active_map`, `set_active_map`, `rename_map`, `delete_map`,
`start_mapping`, `start_localizing`, `save_map`, `get_mode`, `get_status`
(live bridge + slam_toolbox health; optional `probe_sensors: false` to skip a
one-shot lidar/odom read; includes measured `scan_hz` / `lidar_read_hz` over
the last ~2 s plus configured `scan_rate_hz` — with `map_when_still`, expect low
`scan_hz` while driving), `set_initial_pose`,
`global_localize` (lidar scan match against saved map; optional `full_map`,
`search_radius_m`, `apply`, `local_yaw_window_deg`, `max_scan_points`,
`auto_full_map_fallback`),
`relocalize` (alias `refine_localization`; optional `pose`, `location`),
`revisit_check` / `get_revisit_check` (mapping-mode revisit watchdog cycle on
demand; optional `apply` to force or dry-run the odom correction; optional
`yaw_flip` to take the opposite corridor heading; `flip_yaw_only` to reverse
the current map heading in place when XY is already right).

## Development

```bash
./setup.sh                 # create venv + verify ROS deps
python -m pytest tests/    # pure-Python unit tests (no ROS needed)
./build.sh                 # package module.tar.gz
```

The geometry/format conversions and the map/location/zone stores have no ROS or
Viam dependency and are unit-tested directly. The ROS bridge, process manager, and
costmap-filter wiring require an on-device ROS2 + Nav2 environment to validate.

## Scope / notes

- 2D ground-robot navigation (slam_toolbox + Nav2 are 2D). One base per service.
- Differential and omnidirectional kinematics supported; Ackermann is out of scope.
- Multi-lidar merging for SLAM assumes roughly coplanar lidars with accurate mount
  transforms; all lidars still contribute to Nav2 obstacle avoidance regardless.
