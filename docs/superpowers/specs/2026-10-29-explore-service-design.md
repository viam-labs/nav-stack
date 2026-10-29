# Explore service: autonomous mapping walk

Date: 2026-10-29

## Intent

Add a generic service to nav-stack that drives a robot around an office
overnight so the builtin SLAM maps it. The robot starts with no prior map and
no trusted localization. The service
must expose the lidars to as much of the floor and as many wall surfaces as
possible within a caller-supplied time budget, while producing a map the
SLAM engine can keep globally consistent.

The first use is a single open-plan corporate floor. Rooms off the open
area have closed doors and are not to be mapped. Cleaning crews move
around overnight, so moving obstacles are expected. There is no charger.

## Goals

- Maximize lidar coverage of reachable floor and wall surface within the
  time budget.
- Produce a map that loop-closes well: drive the way the SLAM engine
  wants (smooth, slow turns, revisits, pauses facing walls) while the
  keyframe budget is fresh.
- Be robust to poor localization: every motion is a short goal so a
  bad pose estimate wastes a leg, not the run.
- Finish cleanly: save the map, then return to the start pose.
- Testable hardware-free in the existing sim.

## Non-goals

- Mapping with the RealSense cameras. Only the lidars feed SLAM; the
  cameras stay avoidance-only.
- Mapping inside rooms. Doors are closed; the lidar sees walls and no
  frontier opens. No room-detection logic.
- Using a floor plan. Exploration works from the live grid only.
- Charging, docking, or battery management.
- Changing SLAM or navigation internals. The service is a client of them.

## Anti-goals

- A run that drives a lot but leaves the map with doubled walls because
  loop closure never fired. Coverage without consistency is a failure.
- Getting stuck on one unreachable goal and burning the night retrying it.
- Painting cleaners into the map as permanent obstacles. The service
  cannot fully prevent this (SLAM decides what to insert) but must not
  make it worse by parking next to moving people.
- Driving faster or turning faster than the SLAM engine accepts scans.

## Target robot (tracer2a, read from its machine config 2026-10-29)

- One RPLidar (`viam-labs:nav-stack:rplidar`), 360 deg, `max_range` 9 m. It is
  the only sensor that builds the map.
- Two RealSense cameras configured as SLAM lidars with `obstacles_only: true`,
  so avoidance and local costmap only.
- `map_when_still` off and lidar odometry off (both default off for a scan
  lidar; `src/config.py:782, 1201`). Mapping is continuous; the yaw gate is
  45 deg/s.
- Navigation has no angular cap configured, so it defaults to 1.5 rad/s
  (`src/config.py:1360`), above the SLAM yaw gate. The machine config must
  set the nav angular limit at or below the explore cap before a run.
- SLAM is in `localizing` mode against `new_map`. A run requires `mapping`.
- Nav `max_linear_velocity` 0.5 m/s, local planner 0.35 m/s.

## Constraints from the stack (verified in code)

- Only lidars in the SLAM `lidars` list build the map
  (`src/slam_builtin/io_sensors.py`).
- Scans are not inserted while yawing faster than 45 deg/s, 15 deg/s with
  lidar odometry (`src/slam_builtin/engine.py` ~1084-1114).
- Keyframes every 0.5 m or 20 deg. After `builtin_mapping_keyframe_max`
  (default 2000, roughly 1 km of path including turning) loop closure
  stops for the session; mapping continues (`src/config.py:849`,
  `src/slam_builtin/pose_graph.py:180`).
- Loop closure only matches keyframes at least 6 m of path behind, within
  4 m (`README.md:156-158`).
- The planner treats unknown cells as lethal and rejects goals in them
  (`src/nav_builtin/planner.py:780-787`). Goals must be on known free
  cells.
- Navigation works in mapping mode. `go_to_point` drives without the
  planner; `plan_to_point` returns feasibility and length without driving;
  `cancel` stops (`src/models/nav_api.py:708, 796, 844`).
- SLAM `get_grid` DoCommand returns the full occupancy grid (-1 unknown,
  0 free, 100 occupied) with cell size and origin (`src/models/slam.py:3006`).
- SLAM `save_map` and `optimize` DoCommands exist (`src/models/slam.py:3113,
  3122`).
- Mapping mode loads an existing `map.pgm`/`map.yaml` from the active map
  directory and extends it (`src/slam_builtin/engine.py:215-283`). A floor
  plan rendered as a PGM in that directory is therefore usable as a prior.
- SLAM and nav share an in-process registry (`src/runtime.py`); nav looks
  the SLAM runtime up by resource name.
- The sim (`sim-base` plus `sim.enabled`) runs mapping hardware-free from
  a `.npy` world grid with perfect odometry (`src/sim/world.py`,
  `sample_configs/local_sim.json`).

## The service

**Model:** `viam-labs:nav-stack:explore`, API `rdk:service:generic`.
Registered in `src/main.py` and `meta.json` like the other models.

**Config attributes:**

- `slam_service` (required): name of the SLAM service. Must be in
  mapping mode when a run starts.
- `navigation_service` (required): name of the navigation service.
- `base` (required): base name, used only for an emergency stop.
- `max_linear_mps` (default 0.4): passed to each leg.
- `max_angular_dps` (default 30): `start` fails unless the navigation
  service's configured angular limit is at or below this. The explore
  service never drives the base itself except to stop.
- `home_tolerance_m` (default 0.5).
- `structure_fraction` (default 0.15): share of the budget reserved for
  the structure phase, bounded by the keyframe budget (see below).

**DoCommands:**

- `start {duration_s}`: begins a run. Errors if a run
  is active or SLAM is not in mapping mode. Records the current pose as
  home.
- `stop`: ends the run early. Same finish sequence as timeout.
- `status`: `{phase, elapsed_s, remaining_s, legs, legs_failed,
  coverage_cells, frontier_count, keyframes_used, last_error}`.

The service never auto-starts. It is idle until `start`.

## The run

Three phases, then finish.

### Phase 1: structure

Purpose: spend the fresh keyframe budget on the loop closures that make
the map globally consistent.

1. Outer circuit. Walk the perimeter of the reachable open area, keeping
   a wall on one side, until the start is revisited. Implementation is
   wall-following over successive short go-to goals, not a planned route,
   since there is no trusted map yet.
2. Inner circuits. Two or three loops through the open area that cross
   the outer circuit, each at least 12 m of path so loop closure can
   match against the far side.
3. Between legs, pause for 2 s facing the nearest wall segment.

Phase 1 ends when its time share is used, or when the keyframe count
reported by SLAM status reaches 60% of `builtin_mapping_keyframe_max`,
whichever is first. The remaining 40% is left for revisits later.

The outer circuit is pure wall-following and the inner circuits are
chosen from the live grid after the outer circuit closes.

### Phase 2: fill

Purpose: cover everything the circuits did not see.

Loop until the budget for this phase runs out:

1. Read the grid via `get_grid`.
2. Build candidate goals:
   - Frontier goals: clusters of free cells adjacent to unknown cells.
     The goal is the free cell one robot-radius inside the cluster, so
     the planner accepts it.
   - Under-seen goals: free cells whose nearest observation pose is
     farther than `well_seen_m` (default 4 m). The service tracks its own
     pose history to compute this; SLAM does not expose per-cell
     observation counts.
3. Score each candidate: expected new cells visible from the goal
   (raycast on the grid, lidar range capped at 8 m for scoring) divided
   by `plan_to_point` length. Reject infeasible plans. Penalize
   candidates near recently failed goals and candidates in the
   direction of currently observed moving obstacles.
4. `navigate_to_point` (planned) to the best candidate, wait, then pause
   1 s. Planned legs route around furniture; the straight-line
   `go_to_point` is used only for short wall-following hops in phase 1.
5. No turn on arrival: the lidar is 360 deg.

### Phase 3: revisit

Purpose: use the remaining keyframe budget for late loop closures.

Starts when phase 2 produces no candidates above a minimum score, or when
`revisit_fraction` (default 0.1) of the budget remains. Replays a subset
of phase 1 circuits, prioritizing ones whose loop closure never confirmed
according to SLAM status diagnostics. If the keyframe cap is already
reached this phase is skipped and phase 2 continues.

### Finish

Triggered by timeout, `stop`, or an unrecoverable error.

1. `cancel` any active goal.
2. SLAM `optimize`, then `save_map`. The map is safe after this step.
3. `navigate_to_point` home (planned) and wait. If it fails, retry once
   from the current pose, then stop in place.
4. Report final status.

## Failure handling

- A leg that errors, times out (3x the planned length at the speed cap,
  minimum 20 s), or ends more than 1 m from its goal is marked failed.
  The goal and cells within 1 m are blacklisted for 10 minutes.
- Five consecutive failed legs: stop, wait 30 s, then resume. Twenty in a
  row: finish the run early with `last_error` set.
- Lost pose (nav reports untrusted or no pose): wait up to 60 s for it to
  return, then continue. If it does not return, finish the run.
- SLAM status unhealthy: same as lost pose.
- `stop` and `timeout` always run the finish sequence even if the service
  is in the middle of a leg.

## Testing

- Unit tests for frontier extraction, under-seen tracking, raycast
  scoring, and blacklisting against hand-built grids.
- A sim test using `sim-base` with a multi-room world grid, closed doors,
  and a time budget, asserting the fraction of ground-truth free cells
  that end up known in the SLAM grid. The sim has perfect odometry, so
  the test covers coverage logic, not loop closure.
- A sim test with a moving obstacle to check that legs fail gracefully and
  get blacklisted. The sim has no dynamic obstacles today; a small
  test-only obstacle-injection DoCommand on `sim-base` is in scope.

## Decided by Benji

- Generic service inside nav-stack, not a standalone script.
- Biased exploration, not a random walk.
- Finish is save map first, then return home.
- Rooms are excluded by their closed doors, not by any detection.
- Time budget is a parameter; the first use is 5 to 10 hours overnight.
- No floor plan input; raw exploration only.

## My calls (Claude)

- Phases, thresholds, and fractions above are starting values and are all
  config attributes. The builder should keep them adjustable.
- The return home drives after the save, and stops in place on failure.
- The service keeps its own pose history for under-seen scoring rather
  than extending SLAM.
- Keyframe budget is read from SLAM status (`keyframes` and
  `loop_closure.graph_full`); the cap itself is an explore config attribute
  mirroring `builtin_mapping_keyframe_max` since status does not expose it.
- Fill legs use planned navigation; direct go-to is only for wall-following.
- The angular cap is enforced by validating nav config, not by driving the
  base. Nav `get_status` gains a `limits` field reporting its configured
  linear and angular caps so the explore service can read them.
  Checkpointing by save and restart mid-run is explicitly not relied on.

## Left to the builder

- Wall-following implementation details for the outer circuit.
- Frontier clustering method and minimum cluster size.
- Raycast implementation for scoring, and whether to cache per grid read.
- How moving obstacles are detected for scoring penalties; using the nav
  local costmap is acceptable.
- Internal threading model, matching how the navigation model runs its
  supervisor.
