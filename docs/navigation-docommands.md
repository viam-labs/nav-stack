# Navigation service DoCommands

Reference for `DoCommand` on the motion services:

- `viam-labs:nav-stack:navigation`
- `viam-labs:nav-stack:navigation-external`

Both share `[NavServiceBase.do_command](../src/models/nav_core.py)`. Payload shape is always:

```json
{ "command": "<name>", ...args }
```

**Units (DoCommand):** map frame, **meters** and **radians**, unless a field name says otherwise (e.g. `angular_z_deg_s`).  
Viam `MoveOnMap` uses millimeters / degrees — that is a separate Motion API path, not listed here.

Unknown `command` values raise `ValueError`.

---

## Locations (CRUD)

Named poses stored per active map.


| Command                               | Args                                                                                                            | Returns                   |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `add_location`                        | `**name**`. Pose via `**pose**` `{x,y,theta?}` or top-level `**x`/`y`/`theta?**`. Omit pose → current map pose. | `{ "location": {...} }`   |
| `get_location`                        | `**name**`                                                                                                      | `{ "location": {...} }`   |
| `list_locations`                      | —                                                                                                               | `{ "locations": [...] }`  |
| `update_location`                     | `**name**`; optional `x`, `y`, `theta`, `new_name`                                                              | `{ "location": {...} }`   |
| `delete_location` / `remove_location` | `**name**`                                                                                                      | `{ "status": "deleted" }` |
| `delete_all_locations`                | —                                                                                                               | `{ "status": "deleted" }` |


```python
await nav.do_command({"command": "add_location", "name": "kitchen"})
await nav.do_command({
    "command": "add_location",
    "name": "dock",
    "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
})
await nav.do_command({"command": "list_locations"})
```

Renaming a location (`update_location` + `new_name`) rewrites matching waypoints on all routes. Deleting a location removes it from every route’s waypoint list. `delete_all_locations` clears waypoints on all routes (routes themselves remain).

---

## Routes & waypoints (CRUD)

A **route** is an ordered series of **waypoints**. Each waypoint is a **location name** on the same map (pose comes from the location store). Persisted per map as `routes.json`.

Returned route shape (poses resolved when the location exists):

```json
{
  "name": "patrol",
  "waypoints": [
    {"location": "dock", "x": 1.0, "y": 2.0, "theta": 0.0},
    {"location": "kitchen", "x": 3.5, "y": -1.0, "theta": 1.57}
  ]
}
```

If a referenced location is missing, that waypoint is `{"location": "...", "missing": true}`.

### Routes


| Command                         | Args                                                                          | Returns                   |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------- |
| `add_route`                     | `**name**`; optional `**waypoints**` (location names or `{location}` objects) | `{ "route": {...} }`      |
| `get_route`                     | `**name**`                                                                    | `{ "route": {...} }`      |
| `list_routes`                   | —                                                                             | `{ "routes": [...] }`     |
| `update_route`                  | `**name**`; optional `new_name`                                               | `{ "route": {...} }`      |
| `delete_route` / `remove_route` | `**name**`                                                                    | `{ "status": "deleted" }` |
| `delete_all_routes`             | —                                                                             | `{ "status": "deleted" }` |


Adding a route (or setting waypoints) **requires** each location to already exist.

### Waypoints


| Command                               | Args                                                              | Returns                           |
| ------------------------------------- | ----------------------------------------------------------------- | --------------------------------- |
| `list_waypoints`                      | `**route**`                                                       | `{ "route", "waypoints": [...] }` |
| `set_waypoints`                       | `**route**`, `**waypoints**`                                      | `{ "route": {...} }`              |
| `clear_waypoints`                     | `**route**`                                                       | `{ "route": {...} }`              |
| `add_waypoint`                        | `**route**`, `**location**`; optional `index` (append if omitted) | `{ "route": {...} }`              |
| `update_waypoint`                     | `**route**`, `**index**`, `**location**`                          | `{ "route": {...} }`              |
| `remove_waypoint` / `delete_waypoint` | `**route**`; `**index**` *or* `**location*`*                      | `{ "route": {...} }`              |
| `move_waypoint`                       | `**route**`, `**from_index**`, `**to_index**`                     | `{ "route": {...} }`              |


```python
await nav.do_command({"command": "add_location", "name": "dock", "x": 0, "y": 0})
await nav.do_command({"command": "add_location", "name": "kitchen", "x": 3, "y": 1})
await nav.do_command({
    "command": "add_route",
    "name": "patrol",
    "waypoints": ["dock", "kitchen"],
})
await nav.do_command({
    "command": "add_waypoint",
    "route": "patrol",
    "location": "lobby",  # must already exist as a location
})
await nav.do_command({
    "command": "move_waypoint",
    "route": "patrol",
    "from_index": 2,
    "to_index": 1,
})
await nav.do_command({"command": "list_routes"})
```

---

## Zones (CRUD)

Virtual regions on the active map. Types: `keepout`, `speed_limit` (requires `speed_pct` 1–100).

Geometry (map frame, meters):

- `{"type": "circle", "center": [x, y], "radius": r}`
- `{"type": "box", "center": [x, y], "size": [w, h], "rotation": theta}` — `rotation` optional (rad)
- `{"type": "polygon", "points": [[x, y], ...]}` — ≥ 3 points


| Command            | Args                                                               | Returns                   |
| ------------------ | ------------------------------------------------------------------ | ------------------------- |
| `add_zone`         | `**name**`, `**type**`, `**geometry**`; `speed_pct` if speed_limit | `{ "zone": {...} }`       |
| `get_zone`         | `**name**`                                                         | `{ "zone": {...} }`       |
| `list_zones`       | optional `type` filter                                             | `{ "zones": [...] }`      |
| `update_zone`      | `**name**`; optional `type`, `geometry`, `speed_pct`, `new_name`   | `{ "zone": {...} }`       |
| `delete_zone`      | `**name**`                                                         | `{ "status": "deleted" }` |
| `delete_all_zones` | optional `type`                                                    | `{ "status": "deleted" }` |


```python
await nav.do_command({
    "command": "add_zone",
    "name": "fragile-display",
    "type": "keepout",
    "geometry": {"type": "circle", "center": [4.0, 1.5], "radius": 0.8},
})
await nav.do_command({
    "command": "add_zone",
    "name": "busy-aisle",
    "type": "speed_limit",
    "speed_pct": 30,
    "geometry": {
        "type": "polygon",
        "points": [[0, 0], [2, 0], [2, 3], [0, 3]],
    },
})
```

---

## Builtin path following

Plans and drives with the in-module navigator (obstacle-aware). Non-blocking: returns once the goal is accepted / planning starts on a worker thread.


| Command                                         | Args                                                                                                | Returns                                                                |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `navigate_to_location`                          | `**name**`                                                                                          | `{ "status": "navigating", "target": <location> }`                     |
| `navigate_to_point`                             | `**x**`, `**y**`; `theta` (default `0`)                                                             | `{ "status": "navigating", "target": {x,y,theta} }`                    |
| `navigate_route`                                | `**name**` *or* `**waypoints*`*; optional `start_index`, `start_nearest`, `loop`, `wait` | `{ "status": "navigating", "motion": "route", ... }`                   |
| `plan_to_point` / `compute_path_to_point`       | `**x**`, `**y**`; `theta`; optional `planner_id`, `timeout_s`, `max_points`, `start` `{x,y,theta?}` | Preview dict + `"status": "planned"|"infeasible"`                      |
| `plan_to_location` / `compute_path_to_location` | `**name**`; same optional planner args                                                              | Preview + `"location"`                                                 |
| `get_last_plan` / `get_preview_plan`            | —                                                                                                   | `{ "plan": <last preview> }`                                           |
| `execute_plan`                                  | — (uses last feasible preview)                                                                      | `{ "status": "navigating", "target": ..., "from_preview": true, ... }` |
| `cancel`                                        | —                                                                                                   | `{ "status": "canceled" }` — clears suspended goal too                 |


### `navigate_route`

Follows an ordered list of waypoints with the builtin navigator, one leg at a time. Cancels any prior route / simple-nav / suspended goal first.


| Arg             | Notes                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `name`          | Saved route name (loads waypoints from the route store)                                        |
| `waypoints`     | Inline list: location names, `{location}`, or `{x,y,theta?,location?}`                         |
| `start_index`   | Skip ahead to this waypoint (default `0`; used by `resume`)                                    |
| `start_nearest` | If `true`, start at the waypoint closest to the current map pose (overrides `start_index`)     |
| `loop`          | If `true`, after the last waypoint wrap to the first and keep going until `cancel` / `suspend` |
| `wait`          | If `true`, block until the route finishes (default `false`). With `loop`, only returns on stop |


`get_status` includes a `route` blob (`state`, `name`, `index`, `total`, `location`, `waypoints`, `loop`, `lap`, `error_msg`). While a route is active, top-level `motion` is `"route"`.

`suspend` remembers the current leg (and `loop`); `resume` restarts the route from that index.

```python
await nav.do_command({"command": "navigate_route", "name": "patrol"})
await nav.do_command({
    "command": "navigate_route",
    "name": "patrol",
    "start_nearest": True,
    "loop": True,
})
await nav.do_command({
    "command": "navigate_route",
    "waypoints": ["dock", "kitchen", "lobby"],
})
```

Preview fields typically include `feasible`, `path` (`[{x,y,theta}, ...]`), `length_m`, `planning_time_s`, `error_msg` / `error_code` when infeasible. Successful previews are drawn on `nav-camera` when configured.

```python
await nav.do_command({"command": "plan_to_point", "x": 3.5, "y": -1.0})
await nav.do_command({"command": "execute_plan"})
await nav.do_command({"command": "cancel"})
```

---

## Optional Jev local-block policy

When the builtin follower sees a **local path block**, it normally chooses among `wait` / `keep_dwa` (inch/peel) / `replan` with fixed heuristics. You can overlay [TypeSafe Jev](https://docs.typesafe.ai/introduction) as an experimental advisor.

Configure under the navigation service `builtin` block (or top-level aliases where applicable):

| Attribute | Default | Notes |
|---|---|---|
| `nav_policy` | `heuristic` | `heuristic` \| `shadow` \| `jev` |
| `jev_min_confidence` | `0.7` | In `jev` mode, fall back to heuristic below this |
| `jev_timeout_s` | `1.25` | Per TypeSafe call |
| `jev_min_period_s` | `1.0` | Min seconds between Jev queries |
| `jev_history_s` | `3.0` | Obstacle-track window for mover-vs-fixed features |
| `jev_model` | `jev-latest` | TypeSafe model id |
| `jev_api_key` | _(env)_ | Or set `TYPESAFE_API_KEY` on the machine |

**Modes**

- `heuristic` — unchanged behavior; no TypeSafe calls.
- `shadow` — call Jev on local blocks, **always log** heuristic vs Jev, still **execute heuristic** (safe for robot trials).
- `jev` — execute Jev’s mapped action when confidence is high enough; otherwise heuristic. Always logs both. Actions: `wait` / `keep_dwa` / `replan` / `backup`.

`get_status` → `progress.jev_policy` (while navigating) includes `heuristic_action`, `jev_action`, `applied_action`, `confidence`, `features` (incl. `motion_score` / `likely_mover`), and answer snippets.

**Robot trial (recommended order)**

1. `pip install typesafe-sdk` (already in `requirements.txt`); export `TYPESAFE_API_KEY`.
2. Set `"nav_policy": "shadow"` on the nav service; restart the module.
3. Drive into a person-crossing and a fixed-box corridor; watch module logs for `jev_policy {...}` and compare `heuristic_action` vs `jev_action`.
4. Only then try `"nav_policy": "jev"` on a supervised run.

Hard stops (lethal footprint, cancel) stay rule-based — Jev only advises the wait/peel/replan fork.

---

## Simple closed-loop go-to

Direct map-frame approach without full builtin planning (useful for short hops / docks). Cancels any prior simple-nav and builtin goal first.


| Command          | Args                                                | Returns                                |
| ---------------- | --------------------------------------------------- | -------------------------------------- |
| `go_to_location` | `**name**`; `wait` (default `true`); `velocity_mps` | status + `motion: "simple"` + `target` |
| `go_to_point`    | `**x**`, `**y**`; `theta`; `wait`; `velocity_mps`   | same                                   |


With `wait: false`, starts a background task and returns `{ "status": "navigating", ... }` immediately.

---

## Suspend / resume


| Command   | Aliases                    | Args              | Returns                                                                                                  |
| --------- | -------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| `suspend` | `pause_nav`, `suspend_nav` | optional `reason` | `{ "status": "suspended", "goal": {...} }`                                                               |
| `resume`  | `resume_nav`               | —                 | Re-issues saved goal (`builtin` → navigate, `simple` → go-to, route → `navigate_route` from current leg) |


Suspend cancels motion but remembers the goal (including multi-waypoint route index). A new navigate / `cancel` / `stop_plan` clears the suspended snapshot. `get_status` includes `suspended` and `suspended_goal` while held.

```python
await nav.do_command({"command": "suspend", "reason": "safety"})
await nav.do_command({"command": "resume"})
```

---

## Status & diagnostics


| Command           | Aliases           | Args                  | Returns                                                  |
| ----------------- | ----------------- | --------------------- | -------------------------------------------------------- |
| `get_status`      | —                 | —                     | Full nav status (see below)                              |
| `describe_motion` | `what_am_i_doing` | —                     | Plain-English summary fields from `summarize_nav_motion` |
| `get_jev_policy_log` | `list_jev_decisions` | optional `limit`, `queried_only` | Timeline of Jev/heuristic decisions for the current/last run |
| `clear_jev_policy_log` | `clear_jev_decisions` | —              | Clear the decision timeline                              |
| `test_drive`      | —                 | body twist + duration | Echo of sent body / Viam SetVelocity units               |
| `get_costmap`     | —                 | `layer`, `stride`     | Base64 costmap grid for UIs                              |


### `get_status`

Includes (among other fields): `state`, `active`, `motion`, `goal`, `pose`, progress / error fields from the follower (including `progress.jev_policy` when enabled), `simple_nav`, `route`, `localization_check`, `suspended` / `suspended_goal`, footprint (`footprint_length_m`, `footprint_width_m`, `robot_radius_m`), and when available drive / control-loop stats (`last_drive`, `drive`, `control_loop`, `pose_source`, `nav_backend`).

### `get_jev_policy_log`

Returns the decision timeline for the **current or last** navigate run (cleared when a new goal starts). Intended for nav-stack-ui overlays.

| Arg | Notes |
|---|---|
| `limit` | Optional max events (most recent) |
| `queried_only` | If `true`, only entries where TypeSafe was actually called |

Each event includes `seq`, `run_id`, `t_wall`, `t_mono`, `pose` `{x,y}`, `goal` `{x,y}`, `heuristic_action`, `jev_action`, `applied_action`, `confidence`, `queried`, `fallback_reason`, `features`, `answers`, latency/error when present.

```python
log = await nav.do_command({"command": "get_jev_policy_log"})
# log["events"] → plot markers along the path in the UI
await nav.do_command({"command": "clear_jev_policy_log"})
```

### `test_drive`

Probes the SetVelocity path without navigating.


| Arg                                 | Notes                                             |
| ----------------------------------- | ------------------------------------------------- |
| `vx` / `body_vx_mps` / `ros_vx_mps` | Body forward m/s (default `0`)                    |
| `vy` / `body_vy_mps`                | Body lateral m/s                                  |
| `vtheta` / `body_vtheta_rad_s`      | Body yaw rate rad/s                               |
| `angular_z_deg_s`                   | Alt yaw input in deg/s (used if `vtheta` omitted) |
| `duration_s`                        | Clamped to `[0.1, 5.0]` (default `1.5`)           |


### `get_costmap`


| Arg      | Notes                                                                       |
| -------- | --------------------------------------------------------------------------- |
| `layer`  | `auto` (default: local while navigating, else global), `local`, or `global` |
| `stride` | Grid downsample (default `2`)                                               |


Returns `available`, `layer`, `origin_x`/`origin_y`, `resolution`, `width`/`height`, `encoding: "uint8_row_major"`, `unknown: 255`, `data_b64`, `nav_backend`. Cost cells `0..100`; unknown = `255`.

```python
await nav.do_command({"command": "get_status"})
await nav.do_command({"command": "describe_motion"})
await nav.do_command({
    "command": "test_drive",
    "vx": 0.5,
    "angular_z_deg_s": 57.3,
    "duration_s": 2,
})
await nav.do_command({"command": "get_costmap", "layer": "local", "stride": 2})
```

---

## Command index


| Command                | Aliases                    |
| ---------------------- | -------------------------- |
| `add_location`         |                            |
| `get_location`         |                            |
| `list_locations`       |                            |
| `update_location`      |                            |
| `delete_location`      | `remove_location`          |
| `delete_all_locations` |                            |
| `add_route`            |                            |
| `get_route`            |                            |
| `list_routes`          |                            |
| `update_route`         |                            |
| `delete_route`         | `remove_route`             |
| `delete_all_routes`    |                            |
| `list_waypoints`       |                            |
| `set_waypoints`        |                            |
| `clear_waypoints`      |                            |
| `add_waypoint`         |                            |
| `update_waypoint`      |                            |
| `remove_waypoint`      | `delete_waypoint`          |
| `move_waypoint`        |                            |
| `add_zone`             |                            |
| `get_zone`             |                            |
| `list_zones`           |                            |
| `update_zone`          |                            |
| `delete_zone`          |                            |
| `delete_all_zones`     |                            |
| `navigate_to_location` |                            |
| `navigate_to_point`    |                            |
| `navigate_route`       |                            |
| `plan_to_point`        | `compute_path_to_point`    |
| `plan_to_location`     | `compute_path_to_location` |
| `get_last_plan`        | `get_preview_plan`         |
| `execute_plan`         |                            |
| `go_to_location`       |                            |
| `go_to_point`          |                            |
| `suspend`              | `pause_nav`, `suspend_nav` |
| `resume`               | `resume_nav`               |
| `cancel`               |                            |
| `test_drive`           |                            |
| `get_status`           |                            |
| `describe_motion`      | `what_am_i_doing`          |
| `get_jev_policy_log`   | `list_jev_decisions`       |
| `clear_jev_policy_log` | `clear_jev_decisions`      |
| `get_costmap`          |                            |


---

## Related (not DoCommand)

- `**MoveOnMap` / plan queries** on the same Motion service — Viam mm/deg; optional `extra={"preview": true}` for a path preview.
- **SLAM service DoCommands** (`list_maps`, `global_localize`, `save_map`, …) — see README “Map management”.

Source of truth: `src/models/nav_core.py` (`NavServiceBase.do_command`).