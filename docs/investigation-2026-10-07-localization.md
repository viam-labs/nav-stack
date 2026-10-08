# Investigation: localization quality and hesitant re-routing on tracer2a

Started 2026-10-07. Machine `tracer2a` (Agilex Tracer, RPLIDAR + RealSense +
WitMotion IMU), module `viam-labs:nav-stack` at `latest-with-prerelease`
(1.0.57 → 1.0.59 deployed during the session). Map `new_map`, mode
`localizing`.

Symptom reported: robot stalls mid-route, "tries to reroute but struggles".

## TL;DR so far

- The two goals observed live (18:20:07, 18:24:09) reached XY in ~20 s and
  succeeded. The stall episodes happened earlier; their replan diagnostics
  were not recoverable (see "Observability gap").
- Localization quality is **below the stack's own thresholds for the whole
  session**, but the headline "96% of scan matches rejected" is a **counter
  artifact**, not a failure rate (see "Reading the counters").
- Working hypothesis: a published pose that is off by roughly 0.5–1 m makes
  the static-map planner and the live-lidar guard disagree, so replans get
  rejected by the candidate filters and the mid-nav localization refine keeps
  pausing the robot. Needs confirmation with the route monitor.

## Timeline (UTC, from cloud logs)

| Time | Event |
| --- | --- |
| 14:12–14:18 | Module (re)starts ×3. RealSense "Camera unavailable" on each. |
| 14:15 | RealSense depth frames 1.0–1.2 s stale. |
| 16:36–16:46 | RPLIDAR shm frame stale for ~9 min (`frame too old 5xx s`) while the module kept running. Module warns the lidar shares a USB hub with a CAN adapter so it cannot reset the device. Recovered on module restart 16:46. |
| 17:13 | Module restart. |
| 17:32:33 | Module **crash**: grpclib `Fatal error: protocol.data_received()` KeyError on stream reset (v1.0.57). |
| 17:43–17:47 | Same goal (-25.69, 17.59) re-sent 4× in 4.5 min. Module restart 17:46. |
| 17:58:56–17:59:11 | Periodic relocalize: scores 0.09–0.28, shifts 0.33–0.51 m, holding. |
| 18:06:20 | Module config changed → restart on v1.0.59. `nav-cam` build error: SDK KeyError on `viam-labs:service:action/docking` ResourceName. |
| 18:06:33 | Engine rejected odom jump dth=121°. Startup localize: ambiguous peaks, skipped. |
| 18:06:47 | Startup localize bypassed by drift watchdog: tick score **−0.31**. |
| 18:07–18:08 | Full-map relocalize refused 4× as ambiguous (one with `second_best` > `score`). |
| 18:08:40 | Mid-nav: refusing 1.48 m / 16° jump (score 0.33). |
| 18:09:55 | Large jump 1.65 m / 60° awaiting confirm (prior score **0.075**). |
| 18:10–18:20 | "no trusted match after local" (score 0.40–0.47, ray_mae 0.87–1.21 m) ×8; small forced corrections 0.11 m at 18:13:36, 18:18:50, 18:18:53. |
| 18:17:14, 18:17:37, 18:18:44 | Goals re-sent 23 s and 67 s apart (operator seeing a stall). |
| 18:20:07 | Goal 3.77 m; arrived XY by ~18:20:28; final spin; succeeded. |
| 18:21:27 | **`start_localizing` DoCommand** (not an RDK reconfigure): engine restart, pose restored from disk, startup localize runs 100 s, every pass "ambiguous peak", bypassed 18:23:07. |
| 18:24:09 | Goal 2.76 m; succeeded. |

## Config snapshot (relevant)

```
slam:  mode=localizing  active_map=new_map  movement_sensor=odom  heading_sensor=wit
       lidars: rplidar (0.2–9 m), camera (RealSense, obstacles_only, cloud_frame=camera_optical, z_min 0.08)
nav:   clearance_m=0.07  control_rate_hz=20  kinematics=differential
footprint from framesystem: L=0.72 W=0.59 → robot_radius 0.295 (+0.07 clearance = 0.365 inscribed)
rplidar mount from framesystem: (0.320, 0.225, 0.21) θ=−π   (robot navigates with this; see note on tests)
wit heading yaw from framesystem: −180°
map grid: 1443×1485 @ 0.05 m  (≈72 m × 74 m)
```

Defaults in play: `periodic_relocalize=true` every 20 s, `nav_scan_track=true`,
`nav_loc_refine_on_disagree=true`, `periodic_relocalize_during_navigation=false`,
`periodic_relocalize_min_score=0.5`, `periodic_relocalize_max_ray_mae_m=1.0`,
`periodic_relocalize_recovery_min_score=0.45`, soft hold max 20 s.

## How localization works in this stack (so the numbers make sense)

Five mechanisms touch the published pose:

1. **Tick scan match** (`slam_builtin/engine.py` + `scan_match.refine_pose`).
   Engine ticks at 10 Hz; a match is due every `match_period_s` = 0.3 s
   (~3 Hz, confirmed live: +988 matches in ~348 s). Grid search ±0.30 m /
   ±12° in 5 cm / 2° steps around the predicted pose, shrunk to ±0.20 m / ±8°
   when the prior already scores ≥ 0.35. 120 subsampled beams.
   Score = `hit_rate − 0.7·free_rate − 1.2·out_of_map_rate`, where a hit is an
   endpoint within 2 cells (10 cm) of an occupied cell.
   A candidate is returned **only if it beats the prior by**
   `0.06 (or 0.03 if prior < 0.25) + 0.10·dist_m + 0.08·(dyaw/45°)`.
   Otherwise `refine_pose` returns `None`.
2. **Apply gating** (`engine._apply_match`). While **navigating**: only
   `nav_scan_track` nudges, blended at α=0.3, ≤ 5 cm / 1° per match, and only
   when prior ≥ 0.25, match ≥ 0.35, yaw rate low, scan fresh. While **idle**:
   small corrections (≤ 0.15 m / 8°) applied at α=0.6 if prior ≥ 0.15; large
   jumps need prior ≥ 0.15 and two agreeing frames, then ≤ 0.25 m / 12° per
   match.
3. **Periodic relocalize** (`models/slam.py`, every 20 s, in a subprocess).
   Local window first, full-map escalation when quality is low. Refuses
   "ambiguous" peaks (second-best too close), refuses large jumps during nav,
   and emits `nav_hold` / `awaiting_confirm` statuses. `nav_hold` does **not**
   stop the base when `nav_loc_refine_on_disagree` is on; `awaiting_confirm`
   does.
4. **Mid-nav loc refine** (`nav_builtin/supervisor.py`
   `_maybe_pause_and_refine_localization`). Every 0.75 s / 2 m, if ≥ 22% of
   beams that the map says should hit within 2.5 m are ≥ 0.8 m farther in
   lidar, stop, run `check_localization`, resume. Applies ≤ 1.0 m / 30° when
   score ≥ 0.35.
5. **Startup global localize** on every engine start, including the
   `start_localizing` DoCommand. Up to 3 refine passes; targets score 0.7,
   ray MAE ≤ 0.4 m.

## Reading the counters

`match_accepts` / `match_rejects` from `slam get_status`:

| Read | ticks | accepts | rejects | nav_track_applies | pose |
| --- | --- | --- | --- | --- | --- |
| ~18:27 | 10866 | 119 | 2984 | 73 | (-32.219, 0.036) |
| ~18:33 | 14342 | 119 | 3972 | 73 | (-32.219, 0.036) |

The robot was stationary between reads and accumulated +988 "rejects" and 0
accepts. A reject is counted whenever `_apply_match` returns False, which
includes: `refine_pose` found no candidate beating the prior by the margin
(i.e. **the pose is already the best local explanation, no correction
needed**), prior below trust, or nav gating. **So 96% is not a failure rate.**
During the ~2 min of goals it also covers, only nav-track nudges (≤ 5 cm) can
count as accepts.

The metrics that do describe quality:

| Metric | Live range today | Stack threshold |
| --- | --- | --- |
| `last_match_score` / `tick_match_score` (local, 120 beams) | 0.41 – 0.52 | nav-track needs ≥ 0.35; "still bad" ≤ 0 |
| `localization_check.score` (periodic, 240 beams) | 0.40 – 0.52 | min 0.5, recovery floor 0.45, target 0.7 |
| `localization_check.ray_mae_m` | 0.87 – 1.21 m | max 1.0, target 0.4 |
| full-map peaks | all "ambiguous" | |

Interpretation: a score of ~0.5 means roughly half the sampled beams land
within 10 cm of a mapped occupied cell and a meaningful fraction land in
mapped free space. Ray MAE ~0.9 m means raycasts to the map disagree with
measured ranges by almost a metre on average. That is **mediocre, not
catastrophic**, and consistent with either (a) a map that no longer matches
the space (clutter, moved furniture, open/closed doors, people), or (b) a
systematic pose offset the local matcher cannot climb out of because every
full-map attempt is refused as ambiguous. The stack is not self-healing in
this regime: periodic relocalize sees low quality but refuses to correct.

**Still open:** is the score low everywhere (map mismatch) or only in some
places / after some manoeuvres (odom drift, deskew, turning)? That is what
the route monitor is for.

## Observability gap (why the stall could not be reconstructed)

- `NavSupervisor` has **no logger**. Replan triggers, rejected candidates,
  forced vias and ban lifts exist only in the status dict.
- `last_replan_error` / `last_replan_info` are **cleared at the start of
  every goal**.
- `get_trace` keeps ~60 s of ticks and only while a goal runs; its window is
  relative to the last recorded tick, not wall clock.
- Nav model logs one line per goal ("builtin navigate to"); nothing on
  success / fail / cancel / replan.

## Monitoring plan

`scripts/monitor_nav.py` polls `navigation get_status` and `slam get_status`
at 1 Hz, writes JSONL, prints one line per sample, and summarizes per goal
(duration, obstacle states, every distinct replan attempt list) and per
mode (navigating vs idle: match score, prior score, loc-check score, ray MAE
distributions, accepts vs not-applied vs nav-track nudges).

```
export VIAM_ADDRESS=<tracer2a address>  VIAM_API_KEY_ID=…  VIAM_API_KEY=…
.venv/bin/python scripts/monitor_nav.py --out .local/monitor/route-$(date +%Y%m%d-%H%M%S).jsonl
```

Test protocol:

1. 60 s stationary at point A (baseline score / ray MAE while still).
2. A → B → A → B shuttle (the ~3.5 m pair used today), 3 round trips.
3. One longer route (~20 m, through at least one doorway).
4. 60 s stationary at the far end.
5. Optional: `slam check_localization {"full_map_escalation":"still_bad"}`
   (apply=false) while parked at A and at B, to see whether the full-map
   peak is ambiguous at both.

Questions the data should answer:

- Does score / ray MAE drop at specific places or stay flat (map mismatch)?
- Does it degrade with distance since the last applied correction (odom
  drift) or with yaw rate (scan smear, no deskew)?
- How many nav-track nudges per metre while moving?
- Do replan attempts cluster in low-score windows?
- Do `odom.gap_hold_events` (51 so far) grow while moving?

## Incident 18:40 UTC: SLAM + navigation unavailable after a config save

While setting up the route monitor the robot became un-navigable.

Trigger (cloud logs): at 18:39:57 the machine config gained a new module
`mugger:slackbot` and a `delivery-slackbot` generic service. That save
restarted the `nav-stack`, `agilex-tracer`, `detection-dock` and `apriltag`
module processes ("Module configuration changed. Stopping the existing module
process to reconfigure"). The slackbot module itself failed
("Module config validation error; skipping").

Effect: from 18:40:12 the RDK re-adds resources to the fresh nav-stack process
every 5 s and each attempt fails:

```
rdk:service:slam/slam  resource build error: KeyError <ResourceName rdk:service:slam/slam>
  in nav-stack .venv/.../viam/resource/manager.py  remove_resource  line 104
rdk:service:motion/navigation   dependency slam is not ready yet
rdk:component:camera/nav-cam    dependency navigation is not ready yet
rdk:component:base/avoidance-base   same KeyError in the avoidance-base module
rdk:service:vision/feature-match    KeyError on viam-labs:service:action/docking in the feature-match module
"unable to route foreign message" ×180/min
```

Mechanism (viam-sdk `module/module.py`): `reconfigure_resource` and
`remove_resource` both call `ResourceManager.remove_resource(rn)`, which does
`self.resources[name]` and raises `KeyError` when the name is not registered.
After a process restart the new process has nothing registered, so any
reconfigure/remove the RDK sends for a "known" resource throws, the build is
marked failed, and the RDK retries forever. The same KeyError hit `nav-cam` at
18:06:27 after the previous config change but cleared on retry that time.
Three different Python modules failed identically, so this is an SDK/RDK
interaction, not a nav-stack bug.

Recovery: operator chose a part restart. `viam machines part restart` sent at
18:47:20; SLAM + navigation resources were back and the engine was ticking
with the pose restored from disk by 18:49:06 (~105 s). Removing the broken
slackbot module/service is still open.

Also noted: the `delivery-slackbot` service attributes contain plaintext
third-party credentials (an LLM API key and Slack tokens). Anyone with
machine-config read access can see them. Rotate and move to secrets.

### 18:50 UTC: second restart, not ours

Right after recovery, `viam-agent` found viam-server 1.11.1, downloaded it and
restarted viam-server (18:50:05 "new version found", 18:50:07 "Stopping
viam-server", 18:50:36 "Starting viam-server"). Every module reported
"Stopped module did not exit cleanly". The first route-loop run died here: the
SDK client's auto-reconnect failed with `[Errno 2] No such file or directory`
on its proxy socket, and the runner wrongly counted that as a failed leg.
Runner fixed to rebuild the connection and not count offline time.

Lesson for the field: auto-updates can restart viam-server mid-route. Pin
the viam-server version in agent config during test campaigns.

## Incident 18:54 UTC: pose frozen in yaw, robot spins in place (REPRODUCED LIVE)

First leg of the monitored loop (current spot → kevin-desk, 18.7 m). Within
5 s the goal went to `loc_refine`, then nav commanded a 0.6 rad/s rotate-to-
heading. For the next 50 s:

- `nav.last_drive` issued `viam_angular_z_deg_s = 34.4`, base RTT 3–5 ms.
- Tracer wheel odometry (`sensor_probe.odometry.raw.raw_av_z_deg_s`) = **34.2°/s**,
  `vtheta` = 0.597 rad/s — the base **was** turning.
- SLAM `pose.theta` stayed at **−1.4504322754805317** to the last digit;
  `yaw_rate_deg_s` = 0.
- Tick match scores went from +0.5 to **−0.2 … −0.6** (the scan rotates, the
  pose does not), so the planner saw lethal cost all around the body
  (`path_cost_ahead` 253/254), replan returned "scan+local: no feasible path;
  blocked-corridor: no feasible path; forced-via: none feasible".
- Runner stall detector cancelled at 50 s; the retry did the same thing.
  Operator-side cancel + runner kill at 18:55:59. Robot made several full
  rotations; its true heading is now unknown.

Root cause chain:

1. **Serial port conflict.** `wit` autodetected
   `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` (CH340). The LCUS-2
   relay module (`tray-actuator-retract` / `-extend`, configured with only
   `channel`) opened **`/dev/ttyUSB0` at 9600 baud** at 18:50:45.191 and
   18:50:46.141. The IMU logged "device reports readiness to read but
   returned no data (device disconnected or multiple access on port?)" at
   18:50:45.306, between the two relay opens. Same pattern at 18:48:05–07.
   `wit GetReadings`: `packets: 49`, `gyro_packets: 12`, then nothing;
   `last_error` = that SerialException; `angular_velocity.z = 0`,
   `yaw_deg = 80.87` (the value it also had at 18:33 and 18:54).
   The nav-stack IMU autodetect already knows about this collision
   (`wit_serial._port_is_lcus2_relay`), but the relay module does not know
   about the IMU and both are CH340 `1a86` with no serial number, so
   `/dev/ttyUSB0` and the by-id link can be the same device after a
   re-enumeration.
2. **Driver republishes a dead sample as fresh.** `wit_imu.py` read loop:
   `device.poll()` then `s = device.sample` → `shm.write_sample(...)` every
   period. When the serial stream is dead, `sample` is the last decoded
   packet, but it is written to `/viam-imu-wit` with a **new timestamp**, so
   `imushm.Reader.read_latest(max_age_s=0.5)` never flags it stale.
3. **Engine prefers heading-sensor delta with no cross-check.**
   `engine._predict`: when `odom.heading_rad` is present, `dth =
   heading_delta` (IMU) replaces `odom.vtheta * dt` (wheels) whenever the
   delta is < 40°. Frozen heading ⇒ `dth = 0` even while wheels report
   34°/s. There is no disagreement check and no fallback to wheel yaw.
4. **Nav has no "pose not responding to command" guard.** It kept commanding
   the spin; the footprint guard and planner then reasoned on a scan that
   was rotating under a fixed pose. The code comments describe this exact
   failure as rc23 ("spin commanded, pose frozen, setPower still worked").

**Confirmed on the robot via the shell service (19:04 UTC):**

```
/dev/ttyUSB0  CH340 1a86:7523  usb-0:2  held by lcus-2 module   speed 9600   (relay, correct)
/dev/ttyUSB1  CH340 1a86:7523  usb-0:5  held by nav-stack       speed 9600   (IMU — WRONG, driver opened it at 115200)
/dev/ttyUSB2  CP2102N          usb-0:7  held by nav-stack       speed 1000000 (lidar, correct)
by-id: usb-1a86_USB_Serial-if00-port0 -> ttyUSB1   (only one by-id link for two identical CH340s)
```

The LCUS-2 module's `ports.py` resolves an omitted `serial_path` by listing
CH340 ports and, when several exist, opening each at 9600 to send its
status query. That probe left ttyUSB1's line speed at 9600 under the IMU
driver's open handle. `od` on ttyUSB1 at 9600 returned no bytes.

Also in dmesg: `watchdog: BUG: soft lockup - CPU#12 stuck for 26s!
[viam-server]` at uptime 8915 s ≈ 16:41 UTC (the lidar-stall window), and
Wi-Fi roaming (`wlo1: disconnect from AP ... for new auth`) every few
minutes. Load average 7–8 on the box.

This is a distinct, severe failure mode from the baseline localization
quality issue, and it is what "stuck, tries to reroute, struggles" looks
like from the outside. It recurs on every module restart while the port
conflict exists.

Fixes (in priority order):

- Config (immediate): give `wit` an explicit `serial_path` and give the two
  LCUS-2 relays an explicit port, so they cannot share `/dev/ttyUSB0`.
  Alternatively drop `heading_sensor: wit` so SLAM integrates yaw from the
  Tracer wheel odometry it already reads.
- `wit_imu.py`: only write to shm when `poll()` produced a *new* packet
  (compare `s.packets`), reopen the port after N consecutive errors, and
  expose `age_s` in readings.
- `engine._predict`: when a heading sensor and wheel `vtheta` disagree by
  more than a threshold for more than ~0.5 s, log once and fall back to
  wheel yaw; expose `heading_disagreement` in status.
- Supervisor: if `cmd_vtheta` has been non-zero for > 3 s and
  `|Δpose.theta|` ≈ 0 while odom reports rotation, stop and fail with
  `pose_not_tracking` instead of spinning.

## Incident 19:24 UTC: pose-error deadlock mid-corridor (run 4, leg 3)

Leg 3 (seanp → dock_test, 35.6 m): left the desk cleanly, drove ~25 m south
at 0.6 m/s with match scores up to 0.75, turned east into the aisle at
y≈1.5 and stalled at (-16.3, 2.1) with 13 m to go. Symptoms:

- Tick match score fell from 0.75 to **0.03–0.06**, prior −0.09 … −0.45,
  ray MAE 0.85–1.8 m.
- Periodic relocalize found a better pose: **1.29–1.33 m / 12° away,
  score 0.50–0.54** (and earlier 1.94 m / 84°, score 0.06), but logged
  "refusing large mid-nav jump; continuing on published pose" because a goal
  was active (`nav_loc_refine_apply_max_m` = 1.0 m caps mid-nav fixes).
- nav-camera: the robot footprint sits *inside* a lethal block of the static
  map (the storage-bin / table area); RealSense colour shows open carpet
  ahead. Planner: start cell lethal → `connect_plan_start` escapes,
  `path_cost_ahead` 253/254, states alternate `planning` (12–25 s) /
  `narrow` / `narrow_reverse`; `last_replan_error` empty (plans exist, the
  robot just cannot make progress on them). Runner stall-cancelled at 264 s.
- Guard nearest return: 8 cm off the right flank (body (−0.02, −0.38)) while
  the camera showed bins on the left. Briefly suspected a mirrored scan;
  after the pose correction the robot sits just north of a separate map
  block to its right-rear, so the return is consistent and the mirror
  hypothesis is **unlikely** (25 m straight runs with low cross-track would
  not work mirrored). Cheap confirmation when convenient: park beside a wall
  on one side and compare `near_by` sign to reality.
- nav-camera: the heading arrow appears drawn with its vertical component
  flipped relative to the grid (robot heading north at kevin-desk was drawn
  pointing down). Viz-only bug in `viz/nav_view.py`; worth fixing because it
  misleads anyone reading the stream.

Deadlock: nav holds the goal → SLAM refuses >1 m corrections during nav →
nav cannot plan from a lethal start → nobody yields. The runner's
`check_localization` recovery ran before `cancel` had released
`navigation_active`, so SLAM skipped it (null result).

Fixes:
- Runner: after cancel, wait for `active=false`, then `check_localization`
  up to 3× (idle-mode large jumps need two agreeing matches), log the
  applied shift, then retry.
- Stack: when `path_cost_ahead` is lethal at the *start* for > N s and SLAM
  has a pending refused large jump, nav should suspend (release
  `navigation_active`), let SLAM confirm/apply, replan, resume. Today
  `periodic_relocalize_during_navigation=false` + the 1.0 m cap make this
  unreachable.

## Incident 20:01 UTC: pinned by a return inside its own body (run 7)

Run 7 (branch build) never moved. Both goals failed in place; the runner
aborted after two consecutive failures. `get_trace` showed one guard point at
body coordinates **(0.008, 0.285)** every tick: 8 mm forward of centre, 1 cm
*inside* the left flank (half-width 0.295), distance 0, rock steady — a
static map cell (desk edge) overlapping the body by a centimetre after the
run-6 fallback leg parked there. RealSense: open corridor ahead.

Why it pinned — **corrected 2026-10-08 by the PR review.** The first
write-up blamed `simple_motion.corridor_min_range` reading the in-body
return as an obstacle 8 mm ahead. That is not the live code path: with the
`FootprintGuard` on (default) `compute_path_command` returns from
`_guarded_command` before the reactive layer runs, and all 89 `avoid` ticks
in the run-7 log report `forward_clearance_m = 0.36` — the guard's
`0 + length/2`, never ≈ 0. Lidar returns are also cropped to the body at
ingestion (`prepare_lidar_point_cloud`), so the point was a static lethal
map cell in the guard's point set, not a scan return. The real mechanism is
the guard's near-floor rule (`footprint_guard._split` / `_hits`): points
inside the body are dropped, but the next cell of the same desk edge sits
2.5 cm past the nose with floor `max(min_gap 0.02, dist − slack 0.01)`, so
every forward or rotating motion brings a corner inside that floor and is a
hit (`rotation_blocked`, state `blocked`). Only reversing is free, and the
backup recovery did not fire. Scratch reproduction: a 5 cm cell line 1 cm
inside the left flank that extends past the nose → `free_distance 0`,
`blocked`; the same line ending at the nose → `clear`.

Fix status: the branch briefly carried in-body masking for the reactive
layer (`corridor_min_range` and friends). It was dropped from PR #68 after
the review because that layer does not run with the guard on, so it could
not fix this incident; if the simple go-to path ever shows the symptom it
can return together with the guard fix (backlog item 15). Run 8 left the
pinned pose after the reload because the process restarted at a slightly
different pose with the relaxed backup knobs. Config relaxations applied to
tracer2a at 20:05: `backup_speed_mps 0.2`, `backup_dist_m 0.5`,
`backup_cooldown_s 1`, `backup_max_attempts 3`,
`local_planner_max_vel_x_mps 0.35`, `clearance_preference_m 0.25`.

## PR #68 adversarial review (2026-10-08)

Four independent reviewers (supervisor policies, planner/plumbing,
body-clearing geometry, harness/tests); every finding below was re-verified
by hand before being fixed. Regression tests: `tests/test_review_regressions.py`,
`tests/test_route_loop_guards.py`.

Confirmed and fixed on the branch:

- Goal-blocked finish keyed on any `"no feasible path"`, which the
  blocked-corridor paint produces by design → a person in a doorway 2 m from
  the goal "succeeded" the goal after 15 s and the route layer advanced.
  Now only goal-side verdicts count (`goal blocked`, `goal pose is in
  lethal`), the verdict must come from a replan in the last 5 s, and moving
  more than 0.3 m since the timer armed starts a new episode.
- A cancel landing during the initial plan (or an Nth failed replan) ended as
  `failed: planning aborted` and failed the whole route. `_set_status` now
  reports `canceled` whenever the cancel event is set;
  `connect_plan_start` keeps the abort verdict instead of relabelling it
  "cannot reach plan start".
- Localization yield: the 2 s inter-attempt pause ignored cancel (now
  `cancel.wait`); default flipped to **off** until it has a live run. Each
  SLAM round trip can still block the control thread up to the 20 s RPC
  timeout (backlog 16).
- Replan budget restarted on the detour-ban-lift recursion (shared now).
- Harness: goals were sent with no localization gate and re-sent after a
  failed check; crash exit left the goal driving; a nav-only status error
  read as idle; a goal could be stacked on an active one; instant successes
  were classed `never_active`. All fixed (`--min-tick-score`, unconditional
  cancel in `finally`, `nav_idle`, `_ensure_nav_idle`, stall exemptions for
  loc holds).
- Body clearing (reactive-layer in-body masking) removed from the PR: dead
  code on the guard path, no live evidence on the paths where it runs.

Not a PR problem, but the cause of the 16:23 pin and the evening's lost
heading: the tracer2a config change `nav_loc_refine_apply_max_m: 2` (default
1.0) let nav force-apply a 1.19 m / 30° local match at score 0.32 while
stationary. Revert to 1.0 before the next laps; the yield code replaces it.

## Incident 20:23 UTC: pinned beside kevin-desk after a forced 1.2 m loc correction (run 8)

- 20:23:05 kevin-desk reached (0.89 m, `succeeded`). 20:23:26 seanp leg: nav
  briefly unavailable (`resource not initialized yet`, ~5 s, leg ended
  `unknown`); runner retried at 20:23:46.
- 20:23:49 periodic relocalize: `corrected via local (shift=1.19 m, 30.0 deg,
  jump=forced)` while the robot sat still against the desk. The new pose
  (-25.39, 17.50, -2.39) puts the desk's map cells under the footprint:
  `path_cost_ahead` 39-43, `nose_clear=False`, `spin_blocked=True`, trace
  nearest point at body (0.279, 0.116) with `near_m=0` (inside the rectangle),
  23-35 depth-memory points.
- Replan ping-pong: `local_blocked` -> `_try_replan` returns a 2-point path ->
  `local_blocked` again next tick. 40+ `planning` ticks per leg with
  `vx=w=0`. Backup / narrow-reverse never fired because every replan
  "succeeded". Three legs stalled on the runner's 45 s guard, the fallback to
  kevin-desk "succeeded" at 0.32 m without moving, loop aborted 20:27:34.
- RealSense colour frame: desk leg ~10-15 cm off the front-left corner. Tight,
  but not touching; the pose jump is what put the body "inside" the desk.
- Open (code): (a) footprint on lethal/inflated cells + replans that succeed
  without progress should trigger the backup recovery, not another replan
  (`_recover_unreachable_start` only handles an unreachable start cell);
  (b) refuse a forced loc correction that lands the footprint in occupied
  cells while the robot is stationary; (c) confirm the local costmap drops
  in-body points the way `FootprintGuard._hits` does (`nearest()` still
  reports them, which is fine for the trace).

## Run 9 (2026-10-08 12:23 local): review build, tight-space tuning

Branch head `b6e14a9` reloaded 12:22 (reload_time 16:22:28Z). Loop: seanp,
dock_test, kevin-desk x2 with the gated harness (`--min-tick-score 0.2`).

- External commands during the run: `navigate` to **charge** at 12:23:31
  (6 s before the loop's first goal) and `cancel` at 12:23:42 and 12:27:45.
  None came from the harness (no `leg_cancel` event; the supervisor only
  sets its cancel flag via `request_cancel`). Source unconfirmed: app,
  delivery Slack bot, or the docking service rebuilding after the reload.
- Blocked seanp approach (people in the corridor), 4.5 m out, 127 s with
  zero motion: states alternated ~10 s `planning` / 8 s `wait`; backup and
  narrow-reverse never fired. 64 of 79 replan cycles ended with
  "blocked-corridor: skipped (replan budget 3.0s exhausted)" — one scan+local
  plan already exceeds 3 s on this map, so the forced-via detours ran in only
  12 cycles. The goal-blocked finish correctly did **not** fire (plain "no
  feasible path" no longer counts; 4.5 m > `goal_blocked_accept_m`).
- Why the unstick is slow: from the blocked-nose branch, reverse needs the
  rear cone >= 0.445 m *and* `reverse_path_clear` in the inflated local
  costmap; turn needs the 0.52 m spin disc clear; the main `backup` maneuver
  only engages while DWA is active and spinning in place. In a tight spot all
  three are vetoed by the same inflation that caused the block.
- Why it stops at gaps it fits through: hard clearance radius = inscribed
  0.295 + `clearance_m` 0.07 = 0.365 m, so a gap must be ~0.73 m (+ cell +
  noise, ~0.8 m) to cost < 253 while the body is 0.59 m wide;
  `local_planner_activate_cost` 200 then declares `local_blocked`.

**Config change applied to tracer2a at 12:33 local (navigation `builtin`):**
`replan_budget_s: 10` (was default 3.0) so the corridor paint and forced-via
detours actually run; `backup_stuck_time_s: 1.5` (was default 3.0).
Reversible; both are nav-only knobs.

Code follow-ups (not in PR #68): (a) blocked-nose branch may use the backup
maneuver after two failed replans regardless of DWA state; (b) judge the
reverse with the footprint guard's rectangle sweep instead of the inflated
costmap; (c) when the guard's straight free distance covers the gap and the
nose cone is clear, crawl through a cost-253 corridor instead of stopping;
(d) budget the first replan attempt separately from the escalations.

## Runs 10-11 (2026-10-08 13:03-13:18 local): unstick builds

Build `315a6ae` (guard-sweep reverse, yaw away) then `d63a72a` (unstick
before the next replan). Both loops cancelled/aborted early; no lap completed.

- **External cancels again.** seanp attempt 1 of run 11 flipped to
  `canceled` at 13:15:36 (140 s in) with no harness `leg_cancel` and no other
  navigate in the robot log. Third time across runs 9 and 11. Source still
  unconfirmed; find it before unattended runs.
- **Why the first unstick build never reversed at seanp (run 10, 5 min, loop
  timeout):** with the nose blocked the local replan cooldown was 0 and each
  replan took 5-10 s, so the control thread was always inside `_try_replan`;
  the unstick lives in the cooldown wait and never got a tick. Fixed in
  `d63a72a`: a replan that leaves the robot in place with a blocked nose sets
  `unstick_pending`, which holds the next replan until the reverse + yaw has
  run (or reported impossible). Run 11 then showed one reverse and a 90 deg
  yaw in the pocket, still interleaved with 5-10 s replans.
- **Reactive wedge (run 11, 2 min, no motion):** rear-left corner 9 cm off
  the desk row, guard `avoid`, nose blocked, spin blocked, path cost 32 — the
  costmap-driven blocked branch (and its unstick) never engages below
  `local_planner_activate_cost`, and the bumper reverse did not fire. Both
  dock_test attempts stalled without moving; loop aborted. Fix (uncommitted
  at the time of the stop): the reactive `avoid` wedge arms the same
  unstick after `backup_stuck_time_s`; the unstick overrides the follower
  command while pending; abandoned only after the nose has read clear for
  0.5 s; at most `backup_max_attempts` rounds per 2 m area, then the normal
  bounded-retry backoff decides. Tight-space simulations (dead end, person in
  hallway for 2 s / 15 s) pin the timing.
- **Planning time is the remaining cost.** 105 of 140 one-second samples in
  the run 11 seanp approach were `planning`; a "no feasible path" verdict
  floods the 2.1 M-cell grid and takes 5-10 s on the robot. Per-attempt
  planning times are now appended to the replan reasons; a search budget in
  the planner is the next lever (backlog 19).
- Trace now carries `rear_free_m` (guard sweep backward), `spin_blocked`
  and the unstick state, so the next wedge can be read from `get_trace`.

Robot left idle at 13:18 (goal cancelled); teammates took the robot after.

## Findings log

- 2026-10-07 19:59 — **Deploy path that works:** `viam module reload-local
  --file module.tar.gz --part-id <part>` (CLI ≥ 1.11). It ships the locally
  built tarball (`./build.sh`, 110 files, no bytecode) over the Viam
  connection, unpacks under
  `/root/.viam/packages-local/data/module/synthetic-viam-labs_nav-stack-0_0_0/`,
  runs `setup.sh` there (venv Python 3.12.3, viam-sdk 0.84.0), sets
  `reload_enabled/reload_path` on the machine's module entry and restarts
  the module. Took 16 s; SLAM/nav back by 19:59:43. Verified via shell: new
  functions present in the deployed `supervisor.py`. To revert, drop
  `reload_enabled`/`reload_path` from the module entry (registry
  `latest-with-prerelease` resumes). Run 7 (branch build) started 20:01.
- 2026-10-07 19:58 — Second reload (CLI 1.11.1, no stale tarball) failed
  identically: the builder's "already built" heuristic keys on the
  `entrypoint` file existing in the source tree, so `viam module reload`
  cannot deploy this module without restructuring it (e.g. entrypoint under
  a build-generated `dist/`). Deploy path for the test build is therefore
  the team's registry prerelease flow.
- 2026-10-07 19:55 — First `viam module reload` failed in the cloud builder:
  "Check if module is already built: built: executable entrypoint run.sh →
  Skipping build … Skipping module tarball (pre-built module)" → nothing to
  ship → "Reloading module failed". The builder treats a module whose
  `entrypoint` already exists as a file in the source tree as pre-built,
  which is every plain-script Python module. Local CLI was 1.6.0 (builder
  1.11); upgraded to 1.11.1 and retried. Fallback if reload cannot work for
  this layout: the team's existing path, `./build.sh` +
  `viam module upload --version 1.0.60-rc.1 --platform linux/any`, which the
  machine picks up via `latest-with-prerelease`. The failed reload did not
  change the machine's module entry; the module restarted at 19:55:09 and
  was back by 19:55:50.
- 2026-10-07 19:55 — Branch `fix/localization-yield-deadlock` implements
  backlog items 7b (goal-blocked finish), 7c (cancel interrupts replan +
  `replan_budget_s`) and the localization yield (nav flags
  `localization_yield`, `runtime.any_navigation_active` treats it as idle,
  SLAM applies, nav replans). 10 new tests in
  `tests/test_recovery_policies.py`; existing suite green except the 4
  known-stale framesystem tests. `meta.json` gained a `build` block so
  `viam module reload` can cloud-build via `build.sh`. Reload to tracer2a
  started 19:55. Run 6 (config knobs only) before reload: seanp retry
  succeeded 0.37 m, dock_test 106 s / 0.11 m, kevin-desk 200 s / 0.71 m,
  then seanp stalled twice ~9 m out with 60 `wait` ticks (aisle occupied).
- 2026-10-07 19:38 — Run 6 leg 1 (kevin-desk → seanp) with the new knobs:
  stalled at 4.0 m then 2.4 m from seanp with **good** localization (score
  0.71–0.73). Static-map-only preview: feasible, 2.5 m. With live scan +
  local costmap painted: "no feasible path" on all three strategies. So at
  seanp the *approach corridor* between desk and chair is sealed by live
  hits + 0.365 m inflation, not just the goal cell; widening
  `max_goal_snap_m` cannot help when the snapped cell is unreachable. Also:
  `cancel` took >10 s to take effect while the supervisor was inside
  back-to-back replans (84 planning ticks; control tick EMA 172 ms vs 50 ms
  period). Cancel latency during replan storms is its own issue.
- 2026-10-07 19:35 — Config mitigations applied to `navigation` (operator
  approved): `builtin.nav_loc_refine_apply_max_m: 2.0` (was 1.0; would have
  let the mid-nav refine take the 1.29 m / score 0.54 correction) and
  `builtin.max_goal_snap_m: 1.0` (was 0.5; chair-on-goal case). Run 5
  before the change: dock_test reached in 40 s / 0.09 m once the deadlock
  was released; kevin-desk again stalled at 2.2 m then succeeded on retry.
  Branch `fix/localization-yield-deadlock` created for the robustness items
  (yield-to-SLAM, arrived-nearby outcome, stale-IMU rejection); no code on
  it yet.
- 2026-10-07 19:27 — Deadlock confirmed: runner killed + `cancel` sent at
  19:27:07; within seconds (no manual relocalize) SLAM's idle-mode tick
  matcher applied the pending correction on its own. `check_localization`
  at 19:27:40 returned status ok, score 0.58, ray MAE 0.48 m, tick score
  0.66, shift 0. Nav releasing the goal was the only thing needed.
- 2026-10-07 19:20 — Run 4 leg 2 (→ seanp, 9.7 m): clean drive, then 3 min
  stopped 0.64 m short. Replan error was explicit every cycle: **"goal snap
  0.52–0.59 m exceeds max_goal_snap_m=0.50 (goal blocked / over-inflated)"**.
  RealSense colour frame shows an office chair with a jacket over it right at
  the robot's nose; the saved `seanp` pose is under/at that chair. Live hits
  + 0.365 m inscribed radius put the goal cell in lethal; nearest free cell
  was 2–9 cm beyond the snap limit, so the planner refused for 177 s instead
  of stopping short. Runner stall-cancel + retry then "succeeded" at 0.49 m
  (snap landed inside 0.50 m). This is the canonical "cannot finish" case:
  the fix is a goal-blocked policy (stop short and report, or widen snap on
  the final approach), not more replanning. Localization was fine (ray MAE
  0.49–0.61 m) throughout.
- 2026-10-07 19:16 — Run 4 leg 1 (→ kevin-desk, 19 m): drove 17 m at
  0.6 m/s with yaw tracking normally (port fix holds). Final 2 m: path cost
  254, "scan+local/blocked-corridor/forced-via: no feasible path" ×3 cycles,
  stall-cancelled at 115 s; retry succeeded in 101 s via slow/avoid/narrow/
  narrow_reverse creeping (56 planning ticks). Localization *improved* near
  the desk (lc score 0.72, ray MAE 0.56 m). So this goal is a tight-approach /
  clearance problem, not a localization one. Slack posted to
  #build-coffee-delivery. Data: `.local/monitor/loop-20261007-151151.*`.
- 2026-10-07 19:12 — Permanent port fix applied to tracer2a config (operator
  approved): `tray-actuator-retract` and `tray-actuator-extend` →
  `serial_path /dev/serial/by-path/pci-0000:00:14.0-usb-0:2:1.0-port0`;
  `wit` → `serial_path /dev/serial/by-path/pci-0000:00:14.0-usb-0:5:1.0-port0`,
  `serial_autodetect false`. Location `kevinj` renamed back to `kevin-desk`
  (same pose). Previous values recorded in the config API response.
- 2026-10-07 19:06 — IMU restored without a restart: `stty -F /dev/ttyUSB1
  115200` via the shell service. `wit` packets 49 → 912 within a minute,
  yaw 80.87° → 12.55° (−68°); the lidar re-seed had moved pose yaw by −70°,
  so IMU and lidar agree. SLAM heading tracking again. Permanent fix
  (explicit `serial_path` by-path on relays + wit) still to apply.
- 2026-10-07 18:58 — Pose repaired after the spin with `set_initial_pose
  {x:-32.219, y:0.035, theta:-1.45, refine:true}` (360° yaw sweep within
  1 m): applied θ=−2.672 (was −1.450), score 0.497, hit_rate 0.82, ray MAE
  1.05 m, ambiguous vs second-best 0.468. IMU still dead; loop NOT resumed
  until the port conflict is fixed.
- 2026-10-07 18:54 — Yaw-freeze failure reproduced live and root-caused to
  the LCUS-2 relay / WitMotion CH340 port conflict + driver + engine gaps
  (see incident section). Run-2 data: `.local/monitor/loop-20261007-145305.*`.
- 2026-10-07 18:2x — 96% reject figure debunked (stationary interval, +988
  rejects, +0 accepts, pose unchanged). Quality metrics still below
  thresholds all session.
- 2026-10-07 — 18:21:27 engine restart traced to a `start_localizing`
  DoCommand issued during/just after a goal; startup localize never
  converged (ambiguous peaks) and was bypassed after 100 s.
- 2026-10-07 — Live framesystem lidar mount resolves to θ=−π and the robot
  navigates with it, matching what viam-sdk ≥ 0.80 produces locally. The 4
  failing `tests/test_viam_frames.py` cases encode the opposite sign and are
  stale, not a live bug.

## Improvements backlog

Observability (do first, cheap):

1. Log every failed replan's `attempts` list at WARN; keep `last_replan_info`
   until the next replan, not the next goal; log goal outcome
   (succeeded/failed/canceled + error) from the nav model.
2. Split `match_rejects` into `match_no_candidate` (refine_pose returned
   None: no improvement over prior), `match_gated` (apply refused), and keep
   `match_accepts`. Add a rolling mean of `last_match_score` while moving.
3. Dump the trace ring to disk on goal failure.

Localization:

4. Measure with the monitor (above). If score is flat ~0.45 everywhere while
   parked, re-map or verify `new_map` against the current layout.
5. Investigate why every full-map peak is ambiguous: 2D 9 m lidar in a
   72 × 74 m self-similar building. Candidates: longer `max_range`, use the
   RealSense band for matching (currently `obstacles_only`), keyframe
   matching, or a different tie-break than `second_best`.
6. Odometry: 51 `gap_hold_events`, `has_twist=false`. Check the Tracer odom
   publish rate and whether gaps coincide with drift.
7. Guard `start_localizing` while a goal is active (or make nav suspend and
   resume cleanly around it).

Navigation:

7b. **Goal-blocked policy ("close but cannot finish").** Today arrival needs
   the robot within `xy_goal_tolerance` (0.25 m) of the requested goal, or at
   the end of a path whose snapped endpoint is within `max_goal_snap_m`
   (0.5 m) of it. If the nearest free cell is farther than 0.5 m (chair on
   the goal, as at seanp 19:18), `plan_on_costmap` refuses with "goal snap
   … exceeds max_goal_snap_m (goal blocked / over-inflated)", the supervisor
   loops stop/replan, and the goal eventually fails as "replan failed (path
   blocked)" or times out. There is no "arrived nearby" outcome. Proposed:
   when the snap limit is the *only* failure and the robot is within a
   configurable `goal_blocked_accept_m` (≈1.0 m), drive to the nearest free
   cell, stop, and finish with state `succeeded` plus `goal_offset_m` and
   `goal_blocked: true` in status (or a distinct `arrived_near` state), so
   routes continue instead of churning for minutes. Short-term config
   mitigation: `builtin.max_goal_snap_m: 1.0`.

7c. **Cancel must interrupt a replan storm.** `NavSupervisor` checks
   `self._cancel.is_set()` once per control tick (supervisor.py ~1621) and
   nowhere else. `_try_replan` runs up to 2 filtered attempts plus
   `_forced_side_detour` (6 vias × 2 legs = up to 14 full-map plans) with no
   cancel check between them, and `planner.plan_path` / Lazy Theta* have no
   abort hook. Live 19:37: `cancel` took > 10 s to register (84 planning
   ticks, control-tick EMA 172 ms vs 50 ms period). The base itself stops
   immediately (navigator `cancel()` calls `world.stop()`), only the goal
   state lags. Fix (code): check the cancel event before every `plan()` call
   in `_try_replan`, `_forced_side_detour` and `_recover_unreachable_start`
   and return early; thread an optional `should_abort` callable into
   `plan_path` → `_search` checked every ~2k expansions; cap the replan
   budget per cycle (e.g. 2 s or 3 full plans) and skip the via search when a
   cancel is pending.

8. `clearance_m=0.07` leaves no slack for pose error; evaluate 0.12–0.15 vs
   the narrowest doorway on the route.
9. Consider a latency/quality-aware speed cap: slow down when
   `tick_match_score` < 0.35 instead of driving 0.6 m/s on a bad pose.

Platform:

10. Move the CAN adapter off the lidar's USB hub (module asks for this on
    every start); the 16:36 nine-minute lidar stall could not be reset.
11. `nav-cam` build KeyError on custom action API dependency (SDK
    dependency resolution).
12. grpclib crash at 17:32 (stream reset KeyError) — check if 1.0.59 still
    has it.
13. Pin `viam-sdk` (needs ≥ 0.80 for `viam.spatialmath`), fix the 4 stale
    framesystem tests, declare pytest / pytest-asyncio.

Navigation (from run 8):

14. Departure pin after a loc jump at a desk: back up when the footprint sits
    on lethal/inflated cells and replans keep succeeding without progress;
    refuse loc corrections that land the footprint in occupied cells.
15. Footprint guard near-floor pin (run 7, real cause): a static cell just
    past the nose gets a floor below the first motion step, so forward and
    rotation are both hits and only reverse is free. Either exempt static map
    cells that are within `min_gap_m` of the body at the start pose from the
    near-floor rule (they can be passed alongside, like in-body points), or
    make the blocked state trigger the backup recovery directly.
16. Loc yield: bound each `check_localization` round trip to the remaining
    `loc_yield_wait_s` (pass a timeout through `WorldIO.check_localization`)
    and extend the goal deadline by the yield time.
17. tracer2a: revert `nav_loc_refine_apply_max_m` to 1.0 (see review).
18. Tight-space recovery (run 9): backup from the blocked-nose branch,
    guard-based reverse check, crawl through passable cost-253 gaps, and a
    replan budget that does not starve the escalations. Live interim:
    `replan_budget_s 10`, `backup_stuck_time_s 1.5` on tracer2a.
19. Planner search budget: bound the expansion count (or plan within a
    window around start/goal) so "no feasible path" returns in < 1 s instead
    of flooding the map; use the new per-attempt timings in the replan
    reasons to size it.
20. Crawl through passable cost-253 gaps when the guard's straight free
    distance covers the gap and the nose cone is clear (the "it fits but
    stops" case), instead of stop-replanning.
