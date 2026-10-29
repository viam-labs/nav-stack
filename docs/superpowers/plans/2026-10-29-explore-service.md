# Explore Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `viam-labs:nav-stack:explore` generic service that drives a robot around an unmapped office for a caller-supplied time budget so the builtin SLAM maps it, then saves the map and returns home.

**Architecture:** Pure-Python exploration logic under `src/explore/` (grid analysis, scoring, circuits, leg execution, run state machine) that talks to SLAM and navigation only through two small client protocols. The Viam model in `src/models/explore.py` wraps the SLAM and navigation gRPC dependencies in those clients and exposes `start` / `stop` / `status` DoCommands. Tests drive the logic with fakes built on the existing `SimWorld` raycaster, so no viam-server is needed.

**Tech Stack:** Python 3.13, viam-sdk, numpy, pytest. Existing modules: `src/sim/world.py` (`SimWorld`, `SimMap`), `src/geom/conversions.py` (`Pose2D`), `src/slam_client.py` (`parse_get_grid`).

**Spec:** `docs/superpowers/specs/2026-10-29-explore-service-design.md`

## Global Constraints

- Model name `viam-labs:nav-stack:explore`, API `rdk:service:generic`.
- Config attributes and defaults: `slam_service`, `navigation_service`, `base` (required); `max_linear_mps` 0.4; `max_angular_dps` 30; `home_tolerance_m` 0.5; `structure_fraction` 0.15; `revisit_fraction` 0.1; `well_seen_m` 4.0; `keyframe_max` 2000.
- Lidar range used for scoring raycasts: 8 m. Score = expected new cells / plan length.
- Leg timeout: `max(20, 3 * planned_length_m / max_linear_mps)` seconds. A leg fails if it ends more than 1 m from its goal.
- Blacklist radius 1 m, TTL 600 s. Five consecutive failures: wait 30 s. Twenty: finish early.
- Lost pose or unhealthy SLAM: wait up to 60 s, then finish.
- Phase 1 ends at its time share or at 60% of `keyframe_max`, whichever is first. Inner circuits are at least 12 m of path. Pause 2 s facing the nearest wall between structure legs, 1 s after fill legs.
- Finish order: cancel, SLAM `optimize`, SLAM `save_map`, planned navigate home (retry once), stop in place.
- `start` refuses unless SLAM mode is `mapping` and nav `limits.max_angular_rad_s` is at or below `max_angular_dps`.
- Units: DoCommand payloads are meters and radians. Grid cells: -1 unknown, 0 free, 100 occupied.
- Commit messages are plain imperative sentences in the repo's style (e.g. "Add simulated base model for hardware-free navigation"), ending with the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer.
- Tests run with `.venv/bin/python -m pytest` from the repo root. Code comments are minimal; code should self-document.

## Review Focus

1. A grid where the robot's cell is surrounded by unknown (first seconds of mapping): `frontier_goals` must return goals on known free cells only. Test in Task 2.
2. Nav `get_status` with no `pose` key or with `localization_check.status` untrusted: the run must enter the lost-pose wait, never issue a leg. Test in Task 7.
3. `stop` arriving mid-leg: the finish sequence runs exactly once and `status` reports `finishing` then `done`. Test in Task 7.
4. Keyframe cap reached before phase 1's time share: phase 1 ends early and phase 3 is skipped. Test in Task 7.
5. `start` while SLAM is in `localizing` mode or nav angular limit is 1.5 rad/s (the live robot's current config): `start` returns an error naming the cause and nothing moves. Tests in Tasks 7 and 8.

---

### Task 1: Dev environment, config, and model skeleton

**Files:**
- Modify: `src/config.py` (append after `ExternalNavConfig`, ~line 1580)
- Create: `src/models/explore.py`
- Modify: `src/main.py:11-19, 27-35`
- Modify: `meta.json` (models list)
- Modify: `src/models/nav_core.py:900-918` (`get_status` block)
- Test: `tests/test_explore_config.py`, `tests/test_nav_status_limits.py`

**Interfaces:**
- Produces: `ExploreConfig` dataclass with fields `slam_service: str`, `navigation_service: str`, `base: str`, `max_linear_mps: float = 0.4`, `max_angular_dps: float = 30.0`, `home_tolerance_m: float = 0.5`, `structure_fraction: float = 0.15`, `revisit_fraction: float = 0.1`, `well_seen_m: float = 4.0`, `keyframe_max: int = 2000`; `from_dict(cls, d: Mapping) -> ExploreConfig`; `required_dependencies(self) -> List[str]` returning the three names in order.
- Produces: `ExploreService(Generic)` with `MODEL`, `new`, `validate_config`, `reconfigure`, `do_command`. `do_command` returns `{"error": "unknown command"}` for unknown commands and `{"phase": "idle"}` for `status`.
- Produces: `nav_limits_for_status(cfg: NavConfig) -> dict` and a `"limits"` key in nav `get_status` = `{"max_linear_mps": float, "max_angular_rad_s": float}`.

- [ ] **Step 1: Create the venv**

Run: `PYTHON=/opt/homebrew/opt/python@3.13/bin/python3.13 ./setup.sh && .venv/bin/pip install -q pytest && .venv/bin/python -m pytest tests/test_config.py -q`
Expected: existing config tests pass.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_explore_config.py
import pytest
from src.config import ExploreConfig

def test_explore_config_defaults():
    cfg = ExploreConfig.from_dict({"slam_service": "slam", "navigation_service": "navigation", "base": "base"})
    assert cfg.max_linear_mps == 0.4
    assert cfg.max_angular_dps == 30.0
    assert cfg.home_tolerance_m == 0.5
    assert cfg.structure_fraction == 0.15
    assert cfg.revisit_fraction == 0.1
    assert cfg.well_seen_m == 4.0
    assert cfg.keyframe_max == 2000
    assert cfg.required_dependencies() == ["slam", "navigation", "base"]

def test_explore_config_requires_names():
    with pytest.raises(ValueError):
        ExploreConfig.from_dict({"slam_service": "slam"})
```

```python
# tests/test_nav_status_limits.py
from src.config import NavConfig
from src.models.nav_core import nav_limits_for_status

def test_nav_limits_from_config():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "base", "max_linear_velocity": 0.5})
    assert nav_limits_for_status(cfg) == {"max_linear_mps": 0.5, "max_angular_rad_s": 1.5}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_config.py tests/test_nav_status_limits.py -q`
Expected: FAIL with ImportError on `ExploreConfig` and `nav_limits_for_status`.

- [ ] **Step 4: Implement `ExploreConfig` in `src/config.py`**

Follow the `NavCameraConfig` dataclass pattern at `src/config.py:1581`. `from_dict` raises `ValueError` naming the missing key when any of the three names is absent or empty.

- [ ] **Step 5: Implement `nav_limits_for_status` in `src/models/nav_core.py` and add `status["limits"] = nav_limits_for_status(cfg)` inside the `if cmd == "get_status":` block next to `robot_radius_m`**

Values come from `cfg.max_linear_velocity` and `cfg.max_vel_theta`.

- [ ] **Step 6: Implement the `ExploreService` skeleton in `src/models/explore.py`**

Subclass `viam.services.generic.Generic`. `MODEL = Model(ModelFamily("viam-labs", "nav-stack"), "explore")`. `validate_config` returns `(cfg.required_dependencies(), [])`. `reconfigure` stores `self._cfg`, `self._slam = cast(SLAM, dependencies[SLAM.get_resource_name(cfg.slam_service)])`, `self._nav = cast(Navigation, dependencies[Navigation.get_resource_name(cfg.navigation_service)])`, `self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])`. Register at module bottom exactly like `src/models/nav_camera.py:183`, with `Generic.API`. Import and `add_model_from_registry` in `src/main.py`. Add the model to `meta.json` with short description "Autonomous exploration drive for mapping: frontier and revisit goals under a time budget."

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_config.py tests/test_nav_status_limits.py -q && .venv/bin/python -c "import src.main"`
Expected: PASS, and the import succeeds.

- [ ] **Step 8: Commit**

```bash
git add src/config.py src/models/explore.py src/main.py meta.json src/models/nav_core.py tests/test_explore_config.py tests/test_nav_status_limits.py
git commit -m "Add explore service skeleton and expose nav velocity limits in status"
```

---

### Task 2: Grid analysis: frontiers, visibility gain, under-seen cells

**Files:**
- Create: `src/explore/__init__.py` (empty), `src/explore/grid.py`
- Test: `tests/test_explore_grid.py`

**Interfaces:**
- Produces: `OccGrid` dataclass: `cells: np.ndarray` shape (rows, cols) int8, `resolution: float`, `x_min: float`, `y_min: float`; `from_get_grid(cls, resp: Mapping) -> Optional[OccGrid]` (uses `src.slam_client.parse_get_grid`); `to_cell(self, x, y) -> tuple[int, int]` returning (row, col); `to_world(self, row, col) -> tuple[float, float]` returning the cell center; `known_free_count(self) -> int`.
- Produces: `Candidate` dataclass: `x: float`, `y: float`, `kind: str` (`"frontier"`, `"under_seen"`, or `"circuit"`).
- Produces: `frontier_goals(grid: OccGrid, robot_radius_m: float, min_cluster_cells: int = 6) -> list[Candidate]`.
- Produces: `visible_new_cells(grid: OccGrid, x: float, y: float, max_range_m: float = 8.0, n_rays: int = 72) -> int`.
- Produces: `under_seen_goals(grid: OccGrid, visited: Sequence[tuple[float, float]], well_seen_m: float, spacing_m: float = 2.0) -> list[Candidate]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_grid.py
import base64
import numpy as np
from src.explore.grid import OccGrid, frontier_goals, visible_new_cells, under_seen_goals

def room(rows=60, cols=80, res=0.05):
    return OccGrid(cells=np.full((rows, cols), -1, dtype=np.int8), resolution=res, x_min=0.0, y_min=0.0)

def enclose(g):
    g.cells[20:40, 20:40] = 0
    g.cells[19, 19:41] = 100; g.cells[40, 19:41] = 100
    g.cells[19:41, 19] = 100; g.cells[19:41, 40] = 100

def test_from_get_grid_roundtrip():
    cells = np.zeros(12, dtype=np.int8); cells[5] = 100; cells[0] = -1
    resp = {"rows": 3, "cols": 4, "cellSize": 0.05, "xMin": -1.0, "yMin": 2.0,
            "data": base64.b64encode(cells.tobytes()).decode()}
    g = OccGrid.from_get_grid(resp)
    assert g.cells.shape == (3, 4) and g.cells[1, 1] == 100 and g.cells[0, 0] == -1
    assert g.to_cell(-1.0 + 0.075, 2.0 + 0.025) == (0, 1)
    assert g.to_world(0, 1) == (-0.925, 2.025)
    assert OccGrid.from_get_grid({"available": False}) is None

def test_frontier_goals_sit_on_known_free_cells():
    g = room(); g.cells[20:40, 20:40] = 0
    goals = frontier_goals(g, robot_radius_m=0.25)
    assert goals
    for c in goals:
        r, col = g.to_cell(c.x, c.y)
        assert g.cells[r, col] == 0

def test_frontier_goals_ignore_walls():
    g = room(); enclose(g)
    assert frontier_goals(g, robot_radius_m=0.25) == []

def test_frontier_goals_drop_small_clusters():
    g = room(); enclose(g)
    g.cells[19, 30] = 0
    assert frontier_goals(g, robot_radius_m=0.25, min_cluster_cells=6) == []
    assert len(frontier_goals(g, robot_radius_m=0.25, min_cluster_cells=1)) == 1

def test_visible_new_cells_counts_unknown_only():
    g = room(); g.cells[20:40, 20:40] = 0
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) > 0
    g.cells[:, :] = 0
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) == 0

def test_visible_new_cells_blocked_by_walls():
    g = room(); enclose(g)
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) == 0

def test_under_seen_goals_far_from_visits():
    g = room(); g.cells[:, :] = 0
    goals = under_seen_goals(g, visited=[(0.5, 0.5)], well_seen_m=1.0, spacing_m=1.0)
    assert goals and all(np.hypot(c.x - 0.5, c.y - 0.5) > 1.0 for c in goals)
    assert under_seen_goals(g, visited=[(2.0, 1.5)], well_seen_m=10.0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_grid.py -q`
Expected: FAIL with ModuleNotFoundError `src.explore`.

- [ ] **Step 3: Implement `src/explore/grid.py`**

Frontier cells are free cells with a 4-neighbor unknown cell. Cluster them by 8-connectivity with a BFS (scipy is not a dependency). For each cluster of at least `min_cluster_cells`, the goal is the free cell nearest the cluster centroid that is at least `robot_radius_m` from any occupied or unknown cell; if none exists within 1 m of the centroid, drop the cluster. `visible_new_cells` casts `n_rays` rays stepping one cell at a time up to `max_range_m`, stops at occupied, continues through unknown, and counts distinct unknown cells crossed. `under_seen_goals` samples free cells on a `spacing_m` lattice and keeps those farther than `well_seen_m` from every visited point.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_grid.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/explore tests/test_explore_grid.py
git commit -m "Add occupancy grid frontier, visibility and under-seen analysis for exploration"
```

---

### Task 3: Scoring and blacklist

**Files:**
- Create: `src/explore/scoring.py`
- Test: `tests/test_explore_scoring.py`

**Interfaces:**
- Consumes: `Candidate` from Task 2.
- Produces: `Blacklist(radius_m: float = 1.0, ttl_s: float = 600.0)` with `add(self, x, y, now: float) -> None` and `blocked(self, x, y, now: float) -> bool`.
- Produces: `score(gain_cells: int, path_len_m: float, min_path_m: float = 0.5) -> float` = `gain_cells / max(path_len_m, min_path_m)`.
- Produces: `Scored` dataclass: `candidate: Candidate`, `gain: int`, `path_len_m: float`, `score: float`.
- Produces: `rank(scored: Sequence[Scored], min_score: float) -> list[Scored]` sorted descending, dropping entries below `min_score`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_scoring.py
from src.explore.grid import Candidate
from src.explore.scoring import Blacklist, Scored, score, rank

def test_blacklist_radius_and_ttl():
    b = Blacklist(radius_m=1.0, ttl_s=600.0)
    b.add(5.0, 5.0, now=100.0)
    assert b.blocked(5.5, 5.0, now=200.0)
    assert not b.blocked(6.5, 5.0, now=200.0)
    assert not b.blocked(5.0, 5.0, now=701.0)

def test_score_prefers_gain_per_meter():
    assert score(100, 10.0) == 10.0
    assert score(100, 0.1) == 200.0

def test_rank_sorts_and_filters():
    a = Scored(Candidate(0, 0, "frontier"), gain=10, path_len_m=1.0, score=10.0)
    b = Scored(Candidate(1, 0, "frontier"), gain=50, path_len_m=1.0, score=50.0)
    c = Scored(Candidate(2, 0, "under_seen"), gain=1, path_len_m=1.0, score=1.0)
    assert [s.score for s in rank([a, b, c], min_score=5.0)] == [50.0, 10.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_scoring.py -q`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/explore/scoring.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_scoring.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/explore/scoring.py tests/test_explore_scoring.py
git commit -m "Add exploration goal scoring and failed-goal blacklist"
```

---

### Task 4: Client protocols, Viam adapters, and leg runner

**Files:**
- Create: `src/explore/clients.py`, `src/explore/legs.py`
- Test: `tests/test_explore_clients.py`, `tests/test_explore_legs.py`

**Interfaces:**
- Produces in `clients.py`:
  - `Pose` dataclass: `x: float`, `y: float`, `theta: float`.
  - `NavStatus` dataclass: `active: bool`, `state: str`, `pose: Optional[Pose]`, `localized: bool`, `max_angular_rad_s: Optional[float]`, `robot_radius_m: Optional[float]`, `error: str`.
  - `SlamStatus` dataclass: `mode: str`, `running: bool`, `keyframes: int`, `graph_full: bool`, `closures: int`.
  - `class NavClient(Protocol)`: `async def status(self) -> NavStatus`; `async def plan_to_point(self, x, y) -> tuple[bool, float]` returning (feasible, length_m); `async def navigate_to_point(self, x, y, theta) -> None`; `async def go_to_point(self, x, y, theta, velocity_mps: float) -> None`; `async def cancel(self) -> None`.
  - `class SlamClient(Protocol)`: `async def grid(self) -> Optional[Mapping]` (raw `get_grid` response, None when unavailable); `async def status(self) -> SlamStatus`; `async def optimize(self) -> None`; `async def save_map(self) -> None`.
  - `ViamNavClient(resource)` and `ViamSlamClient(resource)` implementing the protocols over `resource.do_command({...})`. Nav commands: `get_status`, `plan_to_point` (reads `feasible`, `length_m`), `navigate_to_point`, `go_to_point` (sends `wait: False` and `velocity_mps`), `cancel`. SLAM commands: `get_grid` (None when `available` is False), `get_status` (reads `mode`, `running`, `keyframes`, `loop_closure.graph_full`, `loop_closure.closures`), `optimize`, `save_map`.
  - `localized_from_status(raw: Mapping) -> bool` delegating to `src.models.nav_api._localized`.
- Produces in `legs.py`:
  - `LegResult` dataclass: `ok: bool`, `reason: str` (one of `""`, `"timeout"`, `"failed"`, `"far_from_goal"`, `"lost_pose"`, `"canceled"`), `end_pose: Optional[Pose]`, `elapsed_s: float`.
  - `leg_timeout_s(planned_length_m: float, max_linear_mps: float) -> float` = `max(20.0, 3.0 * planned_length_m / max_linear_mps)`.
  - `LegRunner(nav: NavClient, *, max_linear_mps: float, clock: Callable[[], float], sleep: Callable[[float], Awaitable[None]], poll_s: float = 0.5, should_stop: Callable[[], bool] = lambda: False)` with `async def run(self, x, y, theta, *, planned: bool, planned_length_m: float, goal_tolerance_m: float = 1.0) -> LegResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_clients.py
import asyncio
from src.explore.clients import ViamNavClient, ViamSlamClient

class Recorder:
    def __init__(self, reply): self.reply, self.calls = reply, []
    async def do_command(self, command, **kw):
        self.calls.append(dict(command)); return self.reply

def test_nav_status_parses_pose_limits_and_localized():
    raw = {"active": False, "state": "succeeded", "pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
           "limits": {"max_linear_mps": 0.5, "max_angular_rad_s": 1.5}, "robot_radius_m": 0.3, "error_msg": ""}
    st = asyncio.run(ViamNavClient(Recorder(raw)).status())
    assert st.pose.x == 1.0 and st.max_angular_rad_s == 1.5 and st.robot_radius_m == 0.3
    assert st.localized and st.state == "succeeded"

def test_nav_status_without_pose_is_not_localized():
    st = asyncio.run(ViamNavClient(Recorder({"active": False, "state": "idle"})).status())
    assert st.pose is None and not st.localized and st.max_angular_rad_s is None

def test_nav_status_untrusted_is_not_localized():
    raw = {"active": False, "state": "idle", "pose": {"x": 0, "y": 0, "theta": 0},
           "localization_check": {"status": "untrusted"}}
    assert not asyncio.run(ViamNavClient(Recorder(raw)).status()).localized

def test_plan_to_point_returns_feasible_and_length():
    rec = Recorder({"status": "planned", "feasible": True, "length_m": 7.5})
    assert asyncio.run(ViamNavClient(rec).plan_to_point(1.0, 2.0)) == (True, 7.5)
    assert rec.calls[0]["command"] == "plan_to_point" and rec.calls[0]["x"] == 1.0

def test_go_to_point_sends_velocity_and_no_wait():
    rec = Recorder({"status": "ok"})
    asyncio.run(ViamNavClient(rec).go_to_point(1.0, 2.0, 0.0, velocity_mps=0.4))
    assert rec.calls[0] == {"command": "go_to_point", "x": 1.0, "y": 2.0, "theta": 0.0, "wait": False, "velocity_mps": 0.4}

def test_slam_status_and_grid():
    raw = {"mode": "mapping", "running": True, "keyframes": 12, "loop_closure": {"graph_full": False, "closures": 2}}
    st = asyncio.run(ViamSlamClient(Recorder(raw)).status())
    assert (st.mode, st.keyframes, st.graph_full, st.closures) == ("mapping", 12, False, 2)
    assert asyncio.run(ViamSlamClient(Recorder({"available": False, "reason": "no_map"})).grid()) is None
```

```python
# tests/test_explore_legs.py
import asyncio
from src.explore.clients import NavStatus, Pose
from src.explore.legs import LegRunner, leg_timeout_s

class FakeNav:
    def __init__(self, statuses): self.statuses, self.calls = list(statuses), []
    async def status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
    async def plan_to_point(self, x, y): return True, 1.0
    async def navigate_to_point(self, x, y, theta): self.calls.append(("nav", x, y))
    async def go_to_point(self, x, y, theta, velocity_mps): self.calls.append(("go", x, y, velocity_mps))
    async def cancel(self): self.calls.append(("cancel",))

class FakeClock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    async def sleep(self, s): self.t += s

def st(active, state, x=0.0, y=0.0, localized=True):
    return NavStatus(active=active, state=state, pose=Pose(x, y, 0.0), localized=localized,
                     max_angular_rad_s=0.5, robot_radius_m=0.3, error="")

def runner(nav, clk, **kw):
    return LegRunner(nav, max_linear_mps=0.4, clock=clk.now, sleep=clk.sleep, **kw)

def go(r, planned=True):
    return asyncio.run(r.run(2.0, 0.0, 0.0, planned=planned, planned_length_m=2.0))

def test_timeout_formula():
    assert leg_timeout_s(2.0, 0.4) == 20.0
    assert leg_timeout_s(40.0, 0.4) == 300.0

def test_leg_succeeds_when_nav_reports_succeeded_near_goal():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 1.9, 0.1)]), FakeClock()
    r = go(runner(nav, clk))
    assert r.ok and r.end_pose.x == 1.9 and nav.calls[0][0] == "nav"

def test_leg_far_from_goal_fails():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 0.0, 0.0)]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "far_from_goal"

def test_leg_times_out_and_cancels():
    nav, clk = FakeNav([st(True, "active")]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "timeout" and ("cancel",) in nav.calls and clk.t >= 20.0

def test_leg_lost_pose_cancels():
    nav, clk = FakeNav([st(True, "active"), st(True, "active", localized=False)]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "lost_pose" and ("cancel",) in nav.calls

def test_leg_failed_state_reports_failed():
    nav, clk = FakeNav([st(True, "active"), st(False, "failed")]), FakeClock()
    assert go(runner(nav, clk)).reason == "failed"

def test_should_stop_cancels_leg():
    nav, clk = FakeNav([st(True, "active")]), FakeClock()
    r = go(runner(nav, clk, should_stop=lambda: True))
    assert r.reason == "canceled" and ("cancel",) in nav.calls

def test_unplanned_leg_uses_go_to_point():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 2.0, 0.0)]), FakeClock()
    r = go(runner(nav, clk), planned=False)
    assert r.ok and nav.calls[0] == ("go", 2.0, 0.0, 0.4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_clients.py tests/test_explore_legs.py -q`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/explore/clients.py`**

`NavStatus.localized` comes from `localized_from_status(raw)`. `NavStatus.error` reads `error_msg`, then `error`, default `""`. `max_angular_rad_s` and `robot_radius_m` are None when absent.

- [ ] **Step 4: Implement `src/explore/legs.py`**

`run` issues the goal (`navigate_to_point` when `planned`, else `go_to_point` with `velocity_mps=max_linear_mps`), then polls `nav.status()` every `poll_s` until `active` is false. Checks in order on each poll: `should_stop()` true → cancel, `canceled`; `localized` false → cancel, `lost_pose`; elapsed past `leg_timeout_s` → cancel, `timeout`. On inactive: state `failed` or `canceled` → that reason; `succeeded` farther than `goal_tolerance_m` from the goal → `far_from_goal`; otherwise `ok`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_clients.py tests/test_explore_legs.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/explore/clients.py src/explore/legs.py tests/test_explore_clients.py tests/test_explore_legs.py
git commit -m "Add SLAM and navigation client adapters and a leg runner for exploration"
```

---

### Task 5: Wall-following hops and inner circuit selection

**Files:**
- Create: `src/explore/circuits.py`
- Test: `tests/test_explore_circuits.py`

**Interfaces:**
- Consumes: `OccGrid`, `Candidate` (Task 2); `Pose` (Task 4).
- Produces: `next_wall_hop(grid: OccGrid, pose: Pose, *, hop_m: float = 1.5, standoff_m: float = 0.8, search_m: float = 3.0) -> Optional[Candidate]` keeping the nearest wall on the robot's left.
- Produces: `facing_wall_theta(grid: OccGrid, pose: Pose, search_m: float = 3.0) -> Optional[float]`: heading from the pose toward the nearest occupied cell.
- Produces: `circuit_closed(path_m: float, pose: Pose, start: Pose, *, min_path_m: float = 12.0, close_m: float = 1.5) -> bool`.
- Produces: `inner_circuit_goals(grid: OccGrid, start: Pose, *, n: int = 3, clearance_m: float = 1.0) -> list[Candidate]`: greedy farthest-point picks among free cells at least `clearance_m` from occupied or unknown, in visiting order, ending with a `"circuit"` candidate at `start`.
- Produces: `path_length(points: Sequence[tuple[float, float]]) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_circuits.py
import math
import numpy as np
from src.explore.clients import Pose
from src.explore.grid import OccGrid
from src.explore.circuits import next_wall_hop, facing_wall_theta, circuit_closed, inner_circuit_goals, path_length

def walled_room(w_m=8.0, h_m=6.0, res=0.05):
    rows, cols = int(h_m / res), int(w_m / res)
    g = np.zeros((rows, cols), dtype=np.int8)
    g[0, :] = 100; g[-1, :] = 100; g[:, 0] = 100; g[:, -1] = 100
    return OccGrid(cells=g, resolution=res, x_min=0.0, y_min=0.0)

def test_wall_hop_keeps_wall_on_left_and_moves_along_it():
    g = walled_room()
    hop = next_wall_hop(g, Pose(4.0, 0.8, math.pi))   # bottom wall is on the left when heading -x
    assert hop is not None
    assert hop.x < 3.0 and abs(hop.y - 0.8) < 0.3

def test_wall_hop_turns_corner():
    g = walled_room()
    hop = next_wall_hop(g, Pose(1.0, 0.8, math.pi))   # about to hit the left wall
    assert hop is not None and hop.x >= 0.6 and hop.y > 1.3

def test_wall_hop_none_without_wall_in_range():
    g = walled_room(w_m=20.0, h_m=20.0)
    assert next_wall_hop(g, Pose(10.0, 10.0, 0.0), search_m=3.0) is None

def test_facing_wall_theta_points_at_nearest_wall():
    g = walled_room()
    th = facing_wall_theta(g, Pose(4.0, 0.8, 0.0))
    assert abs(th - (-math.pi / 2)) < 0.2

def test_circuit_closed():
    s = Pose(1.0, 1.0, 0.0)
    assert not circuit_closed(5.0, Pose(1.2, 1.0, 0.0), s)
    assert not circuit_closed(20.0, Pose(5.0, 1.0, 0.0), s)
    assert circuit_closed(20.0, Pose(1.2, 1.0, 0.0), s)

def test_inner_circuit_goals_are_clear_of_walls_and_long_enough():
    g = walled_room()
    goals = inner_circuit_goals(g, Pose(1.5, 1.5, 0.0), n=3, clearance_m=1.0)
    assert len(goals) == 4 and (goals[-1].x, goals[-1].y, goals[-1].kind) == (1.5, 1.5, "circuit")
    for c in goals[:-1]:
        assert 1.0 <= c.x <= 7.0 and 1.0 <= c.y <= 5.0
    assert path_length([(1.5, 1.5)] + [(c.x, c.y) for c in goals]) >= 12.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_circuits.py -q`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/explore/circuits.py`**

`next_wall_hop`: find the nearest occupied cell within `search_m`; the wall normal points from that cell to the robot; travel direction is the normal rotated by -90 degrees so the wall stays on the left; target = wall point + `standoff_m` along the normal + `hop_m` along travel. A target is valid only if the straight segment from the pose to it crosses no occupied or unknown cell and the target itself is at least `standoff_m` from every occupied cell. If invalid, shorten the hop in 0.25 m steps; below 0.5 m, rotate the travel direction by +90 degrees once (corner) and retry from the full hop. `inner_circuit_goals`: sample free cells on a 0.25 m lattice, keep those with no occupied or unknown cell within `clearance_m`, pick `n` by greedy farthest-point starting from `start`, append the start candidate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_circuits.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/explore/circuits.py tests/test_explore_circuits.py
git commit -m "Add wall-following hops and inner circuit selection for the structure phase"
```

---

### Task 6: Sim obstacles and test fakes on the sim world

**Files:**
- Modify: `src/sim/world.py:106-170` (obstacle methods on `SimWorld`)
- Modify: `src/models/sim_base.py:235-258` (two DoCommands)
- Create: `tests/explore_fakes.py`
- Test: `tests/test_explore_fakes.py`

**Interfaces:**
- Produces on `SimWorld`: `add_obstacle(self, x: float, y: float, radius_m: float) -> int` returning an id; `remove_obstacle(self, obstacle_id: int) -> None`; `clear_obstacles(self) -> None`. Obstacles paint 100 into the grid used by raycast and collision; removal restores the original cells.
- Produces on `sim-base` DoCommand: `add_obstacle {x, y, radius_m}` → `{"ok": True, "id": int}`; `clear_obstacles` → `{"ok": True}`.
- Produces in `tests/explore_fakes.py`:
  - `FakeClock` with `now() -> float` and `async sleep(s)` that advances `now`.
  - `FakeSlam(world: SimWorld, *, keyframe_max: int = 2000)` implementing `SlamClient`. Holds an int8 array the shape of `world.map.grid`, initially -1. `reveal(pose: Pose2D)` raycasts from `pose` using the world's scan geometry, marks free along each ray and occupied at the hit. Counts a keyframe on every `reveal` whose pose moved 0.5 m or turned 20 degrees since the last keyframe; `graph_full` when `keyframes >= keyframe_max`. `closures` increments when `reveal` happens within 1.5 m of a keyframe pose recorded more than 6 m of path earlier. Attributes: `mode: str = "mapping"`, `running: bool = True`, `saved: int`, `optimized: int`. `grid()` returns a `get_grid`-shaped dict (base64 int8 `data`, `rows`, `cols`, `cellSize`, `xMin`, `yMin`).
  - `FakeNav(world: SimWorld, slam: FakeSlam, clock: FakeClock, *, speed_mps: float = 0.4, max_angular_rad_s: float = 0.5)` implementing `NavClient`. `plan_to_point` returns `(True, straight-line length)` when the line through `world.map` is unblocked, else `(False, 0.0)`. `navigate_to_point` and `go_to_point` start a leg; each `status()` call advances the world pose `speed_mps * dt` along the line (dt from the clock since the last call), calls `slam.reveal`, and reports `active` until arrival within 0.05 m, then state `succeeded`. A blocked cell on the path ends the leg with state `failed`. Attributes: `lose_pose: bool` (status reports `localized=False`), `block_all: bool` (every plan infeasible, every leg fails immediately), `cancels: int`, `max_angular_rad_s` settable. `cancel()` ends the leg with state `canceled`.
  - `known_fraction(slam: FakeSlam, world: SimWorld) -> float`: fraction of ground-truth free cells known free in the fake SLAM grid.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_fakes.py
import asyncio
from src.geom import conversions as conv
from src.sim.world import SimWorld, make_builtin_corridor
from tests.explore_fakes import FakeClock, FakeSlam, FakeNav, known_fraction

def world():
    return SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))

def test_world_obstacle_add_and_remove():
    w = world()
    before = w.get_scan().ranges.copy()
    oid = w.add_obstacle(2.0, 1.0, 0.2)
    assert w.get_scan().ranges.min() < before.min()
    w.remove_obstacle(oid)
    assert (w.get_scan().ranges == before).all()

def test_fake_slam_reveals_cells_and_counts_keyframes():
    w = world(); s = FakeSlam(w)
    assert known_fraction(s, w) == 0.0
    s.reveal(conv.Pose2D(1.0, 1.0, 0.0))
    assert known_fraction(s, w) > 0.05
    s.reveal(conv.Pose2D(1.6, 1.0, 0.0))
    assert asyncio.run(s.status()).keyframes == 2
    assert asyncio.run(s.grid())["rows"] == w.map.height

def test_fake_nav_drives_to_goal_and_reveals():
    w = world(); clk, s = FakeClock(), FakeSlam(w); nav = FakeNav(w, s, clk)
    async def go():
        assert await nav.plan_to_point(3.0, 1.0) == (True, 2.0)
        await nav.navigate_to_point(3.0, 1.0, 0.0)
        while (await nav.status()).active:
            await clk.sleep(0.5)
        return await nav.status()
    st = asyncio.run(go())
    assert st.state == "succeeded" and abs(st.pose.x - 3.0) < 0.05 and known_fraction(s, w) > 0.1

def test_fake_nav_fails_on_blocked_line():
    w = world(); nav = FakeNav(w, FakeSlam(w), FakeClock())
    assert asyncio.run(nav.plan_to_point(7.0, 1.0)) == (False, 0.0)   # interior wall at x=4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_fakes.py -q`
Expected: FAIL with ModuleNotFoundError `tests.explore_fakes`.

- [ ] **Step 3: Implement obstacle methods on `SimWorld` and the two `sim-base` DoCommands**

- [ ] **Step 4: Implement `tests/explore_fakes.py`**

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_fakes.py tests/test_sim_builtin.py -q`
Expected: PASS, including the existing sim tests.

- [ ] **Step 6: Commit**

```bash
git add src/sim/world.py src/models/sim_base.py tests/explore_fakes.py tests/test_explore_fakes.py
git commit -m "Add sim obstacle injection and fake SLAM and nav clients for exploration tests"
```

---

### Task 7: Run state machine

**Files:**
- Create: `src/explore/run.py`
- Test: `tests/test_explore_run.py`

**Interfaces:**
- Consumes: Tasks 2 to 6.
- Produces: `RunParams` dataclass: `duration_s: float`, `max_linear_mps: float`, `max_angular_dps: float`, `home_tolerance_m: float`, `structure_fraction: float`, `revisit_fraction: float`, `well_seen_m: float`, `keyframe_max: int`, `min_score: float = 2.0`, `robot_radius_m: float = 0.3`.
- Produces: `ExploreRun(nav: NavClient, slam: SlamClient, params: RunParams, *, clock: Callable[[], float], sleep: Callable[[float], Awaitable[None]], stop_base: Callable[[], Awaitable[None]])` with `async def preflight(self) -> Optional[str]` (error text or None), `async def run(self) -> dict` (final status), `def request_stop(self) -> None`, `def status(self) -> dict` with keys `phase` (`idle`, `structure`, `fill`, `revisit`, `finishing`, `done`), `elapsed_s`, `remaining_s`, `legs`, `legs_failed`, `coverage_cells`, `frontier_count`, `keyframes_used`, `last_error`, `home` (dict or None).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_run.py
import asyncio
from src.geom import conversions as conv
from src.sim.world import SimWorld, make_builtin_corridor
from src.explore.run import ExploreRun, RunParams
from tests.explore_fakes import FakeClock, FakeSlam, FakeNav, known_fraction

def params(duration_s=600.0, **kw):
    base = dict(duration_s=duration_s, max_linear_mps=0.4, max_angular_dps=30.0, home_tolerance_m=0.5,
                structure_fraction=0.15, revisit_fraction=0.1, well_seen_m=4.0, keyframe_max=2000)
    base.update(kw)
    return RunParams(**base)

def setup(**kw):
    w = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))
    clk = FakeClock()
    slam = FakeSlam(w, keyframe_max=kw.get("keyframe_max", 2000))
    nav = FakeNav(w, slam, clk)
    stops = []
    async def stop_base(): stops.append(1)
    run = ExploreRun(nav, slam, params(**kw), clock=clk.now, sleep=clk.sleep, stop_base=stop_base)
    return w, clk, slam, nav, run, stops

async def run_watching(run):
    phases = []
    task = asyncio.ensure_future(run.run())
    while not task.done():
        phases.append(run.status()["phase"])
        await asyncio.sleep(0)
    return await task, phases

def test_full_run_covers_corridor_and_finishes_home():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["last_error"] == ""
    assert known_fraction(slam, w) > 0.7
    assert slam.saved == 1 and slam.optimized == 1
    pose = w.get_pose()
    assert abs(pose.x - 1.0) < 0.5 and abs(pose.y - 1.0) < 0.5
    assert final["legs"] > 3 and final["home"] == {"x": 1.0, "y": 1.0, "theta": 0.0}

def test_phases_progress_in_order():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)
    final, seen = asyncio.run(run_watching(run))
    order = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
    assert order[:2] == ["structure", "fill"] and order[-2:] == ["finishing", "done"]

def test_stop_mid_leg_finishes_once():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)
    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["legs"] < 1:
            await asyncio.sleep(0)
        run.request_stop()
        return await task
    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and nav.cancels >= 1

def test_timeout_triggers_finish():
    w, clk, slam, nav, run, _ = setup(duration_s=60.0)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and clk.now() >= 60.0 and slam.saved == 1

def test_lost_pose_waits_then_finishes():
    w, clk, slam, nav, run, stops = setup(duration_s=900.0)
    nav.lose_pose = True
    final = asyncio.run(run.run())
    assert final["last_error"].startswith("lost pose") and 60.0 <= clk.now() < 120.0
    assert slam.saved == 1 and stops == [1] and final["legs"] == 0

def test_wrong_slam_mode_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    slam.mode = "localizing"
    assert "mapping" in asyncio.run(run.preflight())
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and "mapping" in final["last_error"]
    assert final["legs"] == 0 and slam.saved == 0

def test_angular_limit_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    nav.max_angular_rad_s = 1.5
    final = asyncio.run(run.run())
    assert "angular" in final["last_error"] and final["legs"] == 0

def test_keyframe_cap_ends_structure_and_skips_revisit():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0, keyframe_max=10)
    final, phases = asyncio.run(run_watching(run))
    assert "revisit" not in phases and phases.count("structure") < phases.count("fill")

def test_failed_goals_are_blacklisted_and_run_survives():
    w, clk, slam, nav, run, _ = setup(duration_s=600.0)
    w.add_obstacle(2.5, 1.0, 0.3)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["legs_failed"] >= 1 and final["legs"] > final["legs_failed"]

def test_twenty_consecutive_failures_finish_early():
    w, clk, slam, nav, run, _ = setup(duration_s=3600.0)
    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["legs"] < 1:
            await asyncio.sleep(0)
        nav.block_all = True
        return await task
    final = asyncio.run(go())
    assert "failed legs" in final["last_error"] and final["legs_failed"] >= 20 and clk.now() < 3600.0

def test_home_failure_stops_in_place():
    w, clk, slam, nav, run, stops = setup(duration_s=120.0)
    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["phase"] != "finishing":
            await asyncio.sleep(0)
        nav.block_all = True
        return await task
    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and stops == [1] and "home" in final["last_error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_run.py -q`
Expected: FAIL with ModuleNotFoundError `src.explore.run`.

- [ ] **Step 3: Implement `src/explore/run.py`**

`preflight()` returns, in order: `"slam mode must be mapping (got X)"`; `"nav angular limit X rad/s exceeds explore max_angular_dps Y"` (compare against `math.radians(max_angular_dps)`); None otherwise. `run()`:

1. `preflight()`; on error set `last_error`, `phase = done`, return `status()` without the finish sequence. Wait for a localized nav pose (lost-pose rule); record it as `home`.
2. Structure phase while `elapsed < structure_fraction * duration` and `keyframes < 0.6 * keyframe_max`: outer circuit of `next_wall_hop` legs with `planned=False` and `planned_length_m=hop_m` until `circuit_closed` (track path by summing leg displacements); then `inner_circuit_goals` legs with `planned=True`. Record each circuit's waypoints and the `closures` delta across it. Between legs sleep 2 s and use `facing_wall_theta` as the next leg's theta when available. If `next_wall_hop` returns None, end the outer circuit.
3. Fill phase while `remaining > revisit_fraction * duration`: read the grid (`OccGrid.from_get_grid`), gather `frontier_goals(grid, robot_radius_m)` and `under_seen_goals(grid, visited, well_seen_m)`, drop blacklisted, keep the 30 nearest by straight line, `plan_to_point` each, score with `visible_new_cells` gain, `rank(min_score)`. No candidates: break to revisit. Run the best with `planned=True`, sleep 1 s, append the end pose to `visited`.
4. Revisit phase unless `graph_full` or `keyframes >= keyframe_max`: replay recorded circuits, zero-closure ones first, until `remaining <= 0`.
5. Finish (`phase = finishing`): `nav.cancel()`, `slam.optimize()`, `slam.save_map()`, planned leg home with `goal_tolerance_m=home_tolerance_m`, retry once on failure, then `stop_base()` and `last_error = "return home failed: <reason>"` if both fail. `phase = done`.

Failures: a leg with `ok` false increments `legs_failed` and the consecutive counter, blacklists the goal at the current clock. Consecutive 5 → sleep 30 s. Consecutive 20 → `last_error = "too many failed legs"`, finish. Leg reason `lost_pose`, or a nav status without `localized`, or SLAM `running` false → poll every 5 s up to 60 s; if still lost, `last_error = "lost pose for 60 s"`, finish. `request_stop()` sets a flag; it is `LegRunner.should_stop` and is checked before every leg, so the finish sequence runs once.

The spec's penalty for candidates toward currently moving obstacles is not implemented in this plan: the blacklist covers failed goals and the navigation service's local costmap handles live obstacles during a leg. Note this in the README section.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_run.py -q`
Expected: PASS. If the full-run test takes over 30 s wall clock, coarsen the fake nav's step size rather than loosening the coverage assertion.

- [ ] **Step 5: Commit**

```bash
git add src/explore/run.py tests/test_explore_run.py
git commit -m "Add the exploration run state machine with structure, fill, revisit and finish phases"
```

---

### Task 8: Wire the model, document, and sim smoke

**Files:**
- Modify: `src/models/explore.py`
- Modify: `README.md` (models table; new "Explore service" section after the navigation section)
- Modify: `sample_configs/local_sim.json` (add an `explore` service)
- Test: `tests/test_explore_model.py`

**Interfaces:**
- Consumes: `ExploreRun`, `RunParams`, `ViamNavClient`, `ViamSlamClient`, `ExploreConfig`.
- Produces: DoCommands `start {duration_s}` → `{"started": True, "duration_s": float}` or `{"error": str}`; `stop` → `{"stopping": True}` or `{"error": "no active run"}`; `status` → `ExploreRun.status()` of the latest run, or `{"phase": "idle"}` before any run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore_model.py
import asyncio
from src.config import ExploreConfig
from src.models.explore import ExploreService

class Rec:
    def __init__(self, replies): self.replies, self.calls = replies, []
    async def do_command(self, command, **kw):
        self.calls.append(dict(command)); return self.replies.get(command["command"], {})
    async def stop(self): self.calls.append({"command": "base_stop"})

def make(slam_mode="mapping", ang=0.5):
    svc = ExploreService("explore")
    svc._slam = Rec({"get_status": {"mode": slam_mode, "running": True, "keyframes": 0, "loop_closure": {}},
                     "get_grid": {"available": False}})
    svc._nav = Rec({"get_status": {"active": False, "state": "idle", "pose": {"x": 0, "y": 0, "theta": 0},
                                   "limits": {"max_linear_mps": 0.4, "max_angular_rad_s": ang}}})
    svc._base = Rec({})
    svc._cfg = ExploreConfig.from_dict({"slam_service": "slam", "navigation_service": "navigation", "base": "base"})
    return svc

def test_status_idle_before_start():
    assert asyncio.run(make().do_command({"command": "status"})) == {"phase": "idle"}

def test_start_refuses_when_slam_not_mapping():
    svc = make(slam_mode="localizing")
    out = asyncio.run(svc.do_command({"command": "start", "duration_s": 10}))
    assert "mapping" in out["error"]
    assert not any(c["command"] in ("navigate_to_point", "go_to_point") for c in svc._nav.calls)

def test_start_refuses_when_nav_turns_too_fast():
    out = asyncio.run(make(ang=1.5).do_command({"command": "start", "duration_s": 10}))
    assert "angular" in out["error"]

def test_stop_without_run_is_an_error():
    assert asyncio.run(make().do_command({"command": "stop"})) == {"error": "no active run"}

def test_start_twice_is_an_error_and_stop_ends_run():
    svc = make()
    async def go():
        a = await svc.do_command({"command": "start", "duration_s": 300})
        b = await svc.do_command({"command": "start", "duration_s": 300})
        s = await svc.do_command({"command": "stop"})
        await svc._run_task
        return a, b, s, await svc.do_command({"command": "status"})
    a, b, s, st = asyncio.run(go())
    assert a == {"started": True, "duration_s": 300.0} and "already" in b["error"] and s == {"stopping": True}
    assert st["phase"] == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explore_model.py -q`
Expected: FAIL on `start` returning the unknown-command error.

- [ ] **Step 3: Implement `start` / `stop` / `status` in `src/models/explore.py`**

`start`: error if `self._run_task` exists and is not done (`"a run is already active"`). Build `RunParams` from `self._cfg` plus `duration_s` and `robot_radius_m` from nav status when present. Build `ExploreRun` with `ViamNavClient(self._nav)`, `ViamSlamClient(self._slam)`, `clock=time.monotonic`, `sleep=asyncio.sleep`, `stop_base=self._base.stop`. Await `preflight()`; on error return `{"error": msg}`. Otherwise `self._run_task = asyncio.create_task(run.run())`. `stop`: `request_stop()` on the active run. `status`: the latest run's `status()`. `reconfigure` while a run is active calls `request_stop` first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explore_model.py -q`
Expected: PASS

- [ ] **Step 5: Document**

README: a models table row and an "Explore service" section with the config JSON (three names plus defaults), the three DoCommands with payloads and replies, the preflight requirements (SLAM `mode: mapping`; navigation `max_vel_theta` at or below the explore cap, e.g. `"max_vel_theta": 0.5`), the phases in one paragraph, and the finish behavior. Document the sim-base `add_obstacle` / `clear_obstacles` commands next to `reset` / `get_pose`. Add an explore service to `sample_configs/local_sim.json` and set its SLAM service to `mapping` mode if it is not already.

- [ ] **Step 6: Sim smoke**

Run `./scripts/run_local_sim.sh` in one terminal, then in another:

```bash
.venv/bin/python - <<'PY'
import asyncio
from viam.robot.client import RobotClient
from viam.services.generic import Generic
async def main():
    r = await RobotClient.at_address("localhost:8081", RobotClient.Options(disable_sessions=True))
    ex = Generic.from_robot(r, "explore")
    print(await ex.do_command({"command": "start", "duration_s": 120}))
    for _ in range(26):
        await asyncio.sleep(5)
        print(await ex.do_command({"command": "status"}))
    await r.close()
asyncio.run(main())
PY
```

Expected: phases advance `structure` → `fill` → `finishing` → `done`, `coverage_cells` grows, `last_error` is empty.

- [ ] **Step 7: Run the whole suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

```bash
git add src/models/explore.py README.md sample_configs/local_sim.json tests/test_explore_model.py
git commit -m "Wire the explore service DoCommands and document the exploration run"
```
