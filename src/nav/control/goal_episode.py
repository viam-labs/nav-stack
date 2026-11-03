"""Per-goal mutable state for ``NavSupervisor.run_goal``.

Episode-lifetime counters and machines used to live as ~40 locals inside
``run_goal``. Named reset packs encode today's asymmetric clear sites so a
blocked-replan success does not silently match a backup-done success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..motion.simple_motion import DriveCommand
from ..types import Pose2D
from .escape import EscapeMachine


@dataclass
class GoalEpisode:
    """Everything that survives across control ticks for one goal."""

    deadline: float
    last_replan: float
    last_progress_at: float
    last_progress_pose: Optional[Pose2D] = None
    last_progress_dist: float = float("inf")
    last_progress_bearing: float = float("inf")
    spin_stuck_since: Optional[float] = None
    backup_active: bool = False
    backup_start: Optional[Pose2D] = None
    backup_start_cost: int = 0
    backup_attempts: int = 0
    backup_cooldown_until: float = 0.0
    escape_m: EscapeMachine = field(default_factory=EscapeMachine)
    # Map heading of the scan corridor picked for this blocked episode.
    corridor_heading: Optional[float] = None
    local_blocked_since: Optional[float] = None
    last_local_replan_at: float = 0.0
    failed_replan_while_blocked: int = 0
    failed_static_replan: int = 0
    local_planner_active: bool = False
    reactive_avoid_since: Optional[float] = None
    last_obstacle_state: str = ""
    prev_local_cmd: Optional[DriveCommand] = None
    prev_cmd: Optional[DriveCommand] = None
    rotate_active: bool = False
    vx_sign_history: List[Tuple[float, int]] = field(default_factory=list)
    align_best_err: Optional[float] = None
    align_progress_at: Optional[float] = None
    last_tick_pose: Optional[Pose2D] = None
    # Only validate the next few metres — full-path static checks on long
    # goals trip on far unknown/inflation and abort immediately.
    path_block_horizon_m: float = 5.0
    # Abort only when the route stays blocked and lidar is not clear.
    static_replan_fail_limit: int = 5
    # Parked and stopped-goal checks apply up to 1 m.
    pose_jump_replan_m: float = 0.5
    was_loc_holding: bool = False
    pending_loc_replan: bool = False
    # Consecutive failed blocked-replans; reset only by a successful replan
    # or real progress toward the goal (not by a one-tick local_blocked flicker).
    blocked_fail_count: int = 0
    blocked_fail_dist: Optional[float] = None

    # --- named reset packs (asymmetric on purpose) ---------------------

    def on_goal_progress(self) -> None:
        """Goal closed ≥0.5 m since the fail streak started — fresh retries."""
        self.blocked_fail_count = 0
        self.blocked_fail_dist = None

    def on_unblocked(self) -> None:
        """Nose/path no longer locally blocked — drop escape + fail streak."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.escape_m.clear_episode()
        self.corridor_heading = None

    def on_blocked_replan_ok(self) -> None:
        """Successful replan while locally blocked (policy action == replan)."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.backup_attempts = 0
        self.vx_sign_history.clear()
        self.spin_stuck_since = None
        self.reactive_avoid_since = None
        self.last_obstacle_state = ""
        self.prev_local_cmd = None
        self.local_planner_active = False
        self.prev_cmd = None
        self.rotate_active = False
        self.blocked_fail_count = 0
        self.blocked_fail_dist = None
        self.escape_m.clear_episode()
        self.corridor_heading = None

    def on_blocked_replan_fail(self, *, goal_dist_m: float) -> None:
        """Blocked-nose replan returned no path."""
        self.failed_replan_while_blocked += 1
        self.blocked_fail_count += 1
        if self.blocked_fail_dist is None:
            self.blocked_fail_dist = float(goal_dist_m)

    def on_escape_pre_replan_ok(self, *, unstick: bool) -> None:
        """EscapeMachine.tick_pre finished a motion and replan succeeded."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.vx_sign_history.clear()
        if unstick:
            self.backup_attempts = 0
            self.spin_stuck_since = None
            self.reactive_avoid_since = None
            self.last_obstacle_state = ""
            self.prev_local_cmd = None
            self.local_planner_active = False
            self.prev_cmd = None
            self.rotate_active = False
            self.blocked_fail_count = 0
            self.blocked_fail_dist = None

    def on_unstick_replan_fail(self, *, goal_dist_m: float) -> None:
        """blocked_nose_unstick replan failed — count toward abort."""
        self.failed_replan_while_blocked += 1
        self.blocked_fail_count += 1
        if self.blocked_fail_dist is None:
            self.blocked_fail_dist = float(goal_dist_m)

    def on_backup_done_ok(self) -> None:
        """DWA backup finished and replan succeeded (no escape clear)."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.backup_attempts = 0
        self.vx_sign_history.clear()

    def on_periodic_replan_ok(self) -> None:
        """Periodic / pose-jump / loc-exit static replan succeeded."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.failed_static_replan = 0
        self.backup_attempts = 0
        self.vx_sign_history.clear()
        self.spin_stuck_since = None

    def on_escape_post_ok(self) -> None:
        """Undrivable tick_post finished and replan succeeded."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.vx_sign_history.clear()

    def on_stall_replan_ok(self) -> None:
        """Stall recovery replan succeeded."""
        self.local_blocked_since = None
        self.failed_replan_while_blocked = 0
        self.backup_attempts = 0

    def note_replan_finished(self, when: float) -> None:
        self.last_local_replan_at = float(when)
        self.last_replan = float(when)

    def mark_progress(
        self,
        *,
        now: float,
        pose: Pose2D,
        dist_goal: float,
        bearing_err: float = float("inf"),
    ) -> None:
        self.last_progress_at = float(now)
        self.last_progress_pose = pose
        self.last_progress_dist = float(dist_goal)
        self.last_progress_bearing = float(bearing_err)
