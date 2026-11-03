"""Resolve ``NavConfig`` for builtin nav (tests override via ``replace``-compatible kwargs).

Supervisor stores the dataclass and reads ``self._cfg`` / ``self._cfg.builtin``.
Legacy kwargs (``max_vel_x``, ``robot_radius_m``, …) still work for unit tests;
prefer ``dataclasses.replace`` on ``NavConfig`` / ``BuiltinNavConfig`` for new code.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Optional

from ...config import BuiltinNavConfig, NavConfig

# Top-level NavConfig field ← legacy / short override name.
_TOP_ALIASES = {
    "max_vel_x": "max_linear_velocity",
    "avoid_obstacles": "simple_avoid_obstacles",
    "stop_distance_m": "simple_stop_distance",
    "slow_distance_m": "simple_slow_distance",
    "scan_max_age_s": "simple_scan_max_age",
}

# BuiltinNavConfig field ← legacy override name.
_BUILTIN_ALIASES = {
    "algorithm": "planner",
    "xy_tolerance_m": "xy_goal_tolerance",
    "yaw_tolerance_rad": "yaw_goal_tolerance",
}

_TOP_FIELDS = {f.name for f in fields(NavConfig)} - {"builtin"}
_BUILTIN_FIELDS = {f.name for f in fields(BuiltinNavConfig)}


def default_nav_cfg() -> NavConfig:
    """Minimal NavConfig for unit tests that only pass runtime overrides."""
    return NavConfig(slam_service="slam", base="base")


def resolve_nav_cfg(nav_cfg: Optional[NavConfig] = None, **overrides: Any) -> NavConfig:
    """Apply overrides onto a NavConfig via ``dataclasses.replace``.

    Derived footprint kwargs used by older tests:
    - ``robot_radius_m`` → hard clearance (``robot_radius`` + ``clearance_m=0``)
    - ``inflation_radius_m`` → ``inflation_margin_m`` past that hard radius
    - ``poll_interval_s`` → ``control_rate_hz = 1 / period``
    - ``body_radius_m`` → ``robot_radius`` (body / inscribed when no footprint)
    """
    cfg = nav_cfg if nav_cfg is not None else default_nav_cfg()
    overrides = dict(overrides)

    # Drop keys that are not config (supervisor-only construction flags).
    overrides.pop("footprint_guard", None)

    top: dict[str, Any] = {}
    builtin: dict[str, Any] = {}

    if "robot_radius_m" in overrides:
        hard = float(overrides.pop("robot_radius_m"))
        top["robot_radius"] = hard
        top["clearance_m"] = 0.0
        # Rectangular footprint would ignore robot_radius for inscribed width.
        top["footprint_length_m"] = None
        top["footprint_width_m"] = None
    if "body_radius_m" in overrides:
        body = float(overrides.pop("body_radius_m"))
        # Body-only disk when tests pin inscribed radius without a rectangle.
        if "robot_radius" not in top:
            top["robot_radius"] = body
        top["clearance_m"] = max(
            0.0,
            float(top.get("robot_radius", cfg.robot_radius)) - body,
        )
        top["footprint_length_m"] = None
        top["footprint_width_m"] = None
    if "inflation_radius_m" in overrides:
        inf = float(overrides.pop("inflation_radius_m"))
        # Prefer the hard radius implied by overrides already applied.
        if "robot_radius" in top and top.get("clearance_m", cfg.clearance_m) == 0.0:
            hard = float(top["robot_radius"])
        else:
            hard = float(top.get("robot_radius", cfg.robot_radius)) + float(
                top.get("clearance_m", cfg.clearance_m)
            )
        top["inflation_margin_m"] = max(0.0, inf - hard)
    if "poll_interval_s" in overrides:
        period = max(1e-3, float(overrides.pop("poll_interval_s")))
        top["control_rate_hz"] = 1.0 / period

    # spin/nose/wheel overrides: approximate via footprint rectangle when given
    # together; otherwise ignore if they match defaults from robot_radius.
    spin = overrides.pop("spin_radius_m", None)
    nose = overrides.pop("nose_offset_m", None)
    half_track = overrides.pop("wheel_half_track_m", None)
    if nose is not None or spin is not None:
        # Build a rectangle whose half-length ≈ nose and half-diagonal ≈ spin.
        nose_v = (
            float(nose)
            if nose is not None
            else float(top.get("robot_radius", cfg.robot_radius))
        )
        if spin is not None:
            spin_v = float(spin)
            half_w = (
                max(0.01, (spin_v**2 - nose_v**2) ** 0.5) if spin_v > nose_v else nose_v
            )
        else:
            half_w = float(top.get("robot_radius", cfg.robot_radius))
        top["footprint_length_m"] = 2.0 * nose_v
        top["footprint_width_m"] = 2.0 * half_w
        # Keep clearance so hard radius stays consistent if robot_radius_m set.
        if "robot_radius" in top and top.get("clearance_m") == 0.0:
            # hard clearance was the disk; with a rectangle, hard = half_w + clearance.
            # Tests that pass nose/spin/wheel from an existing cfg should pass the cfg
            # itself — this path is for explicit geometry overrides only.
            top["clearance_m"] = max(0.0, float(top["robot_radius"]) - half_w)
    del half_track  # wheel track follows footprint_width via NavConfig method

    for key, value in list(overrides.items()):
        name = _TOP_ALIASES.get(key, key)
        if name in _TOP_FIELDS:
            top[name] = value
            overrides.pop(key, None)
            continue
        name = _BUILTIN_ALIASES.get(key, key)
        if name in _BUILTIN_FIELDS:
            builtin[name] = value
            overrides.pop(key, None)
            continue
        raise TypeError(f"unknown nav config override: {key!r}")

    if builtin:
        cfg = replace(cfg, builtin=replace(cfg.builtin, **builtin))
    if top:
        cfg = replace(cfg, **top)
    return _sanitize_nav_cfg(cfg)


def _sanitize_nav_cfg(cfg: NavConfig) -> NavConfig:
    """Clamp knobs once so callers can read ``cfg`` without re-clamping."""
    b = cfg.builtin
    b2 = replace(
        b,
        clearance_preference_m=max(0.0, float(b.clearance_preference_m)),
        yaw_align_timeout_s=max(0.0, float(b.yaw_align_timeout_s)),
        max_goal_snap_m=max(0.0, float(b.max_goal_snap_m)),
        drive_timeout_streak=max(1, int(b.drive_timeout_streak)),
        backup_max_attempts=max(0, int(b.backup_max_attempts)),
        recovery_wait_duration_s=max(0.0, float(b.recovery_wait_duration_s)),
        replan_local_blocked_time_s=max(0.0, float(b.replan_local_blocked_time_s)),
        replan_local_min_period_s=max(0.1, float(b.replan_local_min_period_s)),
        replan_max_early_detour_ratio=max(1.0, float(b.replan_max_early_detour_ratio)),
        replan_allow_long_detour_after_fails=max(
            0, int(b.replan_allow_long_detour_after_fails)
        ),
        detour_short_flip_budget=max(0, int(b.detour_short_flip_budget)),
        max_linear_accel_mps2=max(0.05, float(b.max_linear_accel_mps2)),
        max_linear_decel_mps2=max(0.05, float(b.max_linear_decel_mps2)),
        max_angular_accel_rad_s2=max(0.1, float(b.max_angular_accel_rad_s2)),
        nav_loc_refine_margin_m=max(0.1, float(b.nav_loc_refine_margin_m)),
        nav_loc_refine_map_max_m=max(0.3, float(b.nav_loc_refine_map_max_m)),
        nav_loc_refine_lidar_min_m=max(0.2, float(b.nav_loc_refine_lidar_min_m)),
        nav_loc_refine_min_frac=min(1.0, max(0.05, float(b.nav_loc_refine_min_frac))),
        nav_loc_refine_min_beams=max(1, int(b.nav_loc_refine_min_beams)),
        nav_loc_refine_max_tries=max(1, int(b.nav_loc_refine_max_tries)),
        nav_loc_refine_cooldown_s=max(0.0, float(b.nav_loc_refine_cooldown_s)),
        nav_loc_refine_period_s=max(0.0, float(b.nav_loc_refine_period_s)),
        nav_loc_refine_check_every_m=max(0.0, float(b.nav_loc_refine_check_every_m)),
        nav_loc_refine_retry_travel_m=max(
            max(0.0, float(b.nav_loc_refine_check_every_m)),
            float(b.nav_loc_refine_retry_travel_m),
        ),
        nav_loc_refine_apply_max_m=max(0.2, float(b.nav_loc_refine_apply_max_m)),
        nav_loc_refine_apply_max_deg=max(5.0, float(b.nav_loc_refine_apply_max_deg)),
        nav_loc_refine_apply_min_score=min(
            1.0, max(0.0, float(b.nav_loc_refine_apply_min_score))
        ),
        nav_loc_refine_settle_s=max(0.0, float(b.nav_loc_refine_settle_s)),
        local_costmap_rate_hz=max(0.5, float(b.local_costmap_rate_hz)),
    )
    return replace(
        cfg,
        builtin=b2,
        max_linear_velocity=max(0.05, float(cfg.max_linear_velocity)),
        control_rate_hz=max(0.5, float(cfg.control_rate_hz)),
    )
