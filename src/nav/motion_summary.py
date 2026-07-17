"""Plain-English summaries of what navigation is commanding the base to do."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple


def _speed_word(abs_mps: float, max_mps: float) -> str:
    if max_mps <= 0:
        max_mps = 1.0
    ratio = abs_mps / max_mps
    if ratio < 0.25:
        return "slowly"
    if ratio < 0.6:
        return "at moderate speed"
    return "quickly"


def _turn_word(abs_rad_s: float, max_rad_s: float) -> str:
    if max_rad_s <= 0:
        max_rad_s = 1.0
    ratio = abs_rad_s / max_rad_s
    if ratio < 0.35:
        return "gently"
    if ratio < 0.7:
        return "moderately"
    return "hard"


def _held_seconds(history: list, last: Mapping[str, Any]) -> Optional[float]:
    """Estimate how long the current cmd signature has been active."""
    if not history:
        return None
    cur_age = last.get("age_s")
    if cur_age is None:
        return None
    # History is oldest→newest. Previous distinct sample's age is a lower bound
    # on when the current command started (deduped refreshes keep current age ~0).
    if len(history) >= 2:
        prev_age = history[-2].get("age_s")
        if prev_age is not None:
            held = float(prev_age) - float(cur_age)
            if held > 0.05:
                return round(held, 1)
    # Single entry / no prior: if it's a stop that's been sitting, use its age.
    if last.get("source") in ("stop", "simple_stop", "watchdog_stop") and float(cur_age) > 0.2:
        return round(float(cur_age), 1)
    return None


def _wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _goal_in_body(
    pose: Mapping[str, Any], goal: Mapping[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    """Return (dist_m, bearing_body_rad, forward_m, left_m) or None."""
    try:
        px = float(pose["x"])
        py = float(pose["y"])
        pth = float(pose["theta"])
        gx = float(goal["x"])
        gy = float(goal["y"])
    except (KeyError, TypeError, ValueError):
        return None
    dx = gx - px
    dy = gy - py
    dist = math.hypot(dx, dy)
    bearing_world = math.atan2(dy, dx) if dist > 1e-6 else pth
    bearing_body = _wrap_pi(bearing_world - pth)
    forward = dist * math.cos(bearing_body)
    left = dist * math.sin(bearing_body)
    return dist, bearing_body, forward, left


def describe_goal_relative(
    pose: Optional[Mapping[str, Any]],
    goal: Optional[Mapping[str, Any]],
    *,
    distance_remaining: Optional[float] = None,
) -> Optional[str]:
    """Plain-English where the goal is relative to the robot."""
    if not goal:
        return None
    label = goal.get("name")
    label_bit = f" '{label}'" if label else ""

    body = _goal_in_body(pose, goal) if pose else None
    if body is None:
        try:
            gx = float(goal["x"])
            gy = float(goal["y"])
        except (KeyError, TypeError, ValueError):
            return None
        dist = distance_remaining
        if isinstance(dist, (int, float)) and math.isfinite(float(dist)) and float(dist) > 0.05:
            return f"goal{label_bit} at map ({gx:.1f}, {gy:.1f}), about {float(dist):.1f} m remaining"
        return f"goal{label_bit} at map ({gx:.1f}, {gy:.1f})"

    dist, bearing, forward, left = body
    if dist < 0.3:
        try:
            gth = float(goal.get("theta", pose["theta"]))
            yaw_err = abs(_wrap_pi(gth - float(pose["theta"])))
        except (KeyError, TypeError, ValueError):
            yaw_err = 0.0
        if yaw_err > math.radians(15):
            return f"at goal{label_bit} position, still need to finish heading"
        return f"at goal{label_bit}"

    # 8-way relative sector from body-frame bearing.
    deg = math.degrees(bearing)
    if abs(deg) <= 22.5:
        sector = "ahead"
    elif abs(deg) >= 157.5:
        sector = "behind"
    elif 22.5 < deg <= 67.5:
        sector = "ahead and to the left"
    elif 67.5 < deg <= 112.5:
        sector = "to the left"
    elif 112.5 < deg < 157.5:
        sector = "behind and to the left"
    elif -67.5 <= deg < -22.5:
        sector = "ahead and to the right"
    elif -112.5 <= deg < -67.5:
        sector = "to the right"
    else:
        sector = "behind and to the right"

    phrase = f"goal{label_bit} is about {dist:.1f} m {sector}"
    # Prefer path distance from Nav2 when it's meaningfully different.
    if (
        isinstance(distance_remaining, (int, float))
        and math.isfinite(float(distance_remaining))
        and float(distance_remaining) > dist + 0.4
    ):
        phrase += f" (path ~{float(distance_remaining):.1f} m)"
    return phrase


def describe_goal_progress(
    last: Mapping[str, Any],
    pose: Optional[Mapping[str, Any]],
    goal: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """How the current cmd_vel relates to reaching the goal."""
    if not goal or not pose:
        return None
    body = _goal_in_body(pose, goal)
    if body is None:
        return None
    dist, bearing, _forward, _left = body
    vx = float(last.get("ros_vx_mps", 0.0) or 0.0)
    vth = float(last.get("ros_vtheta_rad_s", 0.0) or 0.0)
    moving = abs(vx) >= 0.02 or abs(vth) >= 0.05
    if not moving:
        if dist < 0.3:
            return "holding near the goal"
        return "not currently making progress toward the goal"

    try:
        yaw_err = _wrap_pi(float(goal.get("theta", pose["theta"])) - float(pose["theta"]))
    except (KeyError, TypeError, ValueError):
        yaw_err = 0.0

    # Near goal: treat angular motion as final heading alignment.
    if dist < 0.35 and abs(vth) >= 0.05 and abs(vx) < 0.05:
        turn_toward_heading = abs(yaw_err) > math.radians(10) and (vth * yaw_err) > 0
        if turn_toward_heading:
            return "aligning to the goal heading"
        return "adjusting heading at the goal"

    # In-place turn toward goal bearing.
    if abs(vx) < 0.05 and abs(vth) >= 0.05:
        if (vth * bearing) > 0 and abs(bearing) > math.radians(15):
            return "turning to face the goal"
        if (vth * bearing) < 0 and abs(bearing) > math.radians(15):
            return "turning away from the goal (likely avoiding or recovering)"
        return "rotating in place"

    bits: list[str] = []
    if vx > 0.02:
        if abs(bearing) < math.radians(45):
            bits.append("closing distance toward the goal")
        elif abs(bearing) > math.radians(100):
            bits.append("driving forward while the goal is behind (likely on a curved path)")
        else:
            bits.append("advancing while the goal is off to the side")
    elif vx < -0.02:
        if abs(bearing) < math.radians(60):
            bits.append("reversing even though the goal is ahead (likely recovering)")
        else:
            bits.append("backing up as part of the approach")

    if abs(vth) >= 0.05:
        if (vth * bearing) > 0 and abs(bearing) > math.radians(10):
            bits.append("steering toward the goal")
        elif (vth * bearing) < 0 and abs(bearing) > math.radians(10):
            bits.append("steering away from the direct line to the goal")

    if not bits:
        return None
    # Prefer a single concise clause.
    if len(bits) == 1:
        return bits[0]
    return f"{bits[0]}, {bits[1]}"


def describe_cmd_vel(
    last: Mapping[str, Any],
    *,
    max_vel_x: float = 0.75,
    max_vel_theta: float = 1.2,
    held_s: Optional[float] = None,
) -> str:
    """Describe a single ROS body-frame cmd_vel sample."""
    vx = float(last.get("ros_vx_mps", 0.0) or 0.0)
    vy = float(last.get("ros_vy_mps", 0.0) or 0.0)
    vth = float(last.get("ros_vtheta_rad_s", 0.0) or 0.0)
    source = last.get("source") or "unknown"

    if abs(vx) < 0.02 and abs(vy) < 0.02 and abs(vth) < 0.05:
        if source in ("stop", "simple_stop", "watchdog_stop"):
            base = "stopped"
        else:
            base = "commanding zero velocity"
        if held_s is not None and held_s >= 0.5:
            return f"{base} for {held_s:.1f} s"
        return base

    parts: list[str] = []
    # Prefer describing forward/back (ROS +x); mention lateral only if present.
    if abs(vx) >= 0.02:
        direction = "forward" if vx > 0 else "backward"
        parts.append(f"driving {direction} {_speed_word(abs(vx), max_vel_x)}")
    elif abs(vy) >= 0.02:
        side = "left" if vy > 0 else "right"
        parts.append(f"strafing {side} {_speed_word(abs(vy), max_vel_x)}")

    if abs(vth) >= 0.05:
        # ROS +z / CCW is left for a robot facing +x.
        turn = "left" if vth > 0 else "right"
        intensity = _turn_word(abs(vth), max_vel_theta)
        if parts:
            parts.append(f"turning {intensity} {turn}")
        else:
            parts.append(f"spinning {intensity} {turn} in place")

    if not parts:
        parts.append("holding a tiny nonzero velocity")

    # Prefer "driving forward … while turning …"
    if len(parts) == 2 and parts[0].startswith("driving") and parts[1].startswith("turning"):
        phrase = f"{parts[0]} while {parts[1]}"
    else:
        phrase = ", ".join(parts)

    if held_s is not None and held_s >= 0.3:
        phrase = f"{phrase} for about {held_s:.1f} s"
    return phrase


def _resolve_goal(status: Mapping[str, Any]) -> Optional[dict]:
    goal = status.get("goal")
    if isinstance(goal, dict) and "x" in goal and "y" in goal:
        return dict(goal)
    simple = status.get("simple_nav") or {}
    target = simple.get("target")
    if isinstance(target, dict) and "x" in target and "y" in target:
        out = dict(target)
        if goal and isinstance(goal, dict) and goal.get("name"):
            out["name"] = goal["name"]
        return out
    return None


def _resolve_pose(status: Mapping[str, Any]) -> Optional[dict]:
    pose = status.get("pose") or status.get("pose_in_map")
    if isinstance(pose, dict) and "x" in pose and "y" in pose and "theta" in pose:
        return dict(pose)
    return None


def summarize_nav_motion(
    status: Mapping[str, Any],
    *,
    max_vel_x: float = 0.75,
    max_vel_theta: float = 1.2,
) -> dict:
    """Build a plain-English motion summary from a nav ``get_status`` dict."""
    last = dict(status.get("last_cmd_vel") or {})
    history = list(status.get("cmd_vel_history") or [])
    held = _held_seconds(history, last)
    action = describe_cmd_vel(
        last, max_vel_x=max_vel_x, max_vel_theta=max_vel_theta, held_s=held
    )

    active = bool(status.get("active"))
    state = status.get("state") or "idle"
    simple = status.get("simple_nav") or {}
    motion = status.get("motion")
    if simple.get("state") == "active":
        motion = "simple"
    elif active and not motion:
        motion = "nav2"

    dist = status.get("distance_remaining")
    if dist is None:
        dist = simple.get("distance_remaining_m")
    recoveries = status.get("number_of_recoveries")
    goal = _resolve_goal(status)
    pose = _resolve_pose(status)
    goal_where = describe_goal_relative(
        pose,
        goal,
        distance_remaining=float(dist) if isinstance(dist, (int, float)) else None,
    )
    goal_why = describe_goal_progress(last, pose, goal)

    context_bits: list[str] = []
    if not active and state in ("idle", "succeeded", "canceled", None):
        if state == "canceled":
            context_bits.append("navigation canceled")
        elif state == "succeeded":
            context_bits.append("navigation succeeded")
        else:
            context_bits.append("idle")
    elif motion == "simple":
        context_bits.append("simple go_to in progress")
    elif active:
        context_bits.append("Nav2 navigating")
    else:
        context_bits.append(f"state {state}")

    if goal_where:
        context_bits.append(goal_where)
    elif isinstance(dist, (int, float)) and math.isfinite(float(dist)) and float(dist) > 0.05:
        context_bits.append(f"about {float(dist):.1f} m remaining")
    if isinstance(recoveries, int) and recoveries > 0:
        context_bits.append(f"{recoveries} recoveries so far")

    summary = f"{context_bits[0]}: {action}"
    if len(context_bits) > 1:
        summary = f"{context_bits[0]} ({'; '.join(context_bits[1:])}): {action}"
    if goal_why:
        summary = f"{summary} — {goal_why}"

    return {
        "summary": summary,
        "action": action,
        "goal_relative": goal_where,
        "toward_goal": goal_why,
        "goal": goal,
        "pose": pose,
        "active": active,
        "state": state,
        "motion": motion,
        "held_s": held,
        "last_cmd_vel": last,
    }
