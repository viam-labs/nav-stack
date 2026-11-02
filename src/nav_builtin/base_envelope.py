"""Base deadband / crawl envelope shared by drive, guard, and recoveries.

``ViamWorldIO`` snaps speeds below these floors to zero (and turns a tiny
``vx`` plus large ``vθ`` into a pure spin). Every layer that shapes a twist
must use the same numbers, or a command that looks nonzero never reaches
the wheels and the bumper stays put.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from ..nav.simple_motion import DriveCommand

# Snap thresholds in ``sanitize_base_cmd``.
LIN_EPS_MPS = 0.05
ANG_EPS_RAD_S = 0.08
# Below this linear speed with |vθ| above SPIN_W_CAP, the base drops vx.
SPIN_VX_FLOOR_MPS = 0.12
SPIN_W_CAP_RAD_S = 0.25
# Safe crawl that clears SPIN_VX_FLOOR so an arc is not turned into a spin.
CRAWL_FLOOR_MPS = 0.125
# Yaw used when a forward crawl into a blocked nose would be dropped.
BLOCKED_NOSE_TURN_RAD_S = 0.4


def twist_reaches_base(vx: float, vtheta: float) -> bool:
    """False when the base driver will drop both axes."""
    return abs(vx) >= LIN_EPS_MPS or abs(vtheta) >= ANG_EPS_RAD_S


def sanitize_base_cmd(
    vx: float, vy: float, vtheta: float
) -> Tuple[float, float, float]:
    """Snap sub-deadband speeds to zero so wheeled bases do not reject RPM.

    Diff-drive with tiny ``vx`` + large ``vtheta`` also drives one wheel
    through ~0 RPM — prefer pure spin in that case.
    """
    if abs(vx) < LIN_EPS_MPS:
        vx = 0.0
    if abs(vy) < LIN_EPS_MPS:
        vy = 0.0
    if abs(vtheta) < ANG_EPS_RAD_S:
        vtheta = 0.0
    if vx != 0.0 and abs(vx) < SPIN_VX_FLOOR_MPS and abs(vtheta) > SPIN_W_CAP_RAD_S:
        vx = 0.0
    return vx, vy, vtheta


def keep_arc_inside_envelope(
    vx: float, vtheta: float, *, spin_vx_floor: float = CRAWL_FLOOR_MPS
) -> Tuple[float, float]:
    """Cap yaw on a slow arc so the sanitizer does not convert it to a spin."""
    if 0.0 < abs(vx) < spin_vx_floor and abs(vtheta) > SPIN_W_CAP_RAD_S:
        vtheta = math.copysign(SPIN_W_CAP_RAD_S, vtheta)
    return vx, vtheta


def lift_sub_deadband(
    cmd: DriveCommand,
    *,
    forward_clearance_m: Optional[float] = None,
    body_length_m: Optional[float] = None,
) -> DriveCommand:
    """Make a nonzero twist survive ``sanitize_base_cmd``.

    A reverse of a few cm/s is what the cart was publishing while
    ``last_drive`` stayed zero. Raise that reverse to a backup-scale crawl.
    A tiny forward command into a blocked nose is a stop, not a crawl.
    A forward arc whose yaw the base will drop is a straight drive into
    the bumper — turn in place instead.
    """
    if (
        not cmd.done
        and cmd.vx > LIN_EPS_MPS
        and 0.0 < abs(cmd.vtheta) < ANG_EPS_RAD_S
        and body_length_m is not None
        and forward_clearance_m is not None
        and forward_clearance_m <= body_length_m / 2.0 + 0.08
    ):
        return DriveCommand(
            0.0, 0.0, math.copysign(BLOCKED_NOSE_TURN_RAD_S, cmd.vtheta), False
        )

    if cmd.done or twist_reaches_base(cmd.vx, cmd.vtheta):
        return cmd
    if abs(cmd.vx) < 1e-6 and abs(cmd.vtheta) < 1e-6:
        return cmd
    if cmd.vx > 1e-6:
        return DriveCommand(0.0, 0.0, 0.0, False)
    if cmd.vx < -1e-6:
        return DriveCommand(
            -max(0.10, abs(cmd.vx), LIN_EPS_MPS), 0.0, 0.0, False
        )
    return DriveCommand(
        0.0,
        0.0,
        math.copysign(max(ANG_EPS_RAD_S, abs(cmd.vtheta)), cmd.vtheta),
        False,
    )


# Aliases kept for call sites that used the footprint_guard names.
BASE_LIN_EPS_MPS = LIN_EPS_MPS
BASE_ANG_EPS_RAD_S = ANG_EPS_RAD_S
