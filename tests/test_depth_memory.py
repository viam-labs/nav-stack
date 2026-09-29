"""Depth-camera obstacle memory: keep what left the camera's narrow view."""
from __future__ import annotations

import math

import numpy as np

from src.config import LidarConfig
from src.geom import conversions as conv
from src.nav_builtin.depth_memory import DepthObstacleMemory

CAM = LidarConfig(
    name="camera", x=0.30, y=-0.15, min_range=0.2, max_range=2.0, obstacles_only=True
)


def _frame(pose: conv.Pose2D, world_pts) -> conv.LaserScan2D:
    """Base-frame depth scan at ``pose`` seeing ``world_pts``."""
    pts = np.asarray(world_pts, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    dx, dy = pts[:, 0] - pose.x, pts[:, 1] - pose.y
    body = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
    scan = conv.points_to_scan(
        body, angle_min=-math.pi, angle_max=math.pi, num_bins=360, range_min=0.0, range_max=2.0
    )
    return conv.LaserScan2D(
        scan.ranges,
        angle_min=scan.angle_min,
        angle_increment=scan.angle_increment,
        range_min=scan.range_min,
        range_max=scan.range_max,
        capture_pose=pose,
    )


def _mem() -> DepthObstacleMemory:
    return DepthObstacleMemory(length_m=0.72, width_m=0.59)


LEG = (1.0, -0.40)  # low part of a table leg, right of the route


def test_leg_that_slides_out_of_view_is_remembered_beside_the_body():
    m = _mem()
    p0 = conv.Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, _frame(p0, [LEG]), CAM)], p0, 1.0)
    assert len(m) == 1
    # Drove 1 m: the leg is now beside the right flank, far outside the 87 deg
    # view, and the camera frame is empty.
    p1 = conv.Pose2D(1.0, 0.0, 0.0)
    m.update([(2.0, _frame(p1, []), CAM)], p1, 2.0)
    pts = m.points()
    assert len(pts) == 1 and np.allclose(pts[0], LEG, atol=0.01)


def test_camera_looking_again_and_seeing_nothing_clears_it():
    m = _mem()
    p0 = conv.Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, _frame(p0, [LEG]), CAM)], p0, 1.0)
    m.update([(2.0, _frame(p0, []), CAM)], p0, 2.0)
    assert len(m) == 0


def test_still_seen_is_kept_and_same_frame_is_not_reapplied():
    m = _mem()
    p0 = conv.Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, _frame(p0, [LEG]), CAM)], p0, 1.0)
    m.update([(2.0, _frame(p0, [LEG]), CAM)], p0, 2.0)
    assert len(m) == 1
    # The cached frame is re-read every control tick after the robot turned
    # away; it must not be treated as a new look that sees nothing.
    turned = conv.Pose2D(0.0, 0.0, 1.2)
    m.update([(2.0, _frame(p0, [LEG]), CAM)], turned, 2.1)
    assert len(m) == 1


def test_forgets_what_the_body_covers_or_is_far_or_old():
    m = _mem()
    p0 = conv.Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, _frame(p0, [LEG, (1.0, 0.0)]), CAM)], p0, 1.0)
    assert len(m) == 2
    on_top = conv.Pose2D(1.0, 0.0, math.pi / 2)  # body now covers (1, 0)
    m.update([], on_top, 1.5)
    assert len(m) == 1
    away = conv.Pose2D(3.0, 0.0, math.pi)
    m.update([], away, 2.0)
    assert len(m) == 0
    m.update([(3.0, _frame(p0, [LEG]), CAM)], p0, 3.0)
    beside = conv.Pose2D(1.0, 0.0, 0.0)
    m.update([], beside, 3.0 + 25.0)
    assert len(m) == 0
