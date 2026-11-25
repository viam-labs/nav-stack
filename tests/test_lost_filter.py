"""The lost filter: a particle filter that finds a robot with no pose, as it drives."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.geom import conversions as conv
from src.nav.localize.global_localize import OccupancyMap
from src.slam_builtin.lost_filter import (
    CHECKING,
    CONFIRMING,
    FINDING,
    WATCHING,
    LostFilter,
    distance_field,
)

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1

RES = 0.05
# A 14 m x 4 m room. With a cabinet in one corner and a partition halfway
# across it looks different from every place in it; bare, it looks the same
# turned half a turn. (Small features are not enough: a pillar and the
# cabinet left a third of the particles on the half-turned twin.)
ROOM: Box = (-7.0, -2.0, 7.0, 2.0)
FEATURES: List[Box] = [(5.5, 1.2, 6.5, 2.0), (-1.0, -2.0, -0.8, -0.4)]
START = conv.Pose2D(-6.0, 0.3, 0.0)


def _occupancy(boxes: List[Box]) -> OccupancyMap:
    x0, y0, x1, y1 = ROOM
    ox, oy = x0 - 1.0, y0 - 1.0
    w = int(round((x1 - x0 + 2.0) / RES))
    h = int(round((y1 - y0 + 2.0) / RES))
    xs = ox + (np.arange(w) + 0.5) * RES
    ys = oy + (np.arange(h) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys)
    grid = np.full((h, w), -1, dtype=np.int16)
    inside = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
    grid[inside] = 0
    walls = inside & (
        (gx < x0 + RES) | (gx > x1 - RES) | (gy < y0 + RES) | (gy > y1 - RES)
    )
    for bx0, by0, bx1, by1 in boxes:
        walls |= (gx >= bx0) & (gx <= bx1) & (gy >= by0) & (gy <= by1)
    grid[walls] = 100
    return OccupancyMap(grid=grid, resolution=RES, origin_x=ox, origin_y=oy)


def _ray_box(px, py, dx, dy, box: Box) -> np.ndarray:
    """Distance along each ray to a solid box (inf when it misses)."""
    x0, y0, x1, y1 = box
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0, tx1 = (x0 - px) / dx, (x1 - px) / dx
        ty0, ty1 = (y0 - py) / dy, (y1 - py) / dy
    tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def _scan(pose: conv.Pose2D, boxes: List[Box]) -> conv.LaserScan2D:
    angles = -math.pi + np.arange(360) * (2 * math.pi / 360)
    dx, dy = np.cos(angles + pose.theta), np.sin(angles + pose.theta)
    x0, y0, x1, y1 = ROOM
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (x1 - pose.x) / dx, (x0 - pose.x) / dx)
        ty = np.where(dy > 0, (y1 - pose.y) / dy, (y0 - pose.y) / dy)
    r = np.minimum(
        np.where(np.isfinite(tx), tx, np.inf), np.where(np.isfinite(ty), ty, np.inf)
    )
    for box in boxes:
        r = np.minimum(r, _ray_box(pose.x, pose.y, dx, dy, box))
    return conv.LaserScan2D(r, -math.pi, 2 * math.pi / 360, 0.05, 25.0)


def _drive(
    filt: LostFilter,
    boxes: List[Box],
    *,
    seconds: float,
    tracker_after: Optional[conv.Pose2D] = None,
    speed: float = 0.5,
    until_handover: bool = False,
):
    """Drive from ``START`` along +x; the tracker takes each hand-over, carried by odometry.

    Returns the hand-overs (each with the true pose then), and the last time,
    odometry and true pose.
    """
    handovers = []
    tracker = tracker_after or conv.Pose2D(0.0, 0.0, 0.0)
    prev_odom = conv.Pose2D(0.0, 0.0, 0.0)
    t, odom, truth = 0.0, prev_odom, START
    for k in range(int(seconds * 15)):
        t = k / 15.0
        truth = conv.Pose2D(START.x + speed * t, START.y, START.theta)
        odom = conv.Pose2D(speed * t, 0.0, 0.0)
        tracker = conv.compose_poses(
            tracker, conv.compose_poses(conv.invert_pose(prev_odom), odom)
        )
        prev_odom = odom
        out = filt.step(t, odom, _scan(truth, boxes), tracker)
        if out is not None:
            handovers.append((out, truth))
            tracker = out.pose
            if until_handover:
                break
    return handovers, (t, odom, truth)


def _filter(boxes: List[Box], **kw) -> LostFilter:
    occ = _occupancy(boxes)
    kw.setdefault("has_pose", False)
    return LostFilter(occ, field=distance_field(occ, 2.0), **kw)


def test_finds_a_robot_with_no_pose_once_it_has_driven_and_confirms_it():
    filt = _filter(FEATURES)
    assert filt.phase == FINDING
    handovers, _ = _drive(filt, FEATURES, seconds=23.0)
    assert len(handovers) == 1
    out, truth = handovers[0]
    assert math.hypot(out.pose.x - truth.x, out.pose.y - truth.y) < 0.3
    assert abs(conv.normalize_angle(out.pose.theta - truth.theta)) < math.radians(10)
    # Provisional for 2 m, then confirmed.
    assert filt.phase == WATCHING
    assert filt.handovers == 1 and filt.withdrawn == 0
    assert any("confirmed" in line for line in filt.log)


def test_does_not_hand_over_in_a_room_that_looks_the_same_both_ways():
    filt = _filter([])
    handovers, _ = _drive(filt, [], seconds=20.0)
    assert handovers == []
    assert filt.phase == FINDING


def test_a_hand_over_the_scans_stop_fitting_is_withdrawn():
    filt = _filter(FEATURES)
    handovers, (t, odom, truth) = _drive(
        filt, FEATURES, seconds=23.0, until_handover=True
    )
    assert len(handovers) == 1 and filt.phase == CONFIRMING
    # The tracker is somewhere else now (say, a bad match pulled it 2 m off)
    # and the robot has stopped: the scans stop fitting before the hand-over
    # is confirmed.
    out = handovers[0][0]
    wrong = conv.Pose2D(out.pose.x - 2.0, out.pose.y + 1.0, out.pose.theta + 0.5)
    for k in range(1, 60):
        filt.step(t + k / 15.0, odom, _scan(truth, FEATURES), wrong)
    assert filt.withdrawn == 1
    assert filt.phase == FINDING
    assert any("searching again" in line for line in filt.log)


def test_watching_a_tracker_that_fits_does_nothing_and_a_wrong_one_starts_a_check():
    filt = _filter(FEATURES, has_pose=True)
    handovers, (t, odom, truth) = _drive(
        filt, FEATURES, seconds=6.0, tracker_after=START
    )
    assert handovers == [] and filt.phase == WATCHING
    wrong = conv.Pose2D(2.0, -1.0, 1.0)
    for k in range(1, 10 * 15):
        filt.step(t + k / 15.0, odom, _scan(truth, FEATURES), wrong)
    assert filt.phase == CHECKING


def test_the_camera_can_hold_back_a_hand_over_and_point_the_search():
    asked = []

    def camera(pose, now):
        asked.append(pose)
        return "camera backs another place", [conv.Pose2D(0.0, 0.0, 0.0)]

    filt = _filter(FEATURES, camera=camera)
    handovers, _ = _drive(filt, FEATURES, seconds=23.0)
    assert handovers == []
    assert asked and filt.phase == FINDING
    assert any("held back" in line for line in filt.log)


def test_distance_field_is_exact_near_walls_and_capped_beyond():
    occ = _occupancy([])
    field = distance_field(occ, 2.0)
    r, c = int((0.0 - occ.origin_y) / RES), int((ROOM[0] + 0.3 - occ.origin_x) / RES)
    # About 0.3 m from the left wall (cell centers, so within a cell).
    assert abs(float(field[r, c]) - 0.3) < 0.06
    r0 = int((0.0 - occ.origin_y) / RES)
    c0 = int((0.0 - occ.origin_x) / RES)
    assert float(field[r0, c0]) == 2.0  # the middle is far from every wall
