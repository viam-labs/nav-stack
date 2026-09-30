"""Scan-to-local-map ICP: the lidar is the odometry when wheels are absent.

Each scan is aligned (2D point-to-point ICP) against the last few keyscans,
starting from a constant-velocity guess. A cart with only a lidar and an IMU
gets its translation from this, the same way ROS lidar-odometry stacks do.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from ..geom import conversions as conv


@dataclass(frozen=True)
class IcpResult:
    pose: conv.Pose2D
    inlier_ratio: float
    rms_m: float
    inliers: int


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """One mean point per ``voxel_m`` square cell."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0 or voxel_m <= 0.0:
        return pts
    keys = np.floor(pts[:, :2] / voxel_m).astype(np.int64)
    key = keys[:, 0] * 1_000_003 + keys[:, 1]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    out = np.zeros((counts.shape[0], 2), dtype=float)
    np.add.at(out, inverse, pts[:, :2])
    return out / counts[:, None]


def _transform(points: np.ndarray, pose: conv.Pose2D) -> np.ndarray:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x = pose.x + c * points[:, 0] - s * points[:, 1]
    y = pose.y + s * points[:, 0] + c * points[:, 1]
    return np.stack([x, y], axis=1)


def _nearest(query: np.ndarray, ref: np.ndarray, ref_sq: np.ndarray):
    """Nearest ``ref`` index and squared distance for each ``query`` row."""
    d2 = (
        np.sum(query * query, axis=1)[:, None]
        + ref_sq[None, :]
        - 2.0 * (query @ ref.T)
    )
    idx = np.argmin(d2, axis=1)
    best = np.maximum(d2[np.arange(query.shape[0]), idx], 0.0)
    return idx, best


def icp_2d(
    src: np.ndarray,
    ref: np.ndarray,
    guess: conv.Pose2D,
    *,
    max_corr_m: float = 0.40,
    min_corr_m: float = 0.12,
    iterations: int = 20,
    min_inliers: int = 40,
) -> Optional[IcpResult]:
    """Align ``src`` (base_link XY) to ``ref`` (map XY) starting at ``guess``."""
    if src.shape[0] < min_inliers or ref.shape[0] < min_inliers:
        return None
    ref_sq = np.sum(ref * ref, axis=1)
    x, y, th = float(guess.x), float(guess.y), float(guess.theta)
    for it in range(iterations):
        pose = conv.Pose2D(x, y, th)
        moved = _transform(src, pose)
        idx, d2 = _nearest(moved, ref, ref_sq)
        corr = max(min_corr_m, max_corr_m * (0.75 ** it))
        keep = d2 <= corr * corr
        if int(keep.sum()) < min_inliers:
            return None
        a = moved[keep]
        b = ref[idx[keep]]
        ma = a.mean(axis=0)
        mb = b.mean(axis=0)
        h = (a - ma).T @ (b - mb)
        dth = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = math.cos(dth), math.sin(dth)
        tx = mb[0] - (c * ma[0] - s * ma[1])
        ty = mb[1] - (s * ma[0] + c * ma[1])
        x, y = c * x - s * y + tx, s * x + c * y + ty
        th = conv.normalize_angle(th + dth)
        if abs(dth) < 1e-4 and math.hypot(tx, ty) < 1e-4 and corr <= min_corr_m:
            break
    pose = conv.Pose2D(x, y, th)
    _, d2 = _nearest(_transform(src, pose), ref, ref_sq)
    keep = d2 <= min_corr_m * min_corr_m
    n_in = int(keep.sum())
    if n_in == 0:
        return None
    return IcpResult(
        pose=pose,
        inlier_ratio=n_in / float(src.shape[0]),
        rms_m=float(math.sqrt(float(np.mean(d2[keep])))),
        inliers=n_in,
    )


class LidarOdometry:
    """Track the base from scans alone against a rolling local keyscan map."""

    # Brute-force nearest neighbours cost scan_points x ref_points per ICP
    # iteration; these sizes keep one match in the tens of ms on a Pi.
    def __init__(
        self,
        *,
        scan_voxel_m: float = 0.10,
        ref_voxel_m: float = 0.08,
        ref_radius_m: float = 10.0,
        max_keyscans: int = 40,
        key_dist_m: float = 0.15,
        key_yaw_rad: float = math.radians(10.0),
        max_points: int = 250,
        min_inlier_ratio: float = 0.45,
        max_correction_m: float = 0.50,
        max_correction_rad: float = math.radians(15.0),
    ):
        self._scan_voxel = float(scan_voxel_m)
        self._voxel = float(ref_voxel_m)
        self._ref_radius = float(ref_radius_m)
        self._keys: Deque[np.ndarray] = deque(maxlen=int(max_keyscans))
        self._key_pose: Optional[conv.Pose2D] = None
        self._key_dist = float(key_dist_m)
        self._key_yaw = float(key_yaw_rad)
        self._max_points = int(max_points)
        self._min_inlier_ratio = float(min_inlier_ratio)
        self._max_corr_m = float(max_correction_m)
        self._max_corr_rad = float(max_correction_rad)
        self._map_pts: Optional[np.ndarray] = None
        self._ref: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Drop the recent keyscans. Map points stay until replaced."""
        self._keys.clear()
        self._key_pose = None
        self._ref = None

    def set_map_points(self, points: Optional[np.ndarray]) -> None:
        """Occupied map cells (map XY). Matching against them keeps a revisit
        on the walls already drawn after the recent keyscans have rolled off."""
        if points is None or len(points) == 0:
            self._map_pts = None
        else:
            self._map_pts = voxel_downsample(points, self._voxel)
        self._ref = None

    def has_reference(self) -> bool:
        return bool(self._keys)

    def _prepare(self, scan_xy: np.ndarray) -> np.ndarray:
        pts = voxel_downsample(scan_xy, self._scan_voxel)
        if self._max_points > 0 and pts.shape[0] > self._max_points:
            pick = np.linspace(0, pts.shape[0] - 1, self._max_points, dtype=np.int64)
            pts = pts[pick]
        return pts

    def add_keyscan(
        self, scan_xy: np.ndarray, pose: conv.Pose2D, *, force: bool = False
    ) -> bool:
        """Add this scan to the local map once the base has moved enough."""
        last = self._key_pose
        if not force and last is not None:
            moved = math.hypot(pose.x - last.x, pose.y - last.y)
            turned = abs(conv.normalize_angle(pose.theta - last.theta))
            if moved < self._key_dist and turned < self._key_yaw:
                return False
        pts = voxel_downsample(scan_xy, self._voxel)
        if pts.shape[0] == 0:
            return False
        self._keys.append(_transform(pts, pose))
        self._key_pose = pose
        self._ref = None
        return True

    def _reference(self) -> np.ndarray:
        if self._ref is None:
            parts = [voxel_downsample(np.vstack(list(self._keys)), self._voxel)]
            if self._map_pts is not None:
                parts.append(self._map_pts)
            self._ref = np.vstack(parts)
        return self._ref

    def match(
        self, scan_xy: np.ndarray, guess: conv.Pose2D
    ) -> Optional[IcpResult]:
        """ICP ``scan_xy`` against the local map. ``None`` when it does not fit."""
        if not self._keys:
            return None
        src = self._prepare(scan_xy)
        ref = self._reference()
        near = np.hypot(ref[:, 0] - guess.x, ref[:, 1] - guess.y) <= self._ref_radius
        result = icp_2d(src, ref[near], guess)
        if result is None or result.inlier_ratio < self._min_inlier_ratio:
            return None
        dist = math.hypot(result.pose.x - guess.x, result.pose.y - guess.y)
        dyaw = abs(conv.normalize_angle(result.pose.theta - guess.theta))
        if dist > self._max_corr_m or dyaw > self._max_corr_rad:
            return None
        return result
