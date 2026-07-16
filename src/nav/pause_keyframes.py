"""Pause keyframes for mapping-time revisit matching.

With ``map_when_still``, slam_toolbox only sees stop scans. Returning to a
previously mapped area often means a *different* stop pose/angle than before,
so a thin occupancy silhouette can score poorly. Each accepted still publish
stores a compact keyframe — primary-band 2D endpoints plus multi-height slice
points — and revisit matching can snap to those views even when the raster map
is ambiguous.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ros import conversions as conv
from .global_localize import scan_endpoints_base_link


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def _to_map_xy(pts_xy: np.ndarray, pose: conv.Pose2D) -> np.ndarray:
    pts_xy = np.asarray(pts_xy, dtype=float)
    if pts_xy.size == 0:
        return np.empty((0, 2))
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return np.stack(
        [
            pose.x + c * pts_xy[:, 0] - s * pts_xy[:, 1],
            pose.y + s * pts_xy[:, 0] + c * pts_xy[:, 1],
        ],
        axis=1,
    )


def nn_hit_rate(
    query_xy: np.ndarray, ref_xy: np.ndarray, *, tol_m: float = 0.3
) -> Optional[float]:
    """Fraction of query points within ``tol_m`` of some reference point."""
    query_xy = np.asarray(query_xy, dtype=float)
    ref_xy = np.asarray(ref_xy, dtype=float)
    if query_xy.size == 0 or ref_xy.size == 0:
        return None
    # Brute-force is fine for pause keyframes (~few hundred points each).
    d2 = (
        (query_xy[:, None, 0] - ref_xy[None, :, 0]) ** 2
        + (query_xy[:, None, 1] - ref_xy[None, :, 1]) ** 2
    )
    nearest = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(nearest <= tol_m))


@dataclass
class PauseKeyframe:
    pose: conv.Pose2D
    endpoints: np.ndarray  # (N, 2) base_link primary-band
    band_points: List[np.ndarray] = field(default_factory=list)
    wall_time: float = 0.0


@dataclass(frozen=True)
class KeyframeMatchResult:
    pose: conv.Pose2D
    score: float
    keyframe_index: int
    primary_hit_rate: float
    slice_hit_rate: Optional[float]
    keyframes_considered: int


class PauseKeyframeStore:
    """Session store of still-publish keyframes for scan(+slice) matching."""

    def __init__(
        self,
        *,
        min_spacing_m: float = 0.5,
        min_spacing_deg: float = 20.0,
        max_keyframes: int = 250,
        match_tol_m: float = 0.3,
        xy_offsets_m: Sequence[float] = (0.0, 0.4, 0.8),
        yaw_step_deg: float = 15.0,
    ):
        self.min_spacing_m = float(min_spacing_m)
        self.min_spacing_deg = float(min_spacing_deg)
        self.max_keyframes = int(max_keyframes)
        self.match_tol_m = float(match_tol_m)
        self.xy_offsets_m = tuple(float(v) for v in xy_offsets_m)
        self.yaw_step_deg = float(yaw_step_deg)
        self._frames: List[PauseKeyframe] = []

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def add(
        self,
        pose: conv.Pose2D,
        scan: conv.LaserScan2D,
        band_points: Optional[Sequence[np.ndarray]] = None,
    ) -> bool:
        """Store a pause keyframe; False when too close to an existing one."""
        endpoints = scan_endpoints_base_link(scan)
        if endpoints.shape[0] < 16:
            return False
        # Subsample for memory / match cost.
        if endpoints.shape[0] > 240:
            idx = np.linspace(0, endpoints.shape[0] - 1, 240, dtype=np.int64)
            endpoints = endpoints[idx]
        for prev in self._frames:
            dist = math.hypot(pose.x - prev.pose.x, pose.y - prev.pose.y)
            dyaw = abs(
                math.degrees(_normalize_angle(pose.theta - prev.pose.theta))
            )
            if dist < self.min_spacing_m and dyaw < self.min_spacing_deg:
                return False
        bands = [np.asarray(p, dtype=float) for p in (band_points or [])]
        self._frames.append(
            PauseKeyframe(
                pose=pose,
                endpoints=endpoints,
                band_points=bands,
                wall_time=time.monotonic(),
            )
        )
        if len(self._frames) > self.max_keyframes:
            self._frames = self._frames[-self.max_keyframes :]
        return True

    def near(
        self, pose: conv.Pose2D, radius_m: float
    ) -> List[Tuple[int, PauseKeyframe]]:
        r2 = float(radius_m) ** 2
        out: List[Tuple[int, PauseKeyframe]] = []
        for i, kf in enumerate(self._frames):
            if (kf.pose.x - pose.x) ** 2 + (kf.pose.y - pose.y) ** 2 <= r2:
                out.append((i, kf))
        return out

    def match(
        self,
        scan: conv.LaserScan2D,
        band_points: Optional[Sequence[np.ndarray]] = None,
        *,
        hint: Optional[conv.Pose2D] = None,
        search_radius_m: Optional[float] = None,
    ) -> Optional[KeyframeMatchResult]:
        """Best pose explaining ``scan`` against stored pause views.

        When ``search_radius_m`` is set with a ``hint``, only nearby keyframes
        are tried; otherwise every keyframe is a seed (for large odom drift).
        """
        if not self._frames:
            return None
        query = scan_endpoints_base_link(scan)
        if query.shape[0] < 16:
            return None
        if query.shape[0] > 240:
            idx = np.linspace(0, query.shape[0] - 1, 240, dtype=np.int64)
            query = query[idx]
        q_bands = [np.asarray(p, dtype=float) for p in (band_points or [])]

        if search_radius_m is not None and hint is not None:
            targets = self.near(hint, search_radius_m)
        else:
            targets = list(enumerate(self._frames))
        if not targets:
            return None

        yaw_step = math.radians(max(self.yaw_step_deg, 1.0))
        n_yaw = max(1, int(round((2.0 * math.pi) / yaw_step)))
        best: Optional[KeyframeMatchResult] = None

        for idx, kf in targets:
            ref_map = _to_map_xy(kf.endpoints, kf.pose)
            ref_bands_map = [
                _to_map_xy(bp, kf.pose) for bp in kf.band_points if np.asarray(bp).size
            ]
            for dx in self.xy_offsets_m:
                for dy in self.xy_offsets_m:
                    # Keep search cheap: only axis-aligned offsets + origin.
                    if dx != 0.0 and dy != 0.0:
                        continue
                    for iy in range(n_yaw):
                        pose = conv.Pose2D(
                            kf.pose.x + dx,
                            kf.pose.y + dy,
                            _normalize_angle(kf.pose.theta + iy * yaw_step),
                        )
                        q_map = _to_map_xy(query, pose)
                        primary = nn_hit_rate(
                            q_map, ref_map, tol_m=self.match_tol_m
                        )
                        if primary is None:
                            continue
                        slice_rate = self._band_hit_rate(
                            q_bands, ref_bands_map, pose
                        )
                        if slice_rate is None:
                            score = primary
                        else:
                            score = 0.65 * primary + 0.35 * slice_rate
                        if best is None or score > best.score:
                            best = KeyframeMatchResult(
                                pose=pose,
                                score=score,
                                keyframe_index=idx,
                                primary_hit_rate=primary,
                                slice_hit_rate=slice_rate,
                                keyframes_considered=len(targets),
                            )
        return best

    def _band_hit_rate(
        self,
        query_bands: Sequence[np.ndarray],
        ref_bands_map: Sequence[np.ndarray],
        pose: conv.Pose2D,
    ) -> Optional[float]:
        rates: List[float] = []
        for q, ref in zip(query_bands, ref_bands_map):
            q = np.asarray(q, dtype=float)
            if q.size == 0 or np.asarray(ref).size == 0:
                continue
            if q.shape[0] > 200:
                idx = np.linspace(0, q.shape[0] - 1, 200, dtype=np.int64)
                q = q[idx]
            rate = nn_hit_rate(_to_map_xy(q, pose), ref, tol_m=self.match_tol_m)
            if rate is not None:
                rates.append(rate)
        if not rates:
            return None
        return float(sum(rates) / len(rates))

    def status(self) -> Dict:
        return {
            "count": len(self._frames),
            "max": self.max_keyframes,
            "min_spacing_m": self.min_spacing_m,
        }
