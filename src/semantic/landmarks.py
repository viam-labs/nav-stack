"""Embedding landmark map: candidates, promotion, demotion, visible set."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from .types import LandmarkStatus


@dataclass
class EmbeddingLandmark:
    landmark_id: int
    anchor_node: int
    # Position relative to the anchor keyframe pose (base_link of that node).
    rel_xyz: np.ndarray  # (3,)
    normal_rel: np.ndarray  # (3,) facing outward-ish in anchor frame
    # Absolute map-frame position — authoritative for overlay / association.
    # rel_xyz alone is unsafe when anchor 0 is a live pose or later remapped
    # to keyframe 0 (spinning made boxes slide on landmark-camera).
    map_xyz: Optional[np.ndarray] = None
    descriptors: List[np.ndarray] = field(default_factory=list)  # up to 3 (D,)
    observations: int = 0
    # Robot poses that saw this landmark: (x, y, yaw). Yaw matters so a
    # spin-in-place can still count as multi-view.
    view_poses_xy: List[Tuple[float, float, float]] = field(default_factory=list)
    position_std_m: float = float("inf")
    status: LandmarkStatus = LandmarkStatus.CANDIDATE
    # Running mean in map frame while candidate (for std); cleared on confirm.
    _map_samples: List[np.ndarray] = field(default_factory=list, repr=False)


@dataclass
class EmbeddingLandmarkMap:
    """Session landmark store keyed by integer id."""

    associate_radius_m: float = 0.25
    promote_min_obs: int = 3
    promote_min_views: int = 2
    promote_min_view_gap_m: float = 0.5
    # Spin-in-place: promote when yaw span across views reaches this (deg).
    promote_min_view_gap_deg: float = 25.0
    # Depth + association noise routinely exceeds 5 cm on RealSense; 12 cm
    # still rejects wild jumps while letting stable features confirm.
    promote_max_std_m: float = 0.12
    demote_offset_m: float = 0.15
    demote_passes: int = 3
    max_descriptors: int = 3
    cell_m: float = 0.30
    cosine_threshold: float = 0.55
    # Softer gate for candidates: DINOv2 patch tokens drift under approach/zoom,
    # but orthogonal / wrong-surface patches must not steal the same ID
    # (landmark-camera would then jump between objects).
    candidate_cosine_threshold: float = 0.35
    # When False (synthetic RGB stats), associate by distance only so nearby
    # patches collapse instead of spawning one landmark per grid cell.
    require_descriptor_match: bool = True
    # Min XY (m) / yaw (deg) change before recording another view pose.
    view_record_xy_m: float = 0.15
    view_record_yaw_deg: float = 12.0

    def __post_init__(self) -> None:
        self._next_id = 1
        self._landmarks: Dict[int, EmbeddingLandmark] = {}
        self._cell_index: Dict[Tuple[int, int, int], List[int]] = {}
        self._demote_strikes: Dict[int, int] = {}

    def rebuild_cell_index(self) -> None:
        """Re-hash landmarks after ``cell_m`` changes (load / config apply)."""
        self._cell_index = {}
        for lm in self._landmarks.values():
            xyz = self.resolved_map_xyz(lm)
            self._index_add(lm, xyz)

    @staticmethod
    def resolved_map_xyz(
        lm: EmbeddingLandmark, pose_of_node: Optional[Dict[int, conv.Pose2D]] = None
    ) -> np.ndarray:
        """Absolute map position for overlay; prefer locked ``map_xyz``."""
        if lm.map_xyz is not None:
            return np.asarray(lm.map_xyz, dtype=float).reshape(3)
        if pose_of_node is not None:
            ap = pose_of_node.get(lm.anchor_node)
            if ap is not None:
                return EmbeddingLandmarkMap.map_point_from_rel(ap, lm.rel_xyz)
        return np.asarray(lm.rel_xyz, dtype=float).reshape(3)

    @staticmethod
    def association_map_xyz(lm: EmbeddingLandmark) -> Optional[np.ndarray]:
        """Map position used for re-association (running mean while candidate)."""
        if lm.status == LandmarkStatus.CANDIDATE and lm._map_samples:
            return np.stack(lm._map_samples, axis=0).mean(axis=0)
        if lm.map_xyz is not None:
            return np.asarray(lm.map_xyz, dtype=float).reshape(3)
        return None

    def __len__(self) -> int:
        return len(self._landmarks)

    def get(self, landmark_id: int) -> Optional[EmbeddingLandmark]:
        return self._landmarks.get(int(landmark_id))

    def all_landmarks(self) -> List[EmbeddingLandmark]:
        return list(self._landmarks.values())

    def confirmed(self) -> List[EmbeddingLandmark]:
        return [
            lm
            for lm in self._landmarks.values()
            if lm.status == LandmarkStatus.CONFIRMED
        ]

    @staticmethod
    def _cell(xyz: np.ndarray, cell_m: float) -> Tuple[int, int, int]:
        c = max(1e-6, float(cell_m))
        return (
            int(math.floor(float(xyz[0]) / c)),
            int(math.floor(float(xyz[1]) / c)),
            int(math.floor(float(xyz[2]) / c)),
        )

    def _index_add(self, lm: EmbeddingLandmark, map_xyz: np.ndarray) -> None:
        cell = self._cell(map_xyz, self.cell_m)
        self._cell_index.setdefault(cell, []).append(lm.landmark_id)

    def _neighbors(
        self, map_xyz: np.ndarray, *, radius_m: Optional[float] = None
    ) -> List[EmbeddingLandmark]:
        base = self._cell(map_xyz, self.cell_m)
        reach = 1
        if radius_m is not None and radius_m > 0.0:
            reach = max(1, int(math.ceil(float(radius_m) / max(1e-6, float(self.cell_m)))))
        out: List[EmbeddingLandmark] = []
        seen: set = set()
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    key = (base[0] + dx, base[1] + dy, base[2] + dz)
                    for lid in self._cell_index.get(key, ()):
                        if lid in seen:
                            continue
                        lm = self._landmarks.get(lid)
                        if lm is not None and lm.status != LandmarkStatus.MOVED:
                            seen.add(lid)
                            out.append(lm)
        return out

    @staticmethod
    def map_point_from_rel(anchor_pose: conv.Pose2D, rel_xyz: np.ndarray) -> np.ndarray:
        c, s = math.cos(anchor_pose.theta), math.sin(anchor_pose.theta)
        x = anchor_pose.x + c * float(rel_xyz[0]) - s * float(rel_xyz[1])
        y = anchor_pose.y + s * float(rel_xyz[0]) + c * float(rel_xyz[1])
        return np.array([x, y, float(rel_xyz[2])], dtype=float)

    @staticmethod
    def rel_from_map_point(anchor_pose: conv.Pose2D, map_xyz: np.ndarray) -> np.ndarray:
        dx = float(map_xyz[0]) - anchor_pose.x
        dy = float(map_xyz[1]) - anchor_pose.y
        c, s = math.cos(anchor_pose.theta), math.sin(anchor_pose.theta)
        return np.array(
            [c * dx + s * dy, -s * dx + c * dy, float(map_xyz[2])], dtype=float
        )

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def _best_desc_cosine(self, lm: EmbeddingLandmark, desc: np.ndarray) -> float:
        if not lm.descriptors:
            return -1.0
        return max(self._cosine(d, desc) for d in lm.descriptors)

    @staticmethod
    def _normalize_view(
        view_xy: Tuple[float, ...]
    ) -> Tuple[float, float, float]:
        """Accept ``(x, y)`` or ``(x, y, yaw)``."""
        if len(view_xy) >= 3:
            return float(view_xy[0]), float(view_xy[1]), float(view_xy[2])
        return float(view_xy[0]), float(view_xy[1]), 0.0

    def _desc_threshold(self, lm: EmbeddingLandmark) -> float:
        if lm.status == LandmarkStatus.CONFIRMED:
            return float(self.cosine_threshold)
        return float(self.candidate_cosine_threshold)

    def _match_ok(
        self,
        lm: EmbeddingLandmark,
        descriptor: np.ndarray,
        dist: float,
        radius: float,
    ) -> bool:
        if dist > radius:
            return False
        if not self.require_descriptor_match:
            return True
        return self._best_desc_cosine(lm, descriptor) >= self._desc_threshold(lm)

    def _pick_best_neighbor(
        self,
        neighbors: Sequence[EmbeddingLandmark],
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        pose_of: Dict[int, conv.Pose2D],
        same_anchor: Optional[int] = None,
        fallback_pose: Optional[conv.Pose2D] = None,
    ) -> Optional[EmbeddingLandmark]:
        best: Optional[EmbeddingLandmark] = None
        best_score = float("inf")
        for lm in neighbors:
            if same_anchor is not None and lm.anchor_node != same_anchor:
                continue
            # Absolute map frame (not rel⊕live-anchor) so spin cannot reinterpret.
            lm_map = self.association_map_xyz(lm)
            if lm_map is None:
                ap = pose_of.get(lm.anchor_node, fallback_pose)
                if ap is None:
                    continue
                lm_map = self.map_point_from_rel(ap, lm.rel_xyz)
            dist = float(np.linalg.norm(lm_map - map_xyz))
            radius = float(self.associate_radius_m)
            if lm.status == LandmarkStatus.CONFIRMED:
                radius = max(radius, 3.0 * float(self.demote_offset_m))
            if not self._match_ok(lm, descriptor, dist, radius):
                continue
            cos = self._best_desc_cosine(lm, descriptor)
            # Descriptor agreement dominates so nearby wrong patches do not
            # steal an ID; distance is a secondary tie-break.
            score = -cos + 0.15 * dist
            if score < best_score:
                best_score = score
                best = lm
        return best

    def observe(
        self,
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        normal_map: np.ndarray,
        anchor_node: int,
        anchor_pose: conv.Pose2D,
        view_xy: Tuple[float, ...],
    ) -> EmbeddingLandmark:
        """Associate or create a landmark from one measured map-frame point."""
        map_xyz = np.asarray(map_xyz, dtype=float).reshape(3)
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        normal_map = np.asarray(normal_map, dtype=float).reshape(3)
        view = self._normalize_view(view_xy)

        search_r = max(
            float(self.associate_radius_m), 3.0 * float(self.demote_offset_m)
        )
        best = self._pick_best_neighbor(
            self._neighbors(map_xyz, radius_m=search_r),
            map_xyz=map_xyz,
            descriptor=descriptor,
            pose_of={int(anchor_node): anchor_pose},
            same_anchor=int(anchor_node),
            fallback_pose=anchor_pose,
        )

        if best is None:
            lid = self._next_id
            self._next_id += 1
            rel = self.rel_from_map_point(anchor_pose, map_xyz)
            # Normal into anchor frame (rotate only):
            c, s = math.cos(anchor_pose.theta), math.sin(anchor_pose.theta)
            n_rel = np.array(
                [
                    c * float(normal_map[0]) + s * float(normal_map[1]),
                    -s * float(normal_map[0]) + c * float(normal_map[1]),
                    float(normal_map[2]),
                ],
                dtype=float,
            )
            nn = float(np.linalg.norm(n_rel))
            if nn > 1e-9:
                n_rel = n_rel / nn
            lm = EmbeddingLandmark(
                landmark_id=lid,
                anchor_node=int(anchor_node),
                rel_xyz=rel,
                normal_rel=n_rel,
                map_xyz=map_xyz.copy(),
                descriptors=[descriptor.copy()],
                observations=1,
                view_poses_xy=[view],
                _map_samples=[map_xyz.copy()],
            )
            self._landmarks[lid] = lm
            self._index_add(lm, map_xyz)
            return lm

        return self._update_existing(best, map_xyz, descriptor, view, anchor_pose)

    def observe_with_poses(
        self,
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        normal_map: np.ndarray,
        anchor_node: int,
        anchor_pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
        view_xy: Tuple[float, ...],
    ) -> EmbeddingLandmark:
        """Like ``observe`` but resolves map positions via ``pose_of_node``."""
        map_xyz = np.asarray(map_xyz, dtype=float).reshape(3)
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        view = self._normalize_view(view_xy)

        search_r = max(
            float(self.associate_radius_m), 3.0 * float(self.demote_offset_m)
        )
        best = self._pick_best_neighbor(
            self._neighbors(map_xyz, radius_m=search_r),
            map_xyz=map_xyz,
            descriptor=descriptor,
            pose_of=pose_of_node,
            fallback_pose=anchor_pose,
        )

        if best is None:
            return self.observe(
                map_xyz=map_xyz,
                descriptor=descriptor,
                normal_map=normal_map,
                anchor_node=anchor_node,
                anchor_pose=anchor_pose,
                view_xy=view,
            )
        # Demotion needs the landmark's own anchor pose when available.
        update_pose = pose_of_node.get(best.anchor_node, anchor_pose)
        return self._update_existing(best, map_xyz, descriptor, view, update_pose)

    def _update_existing(
        self,
        lm: EmbeddingLandmark,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        view_xy: Tuple[float, float, float],
        anchor_pose: conv.Pose2D,
    ) -> EmbeddingLandmark:
        lm.observations += 1
        lm._map_samples.append(map_xyz.copy())
        # Keep at most 32 samples for std.
        if len(lm._map_samples) > 32:
            lm._map_samples = lm._map_samples[-32:]
        samples = np.stack(lm._map_samples, axis=0)
        mean = samples.mean(axis=0)
        lm.position_std_m = float(samples.std(axis=0).mean())
        # Lock map_xyz after the first observation. Updating it to the running
        # mean made landmark-camera boxes crawl during spins (depth/pose noise).
        # Samples still drive std / promote; confirm refines map_xyz once.
        if lm.status == LandmarkStatus.CANDIDATE and lm.anchor_node >= 0:
            locked = (
                np.asarray(lm.map_xyz, dtype=float).reshape(3)
                if lm.map_xyz is not None
                else mean
            )
            lm.rel_xyz = self.rel_from_map_point(anchor_pose, locked)

        if len(lm.descriptors) < int(self.max_descriptors):
            if all(self._cosine(d, descriptor) < 0.95 for d in lm.descriptors):
                lm.descriptors.append(descriptor.copy())
        elif self._best_desc_cosine(lm, descriptor) < float(self.cosine_threshold) + 0.1:
            # Replace weakest (lowest mean self-sim) — keep simple: replace last.
            lm.descriptors[-1] = descriptor.copy()

        self._maybe_record_view(lm, view_xy)

        if lm.status == LandmarkStatus.CANDIDATE:
            self._try_promote(lm)
        elif lm.status == LandmarkStatus.CONFIRMED:
            self._check_moved(lm, map_xyz, pose_hint=anchor_pose)
        return lm

    def _maybe_record_view(
        self, lm: EmbeddingLandmark, view: Tuple[float, float, float]
    ) -> None:
        vx, vy, vyaw = view
        min_xy = float(self.view_record_xy_m)
        min_yaw = math.radians(float(self.view_record_yaw_deg))
        for px, py, *rest in lm.view_poses_xy:
            pyaw = float(rest[0]) if rest else 0.0
            if math.hypot(vx - px, vy - py) < min_xy and abs(
                conv.normalize_angle(vyaw - pyaw)
            ) < min_yaw:
                return
        lm.view_poses_xy.append((vx, vy, vyaw))

    def _try_promote(self, lm: EmbeddingLandmark) -> None:
        if lm.observations < int(self.promote_min_obs):
            return
        if len(lm.view_poses_xy) < int(self.promote_min_views):
            return
        # Max pairwise XY gap *or* yaw gap (spin-in-place counts).
        gap_m = 0.0
        gap_yaw = 0.0
        for i, (x0, y0, *r0) in enumerate(lm.view_poses_xy):
            yaw0 = float(r0[0]) if r0 else 0.0
            for x1, y1, *r1 in lm.view_poses_xy[i + 1 :]:
                yaw1 = float(r1[0]) if r1 else 0.0
                gap_m = max(gap_m, math.hypot(x0 - x1, y0 - y1))
                gap_yaw = max(gap_yaw, abs(conv.normalize_angle(yaw0 - yaw1)))
        gap_deg = math.degrees(gap_yaw)
        if gap_m < float(self.promote_min_view_gap_m) and gap_deg < float(
            self.promote_min_view_gap_deg
        ):
            return
        if lm.position_std_m > float(self.promote_max_std_m):
            return
        lm.status = LandmarkStatus.CONFIRMED
        if lm._map_samples:
            lm.map_xyz = np.stack(lm._map_samples, axis=0).mean(axis=0).copy()
        lm._map_samples.clear()

    def _check_moved(
        self, lm: EmbeddingLandmark, map_xyz: np.ndarray, *, pose_hint: conv.Pose2D
    ) -> None:
        if lm.map_xyz is not None:
            expected = np.asarray(lm.map_xyz, dtype=float).reshape(3)
        else:
            expected = self.map_point_from_rel(pose_hint, lm.rel_xyz)
        # Only meaningful when pose_hint is the landmark's anchor.
        if lm.anchor_node < 0:
            return
        offset = float(np.linalg.norm(expected - map_xyz))
        if offset > float(self.demote_offset_m):
            strikes = self._demote_strikes.get(lm.landmark_id, 0) + 1
            self._demote_strikes[lm.landmark_id] = strikes
            if strikes >= int(self.demote_passes):
                lm.status = LandmarkStatus.MOVED
        else:
            self._demote_strikes[lm.landmark_id] = 0

    def visible_set(
        self,
        robot_pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
        *,
        camera_xy: Tuple[float, float],
        camera_yaw: float,
        fov_deg: float = 87.0,
        max_range_m: float = 4.0,
        max_facing_deg: float = 70.0,
    ) -> List[Tuple[EmbeddingLandmark, np.ndarray]]:
        """Confirmed landmarks predicted in the camera frustum.

        Returns ``(landmark, map_xyz)`` pairs.
        """
        half = math.radians(float(fov_deg)) * 0.5
        max_facing = math.radians(float(max_facing_deg))
        cx = robot_pose.x + math.cos(robot_pose.theta) * camera_xy[0] - math.sin(
            robot_pose.theta
        ) * camera_xy[1]
        cy = robot_pose.y + math.sin(robot_pose.theta) * camera_xy[0] + math.cos(
            robot_pose.theta
        ) * camera_xy[1]
        cyaw = conv.normalize_angle(robot_pose.theta + float(camera_yaw))
        out: List[Tuple[EmbeddingLandmark, np.ndarray]] = []
        for lm in self.confirmed():
            mxyz = self.resolved_map_xyz(lm, pose_of_node)
            ap = pose_of_node.get(lm.anchor_node)
            dx = float(mxyz[0]) - cx
            dy = float(mxyz[1]) - cy
            dist = math.hypot(dx, dy)
            if dist < 0.2 or dist > float(max_range_m):
                continue
            bearing = math.atan2(dy, dx)
            if abs(conv.normalize_angle(bearing - cyaw)) > half:
                continue
            # Normal in map frame (needs anchor yaw when available).
            if ap is None:
                continue
            c, s = math.cos(ap.theta), math.sin(ap.theta)
            nx = c * float(lm.normal_rel[0]) - s * float(lm.normal_rel[1])
            ny = s * float(lm.normal_rel[0]) + c * float(lm.normal_rel[1])
            # Facing camera: normal · toward_camera > cos(max_facing)
            toward_x = cx - float(mxyz[0])
            toward_y = cy - float(mxyz[1])
            tn = math.hypot(toward_x, toward_y)
            if tn < 1e-9:
                continue
            toward_x /= tn
            toward_y /= tn
            nn = math.hypot(nx, ny)
            if nn < 1e-9:
                continue
            nx /= nn
            ny /= nn
            if nx * toward_x + ny * toward_y < math.cos(max_facing):
                continue
            out.append((lm, mxyz))
        return out

    def status_dict(self) -> dict:
        counts = {s.value: 0 for s in LandmarkStatus}
        for lm in self._landmarks.values():
            counts[lm.status.value] += 1
        return {"total": len(self._landmarks), **counts}
