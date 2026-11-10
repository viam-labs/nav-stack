"""Embedding landmark map: candidates, promotion, demotion, visible set."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from .types import CameraIntrinsics, LandmarkStatus


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
    # True once any observation came from a YOLO-box pooled DINO (not dense).
    # Overlay shows these even before multi-view "stable" geometry trust.
    from_detector: bool = False
    # Approx half-size (m) of the object footprint in XY (from YOLO box + depth).
    extent_m: Optional[float] = None
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
    # Bank several viewpoint descriptors so drive-by re-ID can match a new angle.
    max_descriptors: int = 6
    cell_m: float = 0.30
    cosine_threshold: float = 0.55
    # Softer gate for candidates: DINOv2 patch tokens drift under approach/zoom,
    # but orthogonal / wrong-surface patches must not steal the same ID
    # (landmark-camera would then jump between objects).
    candidate_cosine_threshold: float = 0.35
    # Within this fraction of associate_radius, confirmed landmarks may re-match
    # at the candidate cosine (bank a new view instead of minting a duplicate).
    confirmed_soft_match_frac: float = 0.7
    # Fallback search radius when a landmark has no measured footprint yet.
    confirmed_overlap_m: float = 0.25
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
        self._assoc_hits = 0
        self._assoc_misses = 0
        self._assoc_soft_hits = 0
        self._assoc_overlap_absorbs = 0
        self._overlap_merges = 0

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
        self,
        map_xyz: np.ndarray,
        *,
        radius_m: Optional[float] = None,
        include_moved: bool = False,
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
                        if lm is None:
                            continue
                        if lm.status == LandmarkStatus.MOVED and not include_moved:
                            continue
                        seen.add(lid)
                        out.append(lm)
        return out

    def _reindex(self, lm: EmbeddingLandmark, map_xyz: np.ndarray) -> None:
        """Move a landmark to the cell of ``map_xyz`` (after promote / revive)."""
        lid = int(lm.landmark_id)
        for cell, ids in list(self._cell_index.items()):
            if lid in ids:
                ids[:] = [i for i in ids if i != lid]
                if not ids:
                    del self._cell_index[cell]
        self._index_add(lm, map_xyz)

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

    def _desc_threshold(
        self,
        lm: EmbeddingLandmark,
        dist: float,
        *,
        from_detector: bool = False,
    ) -> float:
        # Detector↔detector: object-level pooled DINO still drifts with angle;
        # keep the soft candidate gate out to a wider radius.
        if from_detector and bool(lm.from_detector):
            return min(
                float(self.candidate_cosine_threshold),
                0.30,
            )
        if lm.status != LandmarkStatus.CONFIRMED:
            return float(self.candidate_cosine_threshold)
        soft_r = float(self.confirmed_soft_match_frac) * float(self.associate_radius_m)
        if from_detector and bool(lm.from_detector):
            soft_r = max(soft_r, 0.45)
        if dist <= soft_r:
            # Geometrically the same spot — allow viewpoint drift so we bank
            # another descriptor instead of creating a parallel candidate.
            return float(self.candidate_cosine_threshold)
        return float(self.cosine_threshold)

    def _match_ok(
        self,
        lm: EmbeddingLandmark,
        descriptor: np.ndarray,
        dist: float,
        radius: float,
        *,
        from_detector: bool = False,
    ) -> bool:
        if dist > radius:
            return False
        if not self.require_descriptor_match:
            return True
        return self._best_desc_cosine(lm, descriptor) >= self._desc_threshold(
            lm, dist, from_detector=from_detector
        )

    def _pick_best_neighbor(
        self,
        neighbors: Sequence[EmbeddingLandmark],
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        pose_of: Dict[int, conv.Pose2D],
        same_anchor: Optional[int] = None,
        fallback_pose: Optional[conv.Pose2D] = None,
        from_detector: bool = False,
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
            # YOLO-pooled centers jitter more than patch tokens; widen the gate.
            if from_detector and bool(lm.from_detector):
                radius = max(radius, 0.55)
            if lm.status == LandmarkStatus.MOVED and from_detector and bool(
                lm.from_detector
            ):
                radius = max(radius, 0.55)
            if not self._match_ok(
                lm, descriptor, dist, radius, from_detector=from_detector
            ):
                continue
            cos = self._best_desc_cosine(lm, descriptor)
            # Descriptor agreement dominates so nearby wrong patches do not
            # steal an ID; distance is a secondary tie-break. Prefer detector
            # landmarks when the observation is detector-anchored.
            score = -cos + 0.15 * dist
            if from_detector and bool(lm.from_detector):
                score -= 0.05
            if score < best_score:
                best_score = score
                best = lm
        return best

    def _note_association(
        self,
        lm: Optional[EmbeddingLandmark],
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
    ) -> None:
        if lm is None:
            self._assoc_misses += 1
            return
        self._assoc_hits += 1
        if lm.status != LandmarkStatus.CONFIRMED:
            return
        soft_r = float(self.confirmed_soft_match_frac) * float(self.associate_radius_m)
        lm_map = self.association_map_xyz(lm)
        if lm_map is None:
            return
        dist = float(np.linalg.norm(lm_map - map_xyz))
        if dist <= soft_r and self._best_desc_cosine(lm, descriptor) < float(
            self.cosine_threshold
        ):
            self._assoc_soft_hits += 1

    def _extent_or_fallback(self, extent_m: Optional[float]) -> float:
        if extent_m is not None and float(extent_m) >= 0.05:
            return float(extent_m)
        return float(self.confirmed_overlap_m)

    def _footprints_overlap(
        self,
        dist_xy: float,
        extent_a: Optional[float],
        extent_b: Optional[float],
    ) -> bool:
        """True when each center lies inside the other's footprint (strong overlap)."""
        ra = self._extent_or_fallback(extent_a)
        rb = self._extent_or_fallback(extent_b)
        d = float(dist_xy)
        return d <= ra and d <= rb

    def _map_xy(self, lm: EmbeddingLandmark) -> Optional[np.ndarray]:
        m = self.association_map_xyz(lm)
        if m is None:
            return None
        return np.asarray(m, dtype=float).reshape(3)

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _prefer_keeper(
        self, a: EmbeddingLandmark, b: EmbeddingLandmark
    ) -> Tuple[EmbeddingLandmark, EmbeddingLandmark]:
        """Return (keeper, donor). Prefer confirmed, then more obs, then lower id."""
        def key(lm: EmbeddingLandmark) -> Tuple[int, int, int]:
            conf = 1 if lm.status == LandmarkStatus.CONFIRMED else 0
            return (conf, int(lm.observations), -int(lm.landmark_id))

        if key(a) >= key(b):
            return a, b
        return b, a

    def _remove_landmark(self, lid: int) -> None:
        lm = self._landmarks.pop(int(lid), None)
        if lm is None:
            return
        for cell, ids in list(self._cell_index.items()):
            if int(lid) in ids:
                ids[:] = [i for i in ids if i != int(lid)]
                if not ids:
                    del self._cell_index[cell]
        self._demote_strikes.pop(int(lid), None)

    def _merge_donor_into_keeper(
        self, keeper: EmbeddingLandmark, donor: EmbeddingLandmark
    ) -> EmbeddingLandmark:
        """Fold ``donor`` into ``keeper`` and drop ``donor`` from the map."""
        if keeper.landmark_id == donor.landmark_id:
            return keeper
        for d in list(donor.descriptors):
            self._bank_descriptor(keeper, d)
        for view in list(donor.view_poses_xy):
            self._maybe_record_view(keeper, self._normalize_view(view))
        keeper.observations += max(0, int(donor.observations))
        if donor.extent_m is not None and float(donor.extent_m) > 0:
            prev = float(keeper.extent_m) if keeper.extent_m is not None else 0.0
            keeper.extent_m = max(prev, float(donor.extent_m))
        if donor.from_detector:
            keeper.from_detector = True
        if keeper.status == LandmarkStatus.CANDIDATE and donor.status == LandmarkStatus.CONFIRMED:
            keeper.status = LandmarkStatus.CONFIRMED
            if keeper.map_xyz is not None:
                self._reindex(keeper, np.asarray(keeper.map_xyz, dtype=float).reshape(3))
        self._remove_landmark(int(donor.landmark_id))
        self._overlap_merges += 1
        self._assoc_overlap_absorbs += 1
        return keeper

    def _nearest_overlapping_landmark(
        self,
        map_xyz: np.ndarray,
        *,
        new_extent_m: Optional[float],
        pose_of: Dict[int, conv.Pose2D],
        fallback_pose: Optional[conv.Pose2D],
        exclude_id: Optional[int] = None,
        detector_only: bool = True,
    ) -> Optional[EmbeddingLandmark]:
        """Existing landmark whose footprint substantially overlaps this observation."""
        search_r = self._extent_or_fallback(new_extent_m) * 2.0
        search_r = max(search_r, float(self.confirmed_overlap_m) * 2.0)
        best: Optional[EmbeddingLandmark] = None
        best_key: Tuple[int, int, int, float] = (-1, -1, 0, float("inf"))
        for lm in self._neighbors(map_xyz, radius_m=search_r, include_moved=False):
            if exclude_id is not None and int(lm.landmark_id) == int(exclude_id):
                continue
            if lm.status == LandmarkStatus.MOVED:
                continue
            if detector_only and not bool(lm.from_detector):
                continue
            lm_map = self.association_map_xyz(lm)
            if lm_map is None:
                ap = pose_of.get(lm.anchor_node, fallback_pose)
                if ap is None:
                    continue
                lm_map = self.map_point_from_rel(ap, lm.rel_xyz)
            dist = self._xy_dist(lm_map, map_xyz)
            # Near-identical centers always count as overlap for detector twins.
            near_same = dist <= 0.12
            if not near_same and not self._footprints_overlap(
                dist, lm.extent_m, new_extent_m
            ):
                continue
            conf = 1 if lm.status == LandmarkStatus.CONFIRMED else 0
            key = (conf, int(lm.observations), -int(lm.landmark_id), -dist)
            if best is None or key > best_key:
                best = lm
                best_key = key
        return best

    def merge_overlapping_detector_landmarks(self) -> int:
        """Collapse detector landmarks whose footprints heavily overlap.

        Fixes twins that both promoted (e.g. DC1 + DC12 on one painting) —
        absorb-at-mint only prevents new mints against an already-confirmed ID.
        """
        merged = 0
        # Repeat until stable — a merge can expose a new pair.
        for _ in range(8):
            lms = [
                lm
                for lm in self._landmarks.values()
                if lm.status != LandmarkStatus.MOVED and bool(lm.from_detector)
            ]
            pair: Optional[Tuple[EmbeddingLandmark, EmbeddingLandmark]] = None
            for i, a in enumerate(lms):
                a_map = self._map_xy(a)
                if a_map is None:
                    continue
                for b in lms[i + 1 :]:
                    b_map = self._map_xy(b)
                    if b_map is None:
                        continue
                    dist = self._xy_dist(a_map, b_map)
                    if dist <= 0.12 or self._footprints_overlap(
                        dist, a.extent_m, b.extent_m
                    ):
                        pair = (a, b)
                        break
                if pair is not None:
                    break
            if pair is None:
                break
            keeper, donor = self._prefer_keeper(pair[0], pair[1])
            self._merge_donor_into_keeper(keeper, donor)
            merged += 1
        return merged

    def observe(
        self,
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        normal_map: np.ndarray,
        anchor_node: int,
        anchor_pose: conv.Pose2D,
        view_xy: Tuple[float, ...],
        from_detector: bool = False,
        extent_m: Optional[float] = None,
    ) -> EmbeddingLandmark:
        """Associate or create a landmark from one measured map-frame point."""
        map_xyz = np.asarray(map_xyz, dtype=float).reshape(3)
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        normal_map = np.asarray(normal_map, dtype=float).reshape(3)
        view = self._normalize_view(view_xy)
        extent = float(extent_m) if extent_m is not None and float(extent_m) > 0 else None

        search_r = max(
            float(self.associate_radius_m), 3.0 * float(self.demote_offset_m)
        )
        if from_detector:
            search_r = max(search_r, 0.55)
        best = self._pick_best_neighbor(
            self._neighbors(
                map_xyz, radius_m=search_r, include_moved=bool(from_detector)
            ),
            map_xyz=map_xyz,
            descriptor=descriptor,
            pose_of={int(anchor_node): anchor_pose},
            # Do NOT filter by same_anchor — keyframes change while driving;
            # map_xyz association must work across anchor ids.
            fallback_pose=anchor_pose,
            from_detector=from_detector,
        )
        self._note_association(best, map_xyz=map_xyz, descriptor=descriptor)

        if best is None:
            # Descriptor gates missed, but another detector landmark's footprint
            # overlaps this observation — absorb rather than mint a twin ID.
            overlap = self._nearest_overlapping_landmark(
                map_xyz,
                new_extent_m=extent,
                pose_of={int(anchor_node): anchor_pose},
                fallback_pose=anchor_pose,
                detector_only=bool(from_detector),
            )
            if overlap is not None and (
                from_detector or overlap.status == LandmarkStatus.CONFIRMED
            ):
                self._assoc_misses = max(0, self._assoc_misses - 1)
                self._assoc_hits += 1
                self._assoc_overlap_absorbs += 1
                cos = self._best_desc_cosine(overlap, descriptor)
                return self._update_existing(
                    overlap,
                    map_xyz,
                    descriptor,
                    view,
                    anchor_pose,
                    from_detector=from_detector,
                    extent_m=extent,
                    # Don't poison the confirmed bank with an unrelated crop.
                    bank_descriptor=cos >= 0.20,
                )
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
                from_detector=bool(from_detector),
                extent_m=extent,
                _map_samples=[map_xyz.copy()],
            )
            self._landmarks[lid] = lm
            self._index_add(lm, map_xyz)
            return lm

        return self._update_existing(
            best,
            map_xyz,
            descriptor,
            view,
            anchor_pose,
            from_detector=from_detector,
            extent_m=extent,
        )

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
        from_detector: bool = False,
        extent_m: Optional[float] = None,
    ) -> EmbeddingLandmark:
        """Like ``observe`` but resolves map positions via ``pose_of_node``."""
        map_xyz = np.asarray(map_xyz, dtype=float).reshape(3)
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        view = self._normalize_view(view_xy)
        extent = float(extent_m) if extent_m is not None and float(extent_m) > 0 else None

        search_r = max(
            float(self.associate_radius_m), 3.0 * float(self.demote_offset_m)
        )
        if from_detector:
            search_r = max(search_r, 0.55)
        best = self._pick_best_neighbor(
            self._neighbors(
                map_xyz, radius_m=search_r, include_moved=bool(from_detector)
            ),
            map_xyz=map_xyz,
            descriptor=descriptor,
            pose_of=pose_of_node,
            fallback_pose=anchor_pose,
            from_detector=from_detector,
        )
        if best is None:
            # Fall through to observe() which notes its own association attempt.
            return self.observe(
                map_xyz=map_xyz,
                descriptor=descriptor,
                normal_map=normal_map,
                anchor_node=anchor_node,
                anchor_pose=anchor_pose,
                view_xy=view,
                from_detector=from_detector,
                extent_m=extent,
            )
        self._note_association(best, map_xyz=map_xyz, descriptor=descriptor)
        # Demotion needs the landmark's own anchor pose when available.
        update_pose = pose_of_node.get(best.anchor_node, anchor_pose)
        return self._update_existing(
            best,
            map_xyz,
            descriptor,
            view,
            update_pose,
            from_detector=from_detector,
            extent_m=extent,
        )

    def _update_existing(
        self,
        lm: EmbeddingLandmark,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        view_xy: Tuple[float, float, float],
        anchor_pose: conv.Pose2D,
        *,
        from_detector: bool = False,
        extent_m: Optional[float] = None,
        bank_descriptor: bool = True,
    ) -> EmbeddingLandmark:
        lm.observations += 1
        if from_detector:
            lm.from_detector = True
        if extent_m is not None and float(extent_m) > 0:
            # Keep the larger footprint so later smaller crops still overlap.
            prev = float(lm.extent_m) if lm.extent_m is not None else 0.0
            lm.extent_m = max(prev, float(extent_m))
        # Revisit of a demoted landmark: revive instead of minting a parallel ID.
        if lm.status == LandmarkStatus.MOVED:
            lm.status = LandmarkStatus.CANDIDATE
            self._demote_strikes[lm.landmark_id] = 0
            if lm.map_xyz is not None:
                self._reindex(lm, np.asarray(lm.map_xyz, dtype=float).reshape(3))
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

        if bank_descriptor:
            self._bank_descriptor(lm, descriptor)

        self._maybe_record_view(lm, view_xy)

        if lm.status == LandmarkStatus.CANDIDATE:
            self._try_promote(lm)
        elif lm.status == LandmarkStatus.CONFIRMED:
            self._check_moved(lm, map_xyz, pose_hint=anchor_pose)
        return lm

    def _bank_descriptor(self, lm: EmbeddingLandmark, descriptor: np.ndarray) -> None:
        """Keep a diverse multi-view descriptor set for drive-by re-ID."""
        if not lm.descriptors:
            lm.descriptors.append(descriptor.copy())
            return
        # Near-duplicate of an existing view — nothing to learn.
        if self._best_desc_cosine(lm, descriptor) >= 0.95:
            return
        if len(lm.descriptors) < int(self.max_descriptors):
            lm.descriptors.append(descriptor.copy())
            return
        # Full: replace the stored view least similar to the new one.
        worst_i = min(
            range(len(lm.descriptors)),
            key=lambda i: self._cosine(lm.descriptors[i], descriptor),
        )
        lm.descriptors[worst_i] = descriptor.copy()

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
        min_gap_m = float(self.promote_min_view_gap_m)
        min_gap_deg = float(self.promote_min_view_gap_deg)
        max_std = float(self.promote_max_std_m)
        if bool(lm.from_detector):
            # Object-level landmarks: tolerate depth jitter; accept shorter baselines.
            min_gap_m = min(min_gap_m, 0.25)
            min_gap_deg = min(min_gap_deg, 18.0)
            max_std = max(max_std, 0.22)
        if gap_m < min_gap_m and gap_deg < min_gap_deg:
            return
        if lm.position_std_m > max_std:
            return
        # About to confirm — if we already have a confirmed twin on this object,
        # fold into it instead of creating DC1+DC12 on the same painting.
        if bool(lm.from_detector) and lm.map_xyz is not None:
            twin = self._nearest_overlapping_landmark(
                np.asarray(lm.map_xyz, dtype=float).reshape(3),
                new_extent_m=lm.extent_m,
                pose_of={},
                fallback_pose=None,
                exclude_id=int(lm.landmark_id),
                detector_only=True,
            )
            if twin is not None and twin.status == LandmarkStatus.CONFIRMED:
                self._merge_donor_into_keeper(twin, lm)
                return
        lm.status = LandmarkStatus.CONFIRMED
        if lm._map_samples:
            new_xyz = np.stack(lm._map_samples, axis=0).mean(axis=0).copy()
            lm.map_xyz = new_xyz
            self._reindex(lm, new_xyz)
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
        demote_m = float(self.demote_offset_m)
        demote_n = int(self.demote_passes)
        if bool(lm.from_detector):
            # Depth-center of a painting wobbles more than a single patch.
            demote_m = max(demote_m, 0.35)
            demote_n = max(demote_n, 5)
        if offset > demote_m:
            strikes = self._demote_strikes.get(lm.landmark_id, 0) + 1
            self._demote_strikes[lm.landmark_id] = strikes
            if strikes >= demote_n:
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

    def image_visible_set(
        self,
        robot_pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
        *,
        intrinsics: CameraIntrinsics,
        mount_x: float = 0.0,
        mount_y: float = 0.0,
        mount_z: float = 0.0,
        mount_theta: float = 0.0,
        mount_pitch: float = 0.0,
        mount_roll: float = 0.0,
        max_range_m: float = 4.0,
        margin_px: float = 20.0,
    ) -> List[Tuple[EmbeddingLandmark, np.ndarray]]:
        """Confirmed landmarks that project into the camera image.

        Matches landmark-camera overlay culling (image projection), not the
        stricter frustum+facing filter in ``visible_set``. Used for writeback
        pairing so greens on the overlay can form inliers.
        """
        # Lazy import: viz already depends on landmarks.
        from ..viz.landmarks_view import project_map_point_to_pixel

        w = int(intrinsics.width)
        h = int(intrinsics.height)
        margin = float(margin_px)
        cx = robot_pose.x + math.cos(robot_pose.theta) * float(mount_x) - math.sin(
            robot_pose.theta
        ) * float(mount_y)
        cy = robot_pose.y + math.sin(robot_pose.theta) * float(mount_x) + math.cos(
            robot_pose.theta
        ) * float(mount_y)
        out: List[Tuple[EmbeddingLandmark, np.ndarray]] = []
        for lm in self.confirmed():
            assoc = self.association_map_xyz(lm)
            if assoc is not None:
                mxyz = assoc
            else:
                mxyz = self.resolved_map_xyz(lm, pose_of_node)
            dist = math.hypot(float(mxyz[0]) - cx, float(mxyz[1]) - cy)
            if dist < 0.2 or dist > float(max_range_m):
                continue
            proj = project_map_point_to_pixel(
                mxyz,
                robot_pose=robot_pose,
                intrinsics=intrinsics,
                mount_x=float(mount_x),
                mount_y=float(mount_y),
                mount_z=float(mount_z),
                mount_theta=float(mount_theta),
                mount_pitch=float(mount_pitch),
                mount_roll=float(mount_roll),
            )
            if proj is None:
                continue
            u, v, _depth = proj
            if u < -margin or v < -margin or u > w + margin or v > h + margin:
                continue
            out.append((lm, mxyz))
        return out

    def status_dict(self) -> dict:
        counts = {s.value: 0 for s in LandmarkStatus}
        n_desc = 0
        for lm in self._landmarks.values():
            counts[lm.status.value] += 1
            n_desc += len(lm.descriptors)
        hits = int(self._assoc_hits)
        misses = int(self._assoc_misses)
        attempts = hits + misses
        from_det = sum(
            1 for lm in self._landmarks.values() if bool(lm.from_detector)
        )
        return {
            "total": len(self._landmarks),
            **counts,
            "from_detector": int(from_det),
            "assoc_hits": hits,
            "assoc_misses": misses,
            "assoc_soft_hits": int(self._assoc_soft_hits),
            "assoc_overlap_absorbs": int(self._assoc_overlap_absorbs),
            "overlap_merges": int(self._overlap_merges),
            "assoc_hit_rate": (hits / attempts) if attempts else None,
            "mean_descriptors": (n_desc / max(1, len(self._landmarks))),
        }
