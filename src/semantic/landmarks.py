"""Embedding landmark map: candidates, promotion, demotion, visible set."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from .types import CameraIntrinsics, LandmarkStatus

# Depth-audit verdicts: what the depth image shows where a landmark is stored.
AUDIT_AGREES = "agrees"  # a surface sits at the stored depth
AUDIT_SEEN_THROUGH = "seen_through"  # the camera sees well past the stored point
AUDIT_OCCLUDED = "occluded"  # something nearer is in the way
AUDIT_NO_DEPTH = "no_depth"  # too few valid depth pixels to judge
AUDIT_UNCLEAR = "unclear"  # mixed readings, no verdict


# A match is refused when another landmark's embedding is this close and the
# two are not clearly separated in distance. Similar nearby objects then stay
# unmatched instead of stealing an id.
UNIQUE_COSINE_MARGIN = 0.08
UNIQUE_DISTANCE_MARGIN_M = 0.15
# Short-term identity between 1 Hz surveys: a box keeps its landmark when it
# still overlaps the previous box and the embedding still agrees.
TRACK_MIN_IOU = 0.25
TRACK_IOU_MARGIN = 0.15


def choose_unique_match(
    scored: Sequence[Tuple[Any, float, float]],
    *,
    cosine_margin: float,
    distance_margin_m: float,
):
    """Pick one ``(item, cosine, distance)`` row, or refuse a tie.

    The higher cosine wins when it leads by ``cosine_margin``. Among rows
    that close, the nearer one wins when it is closer by ``distance_margin_m``.
    A tie on both returns ``None``.
    """
    if not scored:
        return None
    best_cos = max(float(row[1]) for row in scored)
    margin = float(cosine_margin)
    tied = [
        row
        for row in scored
        if float(row[1]) == best_cos or best_cos - float(row[1]) < margin
    ]
    tied.sort(key=lambda row: float(row[2]))
    if len(tied) == 1:
        return tied[0][0]
    if float(tied[1][2]) - float(tied[0][2]) >= float(distance_margin_m):
        return tied[0][0]
    return None


def box_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    if ax1 < ax0:
        ax0, ax1 = ax1, ax0
    if ay1 < ay0:
        ay0, ay1 = ay1, ay0
    if bx1 < bx0:
        bx0, bx1 = bx1, bx0
    if by1 < by0:
        by0, by1 = by1, by0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def assign_box_tracks(
    boxes: Sequence[Tuple[int, Tuple[float, float, float, float], np.ndarray]],
    tracks: Sequence[dict],
    *,
    cosine_of: Callable[[int, np.ndarray], float],
    min_iou: float = TRACK_MIN_IOU,
    iou_margin: float = TRACK_IOU_MARGIN,
    min_cosine: float = 0.35,
) -> Dict[int, int]:
    """Bind this frame's boxes to the previous frame's boxes.

    ``boxes`` is ``(index, xyxy, descriptor)``. Each track has ``xyxy`` and
    ``landmark_id``. A box keeps a landmark only when its overlap leads every
    other landmark by ``iou_margin`` and ``cosine_of`` still clears
    ``min_cosine``. Two boxes cannot take the same landmark.
    """
    by_index: Dict[int, List[Tuple[float, int]]] = {}
    for index, xyxy, desc in boxes:
        desc_arr = np.asarray(desc, dtype=float).reshape(-1)
        for tr in tracks:
            iou = box_iou(xyxy, tuple(tr["xyxy"]))  # type: ignore[arg-type]
            if iou < float(min_iou):
                continue
            lid = int(tr["landmark_id"])
            if float(cosine_of(lid, desc_arr)) < float(min_cosine):
                continue
            by_index.setdefault(int(index), []).append((iou, lid))
    proposals: List[Tuple[float, int, int]] = []
    for index, hits in by_index.items():
        hits.sort(key=lambda row: row[0], reverse=True)
        best_iou, best_lid = hits[0]
        refused = False
        for other_iou, other_lid in hits[1:]:
            if other_lid == best_lid:
                continue
            if best_iou - other_iou < float(iou_margin):
                refused = True
            break
        if refused:
            continue
        proposals.append((best_iou, int(index), int(best_lid)))
    proposals.sort(key=lambda row: row[0], reverse=True)
    used_index: set = set()
    used_lid: set = set()
    assigned: Dict[int, int] = {}
    for _iou, index, lid in proposals:
        if index in used_index or lid in used_lid:
            continue
        assigned[index] = lid
        used_index.add(index)
        used_lid.add(lid)
    return assigned


def display_label(lm: "EmbeddingLandmark") -> str:
    """Name shown on landmark-camera: ``DC12`` / ``Dc12`` / ``C12`` / ``c12``."""
    tag = {
        LandmarkStatus.CONFIRMED: "C",
        LandmarkStatus.CANDIDATE: "c",
        LandmarkStatus.MOVED: "M",
    }.get(lm.status, "?")
    prefix = "D" if bool(lm.from_detector) else ""
    return f"{prefix}{tag}{int(lm.landmark_id)}"


# Surface slope (depth change per metre across the image) past which a
# reading is not a surface. Real indoor backgrounds are steep in at most one
# image direction: a side wall across, a floor down. 4 is about 76 degrees.
_GARBAGE_SLOPE = 4.0


def _depth_is_garbage(win: np.ndarray, depth_m: float, focal_px: float) -> bool:
    """True when a depth window is steep both across and down the image.

    Stereo mismatches on a patterned or glossy object read as a ramp of far
    depths (4 m to 10 m across a painting really at 3.7 m). That is no
    evidence the object is absent. Medians, so a step between two real
    surfaces does not count.
    """
    ok = np.isfinite(win) & (win > 0.0)
    metres_per_px = max(1e-6, float(depth_m)) / max(1.0, float(focal_px))
    slopes = []
    for axis in (0, 1):
        both = (ok[1:, :] & ok[:-1, :]) if axis == 0 else (ok[:, 1:] & ok[:, :-1])
        steps = np.abs(np.diff(win, axis=axis))[both]
        if steps.size < 8:
            return False
        slopes.append(float(np.median(steps)) / metres_per_px)
    return min(slopes) > _GARBAGE_SLOPE


def _depth_verdict(
    depth_m: np.ndarray,
    u: float,
    v: float,
    stored_m: float,
    *,
    half_px: float,
    focal_px: float,
) -> Tuple[str, Optional[float]]:
    """Judge one stored depth against the image; returns (verdict, median depth)."""
    h, w = depth_m.shape
    half = int(round(min(12.0, max(4.0, float(half_px)))))
    ui, vi = int(round(u)), int(round(v))
    win = depth_m[
        max(0, vi - half) : min(h, vi + half + 1),
        max(0, ui - half) : min(w, ui + half + 1),
    ]
    vals = win[np.isfinite(win) & (win > 0.0)]
    # Measured against the full window, so one clipped by the image edge or
    # mostly invalid gives no verdict.
    if vals.size < max(8.0, 0.25 * float((2 * half + 1) ** 2)):
        return AUDIT_NO_DEPTH, None
    measured = float(np.median(vals))
    near_tol = max(0.20, 0.08 * float(stored_m))
    far_tol = max(0.40, 0.15 * float(stored_m))
    n = float(vals.size)
    at = float(np.count_nonzero(np.abs(vals - stored_m) <= near_tol)) / n
    beyond = float(np.count_nonzero(vals > stored_m + far_tol)) / n
    front = float(np.count_nonzero(vals < stored_m - near_tol)) / n
    # A thin object (pole, plant) fills little of the window: any real share
    # at the stored depth counts as present.
    if at >= 0.15:
        return AUDIT_AGREES, measured
    if beyond >= 0.80:
        if _depth_is_garbage(win, measured, float(focal_px)):
            return AUDIT_UNCLEAR, measured
        return AUDIT_SEEN_THROUGH, measured
    if front >= 0.50:
        return AUDIT_OCCLUDED, measured
    return AUDIT_UNCLEAR, measured


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
    # The overlay may show these before geometry is trusted, but only while
    # this frame's detector box is tied to them.
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
    # A detector box used to confirm on 3 observations with up to 22 cm of
    # scatter, which confirmed landmarks on one short pass. It now needs this
    # many observations and the same 12 cm cap as a patch.
    detector_promote_min_obs: int = 8
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
    # Depth audit: frames in a row the camera must see past a landmark's
    # stored point before the landmark is dropped.
    see_through_strikes: int = 3
    # A landmark seen this often is only deleted when the camera reads at
    # least see_through_keep_gap_m past the stored depth. A smaller gap still
    # flags it (red X); throwing away hundreds of observations for a 1 m
    # disagreement was dropping real objects.
    see_through_keep_min_obs: int = 30
    see_through_keep_gap_m: float = 1.2
    # See ``choose_unique_match``. Same numbers gate pose-fix pairing.
    unique_cosine_margin: float = UNIQUE_COSINE_MARGIN
    unique_distance_margin_m: float = UNIQUE_DISTANCE_MARGIN_M

    def __post_init__(self) -> None:
        self._next_id = 1
        self._landmarks: Dict[int, EmbeddingLandmark] = {}
        self._cell_index: Dict[Tuple[int, int, int], List[int]] = {}
        self._demote_strikes: Dict[int, int] = {}
        self._see_through_strikes: Dict[int, int] = {}
        self._seen_through_removed = 0
        self._assoc_hits = 0
        self._assoc_misses = 0
        self._assoc_soft_hits = 0
        self._assoc_overlap_absorbs = 0
        self._assoc_ambiguous = 0
        self._assoc_tracks = 0
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
    ) -> Tuple[Optional[EmbeddingLandmark], bool]:
        """Return ``(match, ambiguous)``.

        ``ambiguous`` means more than one landmark cleared the gate and neither
        the embedding nor the distance separated them. The caller must not
        mint a new id in that case.
        """
        scored: List[Tuple[EmbeddingLandmark, float, float]] = []
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
            scored.append((lm, cos, dist))
        if not scored:
            return None, False
        # Synthetic RGB has no real descriptor. Keep the old nearest-wins score.
        if not self.require_descriptor_match:
            best: Optional[EmbeddingLandmark] = None
            best_score = float("inf")
            for lm, cos, dist in scored:
                score = -cos + 0.15 * dist
                if from_detector and bool(lm.from_detector):
                    score -= 0.05
                if score < best_score:
                    best_score = score
                    best = lm
            return best, False
        margin = float(self.unique_cosine_margin)
        dist_margin = float(self.unique_distance_margin_m)
        picked = choose_unique_match(
            scored, cosine_margin=margin, distance_margin_m=dist_margin
        )
        if picked is not None:
            return picked, False
        # A detector box prefers another detector landmark when a dense patch
        # is the thing making the embedding look tied.
        if from_detector:
            det_scored = [row for row in scored if bool(row[0].from_detector)]
            if 0 < len(det_scored) < len(scored):
                picked = choose_unique_match(
                    det_scored, cosine_margin=margin, distance_margin_m=dist_margin
                )
                if picked is not None:
                    return picked, False
        return None, True

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
        self._see_through_strikes.pop(int(lid), None)

    def seen_through_ids(self, *, min_strikes: Optional[int] = None) -> List[int]:
        """Landmarks the depth audit caught in empty air.

        Default: those at the strike limit (flagged). ``min_strikes=1`` adds
        the ones seen through on their latest audited frame.
        """
        limit = max(1, int(self.see_through_strikes))
        need = limit if min_strikes is None else max(1, int(min_strikes))
        return sorted(
            int(lid)
            for lid, strikes in self._see_through_strikes.items()
            if strikes >= need and lid in self._landmarks
        )

    def audit_against_depth(
        self,
        depth_m: np.ndarray,
        *,
        robot_pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
        intrinsics: CameraIntrinsics,
        mount_x: float = 0.0,
        mount_y: float = 0.0,
        mount_z: float = 0.0,
        mount_theta: float = 0.0,
        mount_pitch: float = 0.0,
        mount_roll: float = 0.0,
        remove: bool = False,
        max_range_m: float = 6.5,
        window_m: float = 0.10,
    ) -> dict:
        """Compare each in-view landmark's stored depth with the depth image.

        A landmark is a claim that a surface exists at ``map_xyz``. When the
        camera reads depth well beyond that point, nothing is there. After
        ``see_through_strikes`` such frames in a row the landmark is flagged,
        and removed when ``remove`` is set. One with at least
        ``see_through_keep_min_obs`` observations is removed only when the
        camera reads ``see_through_keep_gap_m`` or more past the stored depth;
        a smaller gap stays flagged. A frame that agrees clears the count;
        occluded / no-depth / unclear frames leave it unchanged.

        Only pass ``remove=True`` when ``robot_pose`` is trusted (mapping): a
        wrong pose makes every landmark look seen through. As a second guard,
        a frame where most judged landmarks are seen through at once is
        treated as a pose problem (``pose_suspect``) and changes nothing.

        Returns ``{"rows": [...], "pose_suspect": bool}`` with one row per
        landmark that projects into the image.
        """
        # Lazy import: viz already depends on landmarks.
        from ..viz.landmarks_view import project_map_point_to_pixel

        depth = np.asarray(depth_m, dtype=float)
        if depth.ndim != 2 or depth.size == 0:
            return {"rows": [], "pose_suspect": False}
        h, w = depth.shape
        rows: List[dict] = []
        for lm in self._landmarks.values():
            if lm.status == LandmarkStatus.MOVED:
                continue
            mxyz = self.association_map_xyz(lm)
            if mxyz is None:
                mxyz = self.resolved_map_xyz(lm, pose_of_node)
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
            u, v, stored = proj
            if stored > float(max_range_m) or not (0.0 <= u < w and 0.0 <= v < h):
                continue
            verdict, measured = _depth_verdict(
                depth,
                u,
                v,
                stored,
                half_px=float(window_m) * float(intrinsics.fx) / max(0.2, stored),
                focal_px=float(intrinsics.fx),
            )
            rows.append(
                {
                    "id": int(lm.landmark_id),
                    "label": display_label(lm),
                    "status": lm.status.value,
                    "from_detector": bool(lm.from_detector),
                    "observations": int(lm.observations),
                    "u": round(float(u), 1),
                    "v": round(float(v), 1),
                    "stored_depth_m": round(float(stored), 3),
                    "measured_depth_m": (
                        None if measured is None else round(float(measured), 3)
                    ),
                    "verdict": verdict,
                }
            )

        n_seen = sum(1 for row in rows if row["verdict"] == AUDIT_SEEN_THROUGH)
        n_agree = sum(1 for row in rows if row["verdict"] == AUDIT_AGREES)
        pose_suspect = n_seen >= 4 and n_seen > n_agree
        limit = max(1, int(self.see_through_strikes))
        for row in rows:
            lid = int(row["id"])
            strikes = int(self._see_through_strikes.get(lid, 0))
            if not pose_suspect:
                if row["verdict"] == AUDIT_SEEN_THROUGH:
                    strikes = min(strikes + 1, limit)
                    self._see_through_strikes[lid] = strikes
                elif row["verdict"] == AUDIT_AGREES and strikes:
                    strikes = 0
                    self._see_through_strikes.pop(lid, None)
            # Young landmarks drop on the ordinary seen-through test. A
            # well-observed one stays unless the depth gap is large; the red
            # mark still shows the disagreement.
            drops = True
            if int(row["observations"]) >= int(self.see_through_keep_min_obs):
                measured = row["measured_depth_m"]
                gap = (
                    None
                    if measured is None
                    else float(measured) - float(row["stored_depth_m"])
                )
                drops = gap is not None and gap >= float(self.see_through_keep_gap_m)
            removed = bool(remove) and not pose_suspect and strikes >= limit and drops
            row["strikes"] = strikes
            row["removed"] = removed
            if removed:
                self._remove_landmark(lid)
                self._seen_through_removed += 1
        return {"rows": rows, "pose_suspect": pose_suspect}

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
        pose_of_node: Optional[Dict[int, conv.Pose2D]] = None,
    ) -> Optional[EmbeddingLandmark]:
        """Associate or create a landmark from one measured map-frame point.

        Returns ``None`` when two landmarks match and neither the embedding
        nor the distance separates them. Nothing is minted in that case.

        ``pose_of_node`` lets an update find the matched landmark's own anchor
        pose; without it only landmarks on ``anchor_node`` refresh ``rel_xyz``.
        """
        poses = (
            pose_of_node
            if pose_of_node is not None
            else {int(anchor_node): anchor_pose}
        )
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
        best, ambiguous = self._pick_best_neighbor(
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
        if ambiguous:
            self._assoc_ambiguous += 1
            self._note_association(None, map_xyz=map_xyz, descriptor=descriptor)
            return None
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
                    poses.get(overlap.anchor_node),
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
            poses.get(best.anchor_node),
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
    ) -> Optional[EmbeddingLandmark]:
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
        best, ambiguous = self._pick_best_neighbor(
            self._neighbors(
                map_xyz, radius_m=search_r, include_moved=bool(from_detector)
            ),
            map_xyz=map_xyz,
            descriptor=descriptor,
            pose_of=pose_of_node,
            fallback_pose=anchor_pose,
            from_detector=from_detector,
        )
        if ambiguous:
            self._assoc_ambiguous += 1
            self._note_association(None, map_xyz=map_xyz, descriptor=descriptor)
            return None
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
                pose_of_node=pose_of_node,
            )
        self._note_association(best, map_xyz=map_xyz, descriptor=descriptor)
        # rel_xyz is only meaningful against the landmark's own anchor.
        update_pose = pose_of_node.get(best.anchor_node)
        return self._update_existing(
            best,
            map_xyz,
            descriptor,
            view,
            update_pose,
            from_detector=from_detector,
            extent_m=extent,
        )

    def observe_known(
        self,
        landmark_id: int,
        *,
        map_xyz: np.ndarray,
        descriptor: np.ndarray,
        pose_of_node: Optional[Dict[int, conv.Pose2D]],
        view_xy: Tuple[float, ...],
        from_detector: bool = False,
        extent_m: Optional[float] = None,
    ) -> Optional[EmbeddingLandmark]:
        """Fold an observation into a landmark the previous box already named.

        The short-term track is what separates two similar objects. This skips
        the geometric search. The embedding still has to clear the candidate
        cosine, so a different object that slid into the box does not inherit
        the id. Returns ``None`` when the landmark is gone or the embedding
        disagrees.
        """
        lm = self.get(int(landmark_id))
        if lm is None:
            return None
        map_xyz = np.asarray(map_xyz, dtype=float).reshape(3)
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        if self.require_descriptor_match:
            need = float(self.candidate_cosine_threshold)
            if from_detector and bool(lm.from_detector):
                need = min(need, 0.30)
            if self._best_desc_cosine(lm, descriptor) < need:
                return None
        view = self._normalize_view(view_xy)
        extent = float(extent_m) if extent_m is not None and float(extent_m) > 0 else None
        self._note_association(lm, map_xyz=map_xyz, descriptor=descriptor)
        self._assoc_tracks += 1
        update_pose = None
        if pose_of_node is not None:
            update_pose = pose_of_node.get(lm.anchor_node)
        return self._update_existing(
            lm,
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
        anchor_pose: Optional[conv.Pose2D],
        *,
        from_detector: bool = False,
        extent_m: Optional[float] = None,
        bank_descriptor: bool = True,
    ) -> EmbeddingLandmark:
        """Fold one observation into ``lm``.

        ``anchor_pose`` is the pose of ``lm.anchor_node`` (``None`` when
        unknown, which leaves ``rel_xyz`` untouched).
        """
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
        if (
            lm.status == LandmarkStatus.CANDIDATE
            and lm.anchor_node >= 0
            and anchor_pose is not None
        ):
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
            self._try_promote(lm, anchor_pose=anchor_pose)
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

    def _try_promote(
        self, lm: EmbeddingLandmark, *, anchor_pose: Optional[conv.Pose2D] = None
    ) -> None:
        min_obs = int(self.promote_min_obs)
        if bool(lm.from_detector):
            min_obs = max(min_obs, int(self.detector_promote_min_obs))
        if lm.observations < min_obs:
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
            # A box is one object, so a short drive past it is enough baseline.
            # Scatter stays at promote_max_std_m: a 22 cm allowance confirmed
            # landmarks whose position wandered 14–16 cm.
            min_gap_m = min(min_gap_m, 0.25)
            min_gap_deg = min(min_gap_deg, 18.0)
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
            # Confirming moves map_xyz to the sample mean; keep rel_xyz with it.
            if anchor_pose is not None and lm.anchor_node >= 0:
                lm.rel_xyz = self.rel_from_map_point(anchor_pose, new_xyz)
        lm._map_samples.clear()

    def _check_moved(
        self,
        lm: EmbeddingLandmark,
        map_xyz: np.ndarray,
        *,
        pose_hint: Optional[conv.Pose2D],
    ) -> None:
        if lm.map_xyz is not None:
            expected = np.asarray(lm.map_xyz, dtype=float).reshape(3)
        elif pose_hint is not None:
            expected = self.map_point_from_rel(pose_hint, lm.rel_xyz)
        else:
            return
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
            "assoc_ambiguous": int(self._assoc_ambiguous),
            "assoc_tracks": int(self._assoc_tracks),
            "overlap_merges": int(self._overlap_merges),
            "seen_through_flagged": len(self.seen_through_ids()),
            "seen_through_removed": int(self._seen_through_removed),
            "assoc_hit_rate": (hits / attempts) if attempts else None,
            "mean_descriptors": (n_desc / max(1, len(self._landmarks))),
        }
