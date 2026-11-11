"""Build landmark observations by pooling DINOv2 patches inside detector boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .patches import (
    PatchObservation,
    backproject_pixels,
    depth_sigma_m,
    lateral_sigma_m,
    optical_normal_to_base,
)
from .place import l2_normalize
from .types import Detection, PatchToken, VisionFrame


# Indoor-ish defaults. YOLO-World exports should use the same strings (order = class id).
DEFAULT_LANDMARK_LABELS: tuple[str, ...] = (
    "painting",
    "poster",
    "picture",
    "picture frame",
    "framed picture",
    "artwork",
    "tv",
    "television",
    "monitor",
    "door",
    "fire extinguisher",
    "clock",
    "whiteboard",
    "sign",
    "exit sign",
    "potted plant",
    "plant",
    "bookshelf",
    "cabinet",
    "poster board",
)


def label_allowed(label: str, allowed: Set[str]) -> bool:
    lab = str(label).strip().lower()
    if not lab:
        return False
    if lab in allowed:
        return True
    # Soft match: "picture frame" vs "picture"
    for a in allowed:
        if lab in a or a in lab:
            return True
    return False


# Back-compat alias for callers that imported the private name.
_label_allowed = label_allowed


def pool_descriptor(descriptors: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(d, dtype=np.float32).reshape(-1) for d in descriptors], axis=0)
    return l2_normalize(stacked.mean(axis=0).astype(np.float32))


def tokens_inside_detections(
    patches: Sequence[PatchToken],
    detections: Sequence[Detection],
    *,
    shrink: float = 0.0,
) -> List[PatchToken]:
    """Return patch tokens whose centers fall inside any detection box."""
    if not patches or not detections:
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    for det in detections:
        x0, x1 = sorted((float(det.x0), float(det.x1)))
        y0, y1 = sorted((float(det.y0), float(det.y1)))
        if shrink > 0.0:
            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            x0 += shrink * bw
            x1 -= shrink * bw
            y0 += shrink * bh
            y1 -= shrink * bh
        boxes.append((x0, y0, x1, y1))
    out: List[PatchToken] = []
    seen = set()
    for tok in patches:
        u, v = float(tok.u), float(tok.v)
        key = (tok.row, tok.col, round(u, 1), round(v, 1))
        if key in seen:
            continue
        if any(x0 <= u <= x1 and y0 <= v <= y1 for x0, y0, x1, y1 in boxes):
            seen.add(key)
            out.append(tok)
    return out


@dataclass(frozen=True)
class BoxSurface:
    """The one surface a detector box sits on, measured from raw depth."""

    point_base: np.ndarray  # (3,)
    normal_base: np.ndarray  # (3,)
    depth_m: float
    sigma_m: float
    # Share of the box core's valid depth pixels that lie on this surface.
    support: float


# Why a box produced no surface (keys of the ``stats`` dict).
SKIP_SPARSE_DEPTH = "skipped_sparse_depth"
SKIP_MIXED_DEPTH = "skipped_mixed_depth"
SKIP_RANGE = "skipped_range"
SKIP_HEIGHT = "skipped_height"
# Other per-box outcomes (``outcomes`` list of observations_from_detections).
BOX_USED = "used"
BOX_NOT_LANDMARK_LABEL = "not_landmark_label"
BOX_NO_PATCHES = "no_patches"

# A box is located from hundreds of depth pixels, so it stays usable further
# out than a single patch (4 m): on camera-top a poster 5.6 m away repeated
# within one depth step (9 cm) over six frames. Past 6 m a step is 10 cm or
# more and far walls start to read as noise.
BOX_MIN_RANGE_M = 0.4
BOX_MAX_RANGE_M = 6.0


def describe_box_outcome(outcome: str, depth_m: Optional[float] = None) -> str:
    """Short text for landmark-camera: why a box did or did not count."""
    if outcome == SKIP_RANGE and depth_m is not None:
        side = "far" if float(depth_m) > BOX_MAX_RANGE_M else "near"
        return f"{side} {float(depth_m):.1f}m"
    return {
        BOX_USED: "used",
        SKIP_RANGE: "out of range",
        SKIP_MIXED_DEPTH: "mixed depth",
        SKIP_SPARSE_DEPTH: "no depth",
        SKIP_HEIGHT: "height",
        BOX_NOT_LANDMARK_LABEL: "not a landmark class",
        BOX_NO_PATCHES: "too small",
    }.get(str(outcome), str(outcome))


def _cluster_width_m(depth_m: np.ndarray) -> np.ndarray:
    """Depth span still counted as one surface, per starting depth.

    Wide enough for a painting seen at about 45 degrees, narrow enough to keep
    a table apart from the wall a metre behind it.
    """
    return 2.0 * np.maximum(0.12, 0.06 * np.asarray(depth_m, dtype=float))


def box_surface_point(
    frame: VisionFrame,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    core_frac: float = 0.6,
    min_valid_frac: float = 0.30,
    min_valid_px: int = 12,
    min_support: float = 0.60,
    min_range_m: float = BOX_MIN_RANGE_M,
    max_range_m: float = BOX_MAX_RANGE_M,
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
) -> Tuple[Optional[BoxSurface], Optional[str]]:
    """Locate the object in a detector box, or say why it cannot be located.

    Reads raw depth in the central ``core_frac`` of the box and keeps the
    dominant depth cluster. A box that spans two surfaces with no clear
    majority (object plus background, or an occluder in front) returns
    ``None``: a median across them is a point on neither.

    Returns ``(surface, None)`` or ``(None, skip_reason)``.
    """
    surface, why, _depth = _locate_box(
        frame,
        x0,
        y0,
        x1,
        y1,
        core_frac=core_frac,
        min_valid_frac=min_valid_frac,
        min_valid_px=min_valid_px,
        min_support=min_support,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )
    return surface, why


def _locate_box(
    frame: VisionFrame,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    core_frac: float = 0.6,
    min_valid_frac: float = 0.30,
    min_valid_px: int = 12,
    min_support: float = 0.60,
    min_range_m: float = BOX_MIN_RANGE_M,
    max_range_m: float = BOX_MAX_RANGE_M,
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
) -> Tuple[Optional[BoxSurface], Optional[str], Optional[float]]:
    """:func:`box_surface_point` plus the surface depth when one was found.

    The depth is reported even for a surface rejected for range or height,
    so the caller can say how far away the skipped object is.
    """
    depth = np.asarray(frame.depth_m, dtype=float)
    h, w = depth.shape[:2]
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    inset = 0.5 * (1.0 - float(core_frac))
    u0 = int(math.floor(xa + inset * (xb - xa)))
    u1 = int(math.ceil(xb - inset * (xb - xa)))
    v0 = int(math.floor(ya + inset * (yb - ya)))
    v1 = int(math.ceil(yb - inset * (yb - ya)))
    u0, u1 = max(0, u0), min(w, max(u1, u0 + 1))
    v0, v1 = max(0, v0), min(h, max(v1, v0 + 1))
    core = depth[v0:v1, u0:u1]
    if core.size == 0:
        return None, SKIP_SPARSE_DEPTH, None
    valid = np.isfinite(core) & (core > 0.0)
    n_valid = int(valid.sum())
    if n_valid < int(min_valid_px) or n_valid / float(core.size) < float(min_valid_frac):
        return None, SKIP_SPARSE_DEPTH, None

    vs_idx, us_idx = np.nonzero(valid)
    vals = core[valid]
    order = np.argsort(vals)
    sorted_vals = vals[order]
    # Densest depth window: start at each sample, count those within one
    # cluster width behind it.
    ends = np.searchsorted(
        sorted_vals, sorted_vals + _cluster_width_m(sorted_vals), side="right"
    )
    counts = ends - np.arange(n_valid)
    start = int(np.argmax(counts))
    end = int(ends[start])
    support = float(end - start) / float(n_valid)
    if support < float(min_support):
        # Far walls read as noise with no dominant depth; call that range,
        # not a mix of surfaces.
        overall = float(np.median(vals))
        if overall > float(max_range_m):
            return None, SKIP_RANGE, overall
        return None, SKIP_MIXED_DEPTH, None

    members = order[start:end]
    cluster_depths = vals[members]
    depth_med = float(np.median(cluster_depths))
    if depth_med < float(min_range_m) or depth_med > float(max_range_m):
        return None, SKIP_RANGE, depth_med
    us = us_idx[members].astype(float) + float(u0)
    vs = vs_idx[members].astype(float) + float(v0)
    points = backproject_pixels(frame, us, vs, cluster_depths)
    point = np.median(points, axis=0)
    z = float(point[2])
    if z_min_m is not None and z < float(z_min_m):
        return None, SKIP_HEIGHT, depth_med
    if z_max_m is not None and z > float(z_max_m):
        return None, SKIP_HEIGHT, depth_med

    normal_optical = _surface_normal_optical(frame, us, vs, cluster_depths)
    sigma = float(
        math.hypot(
            depth_sigma_m(depth_med),
            lateral_sigma_m(depth_med, frame.intrinsics.fx),
        )
    )
    return (
        BoxSurface(
            point_base=point.astype(float),
            normal_base=optical_normal_to_base(frame, normal_optical),
            depth_m=depth_med,
            sigma_m=sigma,
            support=support,
        ),
        None,
        depth_med,
    )


def _surface_normal_optical(
    frame: VisionFrame,
    us: np.ndarray,
    vs: np.ndarray,
    depths_m: np.ndarray,
    *,
    max_points: int = 400,
) -> np.ndarray:
    """Plane-fit normal of the cluster in the optical frame, facing the camera."""
    toward_camera = np.array([0.0, 0.0, -1.0], dtype=float)
    n = len(depths_m)
    if n < 3:
        return toward_camera
    step = max(1, n // int(max_points))
    K = frame.intrinsics
    d = np.asarray(depths_m, dtype=float)[::step]
    cloud = np.column_stack(
        [
            (us[::step] - float(K.cx)) / float(K.fx) * d,
            (vs[::step] - float(K.cy)) / float(K.fy) * d,
            d,
        ]
    )
    centered = cloud - cloud.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0.0:
        normal = -normal
    nn = float(np.linalg.norm(normal))
    return normal / nn if nn > 1e-9 else toward_camera


def observations_from_detections(
    frame: VisionFrame,
    patches: Sequence[PatchToken],
    detections: Sequence[Detection],
    *,
    landmark_labels: Iterable[str],
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
    min_patches: int = 1,
    shrink: float = 0.05,
    stats: Optional[Dict[str, int]] = None,
    outcomes: Optional[List[dict]] = None,
) -> List[PatchObservation]:
    """One observation per allowed detection: mean DINO + the box's surface.

    Descriptor pooling uses every DINO token whose center lies in the box
    (depth-invalid tokens still contribute appearance). Geometry comes from
    :func:`box_surface_point`; a box with no single clear surface yields no
    observation. ``stats`` (optional) is filled with per-reason skip counts.

    ``outcomes`` (optional) receives one dict per detection, in order:
    ``label``, ``outcome`` (``used`` or why not), ``depth_m`` when a surface
    was found, and ``obs_index`` into the returned list when used.
    """
    allowed = {str(x).strip().lower() for x in landmark_labels if str(x).strip()}
    if not allowed or not detections or not patches:
        return []

    def note(det: Detection, outcome: str, depth_m=None, obs_index=None) -> None:
        if outcomes is not None:
            outcomes.append(
                {
                    "label": str(det.label),
                    "outcome": outcome,
                    "depth_m": None if depth_m is None else float(depth_m),
                    "obs_index": obs_index,
                }
            )

    out: List[PatchObservation] = []
    for det in detections:
        if not label_allowed(det.label, allowed):
            note(det, BOX_NOT_LANDMARK_LABEL)
            continue
        x0, x1 = sorted((float(det.x0), float(det.x1)))
        y0, y1 = sorted((float(det.y0), float(det.y1)))
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        s = max(0.0, float(shrink))
        x0i, x1i = x0 + s * bw, x1 - s * bw
        y0i, y1i = y0 + s * bh, y1 - s * bh
        # Appearance: all tokens in box (even if depth failed).
        in_tokens = [
            t
            for t in patches
            if x0i <= float(t.u) <= x1i and y0i <= float(t.v) <= y1i
        ]
        if len(in_tokens) < int(min_patches):
            # Retry without shrink — small YOLO boxes otherwise starve.
            in_tokens = [
                t
                for t in patches
                if x0 <= float(t.u) <= x1 and y0 <= float(t.v) <= y1
            ]
        if len(in_tokens) < int(min_patches):
            note(det, BOX_NO_PATCHES)
            continue
        desc = pool_descriptor([t.descriptor for t in in_tokens])
        surface, skip_reason, depth_m = _locate_box(
            frame, x0, y0, x1, y1, z_min_m=z_min_m, z_max_m=z_max_m
        )
        if surface is None:
            if stats is not None and skip_reason:
                stats[skip_reason] = int(stats.get(skip_reason, 0)) + 1
            note(det, str(skip_reason), depth_m)
            continue
        note(det, BOX_USED, depth_m, obs_index=len(out))
        point = surface.point_base
        normal = surface.normal_base
        sigma = float(surface.sigma_m)
        tok = PatchToken(
            row=-1,
            col=-1,
            u=0.5 * (x0 + x1),
            v=0.5 * (y0 + y1),
            descriptor=desc,
            texture=float(max((float(t.texture) for t in in_tokens), default=0.0)),
            point_base=point.astype(float),
            normal_base=np.asarray(normal, dtype=float),
            sigma_m=sigma,
            valid=True,
        )
        # Map-plane half-size of the YOLO box (meters) for footprint overlap.
        range_m = max(0.2, float(surface.depth_m))
        fx = max(1.0, float(frame.intrinsics.fx))
        fy = max(1.0, float(frame.intrinsics.fy))
        extent_m = max(0.5 * bw * range_m / fx, 0.5 * bh * range_m / fy)
        out.append(
            PatchObservation(
                token=tok,
                point_base=point.astype(float),
                normal_base=np.asarray(normal, dtype=float),
                sigma_m=sigma,
                extent_m=float(extent_m),
            )
        )
    return out
