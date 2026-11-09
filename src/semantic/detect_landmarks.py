"""Build landmark observations by pooling DINOv2 patches inside detector boxes."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import numpy as np

from .patches import PatchObservation, backproject_patches
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


def observations_from_detections(
    frame: VisionFrame,
    patches: Sequence[PatchToken],
    detections: Sequence[Detection],
    *,
    landmark_labels: Iterable[str],
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
    min_patches: int = 2,
) -> List[PatchObservation]:
    """One observation per allowed detection: mean DINO + median depth in-box."""
    allowed = {str(x).strip().lower() for x in landmark_labels if str(x).strip()}
    if not allowed or not detections or not patches:
        return []

    patch_obs = backproject_patches(
        frame,
        patches,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )
    if not patch_obs:
        return []

    out: List[PatchObservation] = []
    for det in detections:
        if not label_allowed(det.label, allowed):
            continue
        x0, x1 = sorted((float(det.x0), float(det.x1)))
        y0, y1 = sorted((float(det.y0), float(det.y1)))
        # Slightly shrink box to avoid frame/edge depth contamination.
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        x0i, x1i = x0 + 0.08 * bw, x1 - 0.08 * bw
        y0i, y1i = y0 + 0.08 * bh, y1 - 0.08 * bh
        inside = [
            o
            for o in patch_obs
            if x0i <= float(o.token.u) <= x1i and y0i <= float(o.token.v) <= y1i
        ]
        if len(inside) < int(min_patches):
            continue
        desc = pool_descriptor([o.token.descriptor for o in inside])
        pts = np.stack([o.point_base for o in inside], axis=0)
        point = np.median(pts, axis=0)
        normals = np.stack([o.normal_base for o in inside], axis=0)
        normal = normals.mean(axis=0)
        nn = float(np.linalg.norm(normal))
        if nn > 1e-9:
            normal = normal / nn
        else:
            normal = inside[0].normal_base
        sigma = float(np.median([o.sigma_m for o in inside]))
        # Synthetic token at box center carrying the pooled descriptor.
        tok = PatchToken(
            row=-1,
            col=-1,
            u=0.5 * (x0 + x1),
            v=0.5 * (y0 + y1),
            descriptor=desc,
            texture=float(max(o.token.texture for o in inside)),
            point_base=point.astype(float),
            normal_base=normal.astype(float),
            sigma_m=sigma,
            valid=True,
        )
        out.append(
            PatchObservation(
                token=tok,
                point_base=point.astype(float),
                normal_base=normal.astype(float),
                sigma_m=sigma,
            )
        )
    return out
