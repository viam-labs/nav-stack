"""Build landmark observations by pooling DINOv2 patches inside detector boxes."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

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
) -> List[PatchObservation]:
    """One observation per allowed detection: mean DINO + median depth in-box.

    Descriptor pooling uses every DINO token whose center lies in the box
    (depth-invalid tokens still contribute appearance). Geometry prefers
    depth-valid in-box patches; if none survive the depth gate, falls back to
    back-projecting the box center alone.
    """
    allowed = {str(x).strip().lower() for x in landmark_labels if str(x).strip()}
    if not allowed or not detections or not patches:
        return []

    # Depth-valid observations for geometry (may be a subset of in-box tokens).
    patch_obs = backproject_patches(
        frame,
        patches,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )
    obs_by_uv = {(round(float(o.token.u), 1), round(float(o.token.v), 1)): o for o in patch_obs}

    out: List[PatchObservation] = []
    for det in detections:
        if not label_allowed(det.label, allowed):
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
            continue
        desc = pool_descriptor([t.descriptor for t in in_tokens])
        inside_obs = [
            obs_by_uv[(round(float(t.u), 1), round(float(t.v), 1))]
            for t in in_tokens
            if (round(float(t.u), 1), round(float(t.v), 1)) in obs_by_uv
        ]
        if inside_obs:
            pts = np.stack([o.point_base for o in inside_obs], axis=0)
            point = np.median(pts, axis=0)
            normals = np.stack([o.normal_base for o in inside_obs], axis=0)
            normal = normals.mean(axis=0)
            nn = float(np.linalg.norm(normal))
            normal = normal / nn if nn > 1e-9 else inside_obs[0].normal_base
            sigma = float(np.median([o.sigma_m for o in inside_obs]))
        else:
            # Depth gate wiped the box — still landmark from center depth + pooled DINO.
            center = PatchToken(
                row=-1,
                col=-1,
                u=0.5 * (x0 + x1),
                v=0.5 * (y0 + y1),
                descriptor=desc,
            )
            center_obs = backproject_patches(
                frame,
                [center],
                z_min_m=z_min_m,
                z_max_m=z_max_m,
                window_half=2,
            )
            if not center_obs:
                continue
            point = center_obs[0].point_base
            normal = center_obs[0].normal_base
            sigma = float(center_obs[0].sigma_m)
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
        range_m = max(0.2, abs(float(point[0])))
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
