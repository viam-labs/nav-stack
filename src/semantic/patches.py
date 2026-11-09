"""Patch depth back-projection into base_link with validity and sigmas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geom.conversions import camera_optical_to_sensor_frame, transform_lidar_mount_to_base_link
from .types import CameraIntrinsics, PatchToken, VisionFrame

# D435-class range noise: a * r^2 + b (meters). Conservative vs datasheet.
_DEPTH_SIGMA_A = 0.0025
_DEPTH_SIGMA_B = 0.005
_PIXEL_SIGMA = 1.0  # px lateral uncertainty before sub-patch refine


@dataclass(frozen=True)
class PatchObservation:
    """A measured patch ready for landmark association."""

    token: PatchToken
    point_base: np.ndarray  # (3,)
    normal_base: np.ndarray  # (3,)
    sigma_m: float


def depth_sigma_m(range_m: float) -> float:
    r = max(0.0, float(range_m))
    return _DEPTH_SIGMA_A * r * r + _DEPTH_SIGMA_B


def lateral_sigma_m(range_m: float, fx: float) -> float:
    """~1 px lateral error projected to meters at ``range_m``."""
    f = max(1.0, float(fx))
    return (_PIXEL_SIGMA / f) * max(0.0, float(range_m))


def patch_grid(
    height: int,
    width: int,
    *,
    patch: int = 14,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centers (u, v) for a ViT-style patch grid covering the image."""
    stride = int(stride if stride is not None else patch)
    rows = max(1, (int(height) - patch) // stride + 1)
    cols = max(1, (int(width) - patch) // stride + 1)
    vs = np.arange(rows, dtype=float) * stride + patch / 2.0
    us = np.arange(cols, dtype=float) * stride + patch / 2.0
    uu, vv = np.meshgrid(us, vs)
    return uu.reshape(-1), vv.reshape(-1)


def _median_depth_window(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    half: int,
) -> Tuple[float, float, int, int]:
    """Median depth and spread in a window; returns (median, std, valid, total)."""
    h, w = depth_m.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    x0 = max(0, ui - half)
    x1 = min(w, ui + half + 1)
    y0 = max(0, vi - half)
    y1 = min(h, vi + half + 1)
    win = depth_m[y0:y1, x0:x1]
    total = int(win.size)
    if total == 0:
        return float("nan"), float("nan"), 0, 0
    valid_mask = np.isfinite(win) & (win > 0.0)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return float("nan"), float("nan"), 0, total
    vals = win[valid_mask].astype(float)
    return float(np.median(vals)), float(np.std(vals)), n_valid, total


def _depth_support_ok(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    median_m: float,
    support_half: int = 6,
    min_valid_frac: float = 0.75,
    max_span_frac: float = 0.12,
    max_span_m: float = 0.35,
) -> bool:
    """Reject glass / window-edge / through-window depth.

    RealSense often returns the *glass* range while RGB texture is a far
    building. That looks correct from the first viewpoint (same ray) and then
    slides onto the window frame under parallax. Require a solid neighborhood
    of similar depths so outdoor-through-glass and frame edges drop out.
    """
    h, w = depth_m.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    half = int(support_half)
    x0 = max(0, ui - half)
    x1 = min(w, ui + half + 1)
    y0 = max(0, vi - half)
    y1 = min(h, vi + half + 1)
    win = depth_m[y0:y1, x0:x1]
    total = int(win.size)
    if total <= 0:
        return False
    valid = np.isfinite(win) & (win > 0.0)
    n_valid = int(valid.sum())
    if n_valid / float(total) < float(min_valid_frac):
        return False
    vals = win[valid].astype(float)
    span = float(np.percentile(vals, 90) - np.percentile(vals, 10))
    med = max(1e-3, float(median_m))
    if span > float(max_span_m) or span > float(max_span_frac) * med:
        return False
    # Neighborhood median should agree with the patch median (no mixed planes).
    if abs(float(np.median(vals)) - med) > max(0.12, 0.05 * med):
        return False
    return True


def _plane_normal_optical(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    u: float,
    v: float,
    *,
    half: int = 2,
) -> np.ndarray:
    """Rough surface normal in optical frame from a small depth neighborhood."""
    h, w = depth_m.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    pts: List[np.ndarray] = []
    for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
            x = ui + du
            y = vi + dv
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            z = float(depth_m[y, x])
            if not math.isfinite(z) or z <= 0.0:
                continue
            ox, oy, oz = intrinsics.pixel_to_ray(float(x), float(y))
            pts.append(np.array([ox * z, oy * z, oz * z], dtype=float))
    if len(pts) < 3:
        return np.array([0.0, 0.0, -1.0], dtype=float)  # toward camera
    cloud = np.stack(pts, axis=0)
    centered = cloud - cloud.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    # Face the camera (optical -Z toward sensor is +Z of ray? camera looks +Z;
    # surface facing camera has normal with negative Z component in optical).
    if normal[2] > 0.0:
        normal = -normal
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return (normal / n).astype(float)


def backproject_patches(
    frame: VisionFrame,
    tokens: Sequence[PatchToken],
    *,
    min_range_m: float = 0.4,
    max_range_m: float = 4.0,
    min_valid_frac: float = 0.75,
    max_spread_frac: float = 0.06,
    window_half: int = 2,
    support_half: int = 6,
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
) -> List[PatchObservation]:
    """Fill ``point_base`` / validity on tokens; return only valid observations.

    ``z_min_m`` / ``z_max_m`` gate height in base_link (drop floor/ceiling).
    """
    depth = np.asarray(frame.depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got shape {depth.shape}")
    K = frame.intrinsics
    out: List[PatchObservation] = []
    for tok in tokens:
        med, spread, n_valid, total = _median_depth_window(
            depth, tok.u, tok.v, half=window_half
        )
        tok.valid = False
        tok.point_base = None
        tok.normal_base = None
        tok.sigma_m = float("nan")
        if total <= 0 or n_valid / float(total) < float(min_valid_frac):
            continue
        if not math.isfinite(med) or med < float(min_range_m) or med > float(max_range_m):
            continue
        if math.isfinite(spread) and spread > float(max_spread_frac) * med:
            continue
        if not _depth_support_ok(
            depth,
            tok.u,
            tok.v,
            median_m=float(med),
            support_half=int(support_half),
            min_valid_frac=float(min_valid_frac),
        ):
            continue

        ox, oy, oz = K.pixel_to_ray(tok.u, tok.v)
        optical = np.array([[ox * med, oy * med, oz * med]], dtype=float)
        sensor = camera_optical_to_sensor_frame(optical)
        base = transform_lidar_mount_to_base_link(
            sensor,
            x=frame.mount_x,
            y=frame.mount_y,
            z=frame.mount_z,
            theta=frame.mount_theta,
            pitch=frame.mount_pitch,
            roll=frame.mount_roll,
        )
        point = base[0].astype(float)
        z = float(point[2])
        if z_min_m is not None and z < float(z_min_m):
            continue
        if z_max_m is not None and z > float(z_max_m):
            continue

        n_opt = _plane_normal_optical(depth, K, tok.u, tok.v, half=window_half)
        n_sensor = camera_optical_to_sensor_frame(n_opt.reshape(1, 3))[0]
        # Rotate normal by mount yaw/pitch/roll (ignore translation).
        n_base = transform_lidar_mount_to_base_link(
            n_sensor.reshape(1, 3),
            x=0.0,
            y=0.0,
            z=0.0,
            theta=frame.mount_theta,
            pitch=frame.mount_pitch,
            roll=frame.mount_roll,
        )[0]
        nn = float(np.linalg.norm(n_base))
        if nn > 1e-9:
            n_base = n_base / nn

        sig_r = depth_sigma_m(med)
        sig_l = lateral_sigma_m(med, K.fx)
        sigma = float(math.hypot(sig_r, sig_l))

        tok.point_base = point
        tok.normal_base = n_base.astype(float)
        tok.sigma_m = sigma
        tok.valid = True
        out.append(
            PatchObservation(
                token=tok,
                point_base=point,
                normal_base=n_base.astype(float),
                sigma_m=sigma,
            )
        )
    return out


def local_luminance_std(
    color_rgb: np.ndarray,
    u: float,
    v: float,
    *,
    half: int = 7,
) -> float:
    """Local grayscale std around ``(u, v)`` — high on paintings, low on walls."""
    img = np.asarray(color_rgb)
    if img.ndim != 3 or img.size == 0:
        return 0.0
    h, w = img.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    x0 = max(0, ui - int(half))
    x1 = min(w, ui + int(half) + 1)
    y0 = max(0, vi - int(half))
    y1 = min(h, vi + int(half) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    win = img[y0:y1, x0:x1].astype(np.float32)
    gray = win.mean(axis=2)
    return float(gray.std())


def annotate_patch_texture(
    color_rgb: np.ndarray,
    tokens: Sequence[PatchToken],
    *,
    half: int = 14,
) -> None:
    """Fill ``token.texture`` from local luminance contrast."""
    for tok in tokens:
        tok.texture = local_luminance_std(color_rgb, tok.u, tok.v, half=half)


def select_survey_tokens(
    color_rgb: np.ndarray,
    tokens: Sequence[PatchToken],
    *,
    min_texture: float = 8.0,
    max_keep: int = 64,
    distinctiveness_bonus: float = 10.0,
    max_ratio: float = 0.92,
) -> List[PatchToken]:
    """Rank patches for survey: paintings/edges first, flat walls last.

    Hard distinctiveness filtering alone drops painting *interiors* (adjacent
    patches on the same artwork look alike). Texture is the primary signal;
    distinctiveness is only a soft bonus for isolated features.
    """
    toks = list(tokens)
    if not toks:
        return []
    annotate_patch_texture(color_rgb, toks)
    desc = np.stack([t.descriptor for t in toks], axis=0)
    distinctive = distinctiveness_mask(desc, max_ratio=float(max_ratio))
    bonus = float(distinctiveness_bonus)

    def _score(i: int) -> float:
        return float(toks[i].texture) + (bonus if bool(distinctive[i]) else 0.0)

    order = sorted(range(len(toks)), key=_score, reverse=True)
    floor = float(min_texture)
    # Prefer everything above the texture floor; if the scene is bland, still
    # keep the top-scoring patches so survey is not empty.
    above = [toks[i] for i in order if float(toks[i].texture) >= floor]
    ranked = above if above else [toks[i] for i in order]
    limit = int(max_keep)
    if limit <= 0 or len(ranked) <= limit:
        return ranked
    return ranked[:limit]


def filter_textured_tokens(
    tokens: Sequence[PatchToken],
    *,
    min_texture: float = 8.0,
    max_keep: int = 64,
) -> List[PatchToken]:
    """Drop flat wall patches; keep the most textured first.

    ``min_texture`` is luminance std on a 0–255 scale. Plain white walls are
    typically well below 8; framed art / posters sit much higher.
    """
    scored = [t for t in tokens if float(t.texture) >= float(min_texture)]
    if not scored:
        return []
    scored.sort(key=lambda t: float(t.texture), reverse=True)
    limit = int(max_keep)
    if limit <= 0 or len(scored) <= limit:
        return scored
    return scored[:limit]


def subsample_observations(
    observations: Sequence[PatchObservation],
    max_count: int,
) -> List[PatchObservation]:
    """Cap associations per frame, preferring textured + low-uncertainty patches."""
    obs = list(observations)
    limit = int(max_count)
    if limit <= 0 or len(obs) <= limit:
        return obs
    # Texture first (paintings over walls), then tighter depth as a tie-break.
    # Stride through the ranked list for spatial coverage.
    ranked = sorted(
        obs,
        key=lambda o: (-float(getattr(o.token, "texture", 0.0)), float(o.sigma_m)),
    )
    step = max(1, int(math.ceil(len(ranked) / float(limit))))
    picked = ranked[::step][:limit]
    if len(picked) < limit:
        seen = {id(p) for p in picked}
        for o in ranked:
            if id(o) in seen:
                continue
            picked.append(o)
            if len(picked) >= limit:
                break
    return picked


def distinctiveness_mask(
    descriptors: np.ndarray,
    *,
    max_ratio: float = 0.92,
) -> np.ndarray:
    """Keep patches whose best/second-best in-frame cosine ratio is below max.

    ``descriptors`` is (N, D) L2-normalized (or will be normalized here).
    Returns a boolean mask of length N.
    """
    desc = np.asarray(descriptors, dtype=float)
    if desc.ndim != 2 or desc.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    d = desc / norms
    # Cosine similarity matrix; ignore self.
    sim = d @ d.T
    np.fill_diagonal(sim, -np.inf)
    # Partition for top-2.
    if d.shape[0] == 1:
        return np.ones((1,), dtype=bool)
    part = np.partition(sim, -2, axis=1)
    best = part[:, -1]
    second = part[:, -2]
    # No strong neighbor → distinctive. Two strong similar neighbors → drop.
    keep = np.ones(d.shape[0], dtype=bool)
    strong = best > 0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(strong, second / np.maximum(best, 1e-12), 0.0)
    keep[strong] = ratio[strong] < float(max_ratio)
    return keep
