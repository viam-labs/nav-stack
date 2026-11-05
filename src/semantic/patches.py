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
    min_valid_frac: float = 0.5,
    max_spread_frac: float = 0.1,
    window_half: int = 2,
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


def subsample_observations(
    observations: Sequence[PatchObservation],
    max_count: int,
) -> List[PatchObservation]:
    """Uniform stride subsample so a frame cannot flood the landmark map."""
    obs = list(observations)
    limit = int(max_count)
    if limit <= 0 or len(obs) <= limit:
        return obs
    # Prefer lower range uncertainty, then stride through the ranked list so
    # we keep spatial coverage instead of only the closest patches.
    ranked = sorted(obs, key=lambda o: float(o.sigma_m))
    step = max(1, int(math.ceil(len(ranked) / float(limit))))
    picked = ranked[::step][:limit]
    if len(picked) < limit:
        # Fill remaining from the unpicked front of the ranked list.
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
    max_ratio: float = 0.9,
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
