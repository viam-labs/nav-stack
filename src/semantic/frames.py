"""Helpers to build VisionFrame metadata from LidarConfig mounts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .types import CameraIntrinsics, VisionFrame

if TYPE_CHECKING:
    from ..config import LidarConfig


def intrinsics_from_size(
    width: int,
    height: int,
    *,
    fov_deg: float = 87.0,
) -> CameraIntrinsics:
    """Approximate pinhole K from image size + horizontal FOV."""
    w = max(1, int(width))
    h = max(1, int(height))
    fx = (0.5 * w) / max(1e-6, np.tan(np.deg2rad(float(fov_deg) * 0.5)))
    fy = fx
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * w,
        cy=0.5 * h,
        width=w,
        height=h,
    )


def vision_frame_from_arrays(
    *,
    camera_name: str,
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    stamp: float,
    captured_at: float,
    lidar: "LidarConfig",
    intrinsics: CameraIntrinsics | None = None,
) -> VisionFrame:
    """Stamp a decoded RGB-D pair with the obstacles_only lidar mount."""
    h, w = color_rgb.shape[:2]
    K = intrinsics or intrinsics_from_size(w, h, fov_deg=lidar.fov_deg)
    return VisionFrame(
        camera_name=camera_name,
        stamp=float(stamp),
        captured_at=float(captured_at),
        color_rgb=np.asarray(color_rgb),
        depth_m=np.asarray(depth_m, dtype=np.float32),
        intrinsics=K,
        mount_x=float(lidar.x),
        mount_y=float(lidar.y),
        mount_z=float(lidar.z),
        mount_theta=float(lidar.theta),
        mount_pitch=float(lidar.pitch),
        mount_roll=float(lidar.roll),
    )
