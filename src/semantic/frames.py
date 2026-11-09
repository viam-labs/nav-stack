"""Helpers to build VisionFrame metadata from LidarConfig mounts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .types import CameraIntrinsics, VisionFrame

if TYPE_CHECKING:
    from ..config import LidarConfig


def intrinsics_from_size(
    width: int,
    height: int,
    *,
    fov_deg: float = 87.0,
    vfov_deg: float | None = None,
) -> CameraIntrinsics:
    """Approximate pinhole K from image size + FOV.

    ``fov_deg`` is horizontal; ``vfov_deg`` is vertical (defaults to the same
    angular value only when omitted — prefer passing the camera's real vfov,
    otherwise painted landmarks drift vertically).
    """
    w = max(1, int(width))
    h = max(1, int(height))
    fx = (0.5 * w) / max(1e-6, np.tan(np.deg2rad(float(fov_deg) * 0.5)))
    v = float(fov_deg if vfov_deg is None else vfov_deg)
    fy = (0.5 * h) / max(1e-6, np.tan(np.deg2rad(v * 0.5)))
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * w,
        cy=0.5 * h,
        width=w,
        height=h,
    )


def intrinsics_from_camera_properties(
    props: Any,
    *,
    image_width: int,
    image_height: int,
) -> Optional[CameraIntrinsics]:
    """Build K from Viam ``Camera.get_properties().intrinsic_parameters``.

    Scales fx/fy/cx/cy when the live frame size differs from the calibration
    resolution. Returns ``None`` when properties are missing or empty so
    callers can fall back to FOV config.
    """
    ip = getattr(props, "intrinsic_parameters", None)
    if ip is None:
        return None
    fx = float(getattr(ip, "focal_x_px", 0.0) or 0.0)
    fy = float(getattr(ip, "focal_y_px", 0.0) or 0.0)
    cx = float(getattr(ip, "center_x_px", 0.0) or 0.0)
    cy = float(getattr(ip, "center_y_px", 0.0) or 0.0)
    pw = int(getattr(ip, "width_px", 0) or 0)
    ph = int(getattr(ip, "height_px", 0) or 0)
    if fx < 1.0 or fy < 1.0:
        return None
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    if pw > 1 and ph > 1 and (pw != w or ph != h):
        sx = w / float(pw)
        sy = h / float(ph)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
    elif cx <= 0.0 and cy <= 0.0:
        cx, cy = 0.5 * w, 0.5 * h
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
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
    K = intrinsics or intrinsics_from_size(
        w, h, fov_deg=lidar.fov_deg, vfov_deg=lidar.vfov_deg
    )
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
