"""Shared dataclasses for the vision / embedding-landmark side-channel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class LandmarkStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    MOVED = "moved"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a color (or aligned-depth) image."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def pixel_to_ray(self, u: float, v: float) -> Tuple[float, float, float]:
        """Unit-ish optical-frame ray (X right, Y down, Z forward)."""
        x = (float(u) - float(self.cx)) / float(self.fx)
        y = (float(v) - float(self.cy)) / float(self.fy)
        return x, y, 1.0


@dataclass
class VisionFrame:
    """Latest decoded RGB-D observation from one vision camera.

    ``color_rgb`` is HxWx3 uint8. ``depth_m`` is HxW float32 meters (0 = invalid).
    ``captured_at`` is host monotonic seconds (aligned with SLAM odom history).
    """

    camera_name: str
    stamp: float
    captured_at: float
    color_rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    # Mount in base_link (same fields as LidarConfig for obstacles_only cams).
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_z: float = 0.0
    mount_theta: float = 0.0
    mount_pitch: float = 0.0
    mount_roll: float = 0.0


@dataclass
class Detection:
    """Axis-aligned box from the dynamic paint-gate detector."""

    label: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PatchToken:
    """One DINOv2 (or synthetic) patch with optional 3D measurement."""

    row: int
    col: int
    u: float
    v: float
    descriptor: np.ndarray  # (D,) float32
    # Local luminance std (0–255) used to prefer paintings/edges over flat walls.
    texture: float = 0.0
    # Filled by backproject_patches when depth is valid.
    point_base: Optional[np.ndarray] = None  # (3,)
    normal_base: Optional[np.ndarray] = None  # (3,)
    sigma_m: float = float("nan")
    valid: bool = False


@dataclass
class VisionResult:
    """Worker output for one camera frame (latest-only, never queued)."""

    camera_name: str
    stamp: float
    captured_at: float
    patches: List[PatchToken] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    # Whole-image place descriptor (VLAD / mean-pool); may be empty until fit.
    place_descriptor: Optional[np.ndarray] = None
    infer_ms: float = 0.0
    provider: str = "none"
