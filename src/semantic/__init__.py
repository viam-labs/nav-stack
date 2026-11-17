"""Vision side-channel: camera keyframes for lost-pose recovery, and a person mask.

Kept out of the SLAM hot-path import graph. Inference runs in ``worker``; the
geometry is pure numpy and unit-tested here.
"""

from __future__ import annotations

from .fit import RigidFitResult, fit_rigid_2d
from .paint_mask import Disc, filter_scan_endpoints
from .patches import CameraIntrinsics, PatchObservation, backproject_patches
from .supervisor import VisionSupervisor
from .types import VisionFrame

__all__ = [
    "CameraIntrinsics",
    "Disc",
    "PatchObservation",
    "RigidFitResult",
    "VisionFrame",
    "VisionSupervisor",
    "backproject_patches",
    "filter_scan_endpoints",
    "fit_rigid_2d",
]
