"""Vision side-channel: embedding landmarks from RGB-D for exact localization.

Kept out of the SLAM hot-path import graph. Inference runs in a subprocess
(see ``worker``); geometry / landmark math is pure numpy and unit-tested here.
"""

from __future__ import annotations

from .fit import RigidFitResult, fit_rigid_2d
from .landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from .paint_mask import Disc, filter_scan_endpoints
from .patches import CameraIntrinsics, PatchObservation, backproject_patches
from .supervisor import VisionSupervisor
from .types import LandmarkStatus, VisionFrame

__all__ = [
    "CameraIntrinsics",
    "Disc",
    "EmbeddingLandmark",
    "EmbeddingLandmarkMap",
    "LandmarkStatus",
    "PatchObservation",
    "RigidFitResult",
    "VisionFrame",
    "VisionSupervisor",
    "backproject_patches",
    "filter_scan_endpoints",
    "fit_rigid_2d",
]
