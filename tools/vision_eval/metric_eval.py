"""Patch points and camera pose fits for dataset frames.

Thin wrappers over the runtime code (``src/semantic/keyframes.py``,
``src/semantic/relocalize.py``) so experiments measure what ships.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.geom import conversions as conv
from src.semantic.keyframes import mutual_matches
from src.semantic.keyframes import token_points as _token_points
from src.semantic.relocalize import CameraFit
from src.semantic.relocalize import fit_to_keyframe as _fit_to_keyframe

from .features import FrameFeatures
from .lilocbench import LilocSequence


class PointCache:
    """``(tokens, 3)`` base_link points per frame, computed on first use."""

    def __init__(self, seq: LilocSequence, feats: FrameFeatures) -> None:
        self._seq, self._feats = seq, feats
        self._points: Dict[int, np.ndarray] = {}

    def __call__(self, frame: int) -> np.ndarray:
        if frame not in self._points:
            self._points[frame] = _token_points(self._seq.frame(frame), self._feats.uv)
        return self._points[frame]


def fit_to_keyframe(
    q_tokens: np.ndarray,
    q_points: np.ndarray,
    k_tokens: np.ndarray,
    k_points: np.ndarray,
    k_pose: conv.Pose2D,
) -> Optional[CameraFit]:
    """Match two frames' tokens, then fit the live pose from the matched points."""
    return _fit_to_keyframe(
        q_points, k_points, mutual_matches(q_tokens, k_tokens), k_pose
    )
