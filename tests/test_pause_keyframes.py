"""Tests for pause keyframe store / match (src/nav/pause_keyframes.py)."""
import math

import numpy as np
import pytest

from src.nav import pause_keyframes
from src.ros import conversions as conv


def _wall_scan(x0=2.0, y0=-3.0, x1=2.0, y1=3.0, n=80) -> conv.LaserScan2D:
    t = np.linspace(0.0, 1.0, n)
    pts = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)
    return conv.points_to_scan(pts, num_bins=360, range_min=0.1, range_max=25.0)


def test_nn_hit_rate_identity_and_shift():
    a = np.column_stack([np.linspace(0, 2, 40), np.zeros(40)])
    assert pause_keyframes.nn_hit_rate(a, a, tol_m=0.2) == pytest.approx(1.0)
    shifted = a + np.array([2.0, 0.0])
    assert pause_keyframes.nn_hit_rate(a, shifted, tol_m=0.2) < 0.2


def test_store_dedupes_near_duplicates():
    store = pause_keyframes.PauseKeyframeStore(
        min_spacing_m=0.5, min_spacing_deg=20.0
    )
    scan = _wall_scan()
    pose = conv.Pose2D(1.0, 2.0, 0.1)
    assert store.add(pose, scan, []) is True
    assert store.add(conv.Pose2D(1.1, 2.05, 0.12), scan, []) is False
    assert store.add(conv.Pose2D(3.0, 2.0, 0.1), scan, []) is True
    assert len(store) == 2


def test_match_finds_keyframe_from_different_yaw():
    store = pause_keyframes.PauseKeyframeStore(
        match_tol_m=0.35, yaw_step_deg=10.0
    )
    true = conv.Pose2D(0.0, 0.0, math.pi / 2)
    scan = _wall_scan()
    # Head-height "shelf" unique to this stop.
    bands = [
        np.empty((0, 2)),
        np.column_stack([np.linspace(-1, 1, 40), np.full(40, 1.5)]),
    ]
    assert store.add(true, scan, bands) is True

    # Query from the same place but facing the opposite way — matcher should
    # recover a pose near the keyframe with usable score.
    wrong_yaw = conv.Pose2D(0.0, 0.0, -math.pi / 2)
    # Synthesize a query scan as if observed at the true pose (the robot is
    # physically there; odom yaw is wrong). Matching searches yaw at the kf.
    query = _wall_scan()
    hit = store.match(query, bands, hint=wrong_yaw, search_radius_m=2.0)
    assert hit is not None
    assert hit.score >= 0.5
    assert abs(hit.pose.x - true.x) <= 0.8
    assert abs(hit.pose.y - true.y) <= 0.8
