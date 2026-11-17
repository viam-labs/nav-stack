"""When ``global_localize`` applies a match, and which lidar scan it searches with."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.localize.global_localize import GlobalLocalizeResult


def _scan(tag: float) -> conv.LaserScan2D:
    return conv.LaserScan2D(np.full(360, tag), -math.pi, 2 * math.pi / 360, 0.05, 25.0)


def _found(x: float, score: float, *, ray: float = 0.3, ambiguous: bool = False):
    return GlobalLocalizeResult(
        pose=conv.Pose2D(x, 0.0, 0.0),
        score=score,
        candidates_evaluated=100,
        scan_points_used=200,
        in_map_points=200,
        hit_rate=0.5,
        ray_score=0.5,
        ray_mae_m=ray,
        ambiguous=ambiguous,
    )


class _Localizer:
    """Answers each search by which scan it was given."""

    def __init__(self, found: dict, prior: dict) -> None:
        self.found, self.prior = found, prior
        self.searched: list = []

    async def run(self, name, _occ_map, scan, *args, **kwargs):
        if name == "global_localize_scan":
            self.searched.append(id(scan))
            return self.found[id(scan)]
        if name == "score_pose_with_rays":
            return self.prior[id(scan)]
        raise AssertionError(name)


def _service(found: dict, prior: dict, merged, heights: dict):
    from src.models.slam import SlamService

    slam = SlamService("slam")
    slam._cfg = SlamConfig.from_dict(  # noqa: SLF001
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    mgr = MagicMock()
    mgr.get_pose_in_map.return_value = conv.Pose2D(0.0, 0.0, 0.0)
    slam._manager = mgr  # noqa: SLF001
    slam._read_merged_scan = AsyncMock(return_value=merged)  # noqa: SLF001
    slam._load_active_occupancy_map = lambda _source: (object(), "live")  # noqa: SLF001
    localizer = _Localizer(found, prior)
    slam._localizer = localizer  # noqa: SLF001
    slam._engine = SimpleNamespace(height_scans=lambda: dict(heights))  # noqa: SLF001
    slam._mark_pose_seeded = lambda *_a, **_k: None  # noqa: SLF001
    return slam, mgr, localizer


# The "lost and still" auto-refine's command (localize_supervisor).
LOST_AND_STILL = {
    "full_map": False,
    "auto_full_map_fallback": False,
    "apply": True,
    "map_source": "live",
    "search_radius_m": 1.0,
    "local_yaw_window_deg": 360.0,
    "min_apply_score": 0.22,
    "max_apply_ray_mae_m": 1.5,
    "refuse_ambiguous": False,
}


def _run(slam, command):
    return asyncio.run(slam._global_localize(command))  # noqa: SLF001


def test_a_match_that_beats_a_bad_prior_is_not_applied_when_it_is_bad_too():
    # What tracer2a applied four times while lost: -0.11 against a prior of
    # -0.25, with 1.7 m of ray error.
    merged = _scan(1.0)
    found = {id(merged): _found(0.4, -0.11, ray=1.7)}
    slam, mgr, _ = _service(found, {id(merged): (-0.25, 2.0)}, merged, {})
    out = _run(slam, LOST_AND_STILL)
    assert out["applied"] is False and out["apply_blocked_reason"] == "low_score"
    mgr.relocalize.assert_not_called()


def test_a_match_that_clearly_beats_its_prior_still_applies_just_under_the_limit():
    merged = _scan(1.0)
    found = {id(merged): _found(0.4, 0.30, ray=0.8)}
    slam, mgr, _ = _service(found, {id(merged): (0.20, 1.0)}, merged, {})
    out = _run(slam, {**LOST_AND_STILL, "min_apply_score": 0.35})
    assert out["applied"] is True and "apply_blocked_reason" not in out
    mgr.relocalize.assert_called_once()


def test_beating_the_prior_does_not_excuse_a_large_ray_error():
    merged = _scan(1.0)
    found = {id(merged): _found(0.4, 0.30, ray=1.3)}
    slam, mgr, _ = _service(found, {id(merged): (0.10, 2.0)}, merged, {})
    out = _run(
        slam, {**LOST_AND_STILL, "min_apply_score": 0.35, "max_apply_ray_mae_m": 1.0}
    )
    assert out["applied"] is False and out["apply_blocked_reason"] == "high_ray_mae"
    mgr.relocalize.assert_not_called()


def test_a_lidar_height_finds_the_pose_the_merged_scan_cannot():
    # tracer2a, standing 1.8 m from its published pose: the merged scan and
    # the 0.21 m lidar find ambiguous places, the 1.2 m lidar a clear one.
    merged, low, top = _scan(1.0), _scan(2.0), _scan(3.0)
    found = {
        id(merged): _found(-1.9, 0.00, ray=2.5, ambiguous=True),
        id(low): _found(-2.8, 0.06, ray=2.3, ambiguous=True),
        id(top): _found(1.8, 0.48, ray=0.9),
    }
    prior = {id(merged): (0.0, 2.0), id(low): (-0.3, 2.0), id(top): (0.02, 1.5)}
    slam, mgr, localizer = _service(found, prior, merged, {0: low, 4: top})
    out = _run(slam, {"search_radius_m": 3.0, "local_yaw_window_deg": 180.0})
    assert out["applied"] is True and out["pose"]["x"] == 1.8
    assert out["prior_score"] == 0.02  # judged with the scan that matched
    search = out["height_search"]
    assert search["merged_blocked"] == "low_score"
    assert search["used"] == {"z_min_m": 1.0, "z_max_m": 1.25}
    assert [h["blocked"] for h in search["heights"]] == ["low_score", None]
    applied = mgr.relocalize.call_args.args[0]
    assert applied.x == 1.8
    assert localizer.searched == [id(merged), id(low), id(top)]


def test_the_merged_scan_is_used_alone_when_it_finds_an_acceptable_pose():
    merged, low, top = _scan(1.0), _scan(2.0), _scan(3.0)
    found = {id(merged): _found(0.1, 0.60, ray=0.4)}
    slam, _mgr, localizer = _service(
        found, {id(merged): (0.5, 0.5)}, merged, {0: low, 4: top}
    )
    out = _run(slam, {"search_radius_m": 3.0})
    assert out["applied"] is True and "height_search" not in out
    assert localizer.searched == [id(merged)]


def test_when_no_height_does_better_the_merged_answer_stands_and_is_refused():
    merged, low, top = _scan(1.0), _scan(2.0), _scan(3.0)
    found = {
        id(merged): _found(-1.0, 0.10, ray=2.0, ambiguous=True),
        id(low): _found(-2.0, 0.12, ray=2.1),
        id(top): _found(2.0, 0.20, ray=1.9),
    }
    prior = {id(merged): (0.05, 2.0)}
    slam, mgr, _ = _service(found, prior, merged, {0: low, 4: top})
    out = _run(slam, {"search_radius_m": 3.0})
    assert out["applied"] is False and out["pose"]["x"] == -1.0
    assert "used" not in out["height_search"]
    mgr.relocalize.assert_not_called()


def test_one_lidar_height_searches_exactly_as_before():
    merged = _scan(1.0)
    found = {id(merged): _found(-1.0, 0.10, ray=2.0, ambiguous=True)}
    slam, _mgr, localizer = _service(found, {id(merged): (0.05, 2.0)}, merged, {})
    out = _run(slam, {"search_radius_m": 3.0})
    assert "height_search" not in out and localizer.searched == [id(merged)]
