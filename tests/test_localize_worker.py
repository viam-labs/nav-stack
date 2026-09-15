import asyncio
import math

import numpy as np
import pytest

from src.geom import conversions as conv
from src.nav import global_localize as gl
from src.nav.localize_worker import LocalizeWorker, map_token


def _room_map(size_m: float = 6.0, res: float = 0.05) -> gl.OccupancyMap:
    n = int(size_m / res)
    grid = np.zeros((n, n), dtype=np.int16)
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    # Asymmetric feature so yaw is unambiguous.
    grid[n // 2 : n // 2 + 8, n // 4 : n // 4 + 4] = 100
    return gl.OccupancyMap(grid=grid, resolution=res, origin_x=0.0, origin_y=0.0)


def _scan_from(occ_map: gl.OccupancyMap, pose: conv.Pose2D, beams: int = 180) -> conv.LaserScan2D:
    angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
    ranges = np.empty(beams)
    for i, a in enumerate(angles):
        ranges[i] = gl._raycast_distance(  # noqa: SLF001
            occ_map, pose.x, pose.y, pose.theta + a, max_range_m=12.0, step_m=0.02
        )
    return conv.LaserScan2D(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        range_min=0.05,
        range_max=12.0,
    )


@pytest.fixture(scope="module")
def worker():
    w = LocalizeWorker(enabled=True, call_timeout_s=60.0)
    w.warm()
    yield w
    w.close()


def test_subprocess_matches_in_process(worker):
    occ = _room_map()
    truth = conv.Pose2D(2.0, 3.5, 0.4)
    scan = _scan_from(occ, truth)
    hint = conv.Pose2D(2.3, 3.2, 0.2)

    local = gl.global_localize_scan(occ, scan, hint=hint, search_radius_m=1.5)
    remote = worker.call("global_localize_scan", occ, scan, hint=hint, search_radius_m=1.5)

    assert remote.pose.x == pytest.approx(local.pose.x, abs=1e-9)
    assert remote.pose.y == pytest.approx(local.pose.y, abs=1e-9)
    assert remote.pose.theta == pytest.approx(local.pose.theta, abs=1e-9)
    assert remote.score == pytest.approx(local.score)
    assert worker.stats["subprocess_calls"] >= 1
    assert worker.stats["fallback_calls"] == 0

    # Map shipped once, then cached by token.
    assert map_token(occ) in worker._sent_tokens  # noqa: SLF001
    score, mae = worker.call("score_pose_with_rays", occ, scan, truth)
    assert math.isfinite(score) and math.isfinite(mae)
    choice = worker.call("choose_yaw_or_flip", occ, scan, remote.pose, reference_theta=0.4)
    assert choice.pose.theta == pytest.approx(remote.pose.theta, abs=1e-9)


def test_run_awaitable(worker):
    occ = _room_map()
    truth = conv.Pose2D(1.0, 1.0, 0.0)
    scan = _scan_from(occ, truth)

    async def _go():
        return await worker.run("score_pose_with_rays", occ, scan, truth)

    score, _mae = asyncio.run(_go())
    assert score > 0.5


def test_worker_recovers_after_process_killed(worker):
    occ = _room_map()
    truth = conv.Pose2D(4.0, 2.0, 1.0)
    scan = _scan_from(occ, truth)
    worker.call("score_pose_with_rays", occ, scan, truth)

    pool = worker._pool  # noqa: SLF001
    assert pool is not None
    for proc in list(pool._processes.values()):  # noqa: SLF001
        proc.kill()
    restarts_before = worker.stats["pool_restarts"]

    # Broken pool → restart → map re-sent → same answer, or in-process fallback.
    score, _ = worker.call("score_pose_with_rays", occ, scan, truth)
    assert score > 0.5
    assert worker.stats["pool_restarts"] >= restarts_before + 1


def test_disabled_worker_runs_in_process():
    w = LocalizeWorker(enabled=False)
    occ = _room_map()
    truth = conv.Pose2D(1.0, 2.0, 0.0)
    scan = _scan_from(occ, truth)
    score, _ = w.call("score_pose_with_rays", occ, scan, truth)
    assert score > 0.5
    assert w.stats["subprocess_calls"] == 0
    assert w.stats["fallback_calls"] == 1
    assert w.enabled is False
