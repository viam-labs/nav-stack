"""Lidar odometry: ICP tracks a wheel-less cart and only matched scans paint."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.maps import MapStore
from src.slam_builtin import engine as engine_mod
from src.slam_builtin.engine import BuiltinSlamEngine
from src.slam_builtin.lidar_odom import LidarOdometry, icp_2d


def _room_points() -> np.ndarray:
    """8 x 5 m room with a few boxes so motion along the long walls is observable."""
    step = 0.02
    xs = np.arange(-2.0, 6.0, step)
    ys = np.arange(-2.5, 2.5, step)
    walls = [
        np.stack([xs, np.full_like(xs, -2.5)], axis=1),
        np.stack([xs, np.full_like(xs, 2.5)], axis=1),
        np.stack([np.full_like(ys, -2.0), ys], axis=1),
        np.stack([np.full_like(ys, 6.0), ys], axis=1),
    ]
    boxes = []
    for cx, cy in ((1.0, 1.8), (3.0, -1.7), (4.5, 1.5)):
        e = np.arange(-0.2, 0.2, step)
        boxes += [
            np.stack([cx + e, np.full_like(e, cy - 0.2)], axis=1),
            np.stack([cx + e, np.full_like(e, cy + 0.2)], axis=1),
            np.stack([np.full_like(e, cx - 0.2), cy + e], axis=1),
            np.stack([np.full_like(e, cx + 0.2), cy + e], axis=1),
        ]
    return np.vstack(walls + boxes)


_ROOM = _room_points()


def _scan_from(pose: conv.Pose2D) -> conv.LaserScan2D:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    dx = _ROOM[:, 0] - pose.x
    dy = _ROOM[:, 1] - pose.y
    body = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
    return conv.points_to_scan(body, num_bins=720, range_min=0.1, range_max=15.0)


def test_icp_recovers_offset_from_guess():
    true = conv.Pose2D(0.30, -0.10, math.radians(5.0))
    src = _scan_from(true).to_points()
    result = icp_2d(src, _ROOM[::3], conv.Pose2D(0.20, 0.0, math.radians(2.0)))
    assert result is not None
    assert result.pose.x == pytest.approx(true.x, abs=0.02)
    assert result.pose.y == pytest.approx(true.y, abs=0.02)
    assert result.pose.theta == pytest.approx(true.theta, abs=math.radians(0.5))
    assert result.inlier_ratio > 0.8


def test_lidar_odometry_rejects_scan_that_does_not_fit():
    lo = LidarOdometry()
    origin = conv.Pose2D(0.0, 0.0, 0.0)
    lo.add_keyscan(_scan_from(origin).to_points(), origin, force=True)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-6.0, 6.0, size=(400, 2))
    assert lo.match(noise, origin) is None


class _DrivingSensors:
    """A cart driving along +X with an IMU that reports no linear velocity."""

    def __init__(self):
        self.true = conv.Pose2D(0.0, 0.0, 0.0)
        self.clock = 1000.0

    def get_scan(self, max_age_s: float = 2.0, *, fresh: bool = False):
        del max_age_s, fresh
        return _scan_from(self.true)

    def get_odom(self):
        return conv.OdomReading(0.0, 0.0, 0.0)


def _livox_cfg(tmp_path, **extra):
    raw = {
        "base": "b",
        "lidar": {"name": "livox", "scan_source": "point_cloud"},
        "movement_sensor": "imu",
        "maps_dir": str(tmp_path),
        "mode": "mapping",
    }
    raw.update(extra)
    return SlamConfig.from_dict(raw)


def _drive(engine, sensors, monkeypatch, *, speed: float, seconds: float):
    monkeypatch.setattr(
        engine_mod,
        "time",
        SimpleNamespace(monotonic=lambda: sensors.clock, sleep=time.sleep),
    )
    dt = 0.1
    for _ in range(int(round(seconds / dt))):
        engine._tick()  # noqa: SLF001
        sensors.clock += dt
        t = sensors.true
        sensors.true = conv.Pose2D(t.x + speed * dt, t.y, t.theta)


def test_wheelless_cart_pose_follows_the_lidar(tmp_path, monkeypatch):
    cfg = _livox_cfg(tmp_path)
    assert cfg.lidar_odometry is True
    sensors = _DrivingSensors()
    engine = BuiltinSlamEngine(cfg, sensors, MapStore(str(tmp_path)))  # type: ignore[arg-type]
    _drive(engine, sensors, monkeypatch, speed=0.5, seconds=4.0)
    pose = engine.get_pose()
    diag = engine.diagnostics()["lidar_odometry"]
    # The last scan was taken one step before the true pose advanced.
    expected_x = sensors.true.x - 0.05
    assert pose.x == pytest.approx(expected_x, abs=0.06)
    assert pose.y == pytest.approx(0.0, abs=0.05)
    assert abs(pose.theta) < math.radians(1.5)
    assert diag["accepts"] > 30
    assert diag["forward_m_s"] == pytest.approx(0.5, abs=0.15)


def test_wheelless_cart_back_and_forth_returns_home(tmp_path, monkeypatch):
    cfg = _livox_cfg(tmp_path)
    sensors = _DrivingSensors()
    engine = BuiltinSlamEngine(cfg, sensors, MapStore(str(tmp_path)))  # type: ignore[arg-type]
    _drive(engine, sensors, monkeypatch, speed=0.5, seconds=3.0)
    _drive(engine, sensors, monkeypatch, speed=-0.5, seconds=3.0)
    engine._tick()  # noqa: SLF001
    pose = engine.get_pose()
    assert pose.x == pytest.approx(sensors.true.x, abs=0.06)
    assert pose.y == pytest.approx(0.0, abs=0.05)


def test_unmatched_scan_is_not_painted(tmp_path, monkeypatch):
    cfg = _livox_cfg(tmp_path)
    sensors = _DrivingSensors()
    engine = BuiltinSlamEngine(cfg, sensors, MapStore(str(tmp_path)))  # type: ignore[arg-type]
    _drive(engine, sensors, monkeypatch, speed=0.0, seconds=0.3)
    painted = engine._updates  # noqa: SLF001
    assert painted >= 1

    rng = np.random.default_rng(1)
    junk = conv.points_to_scan(rng.uniform(-6.0, 6.0, size=(600, 2)), num_bins=720)
    sensors.get_scan = lambda max_age_s=2.0, fresh=False: junk  # type: ignore[method-assign]
    engine._last_insert_at -= 5.0  # noqa: SLF001  heartbeat would paint
    engine._tick()  # noqa: SLF001
    assert engine._updates == painted  # noqa: SLF001
    assert engine.diagnostics()["lidar_odometry"]["inserts_held"] == 1


def test_wheel_odom_never_uses_lidar_odometry(tmp_path, monkeypatch):
    """A sample with wheel speed (Tracer) keeps the wheel path."""
    cfg = _livox_cfg(tmp_path)
    engine = BuiltinSlamEngine(cfg, _DrivingSensors(), MapStore(str(tmp_path)))  # type: ignore[arg-type]
    wheel = conv.OdomReading(0.4, 0.0, 0.0)
    posed = conv.OdomReading(0.0, 0.0, 0.0, pose=conv.Pose2D(0.0, 0.0, 0.0))
    assert engine._lidar_odom_active(wheel) is False  # noqa: SLF001
    assert engine._lidar_odom_active(posed) is False  # noqa: SLF001
    laser = SlamConfig.from_dict({"base": "b", "lidar": "front", "maps_dir": str(tmp_path)})
    assert laser.lidar_odometry is False
