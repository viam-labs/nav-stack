import asyncio
import math

import pytest

pytest.importorskip("viam")

from viam.proto.common import GeoPoint, Orientation, Vector3
from viam.components.movement_sensor import MovementSensor

from src.ros.odom_source import TypedMovementSensorOdom, TypedOdomConfig


class FakeMovementSensor:
    """Minimal MovementSensor stand-in: typed getters + a call counter."""

    def __init__(
        self,
        *,
        angular_velocity=False,
        linear_acceleration=False,
        orientation=False,
        position=False,
        compass_heading=False,
        linear_velocity=False,
        av=(0.0, 0.0, 0.0),
        la=(0.0, 0.0, 0.0),
        lv=(0.0, 0.0, 0.0),
        orient=(0.0, 0.0, 1.0, 0.0),  # o_x, o_y, o_z, theta(deg): identity
        geo=(0.0, 0.0),  # (latitude, longitude)
        compass=0.0,
    ):
        self._props = MovementSensor.Properties(
            linear_acceleration_supported=linear_acceleration,
            angular_velocity_supported=angular_velocity,
            orientation_supported=orientation,
            position_supported=position,
            compass_heading_supported=compass_heading,
            linear_velocity_supported=linear_velocity,
        )
        self._av, self._la, self._lv = av, la, lv
        self._orient, self._geo, self._compass = orient, geo, compass
        self.calls = {}

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    async def get_properties(self, **kw):
        self._count("get_properties")
        return self._props

    async def get_angular_velocity(self, **kw):
        self._count("get_angular_velocity")
        return Vector3(x=self._av[0], y=self._av[1], z=self._av[2])

    async def get_linear_acceleration(self, **kw):
        self._count("get_linear_acceleration")
        return Vector3(x=self._la[0], y=self._la[1], z=self._la[2])

    async def get_linear_velocity(self, **kw):
        self._count("get_linear_velocity")
        return Vector3(x=self._lv[0], y=self._lv[1], z=self._lv[2])

    async def get_orientation(self, **kw):
        self._count("get_orientation")
        return Orientation(
            o_x=self._orient[0], o_y=self._orient[1], o_z=self._orient[2], theta=self._orient[3]
        )

    async def get_position(self, **kw):
        self._count("get_position")
        return GeoPoint(latitude=self._geo[0], longitude=self._geo[1]), 0.0

    async def get_compass_heading(self, **kw):
        self._count("get_compass_heading")
        return self._compass


def _read(sensor, cfg=None):
    reader = TypedMovementSensorOdom(sensor, cfg)
    return asyncio.run(reader.read()), reader


def test_imu_path_yaw_and_accel_ignores_position():
    # mid360-like IMU: gyro + accel + orientation + position, no linear velocity.
    s = FakeMovementSensor(
        angular_velocity=True,
        linear_acceleration=True,
        orientation=True,
        position=True,
        av=(0.0, 0.0, 30.0),  # 30 deg/s yaw
        la=(0.7, -0.2, 9.81),  # forward accel + gravity on z (level)
    )
    reading, _ = _read(s)
    assert reading.vtheta == pytest.approx(math.radians(30.0))
    assert reading.vx == 0.0 and reading.vy == 0.0
    # Level orientation -> gravity has no horizontal component; ax/ay pass through.
    assert reading.ax == pytest.approx(0.7)
    assert reading.ay == pytest.approx(-0.2)
    assert reading.pose is None  # Position ignored by default (tar pit)
    assert reading.heading_rad is None
    assert "get_position" not in s.calls  # never even queried


def test_wheel_path_uses_linear_velocity_and_skips_accel():
    s = FakeMovementSensor(
        angular_velocity=True,
        linear_velocity=True,
        linear_acceleration=True,
        orientation=True,
        av=(0.0, 0.0, 10.0),
        lv=(0.5, 0.0, 0.0),
        la=(1.0, 1.0, 9.81),
    )
    reading, _ = _read(s)
    assert reading.vx == pytest.approx(0.5)
    assert reading.vy == pytest.approx(0.0)
    assert reading.vtheta == pytest.approx(math.radians(10.0))
    # Wheel twist present -> never double-integrate accel.
    assert reading.ax is None and reading.ay is None
    assert "get_linear_acceleration" not in s.calls


def test_snap_heading_from_orientation():
    # Orientation = pure yaw 90 deg (identity axis +z, theta 90).
    s = FakeMovementSensor(
        angular_velocity=True,
        orientation=True,
        orient=(0.0, 0.0, 1.0, 90.0),
    )
    reading, _ = _read(s, TypedOdomConfig(snap_heading=True))
    assert reading.heading_rad == pytest.approx(math.pi / 2)


def test_trust_pose_reads_geo_overload():
    s = FakeMovementSensor(
        angular_velocity=True,
        orientation=True,
        position=True,
        geo=(2.0, 5.0),  # latitude=y=2, longitude=x=5
    )
    reading, _ = _read(s, TypedOdomConfig(trust_pose=True))
    assert reading.pose is not None
    assert reading.pose.x == pytest.approx(5.0)  # longitude -> x
    assert reading.pose.y == pytest.approx(2.0)  # latitude -> y


def test_properties_fetched_once():
    s = FakeMovementSensor(angular_velocity=True)
    reader = TypedMovementSensorOdom(s)
    asyncio.run(reader.read())
    asyncio.run(reader.read())
    assert s.calls["get_properties"] == 1


def test_empty_sensor_returns_zero_reading():
    s = FakeMovementSensor()  # advertises nothing
    reading, _ = _read(s)
    assert (reading.vx, reading.vy, reading.vtheta) == (0.0, 0.0, 0.0)
    assert reading.ax is None and reading.pose is None
