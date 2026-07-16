"""Typed MovementSensor -> ``OdomReading`` reader.

The built-in SLAM path parses a movement sensor's ``get_readings()`` dict, whose
key shapes are implementation-specific (``linear_velocity_mps``, nested
``orientation`` blocks, ``position_x_m`` aliases, ...). That is brittle across
arbitrary movement sensors.

This reader instead uses the portable contract: call ``get_properties()`` once to
discover which typed getters the sensor implements, then call only those. It
produces the same sensor-frame :class:`~..ros.conversions.OdomReading` the
readings parser does, so the downstream mount-yaw / upside-down / heading
corrections (see ``slam.py``) compose unchanged.

Capability -> field mapping:

* ``angular_velocity``  -> ``vtheta``       (deg/s -> rad/s)
* ``linear_velocity``   -> ``vx, vy``       (body forward/left; wheel-twist path)
* ``linear_acceleration`` + ``orientation`` -> ``ax, ay`` (gravity removed; IMU path)
* ``orientation`` / ``compass_heading`` -> ``heading_rad``  (only if ``snap_heading``)
* ``position`` + ``orientation`` -> ``pose`` (only if ``trust_pose``)

``Position`` is ignored by default: many IMUs advertise it while double-
integrating acceleration (drifts quadratically), which is unusable as odometry.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Optional

from viam.components.movement_sensor import MovementSensor

from . import conversions as conv


@dataclass(frozen=True)
class TypedOdomConfig:
    """Optional behaviors for the typed reader.

    Defaults suit an IMU (yaw + accel) whose translation is supplied elsewhere
    (e.g. lidar odometry): trust the gyro + gravity-removed accel, ignore the
    drift-prone ``Position``.
    """

    # Use LinearVelocity as body twist when the sensor advertises it (wheel /
    # fused odometry). When true and present, the accel/lidar-odom path is
    # bypassed (see BridgeNode ``_has_wheel_twist``).
    use_linear_velocity: bool = True
    # Emit gravity-compensated body accel hints from LinearAcceleration.
    use_linear_acceleration: bool = True
    # Snap yaw from Orientation (or CompassHeading) instead of integrating gyro.
    snap_heading: bool = False
    # Treat Position (+ Orientation) as an absolute odom pose. Off by default:
    # a dead-reckoned IMU Position drifts; only enable for sensors whose
    # Position is a trustworthy fused/wheel estimate. Uses the Viam lat/lng ->
    # (y, x) map-frame overload when read.
    trust_pose: bool = False


class TypedMovementSensorOdom:
    """Build ``OdomReading`` samples from a MovementSensor via its typed API."""

    def __init__(
        self,
        sensor: MovementSensor,
        cfg: Optional[TypedOdomConfig] = None,
        logger=None,
    ):
        self._sensor = sensor
        self._cfg = cfg or TypedOdomConfig()
        self._logger = logger
        self._props: Optional[MovementSensor.Properties] = None

    async def properties(self) -> MovementSensor.Properties:
        """Cache and return the sensor's capabilities (fetched once)."""
        if self._props is None:
            self._props = await self._sensor.get_properties()
            if self._logger is not None:
                self._logger.info(
                    "typed odom reader: sensor properties "
                    f"angular_velocity={self._props.angular_velocity_supported} "
                    f"linear_velocity={self._props.linear_velocity_supported} "
                    f"linear_acceleration={self._props.linear_acceleration_supported} "
                    f"orientation={self._props.orientation_supported} "
                    f"position={self._props.position_supported}"
                )
        return self._props

    async def read(self) -> conv.OdomReading:
        p = await self.properties()
        cfg = self._cfg

        use_lv = cfg.use_linear_velocity and p.linear_velocity_supported
        use_accel = (
            cfg.use_linear_acceleration
            and p.linear_acceleration_supported
            and not use_lv  # never double-integrate accel on top of wheel twist
        )
        # Orientation is needed to remove gravity from accel, to snap heading, or
        # to orient a trusted pose.
        need_orientation = p.orientation_supported and (
            use_accel
            or cfg.snap_heading
            or (cfg.trust_pose and p.position_supported)
        )

        # Fire every needed typed getter concurrently (one round-trip).
        coros = {}
        if p.angular_velocity_supported:
            coros["av"] = self._sensor.get_angular_velocity()
        if use_lv:
            coros["lv"] = self._sensor.get_linear_velocity()
        if use_accel:
            coros["la"] = self._sensor.get_linear_acceleration()
        if need_orientation:
            coros["orient"] = self._sensor.get_orientation()
        if cfg.snap_heading and not need_orientation and p.compass_heading_supported:
            coros["compass"] = self._sensor.get_compass_heading()
        if cfg.trust_pose and p.position_supported:
            coros["pos"] = self._sensor.get_position()

        results = dict(zip(coros.keys(), await asyncio.gather(*coros.values())))

        vx = vy = vtheta = 0.0
        pose = None
        heading_rad = None
        ax = ay = None

        if "av" in results:
            # Viam AngularVelocity is degrees/s (CCW +).
            vtheta = math.radians(float(results["av"].z))

        if "lv" in results:
            lv = results["lv"]  # body frame: x forward, y left (m/s)
            vx, vy = float(lv.x), float(lv.y)

        rpy = None
        if "orient" in results:
            o = results["orient"]
            rpy = conv.euler_from_orientation_vector(o.o_x, o.o_y, o.o_z, o.theta)

        if "la" in results and rpy is not None:
            la = results["la"]
            ax, ay = conv.gravity_compensated_body_accel(
                (float(la.x), float(la.y), float(la.z)), *rpy
            )

        if cfg.snap_heading:
            if rpy is not None:
                heading_rad = rpy[2]
            elif "compass" in results:
                heading_rad = math.radians(float(results["compass"]))

        if "pos" in results and rpy is not None:
            geo, _alt = results["pos"]
            # Viam lat/lng overloaded as map-frame (y, x) — matches the SLAM/nav
            # geo_point convention for non-georeferenced maps.
            pose = conv.Pose2D(float(geo.longitude), float(geo.latitude), rpy[2])

        return conv.OdomReading(
            vx, vy, vtheta, pose=pose, heading_rad=heading_rad, ax=ax, ay=ay
        )
