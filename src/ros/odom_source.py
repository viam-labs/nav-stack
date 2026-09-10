"""Typed MovementSensor -> ``OdomReading`` reader.

The built-in SLAM path historically parsed a movement sensor's ``get_readings()``
dict, whose key shapes are implementation-specific. That is brittle across
arbitrary movement sensors (e.g. Agilex Tracer odometry advertises typed
velocity but may omit or sparsely serialize the same fields in readings).

This reader uses the portable contract: call ``get_properties()`` once to
discover which typed getters the sensor implements, then call only those. It
produces the same sensor-frame :class:`~..ros.conversions.OdomReading` the
readings parser does, so the downstream mount-yaw / upside-down / heading
corrections (see ``slam.py``) compose unchanged.

Capability -> field mapping:

* ``angular_velocity``  -> ``vtheta``       (deg/s -> rad/s)
* ``linear_velocity``   -> ``vx, vy``       (ROS body: x forward, y left)
* ``linear_acceleration`` + ``orientation`` -> ``ax, ay`` (gravity removed; IMU path)
* ``orientation`` / ``compass_heading`` -> ``heading_rad``  (only if ``snap_heading``)
* ``position`` + ``orientation`` -> ``pose`` (only if ``trust_pose``)

``Position`` is ignored by default: many IMUs advertise it while double-
integrating acceleration (drifts quadratically), which is unusable as odometry.

When ``velocity_convention`` is ``viam`` / ``mir`` (Y-forward wheeled bases),
``GetLinearVelocity`` is remapped into ROS body frame: forward on ``y`` becomes
``vx``, lateral on ``x`` becomes ``vy`` (same swap as wheeled-odometry readings).
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from viam.components.movement_sensor import MovementSensor

from ..config import BASE_VELOCITY_VIAM, BASE_VELOCITY_Y_FORWARD
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
    # How GetLinearVelocity is framed. ``viam`` / ``mir``: +y is forward (Tracer,
    # rdk:builtin:wheeled). ``ros``: +x is forward.
    velocity_convention: str = BASE_VELOCITY_VIAM


@dataclass
class TypedOdomDebug:
    """Last raw typed-getter snapshot for ``sensor_probe`` / status."""

    source: str = "typed"
    linear_velocity_supported: bool = False
    angular_velocity_supported: bool = False
    position_supported: bool = False
    raw_lv_x: Optional[float] = None
    raw_lv_y: Optional[float] = None
    raw_lv_z: Optional[float] = None
    raw_av_z_deg_s: Optional[float] = None
    velocity_convention: str = BASE_VELOCITY_VIAM
    remapped: bool = False


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
        self.last_debug: TypedOdomDebug = TypedOdomDebug(
            velocity_convention=self._cfg.velocity_convention
        )

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
                    f"position={self._props.position_supported} "
                    f"velocity_convention={self._cfg.velocity_convention}"
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
        remapped = False
        raw_lv = (None, None, None)
        raw_av_z = None

        if "av" in results:
            # Viam AngularVelocity is degrees/s (CCW +).
            # Unset protobuf fields decode as 0.0 — treat as valid (parked).
            raw_av_z = float(getattr(results["av"], "z", 0.0) or 0.0)
            vtheta = math.radians(raw_av_z)

        if "lv" in results:
            lv = results["lv"]
            # Missing Vector3 components default to 0.0 (proto omit-empty).
            lx = float(getattr(lv, "x", 0.0) or 0.0)
            ly = float(getattr(lv, "y", 0.0) or 0.0)
            lz = float(getattr(lv, "z", 0.0) or 0.0)
            raw_lv = (lx, ly, lz)
            if cfg.velocity_convention in BASE_VELOCITY_Y_FORWARD:
                # Viam Y-forward / X-lateral -> ROS x-forward / y-left.
                # Match wheeled-odometry readings remap: vx=y, vy=-x.
                vx, vy = ly, -lx
                remapped = True
            else:
                vx, vy = lx, ly

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

        self.last_debug = TypedOdomDebug(
            source="typed",
            linear_velocity_supported=bool(p.linear_velocity_supported),
            angular_velocity_supported=bool(p.angular_velocity_supported),
            position_supported=bool(p.position_supported),
            raw_lv_x=raw_lv[0],
            raw_lv_y=raw_lv[1],
            raw_lv_z=raw_lv[2],
            raw_av_z_deg_s=raw_av_z,
            velocity_convention=cfg.velocity_convention,
            remapped=remapped,
        )

        return conv.OdomReading(
            vx, vy, vtheta, pose=pose, heading_rad=heading_rad, ax=ax, ay=ay
        )

    def debug_dict(self) -> Dict[str, Any]:
        d = self.last_debug
        return {
            "source": d.source,
            "linear_velocity_supported": d.linear_velocity_supported,
            "angular_velocity_supported": d.angular_velocity_supported,
            "position_supported": d.position_supported,
            "raw_lv_x": d.raw_lv_x,
            "raw_lv_y": d.raw_lv_y,
            "raw_lv_z": d.raw_lv_z,
            "raw_av_z_deg_s": d.raw_av_z_deg_s,
            "velocity_convention": d.velocity_convention,
            "remapped": d.remapped,
        }
