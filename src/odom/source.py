"""Typed MovementSensor -> ``OdomReading`` reader.

The built-in SLAM path historically parsed a movement sensor's ``get_readings()``
dict, whose key shapes are implementation-specific. That is brittle across
arbitrary movement sensors (e.g. Agilex Tracer odometry advertises typed
velocity but may omit or sparsely serialize the same fields in readings).

This reader uses the portable contract: call ``get_properties()`` once to
discover which typed getters the sensor implements, then call only those. It
produces the same sensor-frame :class:`~..geom.conversions.OdomReading` the
readings parser does, so the downstream mount-yaw / upside-down / heading
corrections (see ``slam.py``) compose unchanged.

Capability -> field mapping:

* ``angular_velocity``  -> ``vtheta``       (deg/s -> rad/s)
* ``linear_velocity``   -> ``vx, vy``       (sensor-native body frame)
* ``linear_acceleration`` + ``orientation`` -> ``ax, ay`` (gravity removed; IMU path)
* ``orientation`` / ``compass_heading`` -> ``heading_rad``  (only if ``snap_heading``)
* ``position`` + ``orientation`` -> ``pose`` (only if ``trust_pose``)

``Position`` is ignored by default: many IMUs advertise it while double-
integrating acceleration (drifts quadratically), which is unusable as odometry.

``velocity_convention`` selects the sensor body frame for twist fields:

* ``viam`` / ``mir``: keep Y-forward (``vy`` = forward, ``vx`` = right). Do **not**
  apply the body X/Y swap.
* ``x_forward`` / ``ros``: keep X-forward (``vx`` = forward, ``vy`` = left).

When linear velocity is available but ``trust_pose`` is off, this reader
dead-reckons an odom ``pose`` from twist (so ``has_pose`` is true for
velocity-only wheeled sensors). Integration uses a world frame where
theta=0 faces +X, with convention-aware body→world kinematics.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from viam.components.movement_sensor import MovementSensor

from ..config import BASE_VELOCITY_VIAM, BASE_VELOCITY_Y_FORWARD
from ..geom import conversions as conv


@dataclass(frozen=True)
class TypedOdomConfig:
    """Optional behaviors for the typed reader.

    Defaults suit an IMU (yaw + accel) whose translation is supplied elsewhere
    (e.g. lidar odometry): trust the gyro + gravity-removed accel, ignore the
    drift-prone ``Position``.
    """

    # Use LinearVelocity as body twist when the sensor advertises it (wheel /
    # fused odometry). When true and present, the accel/lidar-odom path is
    # Wheel-twist deadband for near-stationary bases.
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
        *,
        clock=time.monotonic,
    ):
        self._sensor = sensor
        self._cfg = cfg or TypedOdomConfig()
        self._logger = logger
        self._clock = clock
        self._props: Optional[MovementSensor.Properties] = None
        self.last_debug: TypedOdomDebug = TypedOdomDebug(
            velocity_convention=self._cfg.velocity_convention
        )
        # Dead-reckon pose from twist when the sensor has no trusted Position.
        self._integ_x = 0.0
        self._integ_y = 0.0
        self._integ_th = 0.0
        self._integ_t: Optional[float] = None
        # Last twist actually integrated. A read gap (module loop busy with
        # global_localize, slow base RPC) is bridged by holding this velocity,
        # not by clamping dt — the clamp silently dropped travel and left the
        # map pose behind the robot until a manual refine snapped it forward.
        self._integ_last_twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_gap_s: float = 0.0
        self.gap_hold_events: int = 0

    # Beyond this gap the held velocity is no longer trusted (robot may have
    # been stopped by a command that landed after our last sample).
    MAX_HOLD_GAP_S = 3.0
    # Normal sample spacing: past this, the previous twist is held for the
    # excess instead of attributing the whole gap to the new sample.
    NOMINAL_GAP_S = 0.25

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

    def _integrate_twist_pose(
        self, vx: float, vy: float, vtheta: float, now: float
    ) -> conv.Pose2D:
        """Integrate sensor-native twist into a world-frame odom pose."""
        if self._integ_t is None:
            self._integ_t = now
            self._integ_last_twist = (vx, vy, vtheta)
            return conv.Pose2D(self._integ_x, self._integ_y, self._integ_th)

        gap = max(0.0, now - self._integ_t)
        self._integ_t = now
        self.last_gap_s = gap

        # Split the gap: the excess beyond a normal sample spacing is bridged
        # with the *previous* twist (zero-order hold — the base kept executing
        # its last SetVelocity while we could not read it); the tail uses the
        # new sample.
        hold_dt = 0.0
        if gap > self.NOMINAL_GAP_S:
            hold_dt = min(gap, self.MAX_HOLD_GAP_S) - self.NOMINAL_GAP_S
            self.gap_hold_events += 1
        new_dt = min(gap, self.NOMINAL_GAP_S)

        if hold_dt > 0.0:
            self._step(*self._integ_last_twist, hold_dt)
        if new_dt > 0.0:
            self._step(vx, vy, vtheta, new_dt)
        self._integ_last_twist = (vx, vy, vtheta)
        return conv.Pose2D(self._integ_x, self._integ_y, self._integ_th)

    def _step(self, vx: float, vy: float, vtheta: float, dt: float) -> None:
        c = math.cos(self._integ_th)
        s = math.sin(self._integ_th)
        if self._cfg.velocity_convention in BASE_VELOCITY_Y_FORWARD:
            # Body: +y forward, +x right → world (theta=0 faces +X).
            self._integ_x += (c * vy + s * vx) * dt
            self._integ_y += (s * vy - c * vx) * dt
        else:
            # Body: +x forward, +y left.
            self._integ_x += (c * vx - s * vy) * dt
            self._integ_y += (s * vx + c * vy) * dt
        self._integ_th = conv.normalize_angle(self._integ_th + vtheta * dt)

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
            # Keep sensor-native axes. Body-frame conversion belongs at the
            # drive/publish boundary, not here (builtin probe expects viam
            # forward on vy).
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
        elif use_lv:
            # Velocity-only wheeled sensors: build odom pose from twist so
            # sensor_probe.has_pose / absolute-odom predict both work.
            pose = self._integrate_twist_pose(vx, vy, vtheta, self._clock())

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

    def sync_heading(self, heading_rad: float) -> None:
        """Align the twist integrator yaw with an external absolute heading.

        Called after a dedicated ``heading_sensor`` supplies orientation so
        subsequent dead-reckoned XY steps use the magnetometer/AHRS yaw instead
        of a drifting gyro-only integrator angle.
        """
        self._integ_th = float(heading_rad)

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
            "last_sample_gap_s": round(self.last_gap_s, 3),
            "gap_hold_events": self.gap_hold_events,
        }


async def read_typed_heading(sensor) -> tuple[Optional[float], str, Dict[str, Any]]:
    """Read yaw (rad) from a MovementSensor via typed getters.

    Matches ``viam-modules/wit-motion``: the Wit 0x53 angle packet is Z-Y-X
    euler in degrees, stored as radians (``EulerAngles.Yaw``). Prefer that
    native euler when ``get_readings().orientation.yaw`` is present. Otherwise
    use ``GetOrientation``; for a near-vertical axis, yaw is the OrientationVector
    ``theta`` (rotation about +Z), not an axis-component atan2.

    Returns ``(yaw_rad, source, debug)``.
    """
    debug: Dict[str, Any] = {}

    # Always capture GetOrientation raw OV for sensor_probe diagnosis.
    orient = None
    try:
        props = await sensor.get_properties()
    except Exception:  # noqa: BLE001
        props = None
    if props is not None and getattr(props, "orientation_supported", False):
        try:
            orient = await sensor.get_orientation()
            _yaw_ov, meta = conv.yaw_rad_from_viam_orientation(orient)
            debug["orientation"] = {
                "o_x": meta["o_x"],
                "o_y": meta["o_y"],
                "o_z": meta["o_z"],
                "theta": meta["theta_raw"],
                "theta_unit": meta["theta_unit"],
                "ov_theta_deg": round(meta["ov_theta_deg"], 3),
                "yaw_from_ov_deg": round(meta["yaw_deg"], 3),
            }
            debug["yaw_from_ov_rad"] = _yaw_ov
        except Exception as exc:  # noqa: BLE001
            debug["orientation_error"] = repr(exc)

    # Native AHRS euler (Wit 0x53 / wit-motion EulerAngles.Yaw).
    try:
        readings = await sensor.get_readings()
        orient_block = (
            readings.get("orientation") if isinstance(readings, Mapping) else None
        )
        if isinstance(orient_block, Mapping) and "yaw" in orient_block:
            yaw = float(orient_block["yaw"])
            debug["native_yaw_rad"] = yaw
            debug["native_yaw_deg"] = round(math.degrees(yaw), 3)
            if "roll" in orient_block:
                debug["native_roll_deg"] = round(
                    math.degrees(float(orient_block["roll"])), 3
                )
            if "pitch" in orient_block:
                debug["native_pitch_deg"] = round(
                    math.degrees(float(orient_block["pitch"])), 3
                )
            return yaw, "orientation_euler", debug
        parsed = conv.parse_heading_sensor_readings(readings)
        if parsed is not None:
            debug["readings_yaw_deg"] = round(math.degrees(parsed), 3)
            return float(parsed), "readings", debug
    except Exception:  # noqa: BLE001
        pass

    if "yaw_from_ov_rad" in debug:
        return float(debug["yaw_from_ov_rad"]), "orientation", debug

    if props is not None and getattr(props, "compass_heading_supported", False):
        try:
            deg = float(await sensor.get_compass_heading())
            debug["compass_deg"] = deg
            return math.radians(deg), "compass", debug
        except Exception:  # noqa: BLE001
            pass

    return None, "none", debug
