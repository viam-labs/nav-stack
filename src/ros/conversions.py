"""Pure-Python geometry and format conversions.

This module deliberately has no ROS dependency so it can be unit-tested without a
ROS2 installation. It handles:

* unit conversion between Viam (millimeters, degrees) and ROS (meters, radians)
* 2D yaw <-> quaternion conversion
* occupancy-grid (Nav2 ``OccupancyGrid``) -> PCD point cloud (Viam SLAM API)
* multi-lidar LaserScan merging (scans -> common-frame XY points -> single scan)
* depth/3D point cloud -> 2D LaserScan ranges
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MM_PER_M = 1000.0


# ---------------------------------------------------------------------------
# Unit + orientation helpers
# ---------------------------------------------------------------------------
def m_to_mm(meters: float) -> float:
    return meters * MM_PER_M


def mm_to_m(millimeters: float) -> float:
    return millimeters / MM_PER_M


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    """Return (x, y, z, w) for a rotation of ``yaw_rad`` about +Z."""
    half = yaw_rad / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def normalize_angle(theta: float) -> float:
    """Wrap ``theta`` to ``(-pi, pi]``."""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    return theta


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw (rotation about +Z) from a quaternion, in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_from_orientation_vector(
    o_x: float, o_y: float, o_z: float, theta_deg: float
) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) in radians from a Viam orientation vector.

    Shared by the external-SLAM pose reader and the typed movement-sensor reader
    so the orientation-vector -> Euler projection lives in one place.
    """
    from viam.proto.common import Orientation
    from viam.spatialmath import OrientationVector

    ov = OrientationVector.from_proto(
        Orientation(o_x=o_x, o_y=o_y, o_z=o_z, theta=theta_deg)
    )
    e = ov.to_quaternion().to_euler_angles()
    return e.roll, e.pitch, e.yaw


@dataclass(frozen=True)
class Pose2D:
    """A 2D pose in the map frame, in meters and radians."""

    x: float
    y: float
    theta: float

    def to_matrix(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]])


def compose_poses(a: Pose2D, b: Pose2D) -> Pose2D:
    """Compose 2D transforms: result = a ∘ b (apply ``b``, then ``a``)."""
    c, s = math.cos(a.theta), math.sin(a.theta)
    return Pose2D(
        a.x + c * b.x - s * b.y,
        a.y + s * b.x + c * b.y,
        normalize_angle(a.theta + b.theta),
    )


def invert_pose(p: Pose2D) -> Pose2D:
    """Inverse of a 2D transform."""
    c, s = math.cos(p.theta), math.sin(p.theta)
    return Pose2D(
        -(c * p.x + s * p.y),
        -(-s * p.x + c * p.y),
        normalize_angle(-p.theta),
    )


def map_pose_to_odom_pose(map_pose: Pose2D, map_to_odom: Pose2D) -> Pose2D:
    """Odom-frame pose that makes ``map_to_odom ∘ odom_pose == map_pose``.

    Used by the mapping-time revisit correction: shifting the published odom
    pose to this value lands slam_toolbox's scan-match prior on ``map_pose``
    without touching its map->odom estimate.
    """
    return compose_poses(invert_pose(map_to_odom), map_pose)


def costmap_frame_to_map(cm: dict, frame_to_map: Pose2D) -> dict:
    """Re-rasterize an axis-aligned occupancy grid into the map frame.

    ``frame_to_map`` is the transform from the costmap's source frame (e.g.
    odom for Nav2 ``local_costmap``) into map: ``map_xy = frame_to_map ∘ src_xy``.
    """
    grid = np.asarray(cm["grid"])
    if grid.ndim != 2 or grid.size == 0:
        return cm
    res = float(cm["resolution"])
    ox, oy = float(cm["origin_x"]), float(cm["origin_y"])
    h, w = grid.shape

    map_xs: list[float] = []
    map_ys: list[float] = []
    cells: list[tuple[int, int, int]] = []
    for row in range(h):
        for col in range(w):
            val = int(grid[row, col])
            if val < 0:
                continue
            fx = ox + (col + 0.5) * res
            fy = oy + (row + 0.5) * res
            mp = compose_poses(frame_to_map, Pose2D(fx, fy, 0.0))
            map_xs.append(mp.x)
            map_ys.append(mp.y)
            cells.append((row, col, val))

    if not cells:
        return cm

    min_x = min(map_xs) - res
    min_y = min(map_ys) - res
    max_x = max(map_xs) + res
    max_y = max(map_ys) + res
    out_w = max(1, int(math.ceil((max_x - min_x) / res)))
    out_h = max(1, int(math.ceil((max_y - min_y) / res)))
    out = np.full((out_h, out_w), -1, dtype=np.int16)

    for row, col, val in cells:
        fx = ox + (col + 0.5) * res
        fy = oy + (row + 0.5) * res
        mp = compose_poses(frame_to_map, Pose2D(fx, fy, 0.0))
        oc = int((mp.x - min_x) / res)
        orow = int((mp.y - min_y) / res)
        if 0 <= orow < out_h and 0 <= oc < out_w:
            prev = int(out[orow, oc])
            out[orow, oc] = val if prev < 0 else max(prev, val)

    return {
        "grid": out,
        "resolution": res,
        "origin_x": min_x,
        "origin_y": min_y,
    }


@dataclass(frozen=True)
class OdomReading:
    """Body-frame twist plus optional pose/heading hints from the sensor."""

    vx: float  # m/s, ROS forward
    vy: float  # m/s, ROS lateral
    vtheta: float  # rad/s about +Z
    pose: Optional[Pose2D] = None  # full odom-frame pose; replaces integration
    heading_rad: Optional[float] = None  # snap yaw while integrating x/y (MiR-style)
    ax: Optional[float] = None  # body-frame horizontal linear accel, gravity removed (m/s^2)
    ay: Optional[float] = None


def _angle_to_rad(value: float) -> float:
    """Convert a heading/yaw that may be radians or degrees into radians."""
    if abs(value) > 2.0 * math.pi + 0.01:
        return math.radians(value)
    return float(value)


def _heading_value_to_rad(key: str, value: float) -> float:
    """Radians from a heading reading; keys named ``*_deg`` are always degrees.

    Other keys are ambiguous, so fall back to the magnitude heuristic (small
    values assumed radians). Without this, ``yaw_deg=5`` would stay ~5 rad.
    """
    if key.endswith("_deg"):
        return math.radians(float(value))
    return _angle_to_rad(float(value))


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return quaternion_to_yaw(float(x), float(y), float(z), float(w))


def _parse_ros_odom_pose_block(readings: Mapping) -> Optional[Pose2D]:
    """Parse a nested ROS ``nav_msgs/Odometry``-style pose from readings."""
    pose_block = readings.get("pose")
    if not isinstance(pose_block, Mapping):
        return None
    inner = pose_block.get("pose")
    if not isinstance(inner, Mapping):
        return None
    position = inner.get("position")
    orientation = inner.get("orientation")
    if not isinstance(position, Mapping) or not isinstance(orientation, Mapping):
        return None
    if not all(k in position for k in ("x", "y")):
        return None
    if not all(k in orientation for k in ("x", "y", "z", "w")):
        return None
    theta = _yaw_from_quaternion(
        orientation["x"], orientation["y"], orientation["z"], orientation["w"]
    )
    return Pose2D(float(position["x"]), float(position["y"]), theta)


def _xyz_from_reading(value) -> Optional[Tuple[float, float, float]]:
    """Extract ``(x, y, z)`` from a dict or Viam Vector3-like object."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not all(k in value for k in ("x", "y", "z")):
            return None
        return float(value["x"]), float(value["y"]), float(value["z"])
    try:
        return float(value.x), float(value.y), float(value.z)
    except (AttributeError, TypeError, ValueError):
        return None


def _as_mapping(value) -> Optional[Mapping]:
    """Best-effort Mapping view of a reading value (dict or object with attrs)."""
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    out: dict = {}
    for key in (
        "x",
        "y",
        "z",
        "w",
        "roll",
        "pitch",
        "yaw",
        "o_x",
        "o_y",
        "o_z",
        "theta",
    ):
        if hasattr(value, key):
            try:
                out[key] = getattr(value, key)
            except Exception:  # noqa: BLE001
                pass
    return out or None


def _parse_orientation_yaw_rad(readings: Mapping) -> Optional[float]:
    """Yaw from Viam ``orientation`` / ``rotation`` blocks (euler or quaternion).

    Viam EulerAngles are radians (Z-Y'-X''). OrientationVector ``theta`` in
    protos is degrees, but attribute-style ``theta`` from OV helpers may already
    be radians — ``_angle_to_rad`` only converts clearly-degree magnitudes.
    """
    for qprefix in ("orientation", "rotation"):
        block = _as_mapping(readings.get(qprefix))
        if block is None:
            continue
        if "yaw" in block:
            # Euler yaw is radians per Viam API; do not degree-guess small values.
            return float(block["yaw"])
        if all(k in block for k in ("x", "y", "z", "w")):
            return _yaw_from_quaternion(
                block["x"], block["y"], block["z"], block["w"]
            )
        if all(k in block for k in ("o_x", "o_y", "o_z")):
            return _angle_to_rad(float(block.get("theta", 0.0)))
    return None


def _parse_heading_rad_from_readings(
    readings: Mapping,
    *,
    odom_only: bool = False,
) -> Optional[float]:
    """Extract yaw in radians from movement-sensor readings.

    When ``odom_only`` is True (for publishing ``/odom``), only ``odom_yaw_deg`` /
    explicit odom aliases and IMU-style ``orientation`` blocks are accepted.
    ``yaw_deg`` from ``viam-labs:mir-base`` is map-fused heading and must not
    drive the odom TF.
    """
    if odom_only:
        for tk in ("odom_yaw_deg", "odom_theta", "odom_yaw"):
            if tk in readings:
                return _heading_value_to_rad(tk, float(readings[tk]))
        # mir-base publishes map-fused pose/heading separately from /odom. When
        # its wheel-odometry velocity fields are present, never treat IMU-style
        # orientation as the odom heading.
        if isinstance(readings.get("linear_velocity_mps"), Mapping):
            return None
        if "position_x_m" in readings and "yaw_deg" in readings:
            return None
        return _parse_orientation_yaw_rad(readings)

    for tk in ("odom_yaw_deg", "odom_theta", "odom_yaw", "yaw_deg", "theta", "yaw", "heading", "pose_theta"):
        if tk in readings:
            return _heading_value_to_rad(tk, float(readings[tk]))
    return _parse_orientation_yaw_rad(readings)


def _parse_euler_rpy_rad(readings: Mapping) -> Optional[Tuple[float, float, float]]:
    """Roll, pitch, yaw in radians from a Viam ``orientation`` block.

    Accepts full euler, yaw-only (roll/pitch assumed 0), or quaternion.
    Viam EulerAngles are radians — do not apply degree heuristics here.
    """
    block = _as_mapping(readings.get("orientation"))
    if block is None:
        return None
    if all(k in block for k in ("roll", "pitch", "yaw")):
        return (
            float(block["roll"]),
            float(block["pitch"]),
            float(block["yaw"]),
        )
    if "yaw" in block:
        return (0.0, 0.0, float(block["yaw"]))
    if all(k in block for k in ("x", "y", "z", "w")):
        x = float(block["x"])
        y = float(block["y"])
        z = float(block["z"])
        w = float(block["w"])
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        yaw = quaternion_to_yaw(x, y, z, w)
        return (roll, pitch, yaw)
    return None


def _gravity_specific_force_body(
    roll: float, pitch: float, yaw: float = 0.0, *, g: float = 9.80665
) -> Tuple[float, float, float]:
    """Expected specific-force gravity vector in the body frame (Z-up world).

    Gravity in the body frame depends only on roll/pitch — not yaw. Including
    yaw here invented a heading-dependent horizontal accel bias that curved
    IMU-only odometry into banana/donut maps.
    """
    del yaw  # yaw must not affect gravity in a Z-up world frame
    # Specific force at rest (accelerometer reads +g along body +Z when level).
    gx = -g * math.sin(pitch)
    gy = g * math.sin(roll) * math.cos(pitch)
    gz = g * math.cos(roll) * math.cos(pitch)
    return gx, gy, gz


def parse_odom_body_horizontal_accel_from_readings(
    readings: Mapping,
) -> Optional[Tuple[float, float]]:
    """Gravity-compensated body-frame X/Y linear acceleration from an IMU."""
    # mir-base wheel odometry already provides twist; never double-integrate IMU
    # acceleration on top of that path.
    if isinstance(readings.get("linear_velocity_mps"), Mapping):
        return None
    if _xyz_from_reading(readings.get("linear_velocity_mps")) is not None:
        return None
    xyz = _xyz_from_reading(readings.get("linear_acceleration"))
    if xyz is None:
        return None
    # Build a temporary readings view so orientation objects also work.
    orient = readings.get("orientation")
    orient_map = _as_mapping(orient)
    probe = dict(readings)
    if orient_map is not None:
        probe["orientation"] = orient_map
    rpy = _parse_euler_rpy_rad(probe)
    if rpy is None:
        return None
    roll, pitch, yaw = rpy
    return gravity_compensated_body_accel(xyz, roll, pitch, yaw)


def gravity_compensated_body_accel(
    accel_xyz: Tuple[float, float, float], roll: float, pitch: float, yaw: float
) -> Tuple[float, float]:
    """Body-frame horizontal (x, y) linear accel with gravity removed.

    ``accel_xyz`` is the raw specific force in m/s^2 (sensor/body frame, gravity
    still present); ``roll``/``pitch``/``yaw`` are the sensor orientation in
    radians. Shared by the ``get_readings`` parser and the typed MovementSensor
    reader so both remove gravity identically.
    """
    ax, ay, _az = accel_xyz
    gx, gy, _gz = _gravity_specific_force_body(roll, pitch, yaw)
    return ax - gx, ay - gy


def parse_odom_twist_from_readings(readings: Mapping) -> Tuple[float, float, float]:
    """Extract body-frame twist (vx, vy, vtheta rad/s) from ``get_readings``."""
    lin = readings.get("linear_velocity_mps")
    ang = readings.get("angular_velocity_dps")
    if isinstance(lin, Mapping) and isinstance(ang, Mapping):
        return (
            float(lin.get("x", 0.0)),
            float(lin.get("y", 0.0)),
            math.radians(float(ang.get("z", 0.0))),
        )

    twist_block = readings.get("twist")
    if isinstance(twist_block, Mapping):
        inner = twist_block.get("twist", twist_block)
        if isinstance(inner, Mapping):
            linear = inner.get("linear")
            angular = inner.get("angular")
            if isinstance(linear, Mapping) and isinstance(angular, Mapping):
                return (
                    float(linear.get("x", 0.0)),
                    float(linear.get("y", 0.0)),
                    float(angular.get("z", 0.0)),
                )

    # rdk:builtin:wheeled-odometry (identified by its position_meters_X/Y
    # readings) uses a compass-style body frame: forward velocity is reported
    # on +Y and X stays zero. Rotate into ROS base_link (x forward, y left)
    # and take yaw rate from angular_velocity (deg/s, CCW positive).
    if "position_meters_X" in readings and "position_meters_Y" in readings:
        lin_xyz = _xyz_from_reading(readings.get("linear_velocity"))
        ang_xyz = _xyz_from_reading(readings.get("angular_velocity"))
        vtheta = math.radians(float(ang_xyz[2])) if ang_xyz is not None else 0.0
        if lin_xyz is not None:
            return float(lin_xyz[1]), -float(lin_xyz[0]), vtheta
        return 0.0, 0.0, vtheta

    # Viam MovementSensor API: GetAngularVelocity is degrees/sec (same as
    # mir-base's angular_velocity_dps). Do NOT treat this as rad/s.
    ang_xyz = _xyz_from_reading(readings.get("angular_velocity"))

    for prefix in ("linear_velocity", "velocity"):
        block = readings.get(prefix)
        if isinstance(block, Mapping) and all(k in block for k in ("x", "y", "z")):
            # Ambiguous units (m/s vs mm/s). Prefer explicit ``linear_velocity_mps``.
            if ang_xyz is not None:
                # Yaw rate must come from angular_velocity when present; the
                # linear block's z is vertical velocity, not a turn rate.
                return (
                    float(block["x"]),
                    float(block["y"]),
                    math.radians(float(ang_xyz[2])),
                )
            return (
                float(block["x"]),
                float(block["y"]),
                math.radians(float(block.get("z", 0.0))),
            )

    if ang_xyz is not None:
        vx = vy = 0.0
        lin_xyz = _xyz_from_reading(readings.get("linear_velocity"))
        if lin_xyz is not None:
            vx, vy, _ = lin_xyz
        return (vx, vy, math.radians(float(ang_xyz[2])))
    return 0.0, 0.0, 0.0


def parse_odom_pose_from_readings(readings: Mapping) -> Optional[Pose2D]:
    """Extract a full odom-frame 2D pose from movement-sensor ``get_readings`` data.

    Supports nested ROS odometry messages, flat ``x``/``y``/``theta`` keys, and
    ``odom_x``/``odom_y`` aliases. Does **not** treat ``viam-labs:mir-base``
    ``position_x_m``/``position_y_m`` as odom position (those are map-frame).
    """
    if not readings:
        return None

    ros_pose = _parse_ros_odom_pose_block(readings)
    if ros_pose is not None:
        return ros_pose

    flat: dict = dict(readings)
    pose_block = readings.get("pose")
    if isinstance(pose_block, Mapping) and "pose" not in pose_block:
        flat.update(pose_block)
    position_block = readings.get("position")
    if isinstance(position_block, Mapping):
        flat.update(position_block)

    xy_pairs = (
        ("odom_position_x_m", "odom_position_y_m"),
        ("odom_x", "odom_y"),
        ("x", "y"),
        ("position_x", "position_y"),
        ("pose_x", "pose_y"),
    )
    x = y = None
    for xk, yk in xy_pairs:
        if xk in flat and yk in flat:
            x = float(flat[xk])
            y = float(flat[yk])
            break

    theta = _parse_heading_rad_from_readings(flat, odom_only=True)
    if theta is None:
        theta = _parse_heading_rad_from_readings(flat, odom_only=False)
        # Map-frame aliases from mir-base must not populate /odom position.
        if x is None and "position_x_m" in flat and "position_y_m" in flat:
            theta = None
    if x is not None and y is not None and theta is not None:
        return Pose2D(x, y, theta)
    return None


def parse_heading_sensor_readings(readings: Mapping) -> Optional[float]:
    """Yaw (rad) from a dedicated heading sensor such as an IMU."""
    if not readings:
        return None
    for tk in ("odom_yaw_deg", "odom_theta", "odom_yaw", "yaw_deg", "theta", "yaw", "heading"):
        if tk in readings:
            return _heading_value_to_rad(tk, float(readings[tk]))
    return _parse_orientation_yaw_rad(readings)


def merge_odom_heading(reading: OdomReading, heading_rad: float) -> OdomReading:
    """Apply ``heading_rad`` to a wheel-odometry sample (pose or heading field)."""
    if reading.pose is not None:
        return OdomReading(
            reading.vx,
            reading.vy,
            reading.vtheta,
            pose=Pose2D(reading.pose.x, reading.pose.y, heading_rad),
            ax=reading.ax,
            ay=reading.ay,
        )
    return OdomReading(
        reading.vx,
        reading.vy,
        reading.vtheta,
        heading_rad=heading_rad,
        ax=reading.ax,
        ay=reading.ay,
    )


def apply_sensor_upside_down(reading: OdomReading) -> OdomReading:
    """Correct a movement sensor mounted upside down (flipped about its x axis).

    Flipping inverts the sensor's z axis, so yaw, yaw rate, and the lateral
    (y) axis all read with the opposite sign. This shows up on maps as scans
    rotating opposite to the pose arrow — the room gets stamped in a circle
    while the robot spins in place. Apply before any mount-yaw correction.
    """
    heading = reading.heading_rad
    if heading is not None:
        heading = normalize_angle(-heading)
    pose = reading.pose
    if pose is not None:
        pose = Pose2D(pose.x, pose.y, normalize_angle(-pose.theta))
    return OdomReading(
        reading.vx,
        -reading.vy,
        -reading.vtheta,
        pose=pose,
        heading_rad=heading,
        ax=reading.ax,
        ay=None if reading.ay is None else -reading.ay,
    )


def apply_sensor_mount_yaw(reading: OdomReading, mount_yaw_rad: float) -> OdomReading:
    """Rotate a body-frame odometry sample from the sensor's mount frame into
    ``base_link``.

    ``mount_yaw_rad`` is the yaw of the sensor's +x axis relative to the robot's
    forward axis (e.g. an IMU physically mounted rotated -90 deg about +z).
    Vectors (velocity, accel) rotate by the mount yaw; reported yaw values get
    the mount yaw subtracted so heading refers to the robot's forward axis.
    """
    if abs(mount_yaw_rad) < 1e-9:
        return reading
    c, s = math.cos(mount_yaw_rad), math.sin(mount_yaw_rad)
    vx = c * reading.vx - s * reading.vy
    vy = s * reading.vx + c * reading.vy
    ax = reading.ax
    ay = reading.ay
    if ax is not None and ay is not None:
        ax, ay = c * ax - s * ay, s * ax + c * ay
    heading = reading.heading_rad
    if heading is not None:
        heading = normalize_angle(heading - mount_yaw_rad)
    pose = reading.pose
    if pose is not None:
        pose = Pose2D(pose.x, pose.y, normalize_angle(pose.theta - mount_yaw_rad))
    return OdomReading(
        vx,
        vy,
        reading.vtheta,
        pose=pose,
        heading_rad=heading,
        ax=ax,
        ay=ay,
    )


def parse_odom_from_readings(readings: Mapping) -> OdomReading:
    """Build an ``OdomReading`` from a single movement-sensor ``get_readings`` call."""
    vx, vy, vtheta = parse_odom_twist_from_readings(readings)
    pose = parse_odom_pose_from_readings(readings)
    heading_rad = None
    if pose is None:
        heading_rad = _parse_heading_rad_from_readings(readings, odom_only=True)
    body_xy = parse_odom_body_horizontal_accel_from_readings(readings)
    ax = ay = None
    if body_xy is not None:
        ax, ay = body_xy
    return OdomReading(
        vx,
        vy,
        vtheta,
        pose=pose,
        heading_rad=heading_rad,
        ax=ax,
        ay=ay,
    )


def viam_pose_to_pose2d(x_mm: float, y_mm: float, theta_deg: float) -> Pose2D:
    """Convert a Viam SLAM ``Pose`` (mm, degrees) into a ROS-style ``Pose2D``."""
    return Pose2D(mm_to_m(x_mm), mm_to_m(y_mm), math.radians(theta_deg))


def pose2d_to_viam_pose(pose: Pose2D) -> Tuple[float, float, float]:
    """Convert a ``Pose2D`` (m, rad) into Viam SLAM ``Pose`` fields (mm, degrees).

    Returns ``(x_mm, y_mm, theta_deg)``.
    """
    return (m_to_mm(pose.x), m_to_mm(pose.y), math.degrees(pose.theta))


def pose2d_to_viam_slam_pose(
    pose: Pose2D, *, yaw_offset_deg: float = 0.0
) -> Tuple[float, float, float, float, float, float, float]:
    """Map-frame ``Pose2D`` → Viam SLAM ``Pose`` OrientationVectorDegrees fields.

    Returns ``(x_mm, y_mm, z_mm, o_x, o_y, o_z, theta_deg)``.

    Uses the planar OV convention (``o_z=1``, yaw in ``theta``) that RDK /
    cartographer use. ``yaw_offset_deg`` is added to the reported yaw only —
    useful when the App arrow is a fixed angular bias relative to the map PCD
    while internal ROS TF is already consistent.
    """
    x_mm, y_mm, theta_deg = pose2d_to_viam_pose(pose)
    return (
        x_mm,
        y_mm,
        0.0,
        0.0,
        0.0,
        1.0,
        float(theta_deg + yaw_offset_deg),
    )


# ---------------------------------------------------------------------------
# Occupancy grid -> PCD
# ---------------------------------------------------------------------------
def occupancy_grid_to_pcd(
    grid: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    occupied_threshold: int = 65,
) -> bytes:
    """Convert a Nav2 occupancy grid into a binary PCD point cloud.

    Args:
        grid: 2D int array (row-major, shape ``(height, width)``) using the ROS
            occupancy convention: -1 unknown, 0 free, 100 fully occupied.
        resolution: meters per cell.
        origin_x/origin_y: world coordinates (meters) of cell (0, 0)'s lower-left
            corner, i.e. ``OccupancyGrid.info.origin.position``.
        occupied_threshold: cells with value >= this are emitted as points.

    Returns:
        Bytes of a valid little-endian binary PCD with float32 x/y/z fields. Viam's
        SLAM ``GetPointCloudMap`` chunks this output.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D (height, width)")

    rows, cols = np.where(grid >= occupied_threshold)
    # Cell centers in world coordinates (meters); z is 0 for a 2D map.
    xs = origin_x + (cols.astype(np.float32) + 0.5) * resolution
    ys = origin_y + (rows.astype(np.float32) + 0.5) * resolution
    zs = np.zeros_like(xs, dtype=np.float32)
    points = np.stack([xs, ys, zs], axis=1).astype(np.float32)

    return points_to_pcd(points)


def points_to_pcd(points: np.ndarray) -> bytes:
    """Serialize an ``(N, 3)`` float array into a binary PCD."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    n = points.shape[0]
    header = (
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    return header + points.tobytes()


def chunk_bytes(data: bytes, chunk_size: int = 1 << 20) -> List[bytes]:
    """Split ``data`` into ``chunk_size``-byte chunks (Viam streams PCD in chunks)."""
    if not data:
        return [b""]
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def path_msg_to_points(path_msg, *, max_points: int = 400) -> List[Dict[str, float]]:
    """Convert a ``nav_msgs/Path``-like object into ``[{x,y,theta}, ...]``.

    Downsamples evenly when longer than ``max_points``, keeping endpoints.
    Coordinates are meters / radians in the path's frame (normally ``map``).
    """
    if path_msg is None:
        return []
    poses = list(getattr(path_msg, "poses", None) or [])
    if not poses:
        return []

    def _one(ps) -> Dict[str, float]:
        pose = ps.pose
        q = pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "theta": float(theta),
        }

    if max_points <= 0 or len(poses) <= max_points:
        return [_one(p) for p in poses]
    if max_points == 1:
        return [_one(poses[-1])]
    n = len(poses)
    idxs = [
        int(round(i * (n - 1) / float(max_points - 1))) for i in range(max_points)
    ]
    seen = set()
    ordered: List[int] = []
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return [_one(poses[i]) for i in ordered]


def path_length_m(points: List[Dict[str, float]]) -> float:
    """Polyline length of map-frame path points (meters)."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
    return total


# ---------------------------------------------------------------------------
# LaserScan <-> points, and multi-lidar merge
# ---------------------------------------------------------------------------
@dataclass
class LidarPoints:
    """Lidar returns in sensor frame and in base_link (meters)."""

    sensor: np.ndarray  # (N, 3) in the scanner frame
    base_link: np.ndarray  # (N, 3) in base_link
    sensor_scan: Optional["LaserScan2D"] = None  # native MiR scan for /scan_0
    # Cache age (seconds) of the scan when mir-base returned it, from the
    # get_laser_scan payload's ``age_s``. Lets the bridge stamp the scan at its
    # capture time (read_start - age_s) instead of read time; None when the
    # producer doesn't report it (older mir-base or the direct PCD path).
    age_s: Optional[float] = None


@dataclass
class LaserScan2D:
    """A minimal, ROS-agnostic representation of a 2D laser scan."""

    ranges: np.ndarray  # shape (N,), meters; inf/nan for no return
    angle_min: float  # radians
    angle_increment: float  # radians
    range_min: float = 0.0
    range_max: float = float("inf")
    # Pose of the scanner in the target (e.g. base_link) frame.
    sensor_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)

    def to_points(self) -> np.ndarray:
        """Return valid scan returns as ``(M, 2)`` XY points in the target frame."""
        n = len(self.ranges)
        if n == 0:
            return np.empty((0, 2))
        angles = self.angle_min + np.arange(n) * self.angle_increment
        r = np.asarray(self.ranges, dtype=float)
        valid = np.isfinite(r) & (r >= self.range_min) & (r <= self.range_max)
        r = r[valid]
        angles = angles[valid]
        # Points in the sensor frame.
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)
        # Transform into the target frame using the sensor pose.
        transformed = (self.sensor_pose.to_matrix() @ pts.T).T
        return transformed[:, :2]


def points_to_scan(
    points: np.ndarray,
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
    num_bins: int = 720,
    range_min: float = 0.05,
    range_max: float = 25.0,
) -> LaserScan2D:
    """Project ``(N, 2|3)`` XY(Z) points into a single 2D LaserScan.

    Each angular bin keeps the closest point. Used both for depth-cloud -> scan and
    for merging multiple lidars (after transforming each into a common frame).
    """
    points = np.asarray(points, dtype=float)
    angle_increment = (angle_max - angle_min) / num_bins
    ranges = np.full(num_bins, np.inf)
    if points.size == 0:
        return LaserScan2D(ranges, angle_min, angle_increment, range_min, range_max)

    xy = points[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    valid = (r >= range_min) & (r <= range_max) & (ang >= angle_min) & (ang < angle_max)
    r = r[valid]
    ang = ang[valid]
    bins = ((ang - angle_min) / angle_increment).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)
    # Keep the minimum range per bin.
    for b, rng in zip(bins, r):
        if rng < ranges[b]:
            ranges[b] = rng
    return LaserScan2D(ranges, angle_min, angle_increment, range_min, range_max)


def forward_sector_min_range(
    scan: LaserScan2D, *, half_width_rad: float = math.radians(15.0)
) -> Optional[float]:
    """Nearest finite return within ±``half_width_rad`` of angle 0 (robot forward).

    Used by sensor probes: standing in front of the cart should drop this value
    even when slam_toolbox refuses to update the occupancy map while parked.
    """
    if scan.ranges.size == 0:
        return None
    angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    # Wrap angles into (-pi, pi] so a scan that starts at -pi still matches
    # forward (= 0) correctly.
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    sector = np.abs(wrapped) <= half_width_rad
    vals = scan.ranges[sector]
    finite = vals[np.isfinite(vals) & (vals >= scan.range_min)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def nearest_return_bearing_deg(
    scan: LaserScan2D, *, max_range_m: float = 8.0
) -> Optional[float]:
    """Bearing (degrees) of the nearest in-range return relative to robot +X.

    When parked facing a wall, this should be near ``0``. A persistent ~±45°
    means lidar ``mount.theta`` disagrees with base_link forward.
    """
    if scan.ranges.size == 0:
        return None
    angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    r = np.asarray(scan.ranges, dtype=float)
    valid = (
        np.isfinite(r)
        & (r >= scan.range_min)
        & (r <= min(float(scan.range_max), max_range_m))
    )
    if not np.any(valid):
        return None
    i = int(np.argmin(np.where(valid, r, np.inf)))
    bearing = (float(angles[i]) + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(bearing))


@dataclass(frozen=True)
class WallYawObservation:
    """Dominant side-wall line in ``base_link`` for anti-banana yaw correction."""

    wall_yaw_body: float  # wall tangent angle in base_link (rad), ~0 when parallel
    length_m: float
    inliers: int
    side: str  # "left" or "right"


def _fold_wall_dir_to_body_x(wall_dir_rad: float) -> float:
    """Map a line direction onto (-pi/2, pi/2] nearest to body +X (ambiguous by pi)."""
    a = normalize_angle(wall_dir_rad)
    b = normalize_angle(wall_dir_rad + math.pi)
    return a if abs(a) <= abs(b) else b


def _sector_points(
    scan: LaserScan2D, *, center_rad: float, half_width_rad: float
) -> np.ndarray:
    """Valid scan hits in a bearing sector as ``(N, 2)`` base_link XY."""
    pts = scan.to_points()
    if pts.size == 0:
        return pts
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    # Smallest angle difference to sector center.
    d = (ang - center_rad + math.pi) % (2.0 * math.pi) - math.pi
    return pts[np.abs(d) <= half_width_rad]


def _fit_line_ransac(
    points: np.ndarray,
    *,
    inlier_dist_m: float,
    iters: int,
    rng: np.random.Generator,
) -> Optional[Tuple[float, float, int, float]]:
    """Return ``(dir_rad, length_m, n_inliers, inlier_ratio)`` or None."""
    n = int(points.shape[0])
    if n < 2:
        return None
    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(max(1, iters)):
        i0, i1 = rng.choice(n, size=2, replace=False)
        p0, p1 = points[i0], points[i1]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        seg = math.hypot(dx, dy)
        if seg < 1e-3:
            continue
        # Distance from each point to the infinite line through p0-p1.
        # | (p - p0) x dir | / |dir| in 2D = |dx*(y-y0) - dy*(x-x0)| / seg
        cross = np.abs(dx * (points[:, 1] - p0[1]) - dy * (points[:, 0] - p0[0]))
        dist = cross / seg
        mask = dist <= inlier_dist_m
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_inliers = points[mask]
    if best_inliers is None or best_count < 2:
        return None
    # PCA direction on inliers for a stable tangent angle / length.
    mean = np.mean(best_inliers, axis=0)
    centered = best_inliers - mean
    cov = centered.T @ centered / max(best_count - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ direction
    length = float(np.max(proj) - np.min(proj)) if proj.size else 0.0
    dir_rad = math.atan2(float(direction[1]), float(direction[0]))
    return dir_rad, length, best_count, best_count / float(n)


def extract_dominant_wall(
    scan: LaserScan2D,
    *,
    min_length_m: float = 2.0,
    min_inliers: int = 25,
    min_inlier_ratio: float = 0.35,
    inlier_dist_m: float = 0.08,
    parallel_tol_rad: float = math.radians(35.0),
    side_half_width_rad: float = math.radians(45.0),
    ransac_iters: int = 80,
    seed: int = 0,
) -> Optional[WallYawObservation]:
    """Find the strongest side wall roughly parallel to body +X.

    Used to de-bias gyro yaw while driving along long straight walls (anti-banana).
    Returns ``None`` when no side wall is long/clean enough.
    """
    rng = np.random.default_rng(seed)
    candidates: list[WallYawObservation] = []
    for side, center in (("left", math.pi / 2.0), ("right", -math.pi / 2.0)):
        pts = _sector_points(
            scan, center_rad=center, half_width_rad=side_half_width_rad
        )
        fitted = _fit_line_ransac(
            pts, inlier_dist_m=inlier_dist_m, iters=ransac_iters, rng=rng
        )
        if fitted is None:
            continue
        dir_rad, length_m, n_inliers, inlier_ratio = fitted
        if length_m < min_length_m or n_inliers < min_inliers:
            continue
        if inlier_ratio < min_inlier_ratio:
            continue
        folded = _fold_wall_dir_to_body_x(dir_rad)
        if abs(folded) > parallel_tol_rad:
            continue
        candidates.append(
            WallYawObservation(
                wall_yaw_body=folded,
                length_m=length_m,
                inliers=n_inliers,
                side=side,
            )
        )
    if not candidates:
        return None
    # Prefer longer, denser walls.
    return max(candidates, key=lambda c: (c.length_m, c.inliers))


def wall_yaw_correction_delta(
    wall_yaw_body: float,
    *,
    max_step_rad: float = math.radians(2.0),
    blend: float = 0.5,
) -> float:
    """Delta to **add** to odom yaw so a side wall lines up with body +X.

    ``wall_yaw_body`` is the wall tangent folded near 0 (parallel to travel).
    Gyro still leads; callers should use a small ``max_step_rad`` and ``blend``.
    """
    # Observed wall rotation in body = robot yaw error vs the wall.
    # Rotate body by -wall_yaw to restore parallelism.
    err = _fold_wall_dir_to_body_x(wall_yaw_body)
    raw = -float(blend) * err
    limit = abs(float(max_step_rad))
    return max(-limit, min(limit, raw))


def merge_scans(
    scans: Sequence[LaserScan2D],
    num_bins: int = 720,
    range_min: float = 0.05,
    range_max: float = 25.0,
) -> LaserScan2D:
    """Merge multiple lidar scans (each with its own ``sensor_pose``) into one scan.

    Each input scan is transformed into the common (base_link) frame via its
    ``sensor_pose``, then all points are re-projected into a single 360 scan. This
    is what feeds slam_toolbox, which only accepts a single ``/scan`` input.
    """
    if not scans:
        return LaserScan2D(np.full(num_bins, np.inf), -math.pi, 2 * math.pi / num_bins)
    all_points = [s.to_points() for s in scans]
    stacked = np.vstack([p for p in all_points if p.size]) if any(
        p.size for p in all_points
    ) else np.empty((0, 2))
    return points_to_scan(
        stacked,
        angle_min=-math.pi,
        angle_max=math.pi,
        num_bins=num_bins,
        range_min=range_min,
        range_max=range_max,
    )


def filter_points_by_z(
    points: np.ndarray, z_min: float, z_max: float
) -> np.ndarray:
    """Keep ``(N, 3+)`` points whose Z lies in ``[z_min, z_max]``."""
    points = np.asarray(points, dtype=float)
    if points.size == 0 or points.shape[1] < 3:
        return points
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def pointcloud_to_scan(
    points: np.ndarray,
    z_min: float = -0.2,
    z_max: float = 2.0,
    sensor_pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
    **scan_kwargs,
) -> LaserScan2D:
    """Flatten a 3D point cloud (e.g. from a depth camera) into a 2D LaserScan.

    Points outside ``[z_min, z_max]`` are discarded before projection. For
    depth/3D cameras and point-cloud lidars (e.g. Livox) whose ``get_point_cloud``
    output is in ``base_link``, Z is height above the ground plane.
    """
    points = np.asarray(points, dtype=float)
    if points.size and points.shape[1] >= 3:
        points = filter_points_by_z(points, z_min, z_max)
    xy = points[:, :2] if points.size else np.empty((0, 2))
    if xy.size:
        homog = np.stack([xy[:, 0], xy[:, 1], np.ones(len(xy))], axis=1)
        xy = (sensor_pose.to_matrix() @ homog.T).T[:, :2]
    scan = points_to_scan(xy, **scan_kwargs)
    return scan


@dataclass(frozen=True)
class ScanMotionResult:
    """Scan-to-scan motion estimate with match-quality metrics."""

    dx: float
    dy: float
    dtheta: float
    residual: float
    improvement: float
    match_fraction: float
    method: str = "grid"

    def __iter__(self):
        yield self.dx
        yield self.dy
        yield self.dtheta


def _interpolate_range_at_angle(
    angles: np.ndarray, ranges: np.ndarray, angle: float
) -> Optional[float]:
    """Linearly interpolate range at ``angle`` (angles must be sorted)."""
    if angles.size == 0:
        return None
    angle = normalize_angle(angle)
    # Map angles to (-pi, pi] for comparison with sorted scan angles.
    wrapped = ((angles + math.pi) % (2 * math.pi)) - math.pi
    target = ((angle + math.pi) % (2 * math.pi)) - math.pi
    if target <= wrapped[0] or target >= wrapped[-1]:
        # Extrapolation at scan edges is unreliable for 360° Livox bins.
        return None
    j = int(np.searchsorted(wrapped, target))
    if j <= 0 or j >= wrapped.size:
        return None
    a0, a1 = float(wrapped[j - 1]), float(wrapped[j])
    r0, r1 = float(ranges[j - 1]), float(ranges[j])
    if a1 == a0:
        return r0
    t = (target - a0) / (a1 - a0)
    return r0 + t * (r1 - r0)


def estimate_forward_range_flow(
    prev: LaserScan2D,
    curr: LaserScan2D,
    *,
    dtheta: float = 0.0,
    forward_window_rad: float = math.pi / 3,
    min_beams: int = 25,
    max_median_deviation_m: float = 0.12,
    min_abs_dx: float = 0.005,
) -> Optional[ScanMotionResult]:
    """Estimate forward ``dx`` from range deltas in the front hemisphere.

    Livox 360° scans are sparse and often fail the grid matcher; comparing
    overlapping forward ranges after compensating for ``dtheta`` is more robust
    for slow skid-steer motion.
    """
    prev_r = np.asarray(prev.ranges, dtype=float)
    curr_r = np.asarray(curr.ranges, dtype=float)
    if prev_r.size < 20 or curr_r.size < 20:
        return None

    prev_angles = prev.angle_min + np.arange(prev_r.size, dtype=float) * prev.angle_increment
    curr_angles = curr.angle_min + np.arange(curr_r.size, dtype=float) * curr.angle_increment
    valid_prev = (
        np.isfinite(prev_r)
        & (prev_r >= float(prev.range_min))
        & (prev_r <= float(prev.range_max))
    )
    valid_curr = (
        np.isfinite(curr_r)
        & (curr_r >= float(curr.range_min))
        & (curr_r <= float(curr.range_max))
    )
    curr_angles_v = curr_angles[valid_curr]
    curr_r_v = curr_r[valid_curr]
    if curr_angles_v.size < 20:
        return None
    order = np.argsort(curr_angles_v)
    curr_angles_v = curr_angles_v[order]
    curr_r_v = curr_r_v[order]

    estimates: List[float] = []
    for angle, r0 in zip(prev_angles[valid_prev], prev_r[valid_prev]):
        if abs(angle) > forward_window_rad:
            continue
        cos_a = math.cos(angle)
        if abs(cos_a) < 0.25:
            continue
        r1 = _interpolate_range_at_angle(curr_angles_v, curr_r_v, angle - dtheta)
        if r1 is None:
            continue
        dr = float(r0) - float(r1)
        estimates.append(dr / cos_a)

    if len(estimates) < min_beams:
        return None
    arr = np.asarray(estimates, dtype=float)
    dx = float(np.median(arr))
    mad = float(np.median(np.abs(arr - dx)))
    if mad > max_median_deviation_m:
        return None
    # ``min_abs_dx=0`` lets callers use a near-zero flow as a valid "not
    # moving" measurement (lidar-confirmed ZUPT) instead of a failed match.
    if abs(dx) < min_abs_dx:
        return None
    return ScanMotionResult(
        dx,
        0.0,
        float(dtheta),
        mad,
        mad,
        len(estimates) / max(int(np.sum(valid_prev)), 1),
        method="range_flow",
    )


def _scan_motion_grid_search(
    prev: LaserScan2D,
    curr: LaserScan2D,
    *,
    dtheta: float = 0.0,
    max_translation_m: float = 0.25,
    step_m: float = 0.02,
    max_beams: int = 180,
    max_mean_residual_m: float = 0.15,
    min_improvement: float = 0.012,
    min_match_fraction: float = 0.22,
    allow_lateral: bool = False,
) -> Tuple[Optional[ScanMotionResult], Dict[str, float]]:
    """Grid search with diagnostic stats (for logging when rejected)."""
    prev_r = np.asarray(prev.ranges, dtype=float)
    curr_r = np.asarray(curr.ranges, dtype=float)
    if prev_r.size < 20 or curr_r.size < 20:
        return None, {"reject": "sparse_scan"}
    prev_angles = prev.angle_min + np.arange(prev_r.size, dtype=float) * prev.angle_increment
    valid_prev = (
        np.isfinite(prev_r)
        & (prev_r >= float(prev.range_min))
        & (prev_r <= float(prev.range_max))
    )
    idx = np.flatnonzero(valid_prev)
    if idx.size < 20:
        return None, {"reject": "sparse_prev"}
    if idx.size > max_beams:
        idx = idx[np.linspace(0, idx.size - 1, max_beams, dtype=np.int32)]
    prev_angles = prev_angles[idx]
    prev_r = prev_r[idx]

    curr_angles = curr.angle_min + np.arange(curr_r.size, dtype=float) * curr.angle_increment
    valid_curr = (
        np.isfinite(curr_r)
        & (curr_r >= float(curr.range_min))
        & (curr_r <= float(curr.range_max))
    )
    curr_angles_v = curr_angles[valid_curr]
    curr_r_v = curr_r[valid_curr]
    if curr_angles_v.size < 20:
        return None, {"reject": "sparse_curr"}
    order = np.argsort(curr_angles_v)
    curr_angles_v = curr_angles_v[order]
    curr_r_v = curr_r_v[order]

    c = math.cos(-dtheta)
    s = math.sin(-dtheta)
    px = prev_r * np.cos(prev_angles)
    py = prev_r * np.sin(prev_angles)

    def _residual(dx: float, dy: float) -> Tuple[float, float]:
        qx = px - dx
        qy = py - dy
        rx = c * qx - s * qy
        ry = s * qx + c * qy
        expected_r = np.hypot(rx, ry)
        expected_a = np.arctan2(ry, rx)
        j = np.searchsorted(curr_angles_v, expected_a)
        j = np.clip(j, 1, curr_angles_v.size - 1)
        left = j - 1
        use_left = np.abs(curr_angles_v[left] - expected_a) <= np.abs(
            curr_angles_v[j] - expected_a
        )
        matched = np.where(use_left, curr_r_v[left], curr_r_v[j])
        ang_err = np.minimum(
            np.abs(curr_angles_v[left] - expected_a),
            np.abs(curr_angles_v[j] - expected_a),
        )
        keep = ang_err < 0.20
        if not np.any(keep):
            return float("inf"), 0.0
        return (
            float(np.mean(np.abs(expected_r[keep] - matched[keep]))),
            float(np.mean(keep)),
        )

    zero_residual, zero_fraction = _residual(0.0, 0.0)
    best_residual = zero_residual
    best_fraction = zero_fraction
    best_dx = 0.0
    best_dy = 0.0
    n_steps = int(math.ceil(max_translation_m / step_m))
    y_steps = range(-n_steps, n_steps + 1) if allow_lateral else (0,)
    for ix in range(-n_steps, n_steps + 1):
        for iy in y_steps:
            if ix == 0 and iy == 0:
                continue
            dx = ix * step_m
            dy = iy * step_m
            residual, fraction = _residual(dx, dy)
            if residual < best_residual:
                best_residual = residual
                best_fraction = fraction
                best_dx = dx
                best_dy = dy

    debug = {
        "zero_residual_m": zero_residual,
        "best_residual_m": best_residual,
        "best_dx": best_dx,
        "best_fraction": best_fraction,
    }
    if best_residual > max_mean_residual_m:
        debug["reject"] = "residual"
        return None, debug
    if best_fraction < min_match_fraction:
        debug["reject"] = "fraction"
        return None, debug
    improvement = zero_residual - best_residual
    rel_improvement = improvement / max(zero_residual, 1e-3)
    if improvement < min_improvement and rel_improvement < 0.08:
        debug["reject"] = "improvement"
        debug["improvement_m"] = improvement
        return None, debug
    if best_dx == 0.0 and best_dy == 0.0:
        debug["reject"] = "zero_dx"
        return None, debug
    return (
        ScanMotionResult(
            best_dx,
            best_dy,
            float(dtheta),
            best_residual,
            improvement,
            best_fraction,
            method="grid",
        ),
        debug,
    )


def estimate_scan_motion(
    prev: LaserScan2D,
    curr: LaserScan2D,
    *,
    dtheta: float = 0.0,
    max_translation_m: float = 0.25,
    step_m: float = 0.02,
    max_beams: int = 180,
    max_mean_residual_m: float = 0.15,
    min_improvement: float = 0.012,
    min_match_fraction: float = 0.22,
    allow_lateral: bool = False,
) -> Optional[ScanMotionResult]:
    """Estimate robot motion in the previous base_link frame from two scans.

    Used as lidar odometry when wheel encoders are unavailable (IMU + lidar).
    Tries a translation grid search first, then a forward range-flow fallback
    suited to sparse 360° Livox scans.
    """
    motion, _debug = _scan_motion_grid_search(
        prev,
        curr,
        dtheta=dtheta,
        max_translation_m=max_translation_m,
        step_m=step_m,
        max_beams=max_beams,
        max_mean_residual_m=max_mean_residual_m,
        min_improvement=min_improvement,
        min_match_fraction=min_match_fraction,
        allow_lateral=allow_lateral,
    )
    if motion is not None:
        return motion
    return estimate_forward_range_flow(prev, curr, dtheta=dtheta)


def estimate_scan_motion_with_debug(
    prev: LaserScan2D,
    curr: LaserScan2D,
    **kwargs,
) -> Tuple[Optional[ScanMotionResult], Dict[str, float]]:
    """Like :func:`estimate_scan_motion` but always returns diagnostic stats."""
    motion, debug = _scan_motion_grid_search(prev, curr, **kwargs)
    if motion is not None:
        debug["method"] = motion.method
        return motion, debug
    flow = estimate_forward_range_flow(prev, curr, dtheta=kwargs.get("dtheta", 0.0))
    if flow is not None:
        debug = {
            "method": flow.method,
            "best_dx": flow.dx,
            "best_residual_m": flow.residual,
            "best_fraction": flow.match_fraction,
        }
        return flow, debug
    debug.setdefault("reject", "no_match")
    debug["method"] = "none"
    return None, debug


def scan_has_returns(scan: LaserScan2D) -> bool:
    """True when the scan contains at least one finite in-range return."""
    r = np.asarray(scan.ranges, dtype=float)
    return bool(
        np.any(np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max))
    )


# MiR250 laser mounts (meters, radians) — keep in sync with viam-mir-base mir_rosbridge.py.
_MIR250_FRONT_MOUNT = Pose2D(0.315 - 0.004485, 0.205, 0.25 * math.pi)
_MIR250_BACK_MOUNT = Pose2D(-0.315 - 0.004485, -0.205, -0.75 * math.pi)


def _mir_mount_for_scan(frame_id: str, topic: str = "") -> Pose2D:
    normalized = (frame_id or "").strip().lower()
    if "front" in normalized and "laser" in normalized:
        return _MIR250_FRONT_MOUNT
    if "back" in normalized and "laser" in normalized:
        return _MIR250_BACK_MOUNT
    key = (topic or "").rsplit("/", 1)[-1].lower()
    if key in {"f_raw_scan", "f_scan"}:
        return _MIR250_FRONT_MOUNT
    if key in {"b_raw_scan", "b_scan"}:
        return _MIR250_BACK_MOUNT
    return Pose2D(0.0, 0.0, 0.0)


def _transform_xy(points_xy: np.ndarray, mount: Pose2D) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2))
    c, s = math.cos(mount.theta), math.sin(mount.theta)
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    out_x = mount.x + c * xs - s * ys
    out_y = mount.y + s * xs + c * ys
    return np.stack([out_x, out_y], axis=1)


def _laser_scan_dict_to_points(
    scan: Mapping,
    *,
    topic: str = "",
) -> LidarPoints:
    """Convert a MiR/rosbridge LaserScan dict into sensor- and base-frame points."""
    ranges = scan.get("ranges") or []
    if not ranges:
        return LidarPoints(sensor=np.empty((0, 3)), base_link=np.empty((0, 3)))

    angle_min = float(scan.get("angle_min", 0.0))
    angle_increment = float(scan.get("angle_increment", 0.0))
    range_min = float(scan.get("range_min", 0.05))
    range_max = float(scan.get("range_max", 25.0))
    header = scan.get("header") or {}
    frame_id = str(header.get("frame_id") or "")
    mount = _mir_mount_for_scan(frame_id, topic)

    sensor_xy: List[Tuple[float, float]] = []
    for index, raw_range in enumerate(ranges):
        if raw_range is None:
            continue
        try:
            distance_m = float(raw_range)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            continue
        if distance_m < range_min or distance_m > range_max:
            continue
        angle = angle_min + index * angle_increment
        sensor_xy.append(
            (distance_m * math.cos(angle), distance_m * math.sin(angle))
        )

    if not sensor_xy:
        return LidarPoints(sensor=np.empty((0, 3)), base_link=np.empty((0, 3)))

    sensor_arr = np.asarray(sensor_xy, dtype=float)
    sensor = np.column_stack([sensor_arr[:, 0], sensor_arr[:, 1], np.zeros(len(sensor_arr))])
    base_xy = _transform_xy(sensor_arr, mount)
    base_link = np.column_stack([base_xy[:, 0], base_xy[:, 1], np.zeros(len(base_xy))])
    return LidarPoints(sensor=sensor, base_link=base_link)


def mir_laser_scan_message_to_scan2d(scan: Mapping) -> LaserScan2D:
    """Convert a rosbridge LaserScan dict to ``LaserScan2D`` (keeps native angles)."""
    raw_ranges = scan.get("ranges") or []
    n = len(raw_ranges)
    ranges = np.full(n, np.inf, dtype=float)
    range_min = float(scan.get("range_min", 0.05))
    range_max = float(scan.get("range_max", 25.0))
    for index, raw_range in enumerate(raw_ranges):
        if raw_range is None:
            continue
        try:
            distance_m = float(raw_range)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            continue
        if distance_m < range_min or distance_m > range_max:
            continue
        ranges[index] = distance_m
    return LaserScan2D(
        ranges=ranges,
        angle_min=float(scan.get("angle_min", -math.pi)),
        angle_increment=float(scan.get("angle_increment", 2 * math.pi / max(n, 1))),
        range_min=range_min,
        range_max=range_max,
    )


def transform_points_between_poses(
    points: np.ndarray,
    from_pose: Pose2D,
    to_pose: Pose2D,
) -> np.ndarray:
    """Re-express ``(N, 3)`` points from an old base_link snapshot in the current one."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3))
    if points.shape[1] == 2:
        points = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    c0, s0 = math.cos(from_pose.theta), math.sin(from_pose.theta)
    c1, s1 = math.cos(to_pose.theta), math.sin(to_pose.theta)
    wx = from_pose.x + c0 * points[:, 0] - s0 * points[:, 1]
    wy = from_pose.y + s0 * points[:, 0] + c0 * points[:, 1]
    dx = wx - to_pose.x
    dy = wy - to_pose.y
    out_x = c1 * dx + s1 * dy
    out_y = -s1 * dx + c1 * dy
    return np.column_stack([out_x, out_y, points[:, 2]])


def merge_accumulated_point_clouds(
    frames: Sequence[Tuple[np.ndarray, Pose2D]],
    current_pose: Pose2D,
) -> np.ndarray:
    """Merge time-accumulated base_link clouds into the current body frame."""
    chunks: List[np.ndarray] = []
    for points, pose in frames:
        if points.size:
            chunks.append(transform_points_between_poses(points, pose, current_pose))
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks)


def merge_accumulated_rotation_only(
    frames: Sequence[Tuple[np.ndarray, Pose2D]],
    current_theta: float,
) -> np.ndarray:
    """Merge clouds using yaw deltas only (safe without wheel translation odom)."""
    chunks: List[np.ndarray] = []
    for points, pose in frames:
        if points.size == 0:
            continue
        dtheta = normalize_angle(pose.theta - current_theta)
        c, s = math.cos(dtheta), math.sin(dtheta)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2] if points.shape[1] >= 3 else np.zeros(len(points))
        chunks.append(np.column_stack([c * x - s * y, s * x + c * y, z]))
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks)


def _mount_rotation(theta: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Sensor->base rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (ROS convention).

    Positive pitch tilts the sensor's forward (+x) axis downward.
    """
    cz, sz = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def transform_lidar_mount_to_base_link(
    points: np.ndarray,
    *,
    x: float,
    y: float,
    z: float,
    theta: float,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> np.ndarray:
    """Transform ``(N, 3)`` points from the lidar mount frame into ``base_link``.

    Uses the static mount configured on the SLAM service. Point-cloud cameras
    (Livox, depth) usually return points in the component / sensor frame;
    ``z_min`` / ``z_max`` are interpreted in ``base_link`` (height above the
    floor), so this transform must run before height filtering. ``pitch`` /
    ``roll`` level a tilted sensor — even a ~2 deg mast tilt moves the floor
    into the z band at 15-20 m and imprints phantom borders.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3))
    if points.shape[1] == 2:
        points = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    rot = _mount_rotation(theta, pitch, roll)
    out = (rot @ points.T).T
    out[:, 0] += x
    out[:, 1] += y
    out[:, 2] += z
    return out


def transform_base_link_to_lidar_mount(
    points: np.ndarray,
    *,
    x: float,
    y: float,
    z: float,
    theta: float,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> np.ndarray:
    """Inverse of :func:`transform_lidar_mount_to_base_link`.

    The /scan projection path deliberately calls this with the default
    ``pitch=0, roll=0``: the published laser frame is the *leveled* mount frame
    (translation + yaw only), matching the yaw-only static TF and keeping the
    2D projection free of out-of-plane distortion.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3))
    if points.shape[1] == 2:
        points = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    shifted = points.copy()
    shifted[:, 0] -= x
    shifted[:, 1] -= y
    shifted[:, 2] -= z
    rot_inv = _mount_rotation(theta, pitch, roll).T
    return (rot_inv @ shifted.T).T


def base_link_cloud_to_lidar_scan(
    points: np.ndarray,
    *,
    x: float,
    y: float,
    z: float,
    theta: float,
    z_min: float,
    z_max: float,
    points_in_base_link: bool = False,
    **scan_kwargs,
) -> LaserScan2D:
    """Height-filter in ``base_link``, then project a scan in the lidar frame."""
    points = np.asarray(points, dtype=float)
    if points.size and points.shape[1] >= 3:
        points = filter_points_by_z(points, z_min, z_max)
    if not points_in_base_link and points.size:
        points = transform_base_link_to_lidar_mount(
            points, x=x, y=y, z=z, theta=theta
        )
    xy = points[:, :2] if points.size else np.empty((0, 2))
    return points_to_scan(xy, **scan_kwargs)


def points_from_mir_laser_scan_payload(payload: Mapping) -> LidarPoints:
    """Parse ``viam-labs:mir-base:lidar`` ``get_laser_scan`` output into points."""
    sensor_chunks: List[np.ndarray] = []
    base_chunks: List[np.ndarray] = []
    sensor_scan: Optional[LaserScan2D] = None
    ages: List[float] = []
    for entry in payload.get("scans") or []:
        if not isinstance(entry, Mapping):
            continue
        msg = entry.get("message")
        if not isinstance(msg, Mapping):
            msg = entry
        topic = str(entry.get("topic") or "")
        entry_age = _coerce_optional_float(entry.get("age_s"))
        if entry_age is not None:
            ages.append(entry_age)
        if sensor_scan is None:
            candidate = mir_laser_scan_message_to_scan2d(msg)
            if scan_has_returns(candidate):
                sensor_scan = candidate
        chunk = _laser_scan_dict_to_points(msg, topic=topic)
        if chunk.sensor.size:
            sensor_chunks.append(chunk.sensor)
        if chunk.base_link.size:
            base_chunks.append(chunk.base_link)
    sensor = np.vstack(sensor_chunks) if sensor_chunks else np.empty((0, 3))
    base_link = np.vstack(base_chunks) if base_chunks else np.empty((0, 3))
    # Fall back to a top-level age_s if the producer reports it once for the set.
    if not ages:
        top_age = _coerce_optional_float(payload.get("age_s"))
        if top_age is not None:
            ages.append(top_age)
    # Merged points mix all lidars, so stamp conservatively at the oldest scan.
    age_s = max(ages) if ages else None
    return LidarPoints(
        sensor=sensor, base_link=base_link, sensor_scan=sensor_scan, age_s=age_s
    )


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def parse_pcd(raw: bytes) -> np.ndarray:
    """Parse a PCD point cloud (as returned by Viam cameras) into ``(N, 3)`` XYZ.

    Supports ASCII and uncompressed little-endian binary PCDs with at least x/y/z
    fields. Extra fields (e.g. rgb, intensity) are skipped.
    """
    if not raw:
        return np.empty((0, 3))
    # Split header (ASCII) from data.
    idx = raw.find(b"DATA ")
    if idx < 0:
        return np.empty((0, 3))
    nl = raw.find(b"\n", idx)
    header_text = raw[:nl].decode("ascii", errors="replace")
    data_fmt = raw[idx + 5 : nl].decode("ascii").strip()
    body = raw[nl + 1 :]

    fields: List[str] = []
    sizes: List[int] = []
    types: List[str] = []
    counts: List[int] = []
    npoints = 0
    for line in header_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(x) for x in parts[1:]]
        elif key == "POINTS":
            npoints = int(parts[1])
        elif key == "WIDTH" and npoints == 0:
            npoints = int(parts[1])

    if not counts:
        counts = [1] * len(fields)

    if not fields or not {"x", "y", "z"} <= set(fields):
        return np.empty((0, 3))

    if data_fmt == "ascii":
        rows = [r.split() for r in body.decode("ascii").splitlines() if r.strip()]
        arr = np.array(rows, dtype=float) if rows else np.empty((0, len(fields)))
        col = {f: i for i, f in enumerate(fields)}
        if not {"x", "y", "z"} <= set(col) or arr.size == 0:
            return np.empty((0, 3))
        return arr[:, [col["x"], col["y"], col["z"]]].astype(float)

    # Binary: build a numpy structured dtype matching the PCD record layout.
    type_map = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
                ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
    dtype_fields = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        np_t = type_map.get((t.upper(), s), f"V{s}")
        for k in range(c):
            name = f if c == 1 else f"{f}_{k}"
            dtype_fields.append((name, np.dtype(np_t)))
    if not dtype_fields:
        return np.empty((0, 3))
    record = np.dtype(dtype_fields)
    record_size = record.itemsize
    if record_size == 0:
        return np.empty((0, 3))
    if npoints == 0:
        npoints = len(body) // record_size
    structured = np.frombuffer(body[: npoints * record.itemsize], dtype=record)
    if not {"x", "y", "z"} <= set(structured.dtype.names or ()):
        return np.empty((0, 3))
    return np.stack(
        [structured["x"], structured["y"], structured["z"]], axis=1
    ).astype(float)
