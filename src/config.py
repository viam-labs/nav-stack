"""Typed configuration objects parsed from Viam component attributes.

Keeping these as plain dataclasses (no Viam SDK imports) makes the parsing
logic easy to unit-test and shareable between the models and the runtime layer.
"""
from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional

DIFFERENTIAL = "differential"
OMNI = "omni"
KINEMATICS = {DIFFERENTIAL, OMNI}

MODE_MAPPING = "mapping"
MODE_LOCALIZING = "localizing"
SLAM_MODES = {MODE_MAPPING, MODE_LOCALIZING}

BASE_VELOCITY_VIAM = "viam"
# Body frame with +X forward, +Y left (same as classic ROS cmd_vel).
BASE_VELOCITY_X_FORWARD = "x_forward"
# Legacy aliases accepted in config.
BASE_VELOCITY_ROS = "ros"  # alias of x_forward
BASE_VELOCITY_MIR = "mir"  # alias of viam (Y-forward)
BASE_VELOCITY_CONVENTIONS = {
    BASE_VELOCITY_VIAM,
    BASE_VELOCITY_X_FORWARD,
    BASE_VELOCITY_ROS,
    BASE_VELOCITY_MIR,
}
# Conventions that put body forward (vx) on Viam ``linear.y``.
BASE_VELOCITY_Y_FORWARD = {BASE_VELOCITY_VIAM, BASE_VELOCITY_MIR}

LIDAR_SCAN_AUTO = "auto"
LIDAR_SCAN_GET_LASER_SCAN = "get_laser_scan"
LIDAR_SCAN_POINT_CLOUD = "point_cloud"
LIDAR_SCAN_SOURCES = {
    LIDAR_SCAN_AUTO,
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
}

# Point-cloud axis convention *before* the mount transform.
# ``sensor``: X forward, Y left, Z up (RPLIDAR / Livox body).
# ``camera_optical``: X right, Y down, Z forward (RealSense / OpenCV optical).
CLOUD_FRAME_SENSOR = "sensor"
CLOUD_FRAME_CAMERA_OPTICAL = "camera_optical"
CLOUD_FRAMES = {CLOUD_FRAME_SENSOR, CLOUD_FRAME_CAMERA_OPTICAL}


def default_lidar_shm_name(component_name: str) -> str:
    """Match ``viam-labs:nav-stack:rplidar`` default writer name."""
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "", str(component_name)) or "lidar"
    return f"/viam-pc-{slug}"


def default_imu_shm_name(component_name: str) -> str:
    """Match ``viam-labs:nav-stack:wit-imu`` default writer name."""
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "", str(component_name)) or "imu"
    return f"/viam-imu-{slug}"


IMU_ODOM_COAST = "coast"
IMU_ODOM_ACCEL_ONLY = "accel_only"
IMU_ODOM_NONE = "none"
IMU_ODOM_MODES = {IMU_ODOM_COAST, IMU_ODOM_ACCEL_ONLY, IMU_ODOM_NONE}


def body_linear_to_viam_mm_s(
    vx_mps: float,
    vy_mps: float,
    convention: str = BASE_VELOCITY_VIAM,
) -> tuple[float, float]:
    """Convert body-frame linear speeds (m/s) to Viam base ``SetVelocity`` mm/s.

    Default ``viam`` convention (also ``mir``): Viam ``linear.y`` = body forward
    (``vx``), Viam ``linear.x`` = body lateral (``vy``). Matches ``rdk:builtin:wheeled``
    and MiR bases, which only drive on Y.

    ``x_forward`` / ``ros``: Viam ``linear.x`` = body forward, ``linear.y`` = lateral —
    for bases that consume X as forward.
    """
    if convention in BASE_VELOCITY_Y_FORWARD:
        vx_mps, vy_mps = vy_mps, vx_mps
    return vx_mps * 1000.0, vy_mps * 1000.0


def body_vtheta_to_viam_angular_deg_s(vtheta_rad_s: float) -> float:
    """Convert body angular rate (rad/s) to Viam ``Base.SetVelocity`` deg/s.

    Viam's protobuf / Python SDK document ``angular.z`` as degrees per second.
    Controllers and builtin nav keep rad/s internally
    (``max_vel_theta``, ``min_cmd_vel_theta``, ``cmd_vtheta_rad_s``); convert
    only at the SetVelocity call site — never change those config/status units.
    """
    import math

    return math.degrees(float(vtheta_rad_s))


def body_twist_to_viam_set_velocity(
    vx_mps: float,
    vy_mps: float,
    vtheta_rad_s: float,
    convention: str = BASE_VELOCITY_VIAM,
) -> tuple[float, float, float]:
    """Body twist → Viam SetVelocity ``(linear.x_mm_s, linear.y_mm_s, angular.z_deg_s)``."""
    lx, ly = body_linear_to_viam_mm_s(vx_mps, vy_mps, convention)
    return lx, ly, body_vtheta_to_viam_angular_deg_s(vtheta_rad_s)


def sensor_twist_to_body(
    vx: float,
    vy: float,
    convention: str = BASE_VELOCITY_VIAM,
) -> tuple[float, float]:
    """Map sensor-native body linear twist to nav body (x forward, y left).

    ``viam`` / ``mir``: sensor +y is forward, +x is right → body ``(vy, -vx)``.
    ``x_forward`` / ``ros``: already X-forward / Y-left — pass through.
    """
    if convention in BASE_VELOCITY_Y_FORWARD:
        return float(vy), -float(vx)
    return float(vx), float(vy)


def viam_set_velocity_to_body_twist(
    linear_x_mm_s: float,
    linear_y_mm_s: float,
    angular_z_deg_s: float,
    convention: str = BASE_VELOCITY_VIAM,
) -> tuple[float, float, float]:
    """Inverse of ``body_twist_to_viam_set_velocity`` (mm/s + deg/s → body m/s + rad/s)."""
    import math

    lx = float(linear_x_mm_s) / 1000.0
    ly = float(linear_y_mm_s) / 1000.0
    if convention in BASE_VELOCITY_Y_FORWARD:
        # Forward was packed on Viam linear.y; lateral on linear.x.
        vx_mps, vy_mps = ly, lx
    else:
        vx_mps, vy_mps = lx, ly
    return vx_mps, vy_mps, math.radians(float(angular_z_deg_s))


@dataclass
class SimConfig:
    """Builtin simulation (raycast world + ``sim-base``), nested under slam ``sim``."""

    enabled: bool = False
    # Shared registry key so ``sim-base`` and SLAM ``SimSensors`` find one world.
    world_name: str = "default"
    # Optional ``.npy`` (+ sibling ``.json`` meta). Empty → built-in L-corridor.
    map_path: Optional[str] = None
    seed_x: float = 1.0
    seed_y: float = 1.0
    seed_theta: float = 0.0
    scan_bins: int = 360
    range_min: float = 0.05
    range_max: float = 20.0
    # Velocity convention for decoding Base.SetVelocity into SimWorld (match slam).
    base_velocity_convention: str = BASE_VELOCITY_VIAM

    @classmethod
    def from_dict(cls, d: Mapping) -> "SimConfig":
        if not d:
            return cls()
        convention = d.get("base_velocity_convention", BASE_VELOCITY_VIAM)
        if convention not in BASE_VELOCITY_CONVENTIONS:
            raise ValueError(
                f"sim.base_velocity_convention must be one of "
                f"{sorted(BASE_VELOCITY_CONVENTIONS)}"
            )
        if convention == BASE_VELOCITY_MIR:
            convention = BASE_VELOCITY_VIAM
        if convention == BASE_VELOCITY_ROS:
            convention = BASE_VELOCITY_X_FORWARD
        map_path = d.get("map_path")
        overrides: Dict[str, Any] = {
            "base_velocity_convention": str(convention),
            "map_path": str(map_path).strip() or None if map_path else None,
        }
        if "world_name" in d:
            overrides["world_name"] = str(d.get("world_name") or "default")
        return _dataclass_from_dict(cls, d, overrides=overrides)


@dataclass
class LidarConfig:
    """A single lidar and its mount transform (base_link -> laser_N)."""

    name: str
    x: float = 0.0  # meters, in base_link
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0  # radians, yaw in base_link
    # Mount tilt (radians). Positive pitch = sensor forward axis tilted down.
    # Levels the cloud before z filtering; a ~2 deg mast tilt is enough to pull
    # floor returns into the z band at 15-20 m (phantom borders at max range).
    pitch: float = 0.0
    roll: float = 0.0
    min_range: float = 0.05  # meters
    max_range: float = 25.0  # meters
    # Height band for 3D lidars / depth cameras when ``get_point_cloud`` returns
    # points in ``base_link`` (Z = height above the floor). Keeps floor/ceiling
    # out of the 2D scan fed to builtin SLAM.
    z_min: float = -0.2
    z_max: float = 2.0
    # How to read this lidar: ``auto`` tries mir-base-style ``get_laser_scan``
    # then falls back to ``get_point_cloud``; use ``point_cloud`` for Livox /
    # depth cameras that only expose ``NextPointCloud``.
    scan_source: str = LIDAR_SCAN_AUTO
    # Set true when ``get_point_cloud`` already returns ``base_link`` points
    # (skip the mount transform — avoids double-offset on some Livox setups).
    points_in_base_link: bool = False
    # Optional POSIX shm object (e.g. ``/viam-pc-lidar``) in the
    # viam-shared-memory-test double-buffer layout. When set, scan paths
    # (bridge + builtin ViamWorldIO) try shm before ``get_point_cloud``.
    # Omitted → default ``/viam-pc-<name>`` (matches nav-stack rplidar). Set
    # ``shm_name: ""`` to disable shm and force gRPC.
    shm_name: Optional[str] = None
    shm_region_size: int = 2 * 1024 * 1024
    # If true, never fall back to gRPC GetPointCloud when shm is empty/missing.
    shm_required: bool = False
    # When true, this sensor feeds nav obstacle avoidance / local costmap only —
    # it is excluded from SLAM scan-matching and map updates. Use for a short-
    # range depth camera alongside a real lidar.
    obstacles_only: bool = False
    # Axis convention of ``get_point_cloud`` *before* mount. RealSense / OpenCV
    # depth clouds are ``camera_optical`` (Z forward); treating them as
    # ``sensor`` (X forward) collapses depth into height and paints a blob on
    # the robot in the local costmap.
    cloud_frame: str = CLOUD_FRAME_SENSOR

    @classmethod
    def from_dict(cls, d: Mapping) -> "LidarConfig":
        if isinstance(d, str):
            name = d
            return cls(name=name, shm_name=default_lidar_shm_name(name))
        mount = d.get("mount", {}) or {}
        scan_source = str(d.get("scan_source", LIDAR_SCAN_AUTO))
        if scan_source not in LIDAR_SCAN_SOURCES:
            raise ValueError(
                f"lidar scan_source must be one of {sorted(LIDAR_SCAN_SOURCES)}"
            )
        cloud_frame = str(d.get("cloud_frame", CLOUD_FRAME_SENSOR)).strip().lower()
        if cloud_frame not in CLOUD_FRAMES:
            raise ValueError(
                f"lidar cloud_frame must be one of {sorted(CLOUD_FRAMES)}"
            )
        name = d["name"]
        if "shm_name" in d:
            raw = d.get("shm_name")
            shm_name_s = str(raw).strip() if raw else ""
            shm_name = shm_name_s or None
        else:
            shm_name = default_lidar_shm_name(str(name))
        region = int(
            d.get("shm_region_size", d.get("shm_region_size_bytes", 2 * 1024 * 1024))
        )
        if region <= 0 or region % 2 != 0:
            raise ValueError("lidar shm_region_size must be a positive even byte count")
        return cls(
            name=name,
            x=float(mount.get("x", d.get("x", 0.0))),
            y=float(mount.get("y", d.get("y", 0.0))),
            z=float(mount.get("z", d.get("z", 0.0))),
            theta=float(mount.get("theta", d.get("theta", 0.0))),
            pitch=float(mount.get("pitch", d.get("pitch", 0.0))),
            roll=float(mount.get("roll", d.get("roll", 0.0))),
            min_range=float(d.get("min_range", 0.05)),
            max_range=float(d.get("max_range", 25.0)),
            z_min=float(d.get("z_min", -0.2)),
            z_max=float(d.get("z_max", 2.0)),
            scan_source=scan_source,
            points_in_base_link=bool(d.get("points_in_base_link", False)),
            shm_name=shm_name,
            shm_region_size=region,
            shm_required=bool(d.get("shm_required", False)),
            obstacles_only=bool(d.get("obstacles_only", False)),
            cloud_frame=cloud_frame,
        )


@dataclass
class MapSettings:
    """Map / scan settings used by builtin SLAM (resolution, range, travel gates).

    Config attribute is ``map``. Common knobs ``resolution`` and
    ``max_laser_range`` may also be set at the SLAM service top level.
    The legacy attribute name ``slam_toolbox`` is still accepted in
    ``from_dict`` for existing machine configs.
    """

    resolution: float = 0.05  # meters/cell
    minimum_travel_distance: float = 0.3  # meters before adding a new scan
    minimum_travel_heading: float = 0.3  # radians before adding a new scan
    max_laser_range: float = 25.0  # meters

    @classmethod
    def from_dict(cls, d: Mapping) -> "MapSettings":
        if not d:
            return cls()
        return _dataclass_from_dict(cls, d)


NAV_BACKEND_BUILTIN = "builtin"
NAV_BACKENDS = frozenset({NAV_BACKEND_BUILTIN})

SLAM_BACKEND_BUILTIN = "builtin"
SLAM_BACKENDS = frozenset({SLAM_BACKEND_BUILTIN})

_REMOVED_BACKEND_HINT = (
    "Legacy backends were removed; use slam_backend/nav_backend "
    "'builtin' only. For the last ROS-based release, check out git tag "
    "pre-ros-removal."
)

# Common nav tuning accepted at the service top level (also under ``builtin``).
_TOP_LEVEL_NAV_TUNING_KEYS = ("xy_goal_tolerance", "yaw_goal_tolerance")


def _positive_hz(value, name: str) -> float:
    hz = float(value)
    if hz <= 0.0:
        raise ValueError(f"{name} must be > 0, got {hz}")
    return hz


def _optional_positive(value) -> Optional[float]:
    """Parse an optional positive dimension; absent / non-positive -> None."""
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0.0 else None


def _config_field_defaults(cls) -> Dict[str, Any]:
    """Field name → default value (evaluates ``default_factory`` when needed)."""
    out: Dict[str, Any] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
    return out


def _coerce_config_value(default: Any, raw: Any) -> Any:
    """Cast ``raw`` to the type implied by the dataclass field default."""
    if isinstance(default, bool):
        return bool(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, str):
        return default if raw is None else str(raw)
    if isinstance(default, Mapping):
        return dict(raw) if raw else dict(default)
    if isinstance(default, list):
        return list(raw) if raw is not None else list(default)
    return raw


def _dataclass_from_dict(
    cls,
    d: Mapping,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
):
    """Build ``cls`` from ``d``; missing keys keep dataclass field defaults.

    ``overrides`` supplies values that need custom parsing (normalization,
    validators, or context-dependent defaults). Keys present only in ``d`` are
    coerced from the field default's type.
    """
    overrides = dict(overrides or {})
    defs = _config_field_defaults(cls)
    kwargs: Dict[str, Any] = dict(overrides)
    for name, default in defs.items():
        if name in kwargs:
            continue
        if name not in d:
            continue
        kwargs[name] = _coerce_config_value(default, d[name])
    return cls(**kwargs)


def _merge_top_level_nav_tuning(d: Mapping) -> dict:
    """Build the ``builtin`` attribute dict: nested block wins over top-level."""
    merged = {
        key: d[key] for key in _TOP_LEVEL_NAV_TUNING_KEYS if key in d
    }
    nested = d.get("builtin") or d.get("nav2") or {}
    merged.update(dict(nested))
    return merged

BUILTIN_PLANNER_ASTAR = "astar"
BUILTIN_PLANNER_LAZY_THETA = "lazy_theta_star"
BUILTIN_PLANNERS = frozenset({BUILTIN_PLANNER_ASTAR, BUILTIN_PLANNER_LAZY_THETA})


def normalize_builtin_planner(name: Optional[str]) -> str:
    """Map config aliases onto ``astar`` / ``lazy_theta_star``."""
    if not name:
        return BUILTIN_PLANNER_LAZY_THETA
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "astar": BUILTIN_PLANNER_ASTAR,
        "a_star": BUILTIN_PLANNER_ASTAR,
        "gridbased": BUILTIN_PLANNER_ASTAR,
        "builtinastar": BUILTIN_PLANNER_ASTAR,
        "lazy_theta_star": BUILTIN_PLANNER_LAZY_THETA,
        "lazy_theta*": BUILTIN_PLANNER_LAZY_THETA,
        "lazythetastar": BUILTIN_PLANNER_LAZY_THETA,
        "theta_star": BUILTIN_PLANNER_LAZY_THETA,
        "thetastar": BUILTIN_PLANNER_LAZY_THETA,
    }
    if key in aliases:
        return aliases[key]
    if key in BUILTIN_PLANNERS:
        return key
    raise ValueError(
        f"builtin.planner must be one of {sorted(BUILTIN_PLANNERS)} "
        f"(aliases: LazyThetaStar, AStar); got {name!r}"
    )


@dataclass
class BuiltinNavConfig:
    """Tuning for the in-module navigator.

    Footprint / velocity limits stay top-level on ``NavConfig``. Defaults are
    tuned for builtin SLAM + pure pursuit.

    Config attribute is ``builtin``. Common knobs ``xy_goal_tolerance`` and
    ``yaw_goal_tolerance`` may also be set at the navigation service top
    level. The legacy attribute name ``nav2`` is still accepted in
    ``NavConfig.from_dict`` for existing machine configs.
    """

    # ``lazy_theta_star`` (default) or ``astar``.
    planner: str = BUILTIN_PLANNER_LAZY_THETA
    # Regulated pure pursuit lookahead: velocity-scaled and clamped to
    # [min, max]; ``lookahead_m`` is the fallback when starting from rest.
    # Longer = smoother / less sensitive to SLAM pose jitter but cuts corners
    # more; shorter = tighter tracking. Soft-loc holds + pose noise still
    # hunt below ~1.1 m on skid-steer, so defaults sit a bit longer.
    lookahead_m: float = 1.35
    min_lookahead_m: float = 1.1
    max_lookahead_m: float = 1.55
    replan_period_s: float = 1.0
    timeout_s: float = 300.0
    # Base.SetVelocity wait on the shared module event loop. Mapping+SLAM can
    # briefly starve the loop; 2s was aborting goals on capable hardware.
    drive_timeout_s: float = 5.0
    # Consecutive SetVelocity timeouts before aborting the goal.
    drive_timeout_streak: int = 20
    cost_scaling_factor: float = 4.0
    # Extra planning clearance past inflation_radius (not drawn as soft inflation).
    clearance_preference_m: float = 0.50
    xy_goal_tolerance: float = 0.25  # meters
    yaw_goal_tolerance: float = 0.35  # radians (~20 deg; mugger uses 0.6)
    # After XY is inside tolerance, accept the goal if final yaw still has not
    # settled (noisy heading / goal θ far from approach). 0 disables.
    yaw_align_timeout_s: float = 12.0
    # Reject plans whose free-cell goal snap exceeds this (metres). Live scan
    # inflation used to snap the goal ~1 m away and then "succeed" there.
    max_goal_snap_m: float = 0.5
    # Final approach: cap linear speed within this distance of the goal.
    approach_dist_m: float = 0.35
    # Post-process global plans (shortcut + resample) before following.
    # Coarser densify: cell-scale jogs feed pure-pursuit κ flicker on straights.
    smooth_path: bool = True
    smooth_sample_spacing_m: float = 0.20
    # Rolling local costmap + DWA-style local planner for dynamic obstacles.
    local_costmap_enabled: bool = True
    local_costmap_width_m: float = 4.0
    local_costmap_height_m: float = 4.0
    local_costmap_resolution: float = 0.05
    # Soft outer radius for live scan hits. Absolute (legacy); unset means the
    # footprint alone, which is what the local planner has always used.
    local_inflation_radius_m: Optional[float] = None
    # Additive band past the footprint for live hits (preferred spelling).
    local_inflation_margin_m: Optional[float] = None
    # Local-window refresh rate (Hz). Independent of ``control_rate_hz`` so the
    # follower tick stays cheap; lidar is typically ~10 Hz anyway.
    local_costmap_rate_hz: float = 5.0
    local_planner_enabled: bool = True
    local_planner_sim_time_s: float = 1.5
    local_planner_activate_cost: int = 200
    local_planner_max_vel_x_mps: float = 0.25
    local_planner_max_vel_x_reverse_m: float = 0.15
    # Backup when local planner spins in place with clear rear space.
    backup_enabled: bool = True
    backup_stuck_time_s: float = 3.0
    backup_dist_m: float = 0.30
    backup_speed_mps: float = 0.12
    backup_rear_clear_m: float = 0.45
    backup_max_attempts: int = 1
    backup_cooldown_s: float = 4.0
    # Wait: stop and wait for a dynamic blocker to clear before
    # the first local replan (people crossing). Same grace is used when the
    # nose is clear to prefer DWA on the short path before escalating.
    recovery_wait_duration_s: float = 2.0
    # Legacy grace before local replan; effective wait / DWA grace is
    # max(recovery_wait_duration_s, replan_local_blocked_time_s).
    replan_local_blocked_time_s: float = 0.3
    # Cooldown begins when a blocking plan finishes. Give the local planner
    # time to execute the peel instead of stop/replanning every control tick.
    replan_local_min_period_s: float = 4.0
    # Command slew limits (the base has no onboard ramp). Requests to stop
    # translating are never slewed, so stop distances are unaffected.
    max_linear_accel_mps2: float = 0.8
    max_linear_decel_mps2: float = 1.2
    max_angular_accel_rad_s2: float = 2.0
    # After each route waypoint succeeds (robot still), run SLAM
    # ``check_localization`` before departing for the next leg. Mid-nav
    # periodic relocalize defaults off, so drift otherwise rides until idle.
    route_verify_pose: bool = True

    @classmethod
    def from_dict(cls, d: Mapping) -> "BuiltinNavConfig":
        if not d:
            return cls()
        overrides: Dict[str, Any] = {
            "local_inflation_radius_m": _optional_positive(
                d.get("local_inflation_radius_m")
            ),
            "local_inflation_margin_m": _optional_positive(
                d.get("local_inflation_margin_m")
            ),
        }
        if "planner" in d:
            overrides["planner"] = normalize_builtin_planner(d.get("planner"))
        if "local_costmap_rate_hz" in d:
            overrides["local_costmap_rate_hz"] = _positive_hz(
                d["local_costmap_rate_hz"], "local_costmap_rate_hz"
            )
        return _dataclass_from_dict(cls, d, overrides=overrides)


@dataclass
class Frames:
    map: str = "map"
    odom: str = "odom"
    base_link: str = "base_link"


@dataclass
class SlamConfig:
    base: str
    lidars: List[LidarConfig]
    movement_sensor: Optional[str] = None
    # Optional POSIX shm published by ``viam-labs:nav-stack:wit-imu`` (e.g.
    # ``/viam-imu-wit``). When set, builtin SLAM prefers this over gRPC
    # ``get_readings`` — same pattern as lidar ``shm_name``.
    imu_shm_name: Optional[str] = None
    imu_shm_region_size: int = 4096
    imu_shm_max_age_s: float = 0.5
    # Optional IMU (or other) heading source. When set alongside wheel odometry
    # on ``movement_sensor``, yaw comes from here while translation comes from
    # the movement sensor — useful for skid-steer bases where wheel slip skews
    # turn odometry but straight-line encoder velocity is still useful.
    heading_sensor: Optional[str] = None
    # Yaw (degrees) of the movement sensor's +x axis relative to the robot's
    # forward axis. An IMU mounted rotated -90 deg about +z (its x pointing at
    # the robot's right side) needs -90 here so integrated accel and reported
    # yaw line up with base_link. Applied to velocity/accel vectors and yaw.
    movement_sensor_yaw_deg: float = 0.0
    # Set true when the movement sensor is mounted upside down (flipped about
    # x): yaw, yaw rate, and lateral accel all read with inverted sign. The
    # telltale is the map rotating opposite to the robot (room stamped in a
    # circle while spinning in place). Applied before movement_sensor_yaw_deg.
    movement_sensor_upside_down: bool = False
    # Same mount-yaw correction for the dedicated heading sensor (subtracted
    # from its reported yaw). Set this when heading comes from an IMU that is
    # physically rotated on the chassis.
    heading_sensor_yaw_deg: float = 0.0
    # Negate the dedicated heading sensor's yaw (upside-down heading IMU).
    heading_sensor_invert: bool = False
    # Added to GetPosition yaw only (App arrow vs map PCD). Prefer fixing lidar
    # ``mount.theta`` (see status probe
    # ``nearest_return_bearing_deg``) — a cosmetic ±45 rarely means TF and PCD
    # disagree; more often the Livox +X is off base_link forward.
    map_pose_yaw_offset_deg: float = 0.0
    mode: str = MODE_MAPPING
    # Builtin occupancy SLAM only.
    slam_backend: str = SLAM_BACKEND_BUILTIN
    maps_dir: str = "/root/.viam/nav-stack/maps"
    active_map: Optional[str] = None
    frames: Frames = field(default_factory=Frames)
    # Builtin SLAM tick rate is ``max(scan_rate_hz, odom_rate_hz)`` (scan + odom
    # are read each tick). Defaults keep the historical ~10 Hz loop. Scan
    # matching stays throttled separately (``match_period_s`` ≈ 0.3 s).
    scan_rate_hz: float = 10.0
    odom_rate_hz: float = 10.0
    sensor_read_timeout_s: float = 10.0
    # External-SLAM navigation (navigation-external) poll rates (unused by builtin
    # slam; retained for config compatibility).
    external_pose_rate_hz: float = 10.0
    external_grid_rate_hz: float = 1.5
    external_transform_timeout_s: float = 0.2
    scan_bins: int = 720
    # Merge recent point-cloud frames before /scan projection. Livox Mid-360 and
    # similar non-repetitive 3D lidars need this for stable SLAM input.
    scan_accumulation_s: float = 0.0
    # How IMU dead-reckons forward velocity when wheel encoders are absent.
    # ``accel_only`` integrates only on clear forward accel (Livox carts);
    # ``coast`` keeps velocity at steady speed; ``none`` is yaw-only odom.
    imu_odom_mode: str = IMU_ODOM_COAST
    # Deprecated alias for ``imu_odom_mode=none``.
    heading_only_odom: bool = False
    # Scan-to-scan lidar odometry; Livox uses loose range-flow hints only.
    lidar_odom_enabled: bool = True
    lidar_odom_range_flow_only: bool = False
    # Stop-and-go mapping bias: only publish /scan after the robot has been
    # still for ``map_when_still_dwell_s``, once per stop. For IMU + Livox carts
    # this yields dense scans and lets SLAM match between pauses.
    # Leave false for MiR / wheel-odom robots (continuous mapping).
    map_when_still: bool = False
    map_when_still_dwell_s: float = 1.0
    map_when_still_linear_speed_m_s: float = 0.02
    map_when_still_yaw_rate_rad_s: float = 0.04
    # Mid-pivot scans every N degrees. Default 0 = only publish when fully
    # stopped (strict stop-and-go; safest against ghost walls with IMU odom).
    map_when_still_yaw_step_deg: float = 0.0
    # Abort dwell if pose drifts more than this while "still" (m / deg).
    map_when_still_max_drift_m: float = 0.03
    map_when_still_max_drift_deg: float = 1.5
    # Soft-correct odom yaw from a long side wall in the pause scan (anti-banana
    # for IMU gyro drift along straight walls). Default on with map_when_still
    # + point-cloud lidars.
    wall_yaw_correction: bool = False
    wall_yaw_min_length_m: float = 2.0
    wall_yaw_max_step_deg: float = 2.0
    wall_yaw_blend: float = 0.5
    # Mapping-time revisit check: periodically match the live scan against the
    # live map near the current pose and, on a strong match that disagrees with
    # odom, shift the odom TF so localization links back to the original area
    # instead of mapping a duplicate corridor. Tiered search: local radius
    # first, wider radius on weak match, full map only as a last resort with a
    # stricter score gate. Default on with map_when_still + point-cloud lidars.
    mapping_revisit_check: bool = False
    mapping_revisit_interval_s: float = 20.0
    mapping_revisit_search_radius_m: float = 5.0
    mapping_revisit_wide_radius_m: float = 12.0
    mapping_revisit_min_score: float = 0.6
    mapping_revisit_max_ray_mae_m: float = 0.8
    # Correct only when the match is meaningfully away from the current pose
    # (below = normal jitter) and not absurdly far (above = likely false match).
    mapping_revisit_min_shift_m: float = 1.0
    mapping_revisit_min_shift_deg: float = 10.0
    mapping_revisit_max_shift_m: float = 10.0
    # Full-map fallback needs a stronger score: self-similar offices produce
    # convincing wrong corridors at map scale.
    mapping_revisit_full_map_fallback: bool = True
    mapping_revisit_full_map_min_score: float = 0.75
    # Multi-height-slice verification: accumulate sparse per-band grids from
    # pause scans (3D lidar only) and veto a revisit correction whose pose
    # disagrees with any band that has reference data there. The occupancy map
    # only holds the primary z-band silhouette — desk clutter is self-similar
    # in that band, but head-height structure rarely is.
    mapping_revisit_slice_verify: bool = True
    # Extra bands beyond the primary scan band, [z_min, z_max] in base_link
    # meters. Defaults: knee band below typical desk clutter, head band above.
    mapping_revisit_slice_bands: List = field(
        default_factory=lambda: [[0.15, 0.45], [1.6, 2.4]]
    )
    mapping_revisit_slice_min_hit_rate: float = 0.4
    mapping_revisit_slice_resolution_m: float = 0.15
    # Pause keyframes: store 2D endpoints + height slices on every accepted
    # map_when_still /scan publish, then match against them when occupancy
    # revisit scores are weak (different stop pose/angle than first visit).
    mapping_revisit_keyframes: bool = True
    mapping_revisit_keyframe_min_spacing_m: float = 0.5
    mapping_revisit_keyframe_min_spacing_deg: float = 20.0
    mapping_revisit_keyframe_max: int = 250
    mapping_revisit_keyframe_match_tol_m: float = 0.3
    mapping_revisit_keyframe_min_score: float = 0.55
    # Builtin SLAM: rebuild the occupancy grid from stored scan keyframes when
    # mapping_revisit corrects pose drift (eliminates duplicate corridor smear).
    builtin_rebuild_map_on_revisit: bool = True
    builtin_mapping_keyframe_max: int = 500
    # Allow mapping revisit / loop-closure corrections while the robot is
    # moving. Default on for continuous-scan backends (builtin); off when
    # map_when_still is on, where odom TF shifts mid-hop are unsafe.
    mapping_revisit_while_moving: bool = False
    # Even with while_moving, skip corrections above this yaw rate — spinning
    # motion-distorts the scan and hallway matches are unreliable.
    mapping_revisit_max_yaw_rate_rad_s: float = 0.35
    # Safety cutoff for the SLAM/publish scan path: if mir-base reports a scan
    # cache age (get_laser_scan ``age_s``) above this, the bridge skips publishing
    # it rather than feeding SLAM a misregistered scan. Accurate age-based
    # stamping handles normal latency; this only guards genuinely stale data.
    scan_max_age_s: float = 2.0
    # How body twist (vx forward, vy lateral) maps to the Viam base SetVelocity axes.
    base_velocity_convention: str = BASE_VELOCITY_VIAM
    map: MapSettings = field(default_factory=MapSettings)
    slam_params: Mapping = field(default_factory=dict)
    # Automatically run global_localize shortly after starting in localizing mode.
    global_localize_on_start: bool = True
    global_localize_on_start_delay_s: float = 4.0
    # Max time to wait for slam + scans + map to become usable before the
    # startup auto-localize runs (slow networks can take tens of seconds).
    global_localize_on_start_readiness_timeout_s: float = 90.0
    global_localize_on_start_options: Mapping = field(
        default_factory=lambda: {
            "full_map": True,
            "map_source": "live",
            # Finer than the generic full-map defaults — office maps need it
            # to keep the true pose in the coarse winner set.
            "coarse_position_step_m": 0.35,
            "coarse_yaw_step_deg": 10.0,
            "ray_weight": 0.55,
            "ray_refine_candidates": 48,
        }
    )
    global_localize_on_start_refine: bool = True
    global_localize_on_start_refine_delay_s: float = 8.0
    global_localize_on_start_refine_max_passes: int = 3
    global_localize_on_start_target_score: float = 0.7
    global_localize_on_start_target_ray_mae_m: float = 0.4
    global_localize_on_start_post_apply_refine: bool = True
    global_localize_on_start_post_apply_refine_delay_s: float = 8.0
    global_localize_on_start_post_apply_refine_options: Mapping = field(
        default_factory=lambda: {"map_source": "live"}
    )
    global_localize_on_start_refine_options: Mapping = field(
        default_factory=lambda: {
            "full_map": False,
            "map_source": "live",
            "local_yaw_window_deg": 120.0,
            "search_radius_m": 6.0,
        }
    )
    # Periodic localization drift watchdog (localizing mode only). Runs a cheap
    # local scan-match on an interval and re-localizes when pose has drifted.
    periodic_relocalize: bool = True
    periodic_relocalize_interval_s: float = 20.0
    # While navigating, score less often so matches are not taken mid-whip on
    # motion-distorted scans (still frequent enough to catch soft loc).
    periodic_relocalize_nav_interval_s: float = 25.0
    # Soft loc with no large jump: hold drive briefly, then resume on the
    # published pose (odom continuity). 0 = hold until quality recovers.
    # Large-jump holds are unchanged (awaiting_confirm / nav_hold until clear).
    periodic_relocalize_soft_hold_max_s: float = 20.0
    # Skip a cycle when |yaw rate| is above this (rad/s) — spinning scans smear.
    periodic_relocalize_max_yaw_rate_rad_s: float = 0.35
    # Skip when the latest lidar age exceeds this (s). 0 disables. Matches the
    # builtin SLAM match path's tight age clamp.
    periodic_relocalize_max_scan_age_s: float = 0.75
    # Below this match score, or above this ray MAE (m), the local match is not
    # trusted; the watchdog then tries a full-map global_localize (like manual).
    # ray MAE default is deliberately generous: on real robots a correctly
    # localized pose still has ~0.5-0.9 m ray MAE (lidar noise, map resolution,
    # partial coverage), so a tighter gate makes the routine drift-correction path
    # never fire and drift only gets caught after nav degrades into a recovery.
    periodic_relocalize_min_score: float = 0.5
    periodic_relocalize_max_ray_mae_m: float = 1.0
    # In a recovery situation (full-map match because nav is failing or the local
    # match was low quality) the watchdog mirrors a manual global_localize: it
    # applies the best full-map match when its score clears this floor, ignoring
    # the ray_mae gate. This is what lets a genuinely-lost robot recover even when
    # the environment's baseline ray_mae is above periodic_relocalize_max_ray_mae_m.
    periodic_relocalize_recovery_min_score: float = 0.45
    periodic_relocalize_min_shift_m: float = 0.2
    periodic_relocalize_min_shift_deg: float = 10.0
    # Large automatic pose jumps (periodic relocalize, mapping revisit, seed /
    # startup localize) must agree across N matches before apply. Manual
    # ``relocalize`` and ``apply: true`` bypass this gate.
    localize_jump_confirm_count: int = 2
    localize_jump_agree_m: float = 0.4
    localize_jump_agree_deg: float = 15.0
    localize_jump_large_m: float = 0.75
    localize_jump_large_deg: float = 25.0
    # When navigation reports this many recoveries on the active goal, skip the
    # cheap local match and run full-map global_localize immediately.
    periodic_relocalize_nav_recoveries_threshold: int = 2
    periodic_relocalize_full_map_on_low_quality: bool = True
    periodic_relocalize_during_navigation: bool = False
    # If startup global_localize is still running this long (s), cancel it so
    # the drift watchdog can take over — otherwise a long refine loop leaves
    # the robot on a bad pose with status=skipped forever. 0 disables bypass.
    periodic_relocalize_bypass_startup_after_s: float = 90.0
    # While still, if the continuous tick match score is at/below this, go
    # straight to full-map global_localize (360° lidar already has the view;
    # spinning does not help). Default 0 catches "scan does not explain pose".
    periodic_relocalize_still_bad_score: float = 0.0
    # Run global_localize / ray scoring in a dedicated subprocess so the
    # matcher's Python loops do not hold this process's GIL (which starved the
    # nav control tick and wheel-odom reads). Falls back in-process on error.
    localize_subprocess: bool = True
    periodic_relocalize_options: Mapping = field(
        default_factory=lambda: {
            "full_map": False,
            "map_source": "live",
            "search_radius_m": 3.0,
            "auto_full_map_fallback": True,
        }
    )
    # Builtin simulation: raycast floorplan + in-process SimSensors (see ``sim``).
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_dict(cls, d: Mapping) -> "SlamConfig":
        sim = SimConfig.from_dict(d.get("sim", {}) or {})
        lidars_raw = d.get("lidars") or ([d["lidar"]] if d.get("lidar") else [])
        if not lidars_raw:
            if sim.enabled:
                # Placeholder lidar config for scan_bins/range; not a Viam Camera dep.
                lidars_raw = [
                    {
                        "name": "sim-lidar",
                        "min_range": sim.range_min,
                        "max_range": sim.range_max,
                    }
                ]
            else:
                raise ValueError("at least one lidar is required ('lidars' or 'lidar')")
        lidars = [LidarConfig.from_dict(x) for x in lidars_raw]
        slam_lidars = [lidar for lidar in lidars if not lidar.obstacles_only]
        if not slam_lidars:
            raise ValueError(
                "at least one lidar must have obstacles_only=false "
                "(needed for SLAM matching/mapping)"
            )
        mode = d.get("mode", MODE_MAPPING)
        if mode not in SLAM_MODES:
            raise ValueError(f"mode must be one of {sorted(SLAM_MODES)}")
        slam_backend = str(
            d.get("slam_backend", SLAM_BACKEND_BUILTIN) or SLAM_BACKEND_BUILTIN
        )
        if slam_backend in ("slam_toolbox", "toolbox"):
            raise ValueError(
                f"slam_backend={slam_backend!r} is no longer supported. {_REMOVED_BACKEND_HINT}"
            )
        if slam_backend not in SLAM_BACKENDS:
            raise ValueError(
                f"slam_backend must be {SLAM_BACKEND_BUILTIN!r}, got {slam_backend!r}"
            )
        if sim.enabled and slam_backend != SLAM_BACKEND_BUILTIN:
            raise ValueError(
                "sim.enabled requires slam_backend=builtin "
                f"(got {slam_backend!r})"
            )
        convention = d.get("base_velocity_convention", BASE_VELOCITY_VIAM)
        if convention not in BASE_VELOCITY_CONVENTIONS:
            raise ValueError(
                f"base_velocity_convention must be one of {sorted(BASE_VELOCITY_CONVENTIONS)}"
            )
        # Normalize legacy ``mir`` to the canonical Y-forward name.
        if convention == BASE_VELOCITY_MIR:
            convention = BASE_VELOCITY_VIAM
        if convention == BASE_VELOCITY_ROS:
            convention = BASE_VELOCITY_X_FORWARD
        # Prefer slam-level convention for the sim world when not overridden in sim{}.
        if "base_velocity_convention" not in (d.get("sim") or {}):
            sim = SimConfig(
                enabled=sim.enabled,
                world_name=sim.world_name,
                map_path=sim.map_path,
                seed_x=sim.seed_x,
                seed_y=sim.seed_y,
                seed_theta=sim.seed_theta,
                scan_bins=sim.scan_bins,
                range_min=sim.range_min,
                range_max=sim.range_max,
                base_velocity_convention=convention,
            )
        frames_d = d.get("frames", {}) or {}
        # Point-cloud SLAM defaults follow mapping sensors only — an
        # obstacles_only depth cam must not flip Livox-style tuning on/off.
        all_point_cloud = bool(slam_lidars) and all(
            lidar.scan_source == LIDAR_SCAN_POINT_CLOUD for lidar in slam_lidars
        )
        imu_odom_mode = str(
            d.get(
                "imu_odom_mode",
                IMU_ODOM_ACCEL_ONLY if all_point_cloud else IMU_ODOM_COAST,
            )
        )
        if imu_odom_mode not in IMU_ODOM_MODES:
            raise ValueError(
                f"imu_odom_mode must be one of {sorted(IMU_ODOM_MODES)}"
            )
        heading_only_odom = bool(d.get("heading_only_odom", False))
        if heading_only_odom:
            imu_odom_mode = IMU_ODOM_NONE
        # Prefer ``map``; accept legacy ``slam_toolbox`` block from older configs.
        # Top-level ``resolution`` / ``max_laser_range`` fill in when the nested
        # block omits them (nested wins on conflict).
        stb_raw = dict(d.get("map") or d.get("slam_toolbox") or {})
        for key in ("resolution", "max_laser_range"):
            if key in d and key not in stb_raw:
                stb_raw[key] = d[key]
        slam_params_raw = dict(d.get("slam_params", {}) or {})
        if all_point_cloud:
            max_lidar_range = max(lidar.max_range for lidar in slam_lidars)
            # Real travel gates matter for Livox: with minimum_travel_* at 0,
            # SLAM would otherwise scan-match every noisy non-repetitive frame while
            # parked and imprints walls at slightly different poses each time.
            stb_raw.setdefault("minimum_travel_distance", 0.15)
            stb_raw.setdefault("minimum_travel_heading", 0.12)
            stb_raw.setdefault("max_laser_range", max_lidar_range)
            slam_params_raw.setdefault("minimum_time_interval", 0.3)
            # Keep the correlation search modest: a wide window lets the
            # matcher jump between self-similar noise minima (ghost walls).
            slam_params_raw.setdefault("correlation_search_space_dimension", 0.6)
            slam_params_raw.setdefault("link_scan_maximum_distance", 2.5)
            if heading_only_odom:
                stb_raw["minimum_travel_distance"] = 0.0
                stb_raw["minimum_travel_heading"] = 0.0
        # Default off for MiR (get_laser_scan + wheel odom). Opt-in for Livox /
        # point-cloud carts that lack reliable translation odometry.
        map_when_still = bool(d.get("map_when_still", False))
        if map_when_still and all_point_cloud:
            # Scans arrive only at stops; the bridge already rate-limits.
            # Builtin SLAM uses odom as the scan-match prior —
            # ``use_odometry`` / ``use_tf_scan_transformation`` are NOT real
            # Legacy toolbox params were no-ops here. Widen the correlative angular
            # search: default coarse_search_angle_offset is only ~±20°, so a
            # 45–180° pivot between pauses imprints rotated ghost walls.
            user_sp = dict(d.get("slam_params", {}) or {})
            stb_raw["minimum_travel_distance"] = 0.0
            stb_raw["minimum_travel_heading"] = 0.0
            if "minimum_time_interval" not in user_sp:
                slam_params_raw["minimum_time_interval"] = 0.0
            if "correlation_search_space_dimension" not in user_sp:
                # ~1 m hops; 2.0 let sequential links latch onto neighbors.
                slam_params_raw["correlation_search_space_dimension"] = 1.0
            if "link_scan_maximum_distance" not in user_sp:
                slam_params_raw["link_scan_maximum_distance"] = 3.0
            if "link_match_minimum_response_fine" not in user_sp:
                # Reject weak false peaks (wide angular search ghosts rooms).
                slam_params_raw["link_match_minimum_response_fine"] = 0.25
            # Gyro provides the yaw prior — search ±~30°, not ±180°. Wider than
            # stock ±20° so inter-hop pivots still match; ±45°+ re-orients rooms.
            slam_params_raw["coarse_search_angle_offset"] = float(
                user_sp.get("coarse_search_angle_offset", 0.52)
            )
            slam_params_raw["coarse_angle_resolution"] = float(
                user_sp.get("coarse_angle_resolution", 0.0349)
            )
            slam_params_raw["use_response_expansion"] = bool(
                user_sp.get("use_response_expansion", True)
            )
            # Prefer stock-ish loop closure: chain_size=3 + 12 m search accepted
            # false corridors in self-similar desk spaces (ghost/warp maps).
            # Search a bit farther than stock 3 m for IMU XY drift, but keep
            # stock chain length and strong loop response thresholds.
            if "loop_match_minimum_chain_size" not in user_sp:
                slam_params_raw["loop_match_minimum_chain_size"] = 10
            if "loop_search_maximum_distance" not in user_sp:
                slam_params_raw["loop_search_maximum_distance"] = 5.0
            if "loop_search_space_dimension" not in user_sp:
                slam_params_raw["loop_search_space_dimension"] = 8.0
            if "loop_match_minimum_response_coarse" not in user_sp:
                slam_params_raw["loop_match_minimum_response_coarse"] = 0.35
            if "loop_match_minimum_response_fine" not in user_sp:
                slam_params_raw["loop_match_minimum_response_fine"] = 0.45
            if "do_loop_closing" not in user_sp:
                slam_params_raw["do_loop_closing"] = True
            if "angle_variance_penalty" not in user_sp:
                # Keep gyro prior meaningful; low values let matches flip rooms.
                slam_params_raw["angle_variance_penalty"] = 1.0
        map_when_still_dwell_s = float(d.get("map_when_still_dwell_s", 1.0))
        map_when_still_yaw_step_deg = float(d.get("map_when_still_yaw_step_deg", 0.0))
        default_accum = (
            max(0.6, map_when_still_dwell_s)
            if map_when_still and all_point_cloud
            else (0.3 if all_point_cloud and not heading_only_odom else 0.0)
        )
        lidar_odom_enabled = bool(
            d.get(
                "lidar_odom_enabled",
                all_point_cloud and not map_when_still,
            )
        )
        lidar_odom_range_flow_only = bool(
            d.get("lidar_odom_range_flow_only", all_point_cloud)
        )
        wall_yaw_correction = bool(
            d.get(
                "wall_yaw_correction",
                map_when_still and all_point_cloud,
            )
        )
        mapping_revisit_check = bool(
            d.get(
                "mapping_revisit_check",
                (map_when_still and all_point_cloud)
                or (
                    slam_backend == SLAM_BACKEND_BUILTIN and mode == MODE_MAPPING
                ),
            )
        )
        mapping_revisit_while_moving = bool(
            d.get(
                "mapping_revisit_while_moving",
                # Continuous scanning: safe to correct mid-drive. map_when_still
                # publishes sparse stop scans and shifts odom TF — keep parked.
                (not map_when_still)
                and (
                    slam_backend == SLAM_BACKEND_BUILTIN
                    or not all_point_cloud
                ),
            )
        )
        heading_sensor = d.get("heading_sensor")
        if "imu_shm_name" in d:
            raw_imu = d.get("imu_shm_name")
            imu_shm_name = str(raw_imu).strip() or None if raw_imu else None
        elif heading_sensor:
            # Prefer wit-imu POSIX shm over gRPC heading (keeps the shared
            # module event loop free for Base.SetVelocity during nav).
            imu_shm_name = default_imu_shm_name(str(heading_sensor))
        else:
            imu_shm_name = None
        frames_default = Frames()
        overrides: Dict[str, Any] = {
            "base": d["base"],
            "lidars": lidars,
            "movement_sensor": d.get("movement_sensor"),
            "imu_shm_name": imu_shm_name,
            "heading_sensor": heading_sensor,
            "mode": mode,
            "slam_backend": slam_backend,
            "active_map": d.get("active_map"),
            "frames": Frames(
                map=frames_d.get("map", frames_default.map),
                odom=frames_d.get("odom", frames_default.odom),
                base_link=frames_d.get("base_link", frames_default.base_link),
            ),
            "scan_accumulation_s": float(d.get("scan_accumulation_s", default_accum)),
            "imu_odom_mode": imu_odom_mode,
            "heading_only_odom": heading_only_odom,
            "lidar_odom_enabled": lidar_odom_enabled,
            "lidar_odom_range_flow_only": lidar_odom_range_flow_only,
            "map_when_still": map_when_still,
            "map_when_still_dwell_s": map_when_still_dwell_s,
            "map_when_still_yaw_step_deg": map_when_still_yaw_step_deg,
            "wall_yaw_correction": wall_yaw_correction,
            "mapping_revisit_check": mapping_revisit_check,
            "mapping_revisit_while_moving": mapping_revisit_while_moving,
            "base_velocity_convention": convention,
            "map": MapSettings.from_dict(stb_raw),
            "slam_params": slam_params_raw,
            # Sim seeds SLAM pose from sim.seed_*; startup global_localize is
            # optional and off by default (avoids fighting the ground-truth seed).
            "global_localize_on_start": bool(
                d.get("global_localize_on_start", not sim.enabled)
            ),
            "sim": sim,
        }
        if "scan_rate_hz" in d:
            overrides["scan_rate_hz"] = _positive_hz(d["scan_rate_hz"], "scan_rate_hz")
        if "odom_rate_hz" in d:
            overrides["odom_rate_hz"] = _positive_hz(d["odom_rate_hz"], "odom_rate_hz")
        if "mapping_revisit_slice_bands" in d:
            overrides["mapping_revisit_slice_bands"] = [
                [float(pair[0]), float(pair[1])]
                for pair in d["mapping_revisit_slice_bands"]
            ]
        # Empty mapping blocks fall back to field defaults (same as ``x or default``).
        defs = _config_field_defaults(cls)
        for key in (
            "global_localize_on_start_options",
            "global_localize_on_start_post_apply_refine_options",
            "global_localize_on_start_refine_options",
            "periodic_relocalize_options",
        ):
            if key in d:
                overrides[key] = d[key] or defs[key]
        return _dataclass_from_dict(cls, d, overrides=overrides)

    def required_dependencies(self) -> List[str]:
        if self.sim.enabled:
            # Lidars / movement sensors are in-process SimSensors; only Base
            # (typically viam-labs:nav-stack:sim-base) is a Viam dependency.
            return [self.base]
        deps = [self.base, *[lidar.name for lidar in self.lidars]]
        if self.movement_sensor:
            deps.append(self.movement_sensor)
        if self.heading_sensor:
            deps.append(self.heading_sensor)
        return deps

    def uses_sim(self) -> bool:
        return bool(self.sim.enabled)

    def slam_lidars(self) -> List[LidarConfig]:
        """Lidars used for SLAM matching/mapping (excludes ``obstacles_only``)."""
        return [lidar for lidar in self.lidars if not lidar.obstacles_only]

    def uses_builtin_slam(self) -> bool:
        return self.slam_backend == SLAM_BACKEND_BUILTIN

    def tick_rate_hz(self) -> float:
        """Builtin SLAM predict/update loop rate (Hz)."""
        return max(float(self.scan_rate_hz), float(self.odom_rate_hz))



@dataclass
class NavConfig:
    slam_service: str
    base: str
    kinematics: str = DIFFERENTIAL
    robot_radius: float = 0.22  # meters
    # Rectangular footprint (metres, optional). A single ``robot_radius`` has to
    # cover both driving and spinning, so it must be the half-diagonal — which
    # seals every gap narrower than 2·radius even when the robot easily fits
    # (0.59 m robot refusing an 0.84 m doorway). Given length+width, clearance
    # uses the half-width and rotation uses the half-diagonal instead.
    footprint_length_m: Optional[float] = None
    footprint_width_m: Optional[float] = None
    max_vel_x: float = 0.6  # m/s
    max_vel_y: float = 0.0  # m/s (omni only)
    max_vel_theta: float = 1.5  # rad/s
    acc_lim_x: float = 1.0
    acc_lim_theta: float = 2.0
    # Absolute soft-inflation outer radius, measured from the obstacle (Nav2
    # convention). Values at or below the footprint clearance radius add no soft
    # band at all — a silent no-op. Prefer ``inflation_margin_m``.
    inflation_radius: float = 0.25
    # Soft-inflation band width *past* the footprint (additive), matching how
    # ``clearance_preference_m`` is measured. Wins over ``inflation_radius``.
    inflation_margin_m: Optional[float] = None
    cmd_vel_timeout: float = 2.0  # seconds (watchdog)
    # Builtin nav control rate (Hz). Local costmap refreshes separately via
    # ``builtin.local_costmap_rate_hz`` (default 5) so follower ticks stay cheap.
    control_rate_hz: float = 10.0
    # Background refresh rate for ``obstacles_only`` depth cams (Hz). Nav never
    # awaits GetPointCloud on the control tick — this only throttles the
    # fire-and-forget refresh. Default 5 Hz (~0.2 s) keeps ankle-height depth
    # obstacles fresher on the gRPC path; prefer POSIX shm for 10–20 Hz.
    obstacles_only_rate_hz: float = 5.0
    # Reactive obstacle avoidance for simple go_to_* motion.
    simple_avoid_obstacles: bool = True
    simple_stop_distance: float = 0.4  # meters: stop forward + turn away inside this
    simple_slow_distance: float = 1.0  # meters: scale speed down inside this
    # Max scan age (s) still trusted for avoidance. Larger tolerates slower MiR
    # rosbridge lidar reads; too small makes avoidance fail closed (no drive).
    simple_scan_max_age: float = 2.0
    # Optional stiction floors (m/s and rad/s) for simple go_to_* motion only.
    min_cmd_vel_x: float = 0.0
    min_cmd_vel_theta: float = 0.0
    # Builtin navigator only.
    nav_backend: str = NAV_BACKEND_BUILTIN
    builtin: BuiltinNavConfig = field(default_factory=BuiltinNavConfig)

    @classmethod
    def from_dict(cls, d: Mapping) -> "NavConfig":
        kinematics = d.get("kinematics", DIFFERENTIAL)
        if kinematics not in KINEMATICS:
            raise ValueError(f"kinematics must be one of {sorted(KINEMATICS)}")
        backend = str(d.get("nav_backend", NAV_BACKEND_BUILTIN) or NAV_BACKEND_BUILTIN)
        if backend in ("nav2", "ros"):
            raise ValueError(
                f"nav_backend={backend!r} is no longer supported. {_REMOVED_BACKEND_HINT}"
            )
        if backend not in NAV_BACKENDS:
            raise ValueError(
                f"nav_backend must be {NAV_BACKEND_BUILTIN!r}, got {backend!r}"
            )
        defs = _config_field_defaults(cls)
        overrides: Dict[str, Any] = {
            "slam_service": d["slam_service"],
            "base": d["base"],
            "kinematics": kinematics,
            "footprint_length_m": _optional_positive(d.get("footprint_length_m")),
            "footprint_width_m": _optional_positive(d.get("footprint_width_m")),
            "inflation_margin_m": _optional_positive(d.get("inflation_margin_m")),
            "nav_backend": backend,
            # Prefer ``builtin``; accept legacy ``nav2`` block from older configs.
            # Top-level goal tolerances fill in when the nested block omits them
            # (nested wins on conflict).
            "builtin": BuiltinNavConfig.from_dict(_merge_top_level_nav_tuning(d)),
            # Legacy aliases: simple_min_vel_x / simple_min_vel_theta.
            "min_cmd_vel_x": float(
                d["min_cmd_vel_x"]
                if "min_cmd_vel_x" in d
                else d.get("simple_min_vel_x", defs["min_cmd_vel_x"])
            ),
            "min_cmd_vel_theta": float(
                d["min_cmd_vel_theta"]
                if "min_cmd_vel_theta" in d
                else d.get("simple_min_vel_theta", defs["min_cmd_vel_theta"])
            ),
        }
        if "control_rate_hz" in d:
            overrides["control_rate_hz"] = _positive_hz(
                d["control_rate_hz"], "control_rate_hz"
            )
        if "obstacles_only_rate_hz" in d:
            overrides["obstacles_only_rate_hz"] = _positive_hz(
                d["obstacles_only_rate_hz"], "obstacles_only_rate_hz"
            )
        return _dataclass_from_dict(cls, d, overrides=overrides)

    def inscribed_radius_m(self) -> float:
        """Clearance radius for *driving*: what has to fit through a gap."""
        if self.footprint_width_m:
            return max(0.01, float(self.footprint_width_m) / 2.0)
        return float(self.robot_radius)

    def circumscribed_radius_m(self) -> float:
        """Clearance radius for *rotating*: what the body sweeps turning in place."""
        if self.footprint_width_m and self.footprint_length_m:
            return math.hypot(
                float(self.footprint_length_m) / 2.0,
                float(self.footprint_width_m) / 2.0,
            )
        return max(float(self.robot_radius), self.inscribed_radius_m())

    def nose_offset_m(self) -> float:
        """Distance from the body centre to the bumper (forward stop distance)."""
        if self.footprint_length_m:
            return max(0.01, float(self.footprint_length_m) / 2.0)
        return float(self.robot_radius)

    def wheel_half_track_m(self) -> float:
        """Half the drive track, for the skid-steer arc envelope."""
        if self.footprint_width_m:
            return max(0.08, 0.9 * float(self.footprint_width_m) / 2.0)
        return max(0.08, 0.6 * float(self.robot_radius))

    def effective_inflation_radius_m(self) -> float:
        """Absolute soft-inflation outer radius the costmap should use."""
        if self.inflation_margin_m is not None:
            return self.inscribed_radius_m() + float(self.inflation_margin_m)
        return float(self.inflation_radius)

    def inflation_is_noop(self) -> bool:
        """True when the configured inflation adds no soft band at all."""
        return self.effective_inflation_radius_m() <= self.inscribed_radius_m() + 1e-6

    def effective_local_inflation_radius_m(self) -> float:
        """Absolute soft outer radius for *live* (scan) hits in the local costmap.

        Defaults to the footprint alone, which is what the local planner has
        always used: ``path_cost_ahead`` then means "the route is inside the
        footprint of a live return", not "near one".
        """
        inscribed = self.inscribed_radius_m()
        builtin = self.builtin
        if builtin.local_inflation_margin_m is not None:
            return inscribed + float(builtin.local_inflation_margin_m)
        if builtin.local_inflation_radius_m is not None:
            return max(inscribed, float(builtin.local_inflation_radius_m))
        return inscribed

    def control_period_s(self) -> float:
        """Seconds between builtin nav control ticks."""
        return 1.0 / float(self.control_rate_hz)

    def obstacles_only_period_s(self) -> float:
        """Seconds between background ``obstacles_only`` depth refreshes."""
        return 1.0 / float(self.obstacles_only_rate_hz)

    def uses_builtin_nav(self) -> bool:
        return True

    def required_dependencies(self) -> List[str]:
        return [self.slam_service, self.base]


@dataclass
class ExternalNavConfig:
    """Config for ``viam-labs:nav-stack:navigation-external``.

    Drives builtin navigation from an arbitrary Viam ``rdk:service:slam``.
    One flat attributes block yields both a sensor ``SlamConfig`` (base, lidars,
    movement sensor, odom tuning) and a ``NavConfig``; ``slam_service`` names the
    SLAM dependency.
    """

    slam_service: str
    bridge: SlamConfig
    nav: NavConfig
    # Trust the movement sensor's Position as an absolute odom pose. Off by
    # default: dead-reckoned IMU Position drifts (see odom_source.py).
    trust_movement_sensor_pose: bool = False
    # Snap yaw from the movement sensor's Orientation instead of integrating gyro.
    snap_heading: bool = False

    @classmethod
    def from_dict(cls, d: Mapping) -> "ExternalNavConfig":
        return cls(
            slam_service=d["slam_service"],
            bridge=SlamConfig.from_dict(d),
            nav=NavConfig.from_dict(d),
            trust_movement_sensor_pose=bool(d.get("trust_movement_sensor_pose", False)),
            snap_heading=bool(d.get("snap_heading", False)),
        )

    def required_dependencies(self) -> List[str]:
        # Union of nav deps (slam_service, base) and sensor deps (base, lidars,
        # movement/heading sensors), de-duplicated preserving order.
        deps = [*self.nav.required_dependencies(), *self.bridge.required_dependencies()]
        return list(dict.fromkeys(deps))


@dataclass
class NavCameraConfig:
    """Config for ``viam-labs:nav-stack:nav-camera``.

    A visualization camera that renders the running navigation service's
    global costmap with the active plan(s), robot pose, footprint and goal
    overlaid. ``navigation`` names the ``navigation`` / ``navigation-external``
    service whose in-process viz store supplies the data.
    """

    navigation: str
    max_dim: int = 700
    plan_history_len: int = 8
    robot_radius_m: float = 0.22
    show_global_plan: bool = True
    show_local_plan: bool = True
    show_pose: bool = True
    show_footprint: bool = True
    show_goal: bool = True
    show_history: bool = True
    # Windowing: "full" (whole map), "follow" (window_size_m square tracking the
    # robot), or "region" (fixed map-frame bbox from window_{min,max}_{x,y}).
    window_mode: str = "full"
    window_size_m: float = 6.0
    window_min_x: Optional[float] = None
    window_min_y: Optional[float] = None
    window_max_x: Optional[float] = None
    window_max_y: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping) -> "NavCameraConfig":
        def _optf(key: str) -> Optional[float]:
            v = d.get(key)
            return None if v is None else float(v)

        return cls(
            navigation=d["navigation"],
            max_dim=int(d.get("max_dim", 700)),
            plan_history_len=int(d.get("plan_history_len", 8)),
            robot_radius_m=float(d.get("robot_radius_m", 0.22)),
            show_global_plan=bool(d.get("show_global_plan", True)),
            show_local_plan=bool(d.get("show_local_plan", True)),
            show_pose=bool(d.get("show_pose", True)),
            show_footprint=bool(d.get("show_footprint", True)),
            show_goal=bool(d.get("show_goal", True)),
            show_history=bool(d.get("show_history", True)),
            window_mode=str(d.get("window_mode", "full")).lower(),
            window_size_m=float(d.get("window_size_m", 6.0)),
            window_min_x=_optf("window_min_x"),
            window_min_y=_optf("window_min_y"),
            window_max_x=_optf("window_max_x"),
            window_max_y=_optf("window_max_y"),
        )

    def required_dependencies(self) -> List[str]:
        # Depend on the navigation service so Viam constructs it (and registers
        # its viz store) before this camera.
        return [self.navigation]
