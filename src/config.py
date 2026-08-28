"""Typed configuration objects parsed from Viam component attributes.

Keeping these as plain dataclasses (no Viam or ROS imports) makes the parsing
logic easy to unit-test and shareable between the models and the ROS layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional

DIFFERENTIAL = "differential"
OMNI = "omni"
KINEMATICS = {DIFFERENTIAL, OMNI}

MODE_MAPPING = "mapping"
MODE_LOCALIZING = "localizing"
SLAM_MODES = {MODE_MAPPING, MODE_LOCALIZING}

BASE_VELOCITY_VIAM = "viam"
BASE_VELOCITY_ROS = "ros"
# Legacy alias for ``viam`` (Y-forward); accepted in config, normalized to ``viam``.
BASE_VELOCITY_MIR = "mir"
BASE_VELOCITY_CONVENTIONS = {BASE_VELOCITY_VIAM, BASE_VELOCITY_ROS, BASE_VELOCITY_MIR}
# Conventions that put ROS forward (vx) on Viam ``linear.y``.
BASE_VELOCITY_Y_FORWARD = {BASE_VELOCITY_VIAM, BASE_VELOCITY_MIR}

LIDAR_SCAN_AUTO = "auto"
LIDAR_SCAN_GET_LASER_SCAN = "get_laser_scan"
LIDAR_SCAN_POINT_CLOUD = "point_cloud"
LIDAR_SCAN_SOURCES = {
    LIDAR_SCAN_AUTO,
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
}

IMU_ODOM_COAST = "coast"
IMU_ODOM_ACCEL_ONLY = "accel_only"
IMU_ODOM_NONE = "none"
IMU_ODOM_MODES = {IMU_ODOM_COAST, IMU_ODOM_ACCEL_ONLY, IMU_ODOM_NONE}


def ros_cmd_vel_to_viam_linear_mm_s(
    vx_mps: float,
    vy_mps: float,
    convention: str = BASE_VELOCITY_VIAM,
) -> tuple[float, float]:
    """Convert ROS body-frame linear speeds (m/s) to Viam base ``SetVelocity`` mm/s.

    Default ``viam`` convention (also ``mir``): Viam ``linear.y`` = ROS forward
    (``vx``), Viam ``linear.x`` = ROS lateral (``vy``). Matches ``rdk:builtin:wheeled``
    and MiR bases, which only drive on Y.

    ``ros`` convention: Viam ``linear.x`` = ROS forward, ``linear.y`` = lateral —
    for bases that actually consume X as forward.
    """
    if convention in BASE_VELOCITY_Y_FORWARD:
        vx_mps, vy_mps = vy_mps, vx_mps
    return vx_mps * 1000.0, vy_mps * 1000.0


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
    # out of the 2D scan fed to slam_toolbox.
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
    shm_name: Optional[str] = None
    shm_region_size: int = 2 * 1024 * 1024
    # If true, never fall back to gRPC GetPointCloud when shm is empty/missing.
    shm_required: bool = False

    @classmethod
    def from_dict(cls, d: Mapping) -> "LidarConfig":
        if isinstance(d, str):
            return cls(name=d)
        mount = d.get("mount", {}) or {}
        scan_source = str(d.get("scan_source", LIDAR_SCAN_AUTO))
        if scan_source not in LIDAR_SCAN_SOURCES:
            raise ValueError(
                f"lidar scan_source must be one of {sorted(LIDAR_SCAN_SOURCES)}"
            )
        shm_name = d.get("shm_name")
        shm_name_s = str(shm_name).strip() if shm_name else ""
        region = int(
            d.get("shm_region_size", d.get("shm_region_size_bytes", 2 * 1024 * 1024))
        )
        if region <= 0 or region % 2 != 0:
            raise ValueError("lidar shm_region_size must be a positive even byte count")
        return cls(
            name=d["name"],
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
            shm_name=shm_name_s or None,
            shm_region_size=region,
            shm_required=bool(d.get("shm_required", False)),
        )


@dataclass
class SlamToolboxConfig:
    """Common slam_toolbox ROS parameters exposed via Viam config.

    Top-level ``mode`` on the SLAM service (``mapping`` / ``localizing``) selects
    the slam_toolbox node and sets its ``mode`` parameter. Additional tuning lives
    under the ``slam_toolbox`` attribute block. Use ``slam_params`` for any other
    slam_toolbox keys not listed here.
    """

    resolution: float = 0.05  # meters/cell
    transform_publish_period: float = 0.05  # seconds
    map_update_interval: float = 1.0  # seconds
    minimum_travel_distance: float = 0.3  # meters before adding a new scan
    minimum_travel_heading: float = 0.3  # radians before adding a new scan
    max_laser_range: float = 25.0  # meters
    scan_topic: str = "/scan"
    use_map_saver: bool = True

    def to_ros_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "transform_publish_period": self.transform_publish_period,
            "map_update_interval": self.map_update_interval,
            "minimum_travel_distance": self.minimum_travel_distance,
            "minimum_travel_heading": self.minimum_travel_heading,
            "max_laser_range": self.max_laser_range,
            "scan_topic": self.scan_topic,
            "use_map_saver": self.use_map_saver,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "SlamToolboxConfig":
        if not d:
            return cls()
        return cls(
            resolution=float(d.get("resolution", 0.05)),
            transform_publish_period=float(d.get("transform_publish_period", 0.05)),
            map_update_interval=float(d.get("map_update_interval", 1.0)),
            minimum_travel_distance=float(d.get("minimum_travel_distance", 0.3)),
            minimum_travel_heading=float(d.get("minimum_travel_heading", 0.3)),
            max_laser_range=float(d.get("max_laser_range", 25.0)),
            scan_topic=str(d.get("scan_topic", "/scan")),
            use_map_saver=bool(d.get("use_map_saver", True)),
        )


NAV_BACKEND_BUILTIN = "builtin"
NAV_BACKEND_NAV2 = "nav2"
NAV_BACKENDS = frozenset({NAV_BACKEND_BUILTIN, NAV_BACKEND_NAV2})

SLAM_BACKEND_BUILTIN = "builtin"
SLAM_BACKEND_TOOLBOX = "slam_toolbox"
SLAM_BACKENDS = frozenset({SLAM_BACKEND_BUILTIN, SLAM_BACKEND_TOOLBOX})

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
    """Tuning for the in-module (ROS-free) navigator.

    Footprint / velocity limits stay top-level on ``NavConfig`` so both backends
    share them. Defaults are tuned for builtin SLAM + pure pursuit (not copied
    from Nav2 MPPI/RPP template values).
    """

    # ``lazy_theta_star`` (default) or ``astar``.
    planner: str = BUILTIN_PLANNER_LAZY_THETA
    # Pure-pursuit lookahead (mugger-dds RPP uses 0.25–0.7 m velocity-scaled;
    # 1.0 m was Nav2-cart-ish and cut corners / overshot on diff-drive).
    lookahead_m: float = 0.6
    min_lookahead_m: float = 0.25
    max_lookahead_m: float = 0.7
    replan_period_s: float = 1.0
    timeout_s: float = 300.0
    cost_scaling_factor: float = 4.0
    xy_goal_tolerance: float = 0.25  # meters
    yaw_goal_tolerance: float = 0.35  # radians (~20 deg; mugger uses 0.6)
    # Final approach: cap linear speed within this distance of the goal.
    approach_dist_m: float = 0.35
    # Post-process global plans (shortcut + resample) before following.
    smooth_path: bool = True
    smooth_sample_spacing_m: float = 0.10
    # Rolling local costmap + DWA-style local planner for dynamic obstacles.
    local_costmap_enabled: bool = True
    local_costmap_width_m: float = 4.0
    local_costmap_height_m: float = 4.0
    local_costmap_resolution: float = 0.05
    local_inflation_radius_m: float = 0.25
    local_planner_enabled: bool = True
    local_planner_sim_time_s: float = 1.5
    local_planner_activate_cost: int = 200
    local_planner_max_vel_x_reverse_m: float = 0.15
    # Nav2-style backup when local planner spins in place with clear rear space.
    backup_enabled: bool = True
    backup_stuck_time_s: float = 3.0
    backup_dist_m: float = 0.30
    backup_speed_mps: float = 0.12
    backup_rear_clear_m: float = 0.45
    backup_max_attempts: int = 1
    backup_cooldown_s: float = 4.0
    # Replan when the local costmap sees the path ahead blocked (dynamic obstacles).
    replan_local_blocked_time_s: float = 0.3
    replan_local_min_period_s: float = 0.5

    @classmethod
    def from_dict(cls, d: Mapping) -> "BuiltinNavConfig":
        if not d:
            return cls()
        return cls(
            planner=normalize_builtin_planner(d.get("planner", BUILTIN_PLANNER_LAZY_THETA)),
            lookahead_m=float(d.get("lookahead_m", 0.6)),
            min_lookahead_m=float(d.get("min_lookahead_m", 0.25)),
            max_lookahead_m=float(d.get("max_lookahead_m", 0.7)),
            replan_period_s=float(d.get("replan_period_s", 1.0)),
            timeout_s=float(d.get("timeout_s", 300.0)),
            cost_scaling_factor=float(d.get("cost_scaling_factor", 4.0)),
            xy_goal_tolerance=float(d.get("xy_goal_tolerance", 0.25)),
            yaw_goal_tolerance=float(d.get("yaw_goal_tolerance", 0.35)),
            approach_dist_m=float(d.get("approach_dist_m", 0.35)),
            smooth_path=bool(d.get("smooth_path", True)),
            smooth_sample_spacing_m=float(d.get("smooth_sample_spacing_m", 0.10)),
            local_costmap_enabled=bool(d.get("local_costmap_enabled", True)),
            local_costmap_width_m=float(d.get("local_costmap_width_m", 4.0)),
            local_costmap_height_m=float(d.get("local_costmap_height_m", 4.0)),
            local_costmap_resolution=float(d.get("local_costmap_resolution", 0.05)),
            local_inflation_radius_m=float(d.get("local_inflation_radius_m", 0.25)),
            local_planner_enabled=bool(d.get("local_planner_enabled", True)),
            local_planner_sim_time_s=float(d.get("local_planner_sim_time_s", 1.5)),
            local_planner_activate_cost=int(d.get("local_planner_activate_cost", 200)),
            local_planner_max_vel_x_reverse_m=float(
                d.get("local_planner_max_vel_x_reverse_m", 0.15)
            ),
            backup_enabled=bool(d.get("backup_enabled", True)),
            backup_stuck_time_s=float(d.get("backup_stuck_time_s", 3.0)),
            backup_dist_m=float(d.get("backup_dist_m", 0.30)),
            backup_speed_mps=float(d.get("backup_speed_mps", 0.12)),
            backup_rear_clear_m=float(d.get("backup_rear_clear_m", 0.45)),
            backup_max_attempts=int(d.get("backup_max_attempts", 1)),
            backup_cooldown_s=float(d.get("backup_cooldown_s", 4.0)),
            replan_local_blocked_time_s=float(
                d.get("replan_local_blocked_time_s", 0.3)
            ),
            replan_local_min_period_s=float(d.get("replan_local_min_period_s", 0.5)),
        )


@dataclass
class Nav2Config:
    """Common Nav2 ROS parameters exposed via Viam config.

    Velocity, footprint, and inflation defaults remain top-level on the navigation
    service for convenience. Additional tuning lives under the ``nav2`` attribute
    block. Use ``nav2_params`` for any other Nav2 keys not listed here.
    """

    xy_goal_tolerance: float = 0.25  # meters
    yaw_goal_tolerance: float = 0.25  # radians
    planner_tolerance: float = 0.5  # meters
    cost_scaling_factor: float = 4.0
    local_costmap_width: float = 4.0  # meters
    local_costmap_height: float = 4.0  # meters
    costmap_resolution: float = 0.05  # meters/cell
    # Must match the params template default: this dict always overrides the
    # template, so a stale default here silently clobbers template retuning.
    # 10 Hz (not Nav2's stock 20) keeps MPPI within a Pi 5's budget.
    controller_frequency: float = 10.0  # Hz
    # Global replan rate (BT RateController). Stock Nav2's 1 Hz: each replan
    # costs a plan + smooth + path handoff, and 2 Hz measurably starves the
    # controller loop on a Pi running slam_toolbox alongside. Raise it on
    # faster hardware if you need quicker reaction to blocked paths.
    replan_frequency: float = 1.0  # Hz
    # Progress checker: how long with almost no movement before FollowPath
    # fails into recovery. Stock template was 30 s (very patient); 10 s exits
    # reverse/spin loops sooner on stuck carts.
    progress_movement_time_allowance: float = 10.0  # seconds
    # Outer NavigateRecovery retries (spin/backup/wait/clear cycle). Stock is 6.
    navigate_recovery_retries: int = 4
    # Wait behavior duration inside the recovery RoundRobin (stock 5 s).
    recovery_wait_duration: float = 2.0  # seconds

    def to_override_dict(self) -> dict:
        """Flat leaf keys applied to the generated Nav2 params template.

        Costmap width/height are applied separately as integers (Jazzy rejects
        doubles for those parameters). Replan / recovery-retry knobs are applied
        by rewriting the navigate-to-pose behavior tree XML, not here.
        """
        return {
            "xy_goal_tolerance": self.xy_goal_tolerance,
            "yaw_goal_tolerance": self.yaw_goal_tolerance,
            "tolerance": self.planner_tolerance,
            "cost_scaling_factor": self.cost_scaling_factor,
            "resolution": self.costmap_resolution,
            "controller_frequency": self.controller_frequency,
            "movement_time_allowance": self.progress_movement_time_allowance,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Nav2Config":
        if not d:
            return cls()
        return cls(
            xy_goal_tolerance=float(d.get("xy_goal_tolerance", 0.25)),
            yaw_goal_tolerance=float(d.get("yaw_goal_tolerance", 0.25)),
            planner_tolerance=float(d.get("planner_tolerance", 0.5)),
            cost_scaling_factor=float(d.get("cost_scaling_factor", 4.0)),
            local_costmap_width=float(d.get("local_costmap_width", 4.0)),
            local_costmap_height=float(d.get("local_costmap_height", 4.0)),
            costmap_resolution=float(d.get("costmap_resolution", 0.05)),
            controller_frequency=float(d.get("controller_frequency", 10.0)),
            replan_frequency=float(d.get("replan_frequency", 1.0)),
            progress_movement_time_allowance=float(
                d.get("progress_movement_time_allowance", 10.0)
            ),
            navigate_recovery_retries=int(d.get("navigate_recovery_retries", 4)),
            recovery_wait_duration=float(d.get("recovery_wait_duration", 2.0)),
        )


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
    # Added to GetPosition yaw only (App arrow vs map PCD). Does not change ROS
    # TF / slam_toolbox. Prefer fixing lidar ``mount.theta`` (see status probe
    # ``nearest_return_bearing_deg``) — a cosmetic ±45 rarely means TF and PCD
    # disagree; more often the Livox +X is off base_link forward.
    map_pose_yaw_offset_deg: float = 0.0
    mode: str = MODE_MAPPING
    # Default: in-process occupancy SLAM. Set ``slam_toolbox`` to keep ROS.
    slam_backend: str = SLAM_BACKEND_BUILTIN
    maps_dir: str = "/root/.viam/nav-stack/maps"
    active_map: Optional[str] = None
    frames: Frames = field(default_factory=Frames)
    scan_rate_hz: float = 10.0
    odom_rate_hz: float = 20.0
    sensor_read_timeout_s: float = 10.0
    # External-SLAM navigation (navigation-external) tunables: how fast the
    # ExternalSlamPublisher polls the Viam SLAM service for pose/grid and how far
    # in the future map->odom is stamped. Ignored by the built-in slam_toolbox path.
    external_pose_rate_hz: float = 10.0
    external_grid_rate_hz: float = 1.5
    external_transform_timeout_s: float = 0.2
    scan_bins: int = 720
    # Merge recent point-cloud frames before /scan projection. Livox Mid-360 and
    # similar non-repetitive 3D lidars need this for stable slam_toolbox input.
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
    # this yields dense scans and lets slam_toolbox match between pauses.
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
    # odom, shift the odom TF so slam_toolbox links back to the original area
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
    # Safety cutoff for the SLAM/publish scan path: if mir-base reports a scan
    # cache age (get_laser_scan ``age_s``) above this, the bridge skips publishing
    # it rather than feeding SLAM/Nav2 a misregistered scan. Accurate age-based
    # stamping handles normal latency; this only guards genuinely stale data.
    scan_max_age_s: float = 2.0
    # How ROS /cmd_vel (vx forward, vy lateral) maps to the Viam base SetVelocity axes.
    base_velocity_convention: str = BASE_VELOCITY_VIAM
    slam_toolbox: SlamToolboxConfig = field(default_factory=SlamToolboxConfig)
    slam_params: Mapping = field(default_factory=dict)
    # Automatically run global_localize shortly after starting in localizing mode.
    global_localize_on_start: bool = True
    global_localize_on_start_delay_s: float = 4.0
    # Max time to wait for slam + scans + map to become usable before the
    # startup auto-localize runs (rosbridge sessions can take tens of seconds).
    global_localize_on_start_readiness_timeout_s: float = 90.0
    global_localize_on_start_options: Mapping = field(
        default_factory=lambda: {
            "full_map": True,
            "map_source": "live",
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
    # Shorter interval while Nav2 is active (localization drift shows up as planner
    # failures / recoveries mid-goal).
    periodic_relocalize_nav_interval_s: float = 15.0
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
    # When Nav2 reports this many recoveries on the active goal, skip the cheap
    # local match and run full-map global_localize immediately.
    periodic_relocalize_nav_recoveries_threshold: int = 2
    periodic_relocalize_full_map_on_low_quality: bool = True
    periodic_relocalize_during_navigation: bool = True
    periodic_relocalize_options: Mapping = field(
        default_factory=lambda: {
            "full_map": False,
            "map_source": "live",
            "search_radius_m": 3.0,
            "auto_full_map_fallback": True,
        }
    )

    @classmethod
    def from_dict(cls, d: Mapping) -> "SlamConfig":
        lidars_raw = d.get("lidars") or ([d["lidar"]] if d.get("lidar") else [])
        if not lidars_raw:
            raise ValueError("at least one lidar is required ('lidars' or 'lidar')")
        lidars = [LidarConfig.from_dict(x) for x in lidars_raw]
        mode = d.get("mode", MODE_MAPPING)
        if mode not in SLAM_MODES:
            raise ValueError(f"mode must be one of {sorted(SLAM_MODES)}")
        slam_backend = str(
            d.get("slam_backend", SLAM_BACKEND_BUILTIN) or SLAM_BACKEND_BUILTIN
        )
        if slam_backend not in SLAM_BACKENDS:
            raise ValueError(
                f"slam_backend must be one of {sorted(SLAM_BACKENDS)}, "
                f"got {slam_backend!r}"
            )
        convention = d.get("base_velocity_convention", BASE_VELOCITY_VIAM)
        if convention not in BASE_VELOCITY_CONVENTIONS:
            raise ValueError(
                f"base_velocity_convention must be one of {sorted(BASE_VELOCITY_CONVENTIONS)}"
            )
        # Normalize legacy ``mir`` to the canonical Y-forward name.
        if convention == BASE_VELOCITY_MIR:
            convention = BASE_VELOCITY_VIAM
        frames_d = d.get("frames", {}) or {}
        all_point_cloud = bool(lidars) and all(
            lidar.scan_source == LIDAR_SCAN_POINT_CLOUD for lidar in lidars
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
        stb_raw = dict(d.get("slam_toolbox", {}) or {})
        slam_params_raw = dict(d.get("slam_params", {}) or {})
        if all_point_cloud:
            max_lidar_range = max(lidar.max_range for lidar in lidars)
            # Real travel gates matter for Livox: with minimum_travel_* at 0,
            # slam_toolbox scan-matches every noisy non-repetitive frame while
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
            # slam_toolbox ALWAYS uses odom→base TF as the scan-match prior —
            # ``use_odometry`` / ``use_tf_scan_transformation`` are NOT real
            # slam_toolbox params (no-ops). Widen the *real* correlative angular
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
                map_when_still and all_point_cloud,
            )
        )
        return cls(
            base=d["base"],
            lidars=lidars,
            movement_sensor=d.get("movement_sensor"),
            imu_shm_name=(
                str(d["imu_shm_name"]).strip() or None
                if d.get("imu_shm_name")
                else None
            ),
            imu_shm_region_size=int(d.get("imu_shm_region_size", 4096)),
            imu_shm_max_age_s=float(d.get("imu_shm_max_age_s", 0.5)),
            heading_sensor=d.get("heading_sensor"),
            movement_sensor_yaw_deg=float(d.get("movement_sensor_yaw_deg", 0.0)),
            movement_sensor_upside_down=bool(
                d.get("movement_sensor_upside_down", False)
            ),
            heading_sensor_yaw_deg=float(d.get("heading_sensor_yaw_deg", 0.0)),
            heading_sensor_invert=bool(d.get("heading_sensor_invert", False)),
            map_pose_yaw_offset_deg=float(d.get("map_pose_yaw_offset_deg", 0.0)),
            mode=mode,
            slam_backend=slam_backend,
            maps_dir=d.get("maps_dir", "/root/.viam/nav-stack/maps"),
            active_map=d.get("active_map"),
            frames=Frames(
                map=frames_d.get("map", "map"),
                odom=frames_d.get("odom", "odom"),
                base_link=frames_d.get("base_link", "base_link"),
            ),
            scan_rate_hz=float(d.get("scan_rate_hz", 10.0)),
            odom_rate_hz=float(d.get("odom_rate_hz", 20.0)),
            sensor_read_timeout_s=float(d.get("sensor_read_timeout_s", 10.0)),
            external_pose_rate_hz=float(d.get("external_pose_rate_hz", 10.0)),
            external_grid_rate_hz=float(d.get("external_grid_rate_hz", 1.5)),
            external_transform_timeout_s=float(
                d.get("external_transform_timeout_s", 0.2)
            ),
            scan_bins=int(d.get("scan_bins", 720)),
            scan_accumulation_s=float(
                d.get("scan_accumulation_s", default_accum)
            ),
            imu_odom_mode=imu_odom_mode,
            heading_only_odom=heading_only_odom,
            lidar_odom_enabled=lidar_odom_enabled,
            lidar_odom_range_flow_only=lidar_odom_range_flow_only,
            map_when_still=map_when_still,
            map_when_still_dwell_s=map_when_still_dwell_s,
            map_when_still_linear_speed_m_s=float(
                d.get("map_when_still_linear_speed_m_s", 0.02)
            ),
            map_when_still_yaw_rate_rad_s=float(
                d.get("map_when_still_yaw_rate_rad_s", 0.04)
            ),
            map_when_still_yaw_step_deg=map_when_still_yaw_step_deg,
            map_when_still_max_drift_m=float(
                d.get("map_when_still_max_drift_m", 0.03)
            ),
            map_when_still_max_drift_deg=float(
                d.get("map_when_still_max_drift_deg", 1.5)
            ),
            wall_yaw_correction=wall_yaw_correction,
            wall_yaw_min_length_m=float(d.get("wall_yaw_min_length_m", 2.0)),
            wall_yaw_max_step_deg=float(d.get("wall_yaw_max_step_deg", 2.0)),
            wall_yaw_blend=float(d.get("wall_yaw_blend", 0.5)),
            mapping_revisit_check=mapping_revisit_check,
            mapping_revisit_interval_s=float(
                d.get("mapping_revisit_interval_s", 20.0)
            ),
            mapping_revisit_search_radius_m=float(
                d.get("mapping_revisit_search_radius_m", 5.0)
            ),
            mapping_revisit_wide_radius_m=float(
                d.get("mapping_revisit_wide_radius_m", 12.0)
            ),
            mapping_revisit_min_score=float(d.get("mapping_revisit_min_score", 0.6)),
            mapping_revisit_max_ray_mae_m=float(
                d.get("mapping_revisit_max_ray_mae_m", 0.8)
            ),
            mapping_revisit_min_shift_m=float(
                d.get("mapping_revisit_min_shift_m", 1.0)
            ),
            mapping_revisit_min_shift_deg=float(
                d.get("mapping_revisit_min_shift_deg", 10.0)
            ),
            mapping_revisit_max_shift_m=float(
                d.get("mapping_revisit_max_shift_m", 10.0)
            ),
            mapping_revisit_full_map_fallback=bool(
                d.get("mapping_revisit_full_map_fallback", True)
            ),
            mapping_revisit_full_map_min_score=float(
                d.get("mapping_revisit_full_map_min_score", 0.75)
            ),
            mapping_revisit_slice_verify=bool(
                d.get("mapping_revisit_slice_verify", True)
            ),
            mapping_revisit_slice_bands=[
                [float(pair[0]), float(pair[1])]
                for pair in d.get(
                    "mapping_revisit_slice_bands", [[0.15, 0.45], [1.6, 2.4]]
                )
            ],
            mapping_revisit_slice_min_hit_rate=float(
                d.get("mapping_revisit_slice_min_hit_rate", 0.4)
            ),
            mapping_revisit_slice_resolution_m=float(
                d.get("mapping_revisit_slice_resolution_m", 0.15)
            ),
            mapping_revisit_keyframes=bool(d.get("mapping_revisit_keyframes", True)),
            mapping_revisit_keyframe_min_spacing_m=float(
                d.get("mapping_revisit_keyframe_min_spacing_m", 0.5)
            ),
            mapping_revisit_keyframe_min_spacing_deg=float(
                d.get("mapping_revisit_keyframe_min_spacing_deg", 20.0)
            ),
            mapping_revisit_keyframe_max=int(d.get("mapping_revisit_keyframe_max", 250)),
            mapping_revisit_keyframe_match_tol_m=float(
                d.get("mapping_revisit_keyframe_match_tol_m", 0.3)
            ),
            mapping_revisit_keyframe_min_score=float(
                d.get("mapping_revisit_keyframe_min_score", 0.55)
            ),
            scan_max_age_s=float(d.get("scan_max_age_s", 2.0)),
            base_velocity_convention=convention,
            slam_toolbox=SlamToolboxConfig.from_dict(stb_raw),
            slam_params=slam_params_raw,
            global_localize_on_start=bool(d.get("global_localize_on_start", True)),
            global_localize_on_start_delay_s=float(
                d.get("global_localize_on_start_delay_s", 4.0)
            ),
            global_localize_on_start_readiness_timeout_s=float(
                d.get("global_localize_on_start_readiness_timeout_s", 90.0)
            ),
            global_localize_on_start_options=d.get(
                "global_localize_on_start_options",
                {
                    "full_map": True,
                    "map_source": "live",
                },
            )
            or {
                "full_map": True,
                "map_source": "live",
            },
            global_localize_on_start_refine=bool(
                d.get("global_localize_on_start_refine", True)
            ),
            global_localize_on_start_refine_delay_s=float(
                d.get("global_localize_on_start_refine_delay_s", 8.0)
            ),
            global_localize_on_start_refine_max_passes=int(
                d.get("global_localize_on_start_refine_max_passes", 3)
            ),
            global_localize_on_start_target_score=float(
                d.get("global_localize_on_start_target_score", 0.7)
            ),
            global_localize_on_start_target_ray_mae_m=float(
                d.get("global_localize_on_start_target_ray_mae_m", 0.4)
            ),
            global_localize_on_start_post_apply_refine=bool(
                d.get("global_localize_on_start_post_apply_refine", True)
            ),
            global_localize_on_start_post_apply_refine_delay_s=float(
                d.get("global_localize_on_start_post_apply_refine_delay_s", 8.0)
            ),
            global_localize_on_start_post_apply_refine_options=d.get(
                "global_localize_on_start_post_apply_refine_options",
                {"map_source": "live"},
            )
            or {"map_source": "live"},
            global_localize_on_start_refine_options=d.get(
                "global_localize_on_start_refine_options",
                {
                    "full_map": False,
                    "map_source": "live",
                    "local_yaw_window_deg": 120.0,
                    "search_radius_m": 6.0,
                },
            )
            or {
                "full_map": False,
                "map_source": "live",
                "local_yaw_window_deg": 120.0,
                "search_radius_m": 6.0,
            },
            periodic_relocalize=bool(d.get("periodic_relocalize", True)),
            periodic_relocalize_interval_s=float(
                d.get("periodic_relocalize_interval_s", 20.0)
            ),
            periodic_relocalize_nav_interval_s=float(
                d.get("periodic_relocalize_nav_interval_s", 15.0)
            ),
            periodic_relocalize_min_score=float(
                d.get("periodic_relocalize_min_score", 0.5)
            ),
            periodic_relocalize_max_ray_mae_m=float(
                d.get("periodic_relocalize_max_ray_mae_m", 1.0)
            ),
            periodic_relocalize_recovery_min_score=float(
                d.get("periodic_relocalize_recovery_min_score", 0.45)
            ),
            periodic_relocalize_min_shift_m=float(
                d.get("periodic_relocalize_min_shift_m", 0.2)
            ),
            periodic_relocalize_min_shift_deg=float(
                d.get("periodic_relocalize_min_shift_deg", 10.0)
            ),
            periodic_relocalize_nav_recoveries_threshold=int(
                d.get("periodic_relocalize_nav_recoveries_threshold", 2)
            ),
            periodic_relocalize_full_map_on_low_quality=bool(
                d.get("periodic_relocalize_full_map_on_low_quality", True)
            ),
            periodic_relocalize_during_navigation=bool(
                d.get("periodic_relocalize_during_navigation", True)
            ),
            periodic_relocalize_options=d.get(
                "periodic_relocalize_options",
                {
                    "full_map": False,
                    "map_source": "live",
                    "search_radius_m": 3.0,
                    "auto_full_map_fallback": True,
                },
            )
            or {
                "full_map": False,
                "map_source": "live",
                "search_radius_m": 3.0,
                "auto_full_map_fallback": True,
            },
        )

    def required_dependencies(self) -> List[str]:
        deps = [self.base, *[lidar.name for lidar in self.lidars]]
        if self.movement_sensor:
            deps.append(self.movement_sensor)
        if self.heading_sensor:
            deps.append(self.heading_sensor)
        return deps

    def uses_builtin_slam(self) -> bool:
        return self.slam_backend == SLAM_BACKEND_BUILTIN

    def uses_slam_toolbox(self) -> bool:
        return self.slam_backend == SLAM_BACKEND_TOOLBOX


@dataclass
class NavConfig:
    slam_service: str
    base: str
    kinematics: str = DIFFERENTIAL
    robot_radius: float = 0.22  # meters
    max_vel_x: float = 0.6  # m/s
    max_vel_y: float = 0.0  # m/s (omni only)
    max_vel_theta: float = 1.5  # rad/s
    acc_lim_x: float = 1.0
    acc_lim_theta: float = 2.0
    inflation_radius: float = 0.25
    cmd_vel_timeout: float = 2.0  # seconds (watchdog)
    # Reactive obstacle avoidance for simple (non-Nav2) go_to_* motion.
    simple_avoid_obstacles: bool = True
    simple_stop_distance: float = 0.4  # meters: stop forward + turn away inside this
    simple_slow_distance: float = 1.0  # meters: scale speed down inside this
    # Max scan age (s) still trusted for avoidance. Larger tolerates slower MiR
    # rosbridge lidar reads; too small makes avoidance fail closed (no drive).
    simple_scan_max_age: float = 2.0
    # Optional stiction floors (m/s and rad/s) for simple go_to_* motion only.
    # Nav2 cmd_vel is never floored: independently bumping its linear and angular
    # components distorts MPPI curvature and turns gentle corrections into loops.
    # Not the Nav2 ``min_vel_x`` param (reverse speed limit).
    min_cmd_vel_x: float = 0.0
    min_cmd_vel_theta: float = 0.0
    # Default: in-module planner/controller. Set to "nav2" to keep ROS Nav2.
    nav_backend: str = NAV_BACKEND_BUILTIN
    builtin: BuiltinNavConfig = field(default_factory=BuiltinNavConfig)
    nav2: Nav2Config = field(default_factory=Nav2Config)
    nav2_params: Mapping = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping) -> "NavConfig":
        kinematics = d.get("kinematics", DIFFERENTIAL)
        if kinematics not in KINEMATICS:
            raise ValueError(f"kinematics must be one of {sorted(KINEMATICS)}")
        backend = str(d.get("nav_backend", NAV_BACKEND_BUILTIN) or NAV_BACKEND_BUILTIN)
        if backend not in NAV_BACKENDS:
            raise ValueError(
                f"nav_backend must be one of {sorted(NAV_BACKENDS)}, got {backend!r}"
            )
        return cls(
            slam_service=d["slam_service"],
            base=d["base"],
            kinematics=kinematics,
            robot_radius=float(d.get("robot_radius", 0.22)),
            max_vel_x=float(d.get("max_vel_x", 0.6)),
            max_vel_y=float(d.get("max_vel_y", 0.0)),
            max_vel_theta=float(d.get("max_vel_theta", 1.5)),
            acc_lim_x=float(d.get("acc_lim_x", 1.0)),
            acc_lim_theta=float(d.get("acc_lim_theta", 2.0)),
            inflation_radius=float(d.get("inflation_radius", 0.25)),
            cmd_vel_timeout=float(d.get("cmd_vel_timeout", 2.0)),
            simple_avoid_obstacles=bool(d.get("simple_avoid_obstacles", True)),
            simple_stop_distance=float(d.get("simple_stop_distance", 0.4)),
            simple_slow_distance=float(d.get("simple_slow_distance", 1.0)),
            simple_scan_max_age=float(d.get("simple_scan_max_age", 2.0)),
            # Legacy aliases: simple_min_vel_x / simple_min_vel_theta.
            min_cmd_vel_x=float(
                d.get("min_cmd_vel_x", d.get("simple_min_vel_x", 0.0))
            ),
            min_cmd_vel_theta=float(
                d.get("min_cmd_vel_theta", d.get("simple_min_vel_theta", 0.0))
            ),
            nav_backend=backend,
            builtin=BuiltinNavConfig.from_dict(d.get("builtin", {}) or {}),
            nav2=Nav2Config.from_dict(d.get("nav2", {}) or {}),
            nav2_params=d.get("nav2_params", {}) or {},
        )

    def uses_builtin_nav(self) -> bool:
        return self.nav_backend == NAV_BACKEND_BUILTIN

    def uses_nav2(self) -> bool:
        return self.nav_backend == NAV_BACKEND_NAV2

    def required_dependencies(self) -> List[str]:
        return [self.slam_service, self.base]


@dataclass
class ExternalNavConfig:
    """Config for ``viam-labs:nav-stack:navigation-external``.

    Drives Nav2 from an arbitrary Viam ``rdk:service:slam`` (not the built-in
    slam_toolbox). One flat attributes block yields both a sensor-bridge
    ``SlamConfig`` (base, lidars, movement sensor, odom tuning) and a ``NavConfig``
    (Nav2 + navigation behavior); ``slam_service`` names the SLAM dependency.
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
        # Union of Nav2 deps (slam_service, base) and bridge deps (base, lidars,
        # movement/heading sensors), de-duplicated preserving order.
        deps = [*self.nav.required_dependencies(), *self.bridge.required_dependencies()]
        return list(dict.fromkeys(deps))


@dataclass
class NavCameraConfig:
    """Config for ``viam-labs:nav-stack:nav-camera``.

    A visualization camera that renders the running navigation service's Nav2
    global costmap with the active plan(s), robot pose, footprint and goal
    overlaid. ``navigation`` names the ``navigation`` / ``navigation-external``
    service whose in-process bridge supplies the data.
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
        # its bridge) before this camera.
        return [self.navigation]
