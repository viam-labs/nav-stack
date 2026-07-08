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

BASE_VELOCITY_ROS = "ros"
BASE_VELOCITY_MIR = "mir"
BASE_VELOCITY_CONVENTIONS = {BASE_VELOCITY_ROS, BASE_VELOCITY_MIR}


def ros_cmd_vel_to_viam_linear_mm_s(
    vx_mps: float,
    vy_mps: float,
    convention: str = BASE_VELOCITY_ROS,
) -> tuple[float, float]:
    """Convert ROS body-frame linear speeds (m/s) to Viam base ``SetVelocity`` mm/s.

    Default ``ros`` convention: Viam ``linear.x`` = ROS forward (``vx``),
    Viam ``linear.y`` = ROS lateral (``vy``).

    ``mir`` convention (MiR250 via ``viam-labs:mir-base``): Viam ``linear.y`` is
    forward and ``linear.x`` is lateral — see mir_rosbridge_velocity.viam_velocity_to_ros.
    """
    if convention == BASE_VELOCITY_MIR:
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
    min_range: float = 0.05  # meters
    max_range: float = 25.0  # meters
    # For depth-camera-style lidars: keep only points within this height band.
    z_min: float = -0.2
    z_max: float = 2.0

    @classmethod
    def from_dict(cls, d: Mapping) -> "LidarConfig":
        if isinstance(d, str):
            return cls(name=d)
        mount = d.get("mount", {}) or {}
        return cls(
            name=d["name"],
            x=float(mount.get("x", d.get("x", 0.0))),
            y=float(mount.get("y", d.get("y", 0.0))),
            z=float(mount.get("z", d.get("z", 0.0))),
            theta=float(mount.get("theta", d.get("theta", 0.0))),
            min_range=float(d.get("min_range", 0.05)),
            max_range=float(d.get("max_range", 25.0)),
            z_min=float(d.get("z_min", -0.2)),
            z_max=float(d.get("z_max", 2.0)),
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

    def to_override_dict(self) -> dict:
        """Flat leaf keys applied to the generated Nav2 params template.

        Costmap width/height are applied separately as integers (Jazzy rejects
        doubles for those parameters).
        """
        return {
            "xy_goal_tolerance": self.xy_goal_tolerance,
            "yaw_goal_tolerance": self.yaw_goal_tolerance,
            "tolerance": self.planner_tolerance,
            "cost_scaling_factor": self.cost_scaling_factor,
            "resolution": self.costmap_resolution,
            "controller_frequency": self.controller_frequency,
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
    mode: str = MODE_MAPPING
    maps_dir: str = "/root/.viam/nav-stack/maps"
    active_map: Optional[str] = None
    frames: Frames = field(default_factory=Frames)
    scan_rate_hz: float = 10.0
    odom_rate_hz: float = 20.0
    sensor_read_timeout_s: float = 10.0
    scan_bins: int = 720
    # Safety cutoff for the SLAM/publish scan path: if mir-base reports a scan
    # cache age (get_laser_scan ``age_s``) above this, the bridge skips publishing
    # it rather than feeding SLAM/Nav2 a misregistered scan. Accurate age-based
    # stamping handles normal latency; this only guards genuinely stale data.
    scan_max_age_s: float = 2.0
    ros_env: Optional[str] = None
    # How ROS /cmd_vel (vx forward, vy lateral) maps to the Viam base SetVelocity axes.
    base_velocity_convention: str = BASE_VELOCITY_ROS
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
        convention = d.get("base_velocity_convention", BASE_VELOCITY_ROS)
        if convention not in BASE_VELOCITY_CONVENTIONS:
            raise ValueError(
                f"base_velocity_convention must be one of {sorted(BASE_VELOCITY_CONVENTIONS)}"
            )
        frames_d = d.get("frames", {}) or {}
        return cls(
            base=d["base"],
            lidars=lidars,
            movement_sensor=d.get("movement_sensor"),
            mode=mode,
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
            scan_bins=int(d.get("scan_bins", 720)),
            scan_max_age_s=float(d.get("scan_max_age_s", 2.0)),
            ros_env=d.get("ros_env"),
            base_velocity_convention=convention,
            slam_toolbox=SlamToolboxConfig.from_dict(d.get("slam_toolbox", {}) or {}),
            slam_params=d.get("slam_params", {}) or {},
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
        return deps


@dataclass
class NavConfig:
    slam_service: str
    base: str
    kinematics: str = DIFFERENTIAL
    robot_radius: float = 0.22  # meters
    max_vel_x: float = 0.4  # m/s
    max_vel_y: float = 0.0  # m/s (omni only)
    max_vel_theta: float = 1.0  # rad/s
    acc_lim_x: float = 1.0
    acc_lim_theta: float = 2.0
    inflation_radius: float = 0.55
    cmd_vel_timeout: float = 2.0  # seconds (watchdog)
    # Reactive obstacle avoidance for simple (non-Nav2) go_to_* motion.
    simple_avoid_obstacles: bool = True
    simple_stop_distance: float = 0.4  # meters: stop forward + turn away inside this
    simple_slow_distance: float = 1.0  # meters: scale speed down inside this
    # Max scan age (s) still trusted for avoidance. Larger tolerates slower MiR
    # rosbridge lidar reads; too small makes avoidance fail closed (no drive).
    simple_scan_max_age: float = 2.0
    nav2: Nav2Config = field(default_factory=Nav2Config)
    nav2_params: Mapping = field(default_factory=dict)
    nav2_params_path: Optional[str] = None
    ros_env: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping) -> "NavConfig":
        kinematics = d.get("kinematics", DIFFERENTIAL)
        if kinematics not in KINEMATICS:
            raise ValueError(f"kinematics must be one of {sorted(KINEMATICS)}")
        return cls(
            slam_service=d["slam_service"],
            base=d["base"],
            kinematics=kinematics,
            robot_radius=float(d.get("robot_radius", 0.22)),
            max_vel_x=float(d.get("max_vel_x", 0.4)),
            max_vel_y=float(d.get("max_vel_y", 0.0)),
            max_vel_theta=float(d.get("max_vel_theta", 1.0)),
            acc_lim_x=float(d.get("acc_lim_x", 1.0)),
            acc_lim_theta=float(d.get("acc_lim_theta", 2.0)),
            inflation_radius=float(d.get("inflation_radius", 0.55)),
            cmd_vel_timeout=float(d.get("cmd_vel_timeout", 2.0)),
            simple_avoid_obstacles=bool(d.get("simple_avoid_obstacles", True)),
            simple_stop_distance=float(d.get("simple_stop_distance", 0.4)),
            simple_slow_distance=float(d.get("simple_slow_distance", 1.0)),
            simple_scan_max_age=float(d.get("simple_scan_max_age", 2.0)),
            nav2=Nav2Config.from_dict(d.get("nav2", {}) or {}),
            nav2_params=d.get("nav2_params", {}) or {},
            nav2_params_path=d.get("nav2_params_path"),
            ros_env=d.get("ros_env"),
        )

    def required_dependencies(self) -> List[str]:
        return [self.slam_service, self.base]
