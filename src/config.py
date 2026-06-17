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
    scan_bins: int = 720
    ros_env: Optional[str] = None
    slam_params: Mapping = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping) -> "SlamConfig":
        lidars_raw = d.get("lidars") or ([d["lidar"]] if d.get("lidar") else [])
        if not lidars_raw:
            raise ValueError("at least one lidar is required ('lidars' or 'lidar')")
        lidars = [LidarConfig.from_dict(x) for x in lidars_raw]
        mode = d.get("mode", MODE_MAPPING)
        if mode not in SLAM_MODES:
            raise ValueError(f"mode must be one of {sorted(SLAM_MODES)}")
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
            scan_bins=int(d.get("scan_bins", 720)),
            ros_env=d.get("ros_env"),
            slam_params=d.get("slam_params", {}) or {},
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
    inflation_radius: float = 0.45
    cmd_vel_timeout: float = 0.5  # seconds (watchdog)
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
            inflation_radius=float(d.get("inflation_radius", 0.45)),
            cmd_vel_timeout=float(d.get("cmd_vel_timeout", 0.5)),
            nav2_params_path=d.get("nav2_params_path"),
            ros_env=d.get("ros_env"),
        )

    def required_dependencies(self) -> List[str]:
        return [self.slam_service, self.base]
