"""Resolve mounts and footprint from the Viam robot framesystem.

Viam frames use millimetres and orientation-vector degrees; nav-stack mounts use
metres and yaw/pitch/roll radians. Config JSON still wins when ``mount`` or
footprint fields are set explicitly.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .geom.conversions import mm_to_m

LOGGER = logging.getLogger(__name__)

# Parent RobotClient RPCs must stay on the module event loop. Bound the wait so
# a wedged parent cannot stall slam/nav forever after startup.
FRAME_SYSTEM_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class MountPose:
    """Sensor pose in ``base`` / base_link (metres, radians)."""

    x: float
    y: float
    z: float
    theta: float  # yaw
    pitch: float = 0.0
    roll: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "theta": self.theta,
            "pitch": self.pitch,
            "roll": self.roll,
        }


@dataclass(frozen=True)
class FootprintBox:
    """Rectangular footprint on the floor plane (metres)."""

    length_m: float  # forward
    width_m: float  # lateral

    @property
    def inscribed_radius_m(self) -> float:
        return min(self.length_m, self.width_m) / 2.0


def _pose_to_matrix(pose) -> np.ndarray:
    """4×4 SE(3) from a Viam Pose (mm + OV degrees)."""
    from viam.proto.common import Orientation
    from viam.spatialmath import OrientationVector

    ov = OrientationVector.from_proto(
        Orientation(
            o_x=float(pose.o_x),
            o_y=float(pose.o_y),
            o_z=float(pose.o_z),
            theta=float(pose.theta),
        )
    )
    R = np.asarray(ov.to_quaternion().to_rotation_matrix().elements, dtype=float)
    R = R.reshape(3, 3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[0, 3] = float(pose.x)
    T[1, 3] = float(pose.y)
    T[2, 3] = float(pose.z)
    return T


def _matrix_to_mount(T: np.ndarray) -> MountPose:
    """Convert a base←sensor matrix (mm) into a nav-stack mount pose (m/rad).

    Yaw/pitch/roll are solved for nav-stack's ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``
    (see ``geom.conversions._mount_rotation``). Do **not** use Viam EulerAngles
    / OV θ directly: Viam's yaw sign for the same matrix is opposite ours on
    pure Z rotations (OV θ=-90° ≡ our mount θ=+π/2), which showed up as a
    consistent 90° CW localization error when mounts came from the framesystem.
    """
    R = np.asarray(T[:3, :3], dtype=float)
    # R = Rz(yaw) Ry(pitch) Rx(roll): R[2,0] = -sin(pitch).
    pitch = math.atan2(
        -float(R[2, 0]),
        math.sqrt(float(R[2, 1]) ** 2 + float(R[2, 2]) ** 2),
    )
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        # Gimbal lock: fold into yaw, zero roll.
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return MountPose(
        x=mm_to_m(float(T[0, 3])),
        y=mm_to_m(float(T[1, 3])),
        z=mm_to_m(float(T[2, 3])),
        theta=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
    )


def _frame_entries(configs: Sequence[Any]) -> Dict[str, Any]:
    """Map frame name → FrameSystemConfig / Transform."""
    out: Dict[str, Any] = {}
    for cfg in configs:
        frame = getattr(cfg, "frame", None) or cfg
        name = str(getattr(frame, "reference_frame", "") or "")
        if name:
            out[name] = frame
    return out


def pose_of_frame_in_destination(
    configs: Sequence[Any],
    frame_name: str,
    destination: str,
) -> Optional[MountPose]:
    """Compose framesystem transforms so ``frame_name`` is expressed in ``destination``.

    Each framesystem entry is the named frame's pose in its parent (observer).
    """
    frames = _frame_entries(configs)
    if frame_name not in frames and frame_name != destination:
        return None
    if frame_name == destination:
        return MountPose(0.0, 0.0, 0.0, 0.0)

    # Build dest_T_frame by walking frame → … → destination.
    T = np.eye(4, dtype=float)
    cur = frame_name
    seen = set()
    while cur != destination:
        if cur in seen:
            LOGGER.warning("framesystem cycle while resolving %s → %s", frame_name, destination)
            return None
        seen.add(cur)
        entry = frames.get(cur)
        if entry is None:
            LOGGER.warning(
                "framesystem missing frame %r while resolving %s → %s",
                cur,
                frame_name,
                destination,
            )
            return None
        pif = entry.pose_in_observer_frame
        parent = str(getattr(pif, "reference_frame", "") or "")
        if not parent:
            LOGGER.warning("framesystem frame %r has empty parent", cur)
            return None
        # entry pose is cur in parent → parent_T_cur
        parent_T_cur = _pose_to_matrix(pif.pose)
        # We have …_T_cur accumulating as we walk up: want dest_T_cur.
        # After step: parent_T_cur_full = parent_T_cur @ (old cur_T_leaf)
        T = parent_T_cur @ T
        cur = parent
        if len(seen) > 64:
            LOGGER.warning("framesystem chain too deep for %s → %s", frame_name, destination)
            return None
    return _matrix_to_mount(T)


def footprint_from_base_geometry(
    configs: Sequence[Any],
    base_name: str,
) -> Optional[FootprintBox]:
    """Read ``geometry.box`` on the base frame.

    Horizontal dims: longer side → length (forward), shorter → width (lateral).
    That matches typical differential bases (incl. Tracer 0.72×0.59) even when
    the configured box axes are swapped relative to Viam X-forward.
    """
    frames = _frame_entries(configs)
    entry = frames.get(base_name)
    if entry is None:
        return None
    geom = getattr(entry, "physical_object", None)
    if geom is None or not geom.ByteSize():
        return None
    if geom.WhichOneof("geometry_type") != "box":
        return None
    dims = geom.box.dims_mm
    dx = abs(mm_to_m(float(dims.x)))
    dy = abs(mm_to_m(float(dims.y)))
    if dx < 1e-3 or dy < 1e-3:
        return None
    length_m = max(dx, dy)
    width_m = min(dx, dy)
    return FootprintBox(length_m=length_m, width_m=width_m)


async def fetch_frame_system_config(
    robot, *, timeout_s: float = FRAME_SYSTEM_TIMEOUT_S
) -> List[Any]:
    """``robot.get_frame_system_config()`` on the caller's event loop.

    Must not be driven from a worker thread via ``asyncio.run`` — the module
    parent ``RobotClient`` is bound to the module loop and that pattern can
    deadlock reconfigure (seen after v1.0.40).
    """
    return list(
        await asyncio.wait_for(robot.get_frame_system_config(), timeout=timeout_s)
    )


def apply_framesystem_to_slam_cfg(
    cfg,
    configs: Sequence[Any],
    *,
    raw_attrs: Optional[Mapping] = None,
    logger: Optional[logging.Logger] = None,
):
    """Fill lidar mounts from framesystem when JSON omitted ``mount``.

    Mutates existing ``LidarConfig`` objects in place so live sensor facades
    that hold the same instances pick up mounts without a rebuild.

    Returns ``(cfg, notes)`` where notes are human-readable resolution lines.
    """
    log = logger or LOGGER
    raw = raw_attrs or {}
    raw_lidars = list(raw.get("lidars") or [])
    raw_by_name = {}
    for entry in raw_lidars:
        if isinstance(entry, Mapping) and entry.get("name"):
            raw_by_name[str(entry["name"])] = entry

    base_name = str(getattr(cfg, "base", "") or "base")
    notes: List[str] = []
    for lidar in cfg.lidars:
        raw_l = raw_by_name.get(lidar.name, {})
        explicit_mount = isinstance(raw_l, Mapping) and (
            "mount" in raw_l
            or any(k in raw_l for k in ("x", "y", "z", "theta", "pitch", "roll"))
        )
        if explicit_mount or bool(lidar.points_in_base_link):
            if explicit_mount:
                notes.append(f"lidar {lidar.name}: mount from config (override)")
            else:
                notes.append(f"lidar {lidar.name}: points_in_base_link — skip mount")
            continue
        mount = pose_of_frame_in_destination(configs, lidar.name, base_name)
        if mount is None:
            notes.append(
                f"lidar {lidar.name}: no framesystem frame named {lidar.name!r} "
                f"relative to {base_name!r}; keeping defaults "
                f"({lidar.x:.3f},{lidar.y:.3f},{lidar.z:.3f}) θ={lidar.theta:.3f}"
            )
            continue
        # Framesystem pose is the component frame → base. Point clouds are in
        # that same component frame, so do not also apply camera_optical remap.
        from .config import CLOUD_FRAME_CAMERA_OPTICAL, CLOUD_FRAME_SENSOR

        optical_note = ""
        if lidar.cloud_frame == CLOUD_FRAME_CAMERA_OPTICAL:
            lidar.cloud_frame = CLOUD_FRAME_SENSOR
            optical_note = "; cloud_frame sensor (FS pose is full base transform)"
        lidar.x = mount.x
        lidar.y = mount.y
        lidar.z = mount.z
        lidar.theta = mount.theta
        lidar.pitch = mount.pitch
        lidar.roll = mount.roll
        notes.append(
            f"lidar {lidar.name}: mount from framesystem "
            f"({mount.x:.3f},{mount.y:.3f},{mount.z:.3f}) "
            f"θ={mount.theta:.3f} pitch={mount.pitch:.3f} roll={mount.roll:.3f}"
            f"{optical_note}"
        )
    for line in notes:
        log.info("framesystem: %s", line)
    return cfg, notes


def apply_framesystem_to_nav_cfg(
    cfg,
    configs: Sequence[Any],
    *,
    raw_attrs: Optional[Mapping] = None,
    logger: Optional[logging.Logger] = None,
):
    """Fill footprint from base ``geometry.box`` when JSON omitted footprint_*."""
    from dataclasses import replace

    log = logger or LOGGER
    raw = raw_attrs or {}
    notes: List[str] = []
    explicit = ("footprint_length_m" in raw) or ("footprint_width_m" in raw)
    if explicit:
        notes.append(
            f"footprint from config "
            f"L={cfg.footprint_length_m} W={cfg.footprint_width_m} "
            f"r={cfg.robot_radius}"
        )
        log.info("framesystem: %s", notes[-1])
        return cfg, notes

    base_name = str(getattr(cfg, "base", "") or "base")
    box = footprint_from_base_geometry(configs, base_name)
    if box is None:
        notes.append(
            f"footprint: no box geometry on framesystem frame {base_name!r}; "
            f"keeping robot_radius={cfg.robot_radius}"
        )
        log.info("framesystem: %s", notes[-1])
        return cfg, notes

    # Only replace robot_radius when the user did not set it explicitly — if they
    # set radius alone (no footprint_*), keep radius but still adopt the box so
    # length/width drive gap clearance.
    explicit_radius = "robot_radius" in raw
    new_radius = cfg.robot_radius if explicit_radius else box.inscribed_radius_m
    cfg = replace(
        cfg,
        footprint_length_m=box.length_m,
        footprint_width_m=box.width_m,
        robot_radius=float(new_radius),
    )
    notes.append(
        f"footprint from framesystem box on {base_name!r}: "
        f"L={box.length_m:.3f} W={box.width_m:.3f} "
        f"inscribed={box.inscribed_radius_m:.3f} "
        f"robot_radius={cfg.robot_radius:.3f}"
        + (" (radius kept from config)" if explicit_radius else "")
    )
    log.info("framesystem: %s", notes[-1])
    return cfg, notes
