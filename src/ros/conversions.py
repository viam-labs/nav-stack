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
from typing import List, Sequence, Tuple

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


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw (rotation about +Z) from a quaternion, in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass(frozen=True)
class Pose2D:
    """A 2D pose in the map frame, in meters and radians."""

    x: float
    y: float
    theta: float

    def to_matrix(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]])


def viam_pose_to_pose2d(x_mm: float, y_mm: float, theta_deg: float) -> Pose2D:
    """Convert a Viam SLAM ``Pose`` (mm, degrees) into a ROS-style ``Pose2D``."""
    return Pose2D(mm_to_m(x_mm), mm_to_m(y_mm), math.radians(theta_deg))


def pose2d_to_viam_pose(pose: Pose2D) -> Tuple[float, float, float]:
    """Convert a ``Pose2D`` (m, rad) into Viam SLAM ``Pose`` fields (mm, degrees).

    Returns ``(x_mm, y_mm, theta_deg)``.
    """
    return (m_to_mm(pose.x), m_to_mm(pose.y), math.degrees(pose.theta))


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


# ---------------------------------------------------------------------------
# LaserScan <-> points, and multi-lidar merge
# ---------------------------------------------------------------------------
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


def pointcloud_to_scan(
    points: np.ndarray,
    z_min: float = -0.2,
    z_max: float = 2.0,
    sensor_pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
    **scan_kwargs,
) -> LaserScan2D:
    """Flatten a 3D point cloud (e.g. from a depth camera) into a 2D LaserScan.

    Points outside ``[z_min, z_max]`` (sensor frame) are discarded before projection.
    """
    points = np.asarray(points, dtype=float)
    if points.size and points.shape[1] >= 3:
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        points = points[mask]
    xy = points[:, :2] if points.size else np.empty((0, 2))
    if xy.size:
        homog = np.stack([xy[:, 0], xy[:, 1], np.ones(len(xy))], axis=1)
        xy = (sensor_pose.to_matrix() @ homog.T).T[:, :2]
    scan = points_to_scan(xy, **scan_kwargs)
    return scan


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


def pack_pcd_xyz_from_bytes(raw: bytes) -> np.ndarray:
    """Parse a minimal binary PCD (x/y/z float32) back into an ``(N, 3)`` array.

    Provided mainly for tests / round-tripping ``points_to_pcd``.
    """
    text = raw.split(b"DATA binary\n", 1)
    if len(text) != 2:
        raise ValueError("not a binary PCD produced by points_to_pcd")
    body = text[1]
    count = len(body) // 12
    out = np.frombuffer(body[: count * 12], dtype=np.float32).reshape(-1, 3)
    return out
