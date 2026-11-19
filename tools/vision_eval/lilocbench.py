"""Read a LILocBench sequence (see ``tools/fetch_lilocbench.py``).

One sequence folder holds colour and depth PNGs named by timestamp, lidar
ranges as text, calibration yaml and ground-truth poses of ``base_link`` in a
map frame shared by every sequence.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

from src.geom import conversions as conv
from src.nav.localize.global_localize import OccupancyMap
from src.semantic.types import CameraIntrinsics, VisionFrame

# RealSense depth PNGs are uint16 millimetres.
DEPTH_UNIT_M = 0.001
# A frame is used only when ground truth brackets it this closely.
MAX_GT_GAP_S = 0.25
MAX_SCAN_GAP_S = 0.10


def _quat_matrix(q: Dict[str, float]) -> np.ndarray:
    x, y, z, w = (float(q[k]) for k in ("x", "y", "z", "w"))
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _transform(entry: dict) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _quat_matrix(entry["rotation"])
    t = entry["translation"]
    out[:3, 3] = (float(t["x"]), float(t["y"]), float(t["z"]))
    return out


def mount_from_matrix(
    mat: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """``(x, y, z, theta, pitch, roll)`` for ``Rz(theta) Ry(pitch) Rx(roll)``."""
    rot = mat[:3, :3]
    theta = math.atan2(rot[1, 0], rot[0, 0])
    pitch = -math.asin(max(-1.0, min(1.0, float(rot[2, 0]))))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    return float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]), theta, pitch, roll


def _intrinsics(path: Path) -> CameraIntrinsics:
    data = yaml.safe_load(path.read_text())
    k = [float(v) for v in data["K"]]
    return CameraIntrinsics(
        fx=k[0],
        fy=k[4],
        cx=k[2],
        cy=k[5],
        width=int(data["width"]),
        height=int(data["height"]),
    )


def register_depth_to_color(
    depth_m: np.ndarray,
    depth_k: CameraIntrinsics,
    color_k: CameraIntrinsics,
    color_from_depth: np.ndarray,
) -> np.ndarray:
    """Re-project a depth image into the colour camera (nearest surface wins).

    The two sensors sit about 6 cm apart, so a depth pixel does not belong to
    the colour pixel with the same coordinates.
    """
    h, w = depth_m.shape
    vs, us = np.nonzero(depth_m > 0.0)
    out = np.zeros((color_k.height, color_k.width), dtype=np.float32)
    if us.size == 0:
        return out
    z = depth_m[vs, us].astype(np.float64)
    x = (us - depth_k.cx) / depth_k.fx * z
    y = (vs - depth_k.cy) / depth_k.fy * z
    pts = (
        np.column_stack([x, y, z]) @ color_from_depth[:3, :3].T
        + color_from_depth[:3, 3]
    )
    zc = pts[:, 2]
    ok = zc > 0.05
    pts, zc = pts[ok], zc[ok]
    uc = np.round(pts[:, 0] / zc * color_k.fx + color_k.cx).astype(np.int64)
    vc = np.round(pts[:, 1] / zc * color_k.fy + color_k.cy).astype(np.int64)
    inside = (uc >= 0) & (uc < color_k.width) & (vc >= 0) & (vc < color_k.height)
    flat = vc[inside] * color_k.width + uc[inside]
    buf = np.full(color_k.height * color_k.width, np.inf)
    np.minimum.at(buf, flat, zc[inside])
    buf[~np.isfinite(buf)] = 0.0
    return buf.reshape(color_k.height, color_k.width).astype(np.float32)


def interpolate_pose(
    stamps: np.ndarray, poses: np.ndarray, t: float, *, max_gap_s: float = MAX_GT_GAP_S
) -> Optional[conv.Pose2D]:
    """Pose at ``t`` from ``(N, 3)`` rows of x, y, yaw; ``None`` across a gap."""
    i = bisect.bisect_left(stamps, t)
    if i <= 0 or i >= len(stamps):
        return None
    t0, t1 = float(stamps[i - 1]), float(stamps[i])
    if t1 - t0 > max_gap_s:
        return None
    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    x = (1 - a) * poses[i - 1, 0] + a * poses[i, 0]
    y = (1 - a) * poses[i - 1, 1] + a * poses[i, 1]
    dyaw = conv.normalize_angle(float(poses[i, 2] - poses[i - 1, 2]))
    yaw = conv.normalize_angle(float(poses[i - 1, 2]) + a * dyaw)
    return conv.Pose2D(float(x), float(y), float(yaw))


def load_tum_poses(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """TUM rows ``t x y z qx qy qz qw`` → stamps and ``(N, 3)`` x, y, yaw."""
    raw = np.loadtxt(path)
    qx, qy, qz, qw = raw[:, 4], raw[:, 5], raw[:, 6], raw[:, 7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return raw[:, 0].copy(), np.column_stack([raw[:, 1], raw[:, 2], yaw])


@dataclass
class _Lidar:
    stamps: np.ndarray
    ranges: np.ndarray  # (N, beams) float32
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    sensor_pose: conv.Pose2D


class LilocSequence:
    """Colour/depth frames, merged lidar scans and ground truth for one run."""

    def __init__(self, root: Path, name: str, *, camera: str = "front") -> None:
        self.name = str(name)
        self.dir = Path(root).expanduser() / name
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"{self.dir} not found; run tools/fetch_lilocbench.py --sequences {name}"
            )
        self.camera = str(camera)
        cam_dir = self.dir / f"camera_{camera}"
        self.color_k = _intrinsics(cam_dir / "color" / "intrinsics.yaml")
        self.depth_k = _intrinsics(cam_dir / "depth" / "intrinsics.yaml")

        tf = {
            (e["parent"], e["child"]): _transform(e)
            for e in yaml.safe_load((self.dir / "transformations.yaml").read_text())
        }
        self._tf = tf
        link = f"camera_{camera}_link"
        base_from_link = tf[("base_link", link)]
        link_from_color = tf[(link, f"camera_{camera}_color_frame")]
        link_from_depth = tf[(link, f"camera_{camera}_depth_frame")]
        optical = tf[
            (f"camera_{camera}_color_frame", f"camera_{camera}_color_optical_frame")
        ]
        # Colour sensor body (x forward) in base_link: the stack's mount.
        self.mount = mount_from_matrix(base_from_link @ link_from_color)
        # Depth optical → colour optical, for registration.
        self._color_from_depth = np.linalg.inv(link_from_color @ optical) @ (
            link_from_depth @ optical
        )

        self.gt_stamps, self.gt_poses = load_tum_poses(self.dir / "gt_poses.txt")

        color = {float(p.stem): p for p in (cam_dir / "color" / "images").glob("*.png")}
        depth = {float(p.stem): p for p in (cam_dir / "depth" / "images").glob("*.png")}
        depth_stamps = sorted(depth)
        self.stamps: List[float] = []
        self.poses: List[conv.Pose2D] = []
        self._color_paths: List[Path] = []
        self._depth_paths: List[Path] = []
        for stamp in sorted(color):
            pose = interpolate_pose(self.gt_stamps, self.gt_poses, stamp)
            if pose is None or not depth_stamps:
                continue
            i = bisect.bisect_left(depth_stamps, stamp)
            best = min(
                (j for j in (i - 1, i) if 0 <= j < len(depth_stamps)),
                key=lambda j: abs(depth_stamps[j] - stamp),
            )
            if abs(depth_stamps[best] - stamp) > 0.05:
                continue
            self.stamps.append(stamp)
            self.poses.append(pose)
            self._color_paths.append(color[stamp])
            self._depth_paths.append(depth[depth_stamps[best]])
        self._lidars: Optional[List[_Lidar]] = None

    def __len__(self) -> int:
        return len(self.stamps)

    def color(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self._color_paths[i]).convert("RGB"))

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres, registered to the colour image."""
        raw = (
            np.asarray(Image.open(self._depth_paths[i])).astype(np.float32)
            * DEPTH_UNIT_M
        )
        return register_depth_to_color(
            raw, self.depth_k, self.color_k, self._color_from_depth
        )

    def frame(self, i: int, *, with_depth: bool = True) -> VisionFrame:
        color = self.color(i)
        depth = (
            self.depth(i) if with_depth else np.zeros(color.shape[:2], dtype=np.float32)
        )
        x, y, z, theta, pitch, roll = self.mount
        return VisionFrame(
            camera_name=f"camera_{self.camera}",
            stamp=float(self.stamps[i]),
            captured_at=float(self.stamps[i]),
            color_rgb=color,
            depth_m=depth,
            intrinsics=self.color_k,
            mount_x=x,
            mount_y=y,
            mount_z=z,
            mount_theta=theta,
            mount_pitch=pitch,
            mount_roll=roll,
        )

    def _load_lidars(self) -> List[_Lidar]:
        if self._lidars is None:
            out: List[_Lidar] = []
            for which in ("front", "rear"):
                folder = self.dir / f"laser_scan_{which}"
                if not (folder / "ranges.txt").is_file():
                    continue
                spec = yaml.safe_load((folder / "laser_specs.yaml").read_text())
                raw = np.loadtxt(folder / "ranges.txt", dtype=np.float64)
                mat = self._tf[("base_link", f"scan_{which}_link")]
                yaw = math.atan2(mat[1, 0], mat[0, 0])
                out.append(
                    _Lidar(
                        stamps=raw[:, 0].copy(),
                        ranges=raw[:, 1:].astype(np.float32),
                        angle_min=float(spec["angle_min"]),
                        angle_increment=float(spec["angle_increment"]),
                        range_min=float(spec["range_min"]),
                        range_max=min(float(spec["range_max"]), 25.0),
                        sensor_pose=conv.Pose2D(
                            float(mat[0, 3]), float(mat[1, 3]), yaw
                        ),
                    )
                )
            self._lidars = out
        return self._lidars

    def scan(
        self, stamp: float, *, which: Sequence[str] = ("front", "rear")
    ) -> Optional[conv.LaserScan2D]:
        """Both lidars nearest ``stamp``, merged around base_link like the stack does."""
        del which
        scans = []
        for lidar in self._load_lidars():
            i = int(np.argmin(np.abs(lidar.stamps - stamp)))
            if abs(float(lidar.stamps[i]) - stamp) > MAX_SCAN_GAP_S:
                continue
            scans.append(
                conv.LaserScan2D(
                    ranges=lidar.ranges[i].astype(float),
                    angle_min=lidar.angle_min,
                    angle_increment=lidar.angle_increment,
                    range_min=lidar.range_min,
                    range_max=lidar.range_max,
                    sensor_pose=lidar.sensor_pose,
                )
            )
        if not scans:
            return None
        return conv.merge_scans(scans)

    def odometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wheel odometry: stamps, ``(N, 3)`` x/y/yaw in the odom frame, ``(N, 3)`` vx/vy/wz.

        Rows are ``t pos[3] quat[4] pose_cov[36] vel_t[3] vel_r[3] vel_cov[36]``.
        """
        raw = np.loadtxt(self.dir / "dingo_velocity_controller" / "odom.txt")
        qx, qy, qz, qw = raw[:, 4], raw[:, 5], raw[:, 6], raw[:, 7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        poses = np.column_stack([raw[:, 1], raw[:, 2], yaw])
        twists = np.column_stack([raw[:, 44], raw[:, 45], raw[:, 49]])
        return raw[:, 0].copy(), poses, twists

    def scan_stamps(self) -> np.ndarray:
        """Stamps of the front lidar (the replay's scan clock)."""
        return self._load_lidars()[0].stamps

    def scan_before(self, stamp: float) -> Tuple[Optional[conv.LaserScan2D], float]:
        """Latest merged scan taken at or before ``stamp``, and when it was taken."""
        lidars = self._load_lidars()
        i = int(np.searchsorted(lidars[0].stamps, stamp, side="right")) - 1
        if i < 0:
            return None, float("nan")
        taken = float(lidars[0].stamps[i])
        return self.scan(taken), taken

    def gt_pose(self, stamp: float) -> Optional[conv.Pose2D]:
        return interpolate_pose(self.gt_stamps, self.gt_poses, stamp)


def load_reference_map(root: Path) -> OccupancyMap:
    """The dataset's occupancy grid (png + yaml), in the GT frame."""
    folder = Path(root).expanduser() / "map_office"
    meta = yaml.safe_load((folder / "map_office.yaml").read_text())
    img = np.asarray(Image.open(folder / str(meta["image"])).convert("L")).astype(
        np.float32
    )
    occ_prob = img / 255.0 if int(meta.get("negate", 0)) else (255.0 - img) / 255.0
    grid = np.full(img.shape, -1, dtype=np.int16)
    grid[occ_prob > float(meta["occupied_thresh"])] = 100
    grid[occ_prob < float(meta["free_thresh"])] = 0
    # Image row 0 is the top of the map; grid row 0 is at origin_y.
    grid = np.flipud(grid).copy()
    return OccupancyMap(
        grid=grid,
        resolution=float(meta["resolution"]),
        origin_x=float(meta["origin"][0]),
        origin_y=float(meta["origin"][1]),
    )
