"""Overlay embedding landmarks onto a live RGB frame (Pillow)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..geom import conversions as conv
from ..semantic.landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from ..semantic.types import CameraIntrinsics, Detection, LandmarkStatus


# Status → (outline RGBA, fill RGBA, label)
_COLORS = {
    LandmarkStatus.CONFIRMED: ((40, 220, 90, 255), (40, 220, 90, 50), "C"),
    LandmarkStatus.CANDIDATE: ((240, 200, 40, 255), (240, 200, 40, 40), "c"),
    LandmarkStatus.MOVED: ((240, 60, 60, 255), (240, 60, 60, 40), "M"),
}
# YOLO / open-vocab detector boxes (distinct from landmark green/yellow).
_DET_OUTLINE = (80, 180, 255, 255)
_DET_FILL = (80, 180, 255, 36)


@dataclass
class LandmarkOverlayOptions:
    show_confirmed: bool = True
    show_candidates: bool = True
    show_moved: bool = False
    show_detections: bool = True
    show_labels: bool = True
    max_landmarks: int = 80
    max_detections: int = 40
    box_m: float = 0.12  # approximate landmark size in meters for box scale


def sensor_to_camera_optical(points: np.ndarray) -> np.ndarray:
    """Inverse of ``camera_optical_to_sensor_frame``."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3))
    if points.shape[1] == 2:
        points = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    out = np.empty((len(points), 3), dtype=float)
    out[:, 0] = -points[:, 1]  # optical X = -sensor Y
    out[:, 1] = -points[:, 2]  # optical Y = -sensor Z
    out[:, 2] = points[:, 0]  # optical Z = sensor X
    return out


def project_map_point_to_pixel(
    map_xyz: np.ndarray,
    *,
    robot_pose: conv.Pose2D,
    intrinsics: CameraIntrinsics,
    mount_x: float,
    mount_y: float,
    mount_z: float,
    mount_theta: float,
    mount_pitch: float = 0.0,
    mount_roll: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    """Map-frame point → ``(u, v, depth_m)`` in the camera image, or None."""
    # Map → base_link.
    dx = float(map_xyz[0]) - robot_pose.x
    dy = float(map_xyz[1]) - robot_pose.y
    c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    base = np.array(
        [
            [
                c * dx + s * dy,
                -s * dx + c * dy,
                float(map_xyz[2]),
            ]
        ],
        dtype=float,
    )
    sensor = conv.transform_base_link_to_lidar_mount(
        base,
        x=mount_x,
        y=mount_y,
        z=mount_z,
        theta=mount_theta,
        pitch=mount_pitch,
        roll=mount_roll,
    )
    optical = sensor_to_camera_optical(sensor)[0]
    z = float(optical[2])
    if z < 0.15:
        return None
    u = float(intrinsics.fx) * float(optical[0]) / z + float(intrinsics.cx)
    v = float(intrinsics.fy) * float(optical[1]) / z + float(intrinsics.cy)
    return u, v, z


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def placeholder_landmarks_png(text: str) -> bytes:
    img = Image.new("RGB", (640, 360), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    draw.text((24, 160), text, fill=(220, 220, 220))
    return _encode_png(img)


def render_landmarks_overlay(
    color_rgb: np.ndarray,
    landmarks: Sequence[Tuple[EmbeddingLandmark, np.ndarray]],
    *,
    robot_pose: conv.Pose2D,
    intrinsics: CameraIntrinsics,
    mount_x: float = 0.0,
    mount_y: float = 0.0,
    mount_z: float = 0.0,
    mount_theta: float = 0.0,
    mount_pitch: float = 0.0,
    mount_roll: float = 0.0,
    options: Optional[LandmarkOverlayOptions] = None,
    detections: Optional[Sequence[Detection]] = None,
    detection_frame_size: Optional[Tuple[int, int]] = None,
    yolo_loaded: Optional[bool] = None,
) -> bytes:
    """Draw landmark + optional YOLO boxes on ``color_rgb``; return PNG bytes."""
    opts = options or LandmarkOverlayOptions()
    arr = np.asarray(color_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return placeholder_landmarks_png("vision-camera: bad RGB shape")
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    drawn = 0
    counts = {s: 0 for s in LandmarkStatus}
    for lm, map_xyz in landmarks:
        if drawn >= int(opts.max_landmarks):
            break
        status = lm.status
        if status == LandmarkStatus.CONFIRMED and not opts.show_confirmed:
            continue
        if status == LandmarkStatus.CANDIDATE and not opts.show_candidates:
            continue
        if status == LandmarkStatus.MOVED and not opts.show_moved:
            continue
        proj = project_map_point_to_pixel(
            map_xyz,
            robot_pose=robot_pose,
            intrinsics=intrinsics,
            mount_x=mount_x,
            mount_y=mount_y,
            mount_z=mount_z,
            mount_theta=mount_theta,
            mount_pitch=mount_pitch,
            mount_roll=mount_roll,
        )
        if proj is None:
            continue
        u, v, depth = proj
        if u < -20 or v < -20 or u > w + 20 or v > h + 20:
            continue
        # Box half-size from approximate physical size at this depth.
        half = max(6.0, float(intrinsics.fx) * float(opts.box_m) / max(0.2, depth))
        half = min(half, 48.0)
        x0, y0 = u - half, v - half
        x1, y1 = u + half, v + half
        outline, fill, tag = _COLORS.get(
            status, ((200, 200, 200, 255), (200, 200, 200, 40), "?")
        )
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2, fill=fill)
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=outline)
        if opts.show_labels:
            label = f"{tag}{lm.landmark_id}"
            draw.text((x0 + 2, max(0.0, y0 - 12)), label, fill=outline)
        counts[status] = counts.get(status, 0) + 1
        drawn += 1

    det_drawn = 0
    if opts.show_detections and detections:
        sx = sy = 1.0
        if detection_frame_size is not None:
            src_w, src_h = int(detection_frame_size[0]), int(detection_frame_size[1])
            if src_w > 0 and src_h > 0:
                sx = float(w) / float(src_w)
                sy = float(h) / float(src_h)
        for det in detections:
            if det_drawn >= int(opts.max_detections):
                break
            x0 = float(det.x0) * sx
            y0 = float(det.y0) * sy
            x1 = float(det.x1) * sx
            y1 = float(det.y1) * sy
            if max(x0, x1) < -5 or max(y0, y1) < -5:
                continue
            if min(x0, x1) > w + 5 or min(y0, y1) > h + 5:
                continue
            draw.rectangle(
                [x0, y0, x1, y1], outline=_DET_OUTLINE, width=2, fill=_DET_FILL
            )
            if opts.show_labels:
                label = f"{det.label} {float(det.conf):.2f}"
                draw.text(
                    (min(x0, x1) + 2, max(0.0, min(y0, y1) - 12)),
                    label,
                    fill=_DET_OUTLINE,
                )
            det_drawn += 1

    # HUD strip. Distinguish detector-off from detector-quiet (dense fallback
    # can still mint candidates when YOLO is missing).
    if yolo_loaded is False:
        yolo_hud = "yolo=OFF"
    elif yolo_loaded is True:
        yolo_hud = f"yolo={det_drawn}"
    else:
        yolo_hud = f"yolo={det_drawn}"
    hud = (
        f"landmarks drawn={drawn}  "
        f"C={counts.get(LandmarkStatus.CONFIRMED, 0)} "
        f"c={counts.get(LandmarkStatus.CANDIDATE, 0)} "
        f"M={counts.get(LandmarkStatus.MOVED, 0)}  "
        f"{yolo_hud}  "
        f"pose=({robot_pose.x:.2f},{robot_pose.y:.2f},{math.degrees(robot_pose.theta):.0f}°)"
    )
    draw.rectangle([0, 0, w, 18], fill=(0, 0, 0, 140))
    draw.text((4, 2), hud, fill=(230, 230, 230, 255))

    out = Image.alpha_composite(img, overlay)
    return _encode_png(out)


def _candidate_geometry_trusted(lm: EmbeddingLandmark) -> bool:
    """Skip single-view / high-std candidates — their depth is often glass."""
    if lm.observations < 3:
        return False
    if len(lm.view_poses_xy) < 2:
        return False
    std = float(lm.position_std_m)
    if not math.isfinite(std) or std > 0.12:
        return False
    return True


def collect_landmark_map_points(
    store: EmbeddingLandmarkMap,
    pose_of_node: dict,
    *,
    include_candidates: bool = True,
    include_moved: bool = False,
    stable_candidates_only: bool = True,
) -> List[Tuple[EmbeddingLandmark, np.ndarray]]:
    """Resolve landmark map-frame XYZ for overlay (not frustum-culled)."""
    out: List[Tuple[EmbeddingLandmark, np.ndarray]] = []
    for lm in store.all_landmarks():
        if lm.status == LandmarkStatus.CANDIDATE and not include_candidates:
            continue
        if lm.status == LandmarkStatus.MOVED and not include_moved:
            continue
        if lm.status == LandmarkStatus.CANDIDATE and stable_candidates_only:
            if not _candidate_geometry_trusted(lm):
                continue
        if lm.status == LandmarkStatus.CONFIRMED or (
            lm.status == LandmarkStatus.CANDIDATE and include_candidates
        ) or (lm.status == LandmarkStatus.MOVED and include_moved):
            if lm.map_xyz is None and lm.anchor_node not in pose_of_node:
                continue
            # Prefer running mean for stable candidates (better than first hit).
            assoc = EmbeddingLandmarkMap.association_map_xyz(lm)
            mxyz = (
                assoc
                if assoc is not None
                else EmbeddingLandmarkMap.resolved_map_xyz(lm, pose_of_node)
            )
            out.append((lm, mxyz))
    # Prefer confirmed first so max_landmarks budget keeps the useful ones.
    order = {
        LandmarkStatus.CONFIRMED: 0,
        LandmarkStatus.CANDIDATE: 1,
        LandmarkStatus.MOVED: 2,
    }
    out.sort(key=lambda pair: (order.get(pair[0].status, 9), -pair[0].observations))
    return out
