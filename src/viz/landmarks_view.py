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


# Status → (outline RGBA, fill RGBA, label tag)
# Tags: C/c/M = dense or generic; Dc/DC/DM = YOLO-pooled (from_detector).
_COLORS = {
    LandmarkStatus.CONFIRMED: ((40, 220, 90, 255), (40, 220, 90, 50), "C"),
    LandmarkStatus.CANDIDATE: ((240, 200, 40, 255), (240, 200, 40, 40), "c"),
    LandmarkStatus.MOVED: ((240, 60, 60, 255), (240, 60, 60, 40), "M"),
}
# Stronger translucent fills when nesting a detector landmark inside its YOLO box.
_NESTED_FILL = {
    LandmarkStatus.CONFIRMED: (40, 220, 90, 90),
    LandmarkStatus.CANDIDATE: (240, 200, 40, 85),
    LandmarkStatus.MOVED: (240, 60, 60, 70),
}
# YOLO / open-vocab detector boxes (distinct from landmark green/yellow).
_DET_OUTLINE = (80, 180, 255, 255)
_DET_FILL = (80, 180, 255, 28)
# Writeback match rings (navigation pose correction).
_WB_PAIRED = (255, 80, 220, 255)  # magenta = unique pair used in fit
_WB_VISIBLE = (255, 160, 40, 255)  # orange = in writeback pool, not paired


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
    writeback_visible_ids: Optional[Sequence[int]] = None,
    writeback_paired_ids: Optional[Sequence[int]] = None,
) -> bytes:
    """Draw landmark + optional YOLO boxes on ``color_rgb``; return PNG bytes.

    Legend: cyan = live YOLO box; ``Dc``/``DC`` = YOLO-pooled landmark
    (detector-anchored candidate/confirmed); plain ``c``/``C`` = dense patch
    landmark. Detector landmarks are nested as a translucent fill inside the
    matching YOLO box when the reprojected center lands in (or near) that box —
    a separate approximate square looked "offset" because map reprojection is a
    3D center, not the detection rectangle, and YOLO lags the live RGB.
    """
    opts = options or LandmarkOverlayOptions()
    wb_visible = {int(x) for x in (writeback_visible_ids or ())}
    wb_paired = {int(x) for x in (writeback_paired_ids or ())}
    arr = np.asarray(color_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return placeholder_landmarks_png("vision-camera: bad RGB shape")
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    # Scale YOLO boxes into this frame; draw cyan shells first so nested
    # landmark fills sit visually *inside* them.
    det_boxes: List[Tuple[float, float, float, float, Detection]] = []
    if opts.show_detections and detections:
        sx = sy = 1.0
        if detection_frame_size is not None:
            src_w, src_h = int(detection_frame_size[0]), int(detection_frame_size[1])
            if src_w > 0 and src_h > 0:
                sx = float(w) / float(src_w)
                sy = float(h) / float(src_h)
        for det in detections:
            if len(det_boxes) >= int(opts.max_detections):
                break
            x0 = float(det.x0) * sx
            y0 = float(det.y0) * sy
            x1 = float(det.x1) * sx
            y1 = float(det.y1) * sy
            if max(x0, x1) < -5 or max(y0, y1) < -5:
                continue
            if min(x0, x1) > w + 5 or min(y0, y1) > h + 5:
                continue
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            det_boxes.append((xa, ya, xb, yb, det))
            draw.rectangle(
                [xa, ya, xb, yb], outline=_DET_OUTLINE, width=2, fill=_DET_FILL
            )
            if opts.show_labels:
                draw.text(
                    (xa + 2, max(0.0, ya - 12)),
                    f"{det.label} {float(det.conf):.2f}",
                    fill=_DET_OUTLINE,
                )

    drawn = 0
    nested = 0
    counts = {s: 0 for s in LandmarkStatus}
    used_det: set = set()
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
        outline, fill, tag = _COLORS.get(
            status, ((200, 200, 200, 255), (200, 200, 200, 40), "?")
        )
        is_det = bool(getattr(lm, "from_detector", False))
        matched = None
        if is_det and det_boxes:
            matched = _match_detection_box(u, v, det_boxes, used=used_det)
        if matched is not None:
            # Nest translucent landmark fill inside the YOLO rectangle.
            xa, ya, xb, yb, _det = matched
            used_det.add(id(matched))
            nest_fill = _NESTED_FILL.get(status, fill)
            # Slight inset so cyan outline stays visible around the fill.
            bw, bh = max(1.0, xb - xa), max(1.0, yb - ya)
            ix0, iy0 = xa + 0.04 * bw, ya + 0.04 * bh
            ix1, iy1 = xb - 0.04 * bw, yb - 0.04 * bh
            draw.rectangle([ix0, iy0, ix1, iy1], outline=outline, width=3, fill=nest_fill)
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=outline)
            if opts.show_labels:
                draw.text(
                    (ix0 + 2, min(h - 14.0, iy1 - 14)),
                    f"D{tag}{lm.landmark_id}",
                    fill=outline,
                )
            nested += 1
        else:
            # Dense landmarks, or detector landmarks with no live YOLO match
            # (box dropped / lag): keep a small map-reprojected square.
            box_m = float(opts.box_m)
            half = max(6.0, float(intrinsics.fx) * box_m / max(0.2, depth))
            half = min(half, 48.0)
            x0, y0 = u - half, v - half
            x1, y1 = u + half, v + half
            draw.rectangle([x0, y0, x1, y1], outline=outline, width=2, fill=fill)
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=outline)
            if opts.show_labels:
                prefix = f"D{tag}" if is_det else tag
                draw.text(
                    (x0 + 2, max(0.0, y0 - 12)),
                    f"{prefix}{lm.landmark_id}",
                    fill=outline,
                )
        # Navigation writeback: orange = in match pool, magenta = unique pair.
        lid = int(lm.landmark_id)
        if lid in wb_paired:
            r = 14.0
            draw.ellipse(
                [u - r, v - r, u + r, v + r], outline=_WB_PAIRED, width=3
            )
            if opts.show_labels:
                draw.text((u + r + 2, v - 6), "P", fill=_WB_PAIRED)
        elif lid in wb_visible:
            r = 12.0
            draw.ellipse(
                [u - r, v - r, u + r, v + r], outline=_WB_VISIBLE, width=2
            )
            if opts.show_labels:
                draw.text((u + r + 2, v - 6), "V", fill=_WB_VISIBLE)
        counts[status] = counts.get(status, 0) + 1
        drawn += 1

    det_drawn = len(det_boxes)
    if yolo_loaded is False:
        yolo_hud = "yolo=OFF"
    else:
        yolo_hud = f"yolo={det_drawn} nest={nested}"
    hud = (
        f"landmarks drawn={drawn}  "
        f"C={counts.get(LandmarkStatus.CONFIRMED, 0)} "
        f"c={counts.get(LandmarkStatus.CANDIDATE, 0)} "
        f"M={counts.get(LandmarkStatus.MOVED, 0)}  "
        f"wbV={len(wb_visible)} wbP={len(wb_paired)}  "
        f"{yolo_hud}  "
        f"pose=({robot_pose.x:.2f},{robot_pose.y:.2f},{math.degrees(robot_pose.theta):.0f}°)"
    )
    # Two-line HUD: counts + legend for Dc.
    draw.rectangle([0, 0, w, 32], fill=(0, 0, 0, 150))
    draw.text((4, 2), hud, fill=(230, 230, 230, 255))
    draw.text(
        (4, 16),
        "cyan=YOLO  orange-V=wb pool  magenta-P=wb pair  c/C=map",
        fill=(180, 200, 220, 255),
    )

    out = Image.alpha_composite(img, overlay)
    return _encode_png(out)


def _match_detection_box(
    u: float,
    v: float,
    det_boxes: Sequence[Tuple[float, float, float, float, Detection]],
    *,
    used: set,
    pad_frac: float = 0.12,
) -> Optional[Tuple[float, float, float, float, Detection]]:
    """Pick the YOLO box that contains ``(u,v)``, else nearest center (loose)."""
    containing = []
    nearest = None
    nearest_d = float("inf")
    for box in det_boxes:
        if id(box) in used:
            continue
        x0, y0, x1, y1, _det = box
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        # Allow a little pad — map reprojection often sits just outside the box.
        px0, py0 = x0 - pad_frac * bw, y0 - pad_frac * bh
        px1, py1 = x1 + pad_frac * bw, y1 + pad_frac * bh
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        d = math.hypot(u - cx, v - cy)
        if px0 <= u <= px1 and py0 <= v <= py1:
            containing.append((d, box))
        if d < nearest_d:
            nearest_d = d
            nearest = box
    if containing:
        containing.sort(key=lambda t: t[0])
        return containing[0][1]
    # Fallback: center within ~half-diagonal of the box.
    if nearest is not None:
        x0, y0, x1, y1, _ = nearest
        diag = 0.5 * math.hypot(x1 - x0, y1 - y0)
        if nearest_d <= max(24.0, 0.65 * diag):
            return nearest
    return None


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
            # Detector-anchored landmarks are already object-level; show them
            # immediately so cyan YOLO boxes have matching yellow/green marks.
            if not bool(getattr(lm, "from_detector", False)):
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
    # Prefer confirmed, then detector-anchored, so the budget keeps useful ones.
    order = {
        LandmarkStatus.CONFIRMED: 0,
        LandmarkStatus.CANDIDATE: 1,
        LandmarkStatus.MOVED: 2,
    }
    out.sort(
        key=lambda pair: (
            order.get(pair[0].status, 9),
            0 if bool(getattr(pair[0], "from_detector", False)) else 1,
            -pair[0].observations,
        )
    )
    return out
