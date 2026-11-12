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
# Depth audit: the camera sees well past this landmark's stored point.
_SEEN_THROUGH = (255, 40, 40, 255)
# YOLO / open-vocab detector boxes (distinct from landmark green/yellow).
_DET_OUTLINE = (80, 180, 255, 255)
_DET_FILL = (80, 180, 255, 28)
# A box that made no landmark observation this frame (too far, mixed depth…).
_DET_SKIPPED = (170, 170, 170, 255)
# Two text lines at the top of the frame.
_HUD_HEIGHT = 32
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


def _ray_in_camera_view(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
    *,
    max_range_m: Optional[float],
    fov_deg: Optional[float],
    vfov_deg: Optional[float],
) -> bool:
    """True when the back-projected ray is inside the configured camera volume.

    The pixel test alone keeps every map point in an infinite wedge, and a
    too-wide K pulls bearings that are off to the side back into the frame.
    Reconstruct the optical ray and apply the camera's own range and FOV.
    """
    if max_range_m is None and fov_deg is None and vfov_deg is None:
        return True
    z = float(depth)
    if z <= 1e-6:
        return False
    fx = max(float(intrinsics.fx), 1e-6)
    fy = max(float(intrinsics.fy), 1e-6)
    ox = (float(u) - float(intrinsics.cx)) * z / fx
    oy = (float(v) - float(intrinsics.cy)) * z / fy
    if max_range_m is not None and math.hypot(ox, oy, z) > float(max_range_m):
        return False
    # 1° of slack so a point on the image edge is not dropped by rounding.
    slack = math.radians(1.0)
    if fov_deg is not None and float(fov_deg) > 1.0:
        if abs(math.atan2(ox, z)) > math.radians(float(fov_deg)) * 0.5 + slack:
            return False
    if vfov_deg is not None and float(vfov_deg) > 1.0:
        if abs(math.atan2(oy, z)) > math.radians(float(vfov_deg)) * 0.5 + slack:
            return False
    return True


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
    seen_through_ids: Optional[Sequence[int]] = None,
    detection_notes: Optional[Sequence[Optional[dict]]] = None,
    max_range_m: Optional[float] = None,
    fov_deg: Optional[float] = None,
    vfov_deg: Optional[float] = None,
) -> bytes:
    """Draw landmark + optional YOLO boxes on ``color_rgb``; return PNG bytes.

    Legend: cyan = live YOLO box; ``Dc``/``DC`` = YOLO-pooled landmark
    (detector-anchored candidate/confirmed); plain ``c``/``C`` = dense patch
    landmark. Every landmark is a small square at the projection of its own
    stored point, never the YOLO rectangle: a landmark stored in front of or
    behind an object still projects inside that object's box, and painting the
    box with its ID made a wrong position look right. A red cross marks a
    landmark the depth audit sees past (``seen_through_ids``).

    ``detection_notes`` (one per detection, from the survey) says what each
    box became this frame. A box that fed a landmark stays cyan, is labelled
    ``=DC17`` and gets a line to that landmark's square; a box that fed
    nothing is grey and labelled with the reason (``far 9.6m``,
    ``mixed depth``, ``no depth``).

    Stored points are not drawn just because they fall in an infinite pinhole
    wedge. ``max_range_m`` drops anything farther than the camera is configured
    to see. ``fov_deg`` / ``vfov_deg`` drop rays outside that lens, including
    when a too-wide K would still land the pixel inside the frame. An
    untrusted detector candidate is drawn only when this frame's notes tie a
    box to it — a one-shot fix elsewhere on the map is not reprojected.
    """
    opts = options or LandmarkOverlayOptions()
    seen_through = {int(x) for x in (seen_through_ids or ())}
    wb_visible = {int(x) for x in (writeback_visible_ids or ())}
    wb_paired = {int(x) for x in (writeback_paired_ids or ())}
    arr = np.asarray(color_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return placeholder_landmarks_png("vision-camera: bad RGB shape")
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    # Scale YOLO boxes into this frame; drawn first so landmarks sit on top.
    det_boxes: List[Tuple[float, float, float, float, Detection]] = []
    notes: Sequence[Optional[dict]] = ()
    if detections and detection_notes is not None:
        if len(detection_notes) == len(detections):
            notes = detection_notes
    # Landmark id → centre of the box that fed it this frame.
    fed_from: dict = {}
    boxes_used = 0
    if opts.show_detections and detections:
        sx = sy = 1.0
        if detection_frame_size is not None:
            src_w, src_h = int(detection_frame_size[0]), int(detection_frame_size[1])
            if src_w > 0 and src_h > 0:
                sx = float(w) / float(src_w)
                sy = float(h) / float(src_h)
        for i, det in enumerate(detections):
            if len(det_boxes) >= int(opts.max_detections):
                break
            note = notes[i] if notes else None
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
            used = note is None or note.get("outcome") == "used"
            box_color = _DET_OUTLINE if used else _DET_SKIPPED
            draw.rectangle(
                [xa, ya, xb, yb],
                outline=box_color,
                width=2 if used else 1,
                fill=_DET_FILL if used else None,
            )
            label = f"{det.label} {float(det.conf):.2f}"
            if note is not None:
                label = f"{det.label} {note.get('text') or ''}".rstrip()
                if used:
                    boxes_used += 1
                    if note.get("landmark_id") is not None:
                        fed_from[int(note["landmark_id"])] = (
                            0.5 * (xa + xb),
                            0.5 * (ya + yb),
                        )
            if opts.show_labels:
                # Keep the label on screen: clear of the right edge and, for a
                # box under the HUD, just inside its top edge instead.
                lx = max(0.0, min(xa + 2, float(w) - draw.textlength(label) - 2))
                ly = ya - 12 if ya - 12 >= _HUD_HEIGHT else max(ya, _HUD_HEIGHT) + 2
                draw.text((lx, ly), label, fill=box_color)

    drawn = 0
    flagged = 0
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
        is_det = bool(getattr(lm, "from_detector", False))
        lid = int(lm.landmark_id)
        # A single YOLO hit is stored immediately, often metres off the object.
        # Until geometry is trusted, only draw it when this frame's box fed it.
        if (
            notes
            and status == LandmarkStatus.CANDIDATE
            and is_det
            and lid not in fed_from
            and not _candidate_geometry_trusted(lm)
        ):
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
        if not _ray_in_camera_view(
            u,
            v,
            depth,
            intrinsics,
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            vfov_deg=vfov_deg,
        ):
            continue
        if u < -20 or v < -20 or u > w + 20 or v > h + 20:
            continue
        outline, fill, tag = _COLORS.get(
            status, ((200, 200, 200, 255), (200, 200, 200, 40), "?")
        )
        is_air = lid in seen_through
        if is_air:
            outline = _SEEN_THROUGH
            flagged += 1
        # Square scaled by stored depth: a near (wrong) point draws larger
        # than the far object it sits in front of.
        box_m = float(opts.box_m)
        half = max(6.0, float(intrinsics.fx) * box_m / max(0.2, depth))
        half = min(half, 48.0)
        x0, y0 = u - half, v - half
        x1, y1 = u + half, v + half
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2, fill=fill)
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=outline)
        if lid in fed_from:
            # Tie the square to the box that was matched to it this frame.
            draw.line([fed_from[lid], (u, v)], fill=outline, width=1)
        if is_air:
            draw.line([x0, y0, x1, y1], fill=outline, width=2)
            draw.line([x0, y1, x1, y0], fill=outline, width=2)
        if opts.show_labels:
            prefix = f"D{tag}" if is_det else tag
            draw.text(
                (x0 + 2, max(0.0, y0 - 12)),
                f"{prefix}{lid} {depth:.1f}m",
                fill=outline,
            )
        # Navigation writeback: orange = in match pool, magenta = unique pair.
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
        # Boxes that fed a landmark / boxes drawn, once the frame is surveyed.
        yolo_hud = f"yolo={boxes_used}/{det_drawn}" if notes else f"yolo={det_drawn}"
    hud = (
        f"lm={drawn} "
        f"C={counts.get(LandmarkStatus.CONFIRMED, 0)} "
        f"c={counts.get(LandmarkStatus.CANDIDATE, 0)} "
        f"M={counts.get(LandmarkStatus.MOVED, 0)} "
        f"air={flagged} "
        f"wb={len(wb_visible)}/{len(wb_paired)} "
        f"{yolo_hud} "
        f"pose=({robot_pose.x:.2f},{robot_pose.y:.2f},{math.degrees(robot_pose.theta):.0f}°)"
    )
    # Two-line HUD: counts + legend (kept short enough for a 424 px frame).
    draw.rectangle([0, 0, w, _HUD_HEIGHT], fill=(0, 0, 0, 150))
    draw.text((4, 2), hud, fill=(230, 230, 230, 255))
    draw.text(
        (4, 16),
        "cyan=used box  grey=skipped  red X=seen past  wb=V/P  yolo=used/all",
        fill=(180, 200, 220, 255),
    )

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
            # Detector-anchored landmarks are object-level. Include them here
            # so a box from this frame can be tied to its mark; the overlay
            # drops an untrusted one that this frame did not detect.
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
