"""Background vision loop: grab RGB-D, survey landmarks, optional pose writeback."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import LidarConfig, MODE_LOCALIZING, MODE_MAPPING, SlamConfig
from ..geom import conversions as conv
from .decode import decode_color_image, decode_depth_image, pick_named_images
from .frames import (
    intrinsics_from_camera_properties,
    intrinsics_from_size,
    vision_frame_from_arrays,
)
from .detect_landmarks import (
    BOX_MAX_RANGE_M,
    describe_box_outcome,
    observations_from_detections,
)
from .landmarks import AUDIT_SEEN_THROUGH, EmbeddingLandmarkMap, display_label
from .localize import (
    diagnose_landmark_match,
    estimate_landmark_fix,
)
from .paint_mask import discs_from_detections
from .patches import PatchObservation, backproject_patches, subsample_observations
from .persist import (
    landmarks_path,
    load_landmarks,
    load_place_index,
    save_landmarks,
    save_place_index,
)
from .place import PlaceIndex, descriptors_from_patches
from .types import CameraIntrinsics, Detection, PatchToken, VisionFrame, VisionResult
from .worker import VisionWorker, WorkerConfig


PoseProvider = Callable[[], conv.Pose2D]
PoseAtProvider = Callable[[float], Optional[conv.Pose2D]]
AnchorProvider = Callable[[], Tuple[int, conv.Pose2D, Dict[int, conv.Pose2D]]]
# Same signature as BuiltinSlamHost.apply_map_pose_correction (force always False).
PoseCorrection = Callable[..., Dict[str, Any]]
# True when SLAM localization_state is lost/unknown (place apply only then).
LocLostProvider = Callable[[], bool]
# Path length of each pose-graph node, for landmark loop travel gaps.
NodeTravelProvider = Callable[[], Dict[int, float]]
# ``(matched_pose, anchor, *, score=)`` → engine.add_landmark_loop.
LandmarkLoop = Callable[..., Dict[str, Any]]
# ``(discs, captured_at)`` → engine.note_paint_discs.
PaintDiscNote = Callable[..., None]

# Dense patches used only for writeback pairing (never surveyed) so detector-only
# survey does not starve the rigid fit of inliers.
_WRITEBACK_DENSE_MATCH_CAP = 24
# Graph optimize + re-render is the expensive part; don't repeat it every frame.
_LANDMARK_LOOP_COOLDOWN_S = 8.0
# Place keyframes are a coarse guess (0.75 m apart, heading often off).
# Pair the landmarks that project into the image by appearance; the fit is
# the geometry check and still has to land within the writeback jump cap.
_PLACE_SEED_TOP_K = 5
# Two fits closer than this are the same place (nearby keyframes). Farther
# apart, both explaining the picture, and neither guess is applied.
_PLACE_SEED_AGREE_M = 0.75
_PLACE_SEED_AGREE_DEG = 15.0
# While lost, landmark positions from a tracked survey still scatter more
# than the 15 cm nudge gate. Three matches inside this may move the pose.
_LOST_FIT_RESIDUAL_M = 0.30

# A frame whose capture-time pose cannot be looked up is still usable while it
# is this fresh (no odometry history yet); older than this it is dropped.
_MAX_UNSTAMPED_FRAME_AGE_S = 1.0
# Rows kept from the latest depth audit / recent removals (DoCommand payload).
_AUDIT_MAX_ROWS = 200
_AUDIT_RECENT_REMOVED = 20

# Fetched by setup.sh / run.sh next to this package (same pattern as DINOv2).
_DEFAULT_YOLO_WORLD_ONNX = "yolov8s-worldv2-indoor.onnx"


class VisionSupervisor:
    """Owns the worker + landmark map; never blocks the SLAM tick."""

    def __init__(
        self,
        *,
        cfg: SlamConfig,
        cameras: Dict[str, Any],
        get_pose: PoseProvider,
        get_pose_at: PoseAtProvider,
        get_anchor: AnchorProvider,
        apply_pose_correction: Optional[PoseCorrection] = None,
        is_localization_lost: Optional[LocLostProvider] = None,
        get_node_travel: Optional[NodeTravelProvider] = None,
        add_landmark_loop: Optional[LandmarkLoop] = None,
        note_paint_discs: Optional[PaintDiscNote] = None,
        logger=None,
    ):
        self._cfg = cfg
        self._cameras = dict(cameras or {})
        self._get_pose = get_pose
        self._get_pose_at = get_pose_at
        self._get_anchor = get_anchor
        self._apply_pose_correction = apply_pose_correction
        self._is_localization_lost = is_localization_lost
        self._get_node_travel = get_node_travel
        self._add_landmark_loop = add_landmark_loop
        self._note_paint_discs = note_paint_discs
        self._logger = logger
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        dinov2_path = _resolve_model_path(str(cfg.vision_dinov2_onnx or ""))
        if not str(cfg.vision_dinov2_onnx or "").strip() and not bool(
            cfg.vision_synthetic
        ):
            # setup.sh / run.sh fetch default weights next to this package.
            candidate = _resolve_model_path("dinov2_vits14_patches_448.onnx")
            if candidate and Path(candidate).is_file():
                dinov2_path = candidate
        want_synthetic = bool(cfg.vision_synthetic) or not (
            dinov2_path and Path(dinov2_path).is_file()
        )
        self._dinov2_path = dinov2_path
        self._landmarks = EmbeddingLandmarkMap(
            associate_radius_m=float(cfg.vision_associate_radius_m),
            cell_m=float(cfg.vision_cell_m),
            # Synthetic RGB-stat descriptors are not re-id stable; merge by
            # distance so the density pilot is not drowned in one-per-patch IDs.
            require_descriptor_match=not want_synthetic,
            demote_passes=5 if want_synthetic else 3,
        )
        self._last_result: Optional[VisionResult] = None
        self._last_frame_age_s: Optional[float] = None
        self._last_error: Optional[str] = None
        # Last survey camera geometry — vision-camera overlay must use the same
        # mount/FOV or boxes slide vs the RGB while spinning.
        self._last_camera_geom: Optional[dict] = None
        # Per-camera K from Camera.get_properties (None = not tried / failed).
        self._intrinsics_by_cam: Dict[str, Optional[CameraIntrinsics]] = {}
        self._intrinsics_source_by_cam: Dict[str, str] = {}
        self._frames = 0
        self._observations = 0
        self._inflight: set = set()
        self._rr_index = 0
        self._last_writeback_mono = 0.0
        self._writeback_attempts = 0
        self._writeback_applied = 0
        self._last_writeback: Optional[dict] = None
        self._last_loop_mono = 0.0
        self._loop_attempts = 0
        self._loop_applied = 0
        self._last_landmark_loop: Optional[dict] = None
        self._last_paint_discs = 0
        # Overlay: last writeback visible / paired landmark ids.
        self._last_writeback_match: dict = {
            "visible_ids": [],
            "paired_ids": [],
        }
        self._last_detection_obs = 0
        self._last_detection_boxes = 0
        # Detector boxes turned into observations vs skipped, by reason.
        self._box_depth_counts: Dict[str, int] = {}
        self._last_box_depth: Dict[str, int] = {}
        self._box_label_counts: Dict[str, Dict[str, int]] = {}
        # What became of each box in the last surveyed frame (for the overlay
        # and landmark_audit): ``{"result": VisionResult, "boxes": [...]}``.
        self._last_boxes: Optional[dict] = None
        # Depth audit of stored landmarks against each survey frame.
        self._audit_frames = 0
        self._audit_pose_suspect_frames = 0
        self._last_audit: Optional[dict] = None
        self._audit_recent_removed: List[dict] = []
        # Frames dropped / stamped at grab time for lack of a capture pose.
        self._stale_frames_dropped = 0
        self._grab_pose_fallbacks = 0
        self._place = PlaceIndex(
            cosine_threshold=float(cfg.vision_place_min_score)
        )
        self._place_next_id = 1
        self._place_last_pose: Optional[conv.Pose2D] = None
        self._place_fit_attempted = False
        self._last_place_query: Optional[dict] = None
        self._place_retrieval_attempts = 0
        self._place_retrieval_applied = 0
        self._last_place_retrieval: Optional[dict] = None
        self._last_place_retrieval_mono = 0.0
        self._place_seed_attempts = 0
        self._place_seed_applied = 0
        self._last_place_seed: Optional[dict] = None
        self._last_place_seed_mono = 0.0
        # Pose last published by a seed. A later fit that lands somewhere
        # else is the other corridor twin and is not applied.
        self._last_seed_pose: Optional[conv.Pose2D] = None

        input_size = int(cfg.vision_input_size) if not want_synthetic else 224
        if input_size < 14:
            input_size = 224
        yolo_path = _resolve_model_path(str(cfg.vision_yolo_onnx or ""))
        if not str(cfg.vision_yolo_onnx or "").strip() and not want_synthetic:
            candidate = _resolve_model_path(_DEFAULT_YOLO_WORLD_ONNX)
            if candidate and Path(candidate).is_file():
                yolo_path = candidate
        self._yolo_path = yolo_path or ""
        detector_classes = _resolve_detector_classes(
            yolo_path,
            configured=list(cfg.vision_detector_classes or []),
        )
        worker_cfg = WorkerConfig(
            dinov2_onnx=dinov2_path,
            yolo_onnx=yolo_path,
            synthetic=want_synthetic,
            # One thread — leave cores for SLAM/nav; dual-cam was hesitant.
            cpu_threads=1,
            input_short_side=input_size,
            yolo_conf=float(cfg.vision_yolo_conf),
            detector_classes=detector_classes,
            landmark_labels=tuple(
                str(x).strip() for x in (cfg.vision_landmark_labels or []) if str(x).strip()
            ),
        )
        self._worker = VisionWorker(worker_cfg)
        # If ONNX failed to load, worker flips to synthetic — keep landmark
        # association policy in sync (spatial merge, softer demotion).
        if self._worker.cfg.synthetic and not want_synthetic:
            self._landmarks.require_descriptor_match = False
            self._landmarks.demote_passes = 5
            self._log(
                f"vision: DINOv2 load failed ({self._worker.stats.get('last_error')}); "
                "falling back to synthetic"
            )

    def _log(self, msg: str) -> None:
        if self._logger is None:
            return
        try:
            self._logger(msg)
        except Exception:  # noqa: BLE001
            pass

    @property
    def landmarks(self) -> EmbeddingLandmarkMap:
        return self._landmarks

    def writeback_match_overlay(self) -> dict:
        """Landmark ids from the latest writeback attempt (for landmark-camera)."""
        with self._lock:
            return {
                "visible_ids": list(self._last_writeback_match.get("visible_ids") or []),
                "paired_ids": list(self._last_writeback_match.get("paired_ids") or []),
            }

    def seen_through_ids(self) -> List[int]:
        """Landmarks the camera currently sees past (for landmark-camera)."""
        with self._lock:
            return self._landmarks.seen_through_ids(min_strikes=1)

    def landmark_audit(self) -> dict:
        """Latest depth audit: stored vs measured depth per in-view landmark."""
        with self._lock:
            last = self._last_audit
            out: Dict[str, Any] = {
                "enabled": True,
                "mode": str(self._cfg.mode),
                "removes": self._audit_removes(),
                "frames": int(self._audit_frames),
                "pose_suspect_frames": int(self._audit_pose_suspect_frames),
                "strike_limit": int(self._landmarks.see_through_strikes),
                "flagged_ids": self._landmarks.seen_through_ids(),
                "removed_total": int(
                    self._landmarks.status_dict()["seen_through_removed"]
                ),
                "recent_removed": list(self._audit_recent_removed),
            }
            if last is None:
                out.update({"age_s": None, "summary": {}, "rows": []})
                return out
            out.update(
                {
                    "age_s": round(max(0.0, time.monotonic() - last["mono"]), 2),
                    "camera": last["camera"],
                    "pose": last["pose"],
                    "summary": dict(last["summary"]),
                    "rows": list(last["rows"]),
                    "rows_truncated": bool(last["truncated"]),
                    # Every detector box in the last frame and what it became.
                    "boxes": [
                        {k: v for k, v in box.items() if k != "landmark_id"}
                        for box in (self._last_boxes or {}).get("boxes", [])
                    ],
                }
            )
            return out

    def last_detections(
        self, *, camera_name: Optional[str] = None
    ) -> Tuple[List[Detection], Optional[Tuple[int, int]]]:
        """Latest YOLO boxes for overlay, plus ``(width, height)`` of that frame.

        Boxes are in the survey camera's pixel space. Callers should scale when
        the live overlay RGB differs in size. ``camera_name`` is accepted for
        API compatibility but does not drop boxes — dense-fallback candidates
        can appear without YOLO, so the overlay must still surface whatever the
        detector last produced (or nothing) regardless of source aliasing.
        """
        del camera_name  # overlay always wants the latest survey detections
        detections, size, _boxes = self.last_detections_with_outcomes()
        return detections, size

    def last_detections_with_outcomes(
        self,
    ) -> Tuple[List[Detection], Optional[Tuple[int, int]], Optional[List[dict]]]:
        """:meth:`last_detections` plus what each box became, read together.

        The third value has one dict per detection (``outcome``, ``text``,
        ``depth_m``, ``landmark_id``, ``landmark``), or ``None`` when the
        latest detections have not been surveyed (yet).
        """
        with self._lock:
            last = self._last_result
            geom = self._last_camera_geom
            noted = self._last_boxes
        if last is None or not last.detections:
            return [], None, None
        size = None
        if isinstance(geom, dict):
            gw = int(geom.get("width") or 0)
            gh = int(geom.get("height") or 0)
            if gw > 0 and gh > 0:
                size = (gw, gh)
        boxes = None
        if (
            noted is not None
            and noted["result"] is last
            and len(noted["boxes"]) == len(last.detections)
        ):
            boxes = [dict(box) for box in noted["boxes"]]
        return list(last.detections), size, boxes

    def load_from_map_dir(self, map_dir) -> bool:
        loaded = load_landmarks(map_dir)
        place = load_place_index(map_dir)
        ok = False
        with self._lock:
            if loaded is not None:
                # Keep live merge / height settings; only restore landmark rows.
                loaded.associate_radius_m = self._landmarks.associate_radius_m
                loaded.cell_m = self._landmarks.cell_m
                loaded.require_descriptor_match = self._landmarks.require_descriptor_match
                loaded.candidate_cosine_threshold = (
                    self._landmarks.candidate_cosine_threshold
                )
                loaded.cosine_threshold = self._landmarks.cosine_threshold
                loaded.demote_passes = self._landmarks.demote_passes
                loaded.rebuild_cell_index()
                self._landmarks = loaded
                ok = True
            if place is not None:
                place.cosine_threshold = float(self._cfg.vision_place_min_score)
                self._place = place
                self._place_next_id = (
                    max(self._place.keyframe_ids) + 1
                    if self._place.keyframe_ids
                    else 1
                )
                self._place_fit_attempted = bool(place.fitted)
                ok = True
        if loaded is not None:
            self._log(f"vision: loaded {len(loaded)} landmarks from {map_dir}")
        if place is not None:
            self._log(
                f"vision: loaded place index keyframes={len(place.keyframe_ids)} "
                f"fitted={place.fitted} from {map_dir}"
            )
        return ok

    def save_to_map_dir(self, map_dir) -> Optional[str]:
        with self._lock:
            path = save_landmarks(map_dir, self._landmarks)
            place_path = save_place_index(map_dir, self._place)
        if place_path is not None:
            self._log(f"vision: saved place index → {place_path}")
        return str(path) if path is not None else None

    def _new_landmark_map(self) -> EmbeddingLandmarkMap:
        synthetic = bool(self._worker.cfg.synthetic)
        return EmbeddingLandmarkMap(
            associate_radius_m=float(self._cfg.vision_associate_radius_m),
            cell_m=float(self._cfg.vision_cell_m),
            require_descriptor_match=not synthetic,
            demote_passes=5 if synthetic else 3,
        )

    def clear_landmarks(self, map_dir=None) -> dict:
        """Drop every landmark and its saved file; the place index is kept.

        The saved file goes too, otherwise the next restart loads the old
        landmarks back.
        """
        with self._lock:
            cleared = len(self._landmarks)
            self._landmarks = self._new_landmark_map()
            self._observations = 0
            self._box_depth_counts = {}
            self._last_box_depth = {}
            self._box_label_counts = {}
            self._last_boxes = None
            self._audit_frames = 0
            self._audit_pose_suspect_frames = 0
            self._last_audit = None
            self._audit_recent_removed = []
            self._last_writeback_match = {"visible_ids": [], "paired_ids": []}
            removed_file = False
            if map_dir is not None:
                path = landmarks_path(map_dir)
                if path.is_file():
                    path.unlink()
                    removed_file = True
        self._log(
            f"vision: cleared {cleared} landmarks (saved file removed={removed_file})"
        )
        return {"cleared": int(cleared), "saved_file_removed": removed_file}

    def reset_landmarks(self) -> None:
        """Drop the in-memory landmark map (e.g. after ``clear_map``)."""
        with self._lock:
            self._landmarks = self._new_landmark_map()
            self._place = PlaceIndex(
                cosine_threshold=float(self._cfg.vision_place_min_score)
            )
            self._place_next_id = 1
            self._place_last_pose = None
            self._place_fit_attempted = False
            self._last_place_query = None
            self._place_retrieval_attempts = 0
            self._place_retrieval_applied = 0
            self._last_place_retrieval = None
            self._place_seed_attempts = 0
            self._place_seed_applied = 0
            self._last_place_seed = None
            self._last_place_seed_mono = 0.0
            self._observations = 0
            self._frames = 0
            self._last_result = None
            self._last_error = None
            self._box_depth_counts = {}
            self._last_box_depth = {}
            self._box_label_counts = {}
            self._last_boxes = None
            self._audit_frames = 0
            self._audit_pose_suspect_frames = 0
            self._last_audit = None
            self._audit_recent_removed = []

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._cfg.vision_enabled:
            return
        # Stop the async loop only — do NOT close the worker. Closing drops the
        # loaded DINOv2 session and every later frame silently falls back to
        # synthetic (provider=synthetic, last_error=null).
        self._stop_loop()
        self._stop = asyncio.Event()
        self._ensure_worker_loaded()
        self._task = loop.create_task(self._run(), name="vision-supervisor")
        yolo_loaded = bool(self._worker.stats.get("yolo_loaded"))
        self._log(
            f"vision: started cameras={list(self._cfg.vision_cameras)} "
            f"rate={self._cfg.vision_rate_hz}Hz "
            f"survey_only={self._cfg.vision_survey_only} "
            f"pose_writeback={self._writeback_enabled()} "
            f"round_robin={self._cfg.vision_round_robin} "
            f"z=[{self._cfg.vision_z_min_m},{self._cfg.vision_z_max_m}] "
            f"max_patches={self._cfg.vision_max_patches_per_frame} "
            f"provider={self._worker.stats.get('provider')} "
            f"dinov2={self._dinov2_path or 'none'} "
            f"yolo={self._yolo_path or 'none'} "
            f"yolo_loaded={yolo_loaded} "
            f"yolo_conf={self._cfg.vision_yolo_conf}"
        )
        if self._cfg.vision_prefer_detections and not yolo_loaded:
            self._log(
                "vision: YOLO detector not loaded — landmarks will use dense "
                "DINO fallback only. Ensure setup/run fetched "
                f"{_DEFAULT_YOLO_WORLD_ONNX} (vision_status.yolo_onnx / "
                "worker.yolo_loaded)."
            )

    def _stop_loop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            self._stop.set()
            task.cancel()

    def _ensure_worker_loaded(self) -> None:
        """Reload ONNX if stop/close cleared the session."""
        worker = self._worker
        if worker.cfg.synthetic:
            return
        if worker._dino is not None:  # noqa: SLF001
            return
        worker._load_models()  # noqa: SLF001
        if worker.cfg.synthetic:
            self._landmarks.require_descriptor_match = False
            self._landmarks.demote_passes = 5
            self._log(
                f"vision: DINOv2 reload failed ({worker.stats.get('last_error')}); "
                "falling back to synthetic"
            )

    def stop(self) -> None:
        self._stop_loop()
        self._worker.close()

    def status(self) -> dict:
        with self._lock:
            lm = self._landmarks.status_dict()
            last = self._last_result
            return {
                "enabled": bool(self._cfg.vision_enabled),
                "mode": str(self._cfg.mode),
                "survey_only": bool(self._cfg.vision_survey_only),
                # Config intent vs actually applying (mapping never applies).
                "pose_writeback": bool(self._cfg.vision_pose_writeback),
                "pose_writeback_active": self._writeback_enabled(),
                "cameras": list(self._cfg.vision_cameras),
                "provider": self._worker.stats.get("provider"),
                "frames": self._frames,
                "observations": self._observations,
                "landmarks": lm,
                "last_infer_ms": None if last is None else float(last.infer_ms),
                "last_frame_age_s": self._last_frame_age_s,
                "last_error": self._last_error,
                "worker": dict(self._worker.stats),
                "z_min_m": float(self._cfg.vision_z_min_m),
                "z_max_m": float(self._cfg.vision_z_max_m),
                "max_patches_per_frame": int(self._cfg.vision_max_patches_per_frame),
                "round_robin": bool(self._cfg.vision_round_robin),
                "associate_radius_m": float(self._landmarks.associate_radius_m),
                "dinov2_onnx": self._dinov2_path or None,
                "yolo_onnx": self._yolo_path or None,
                "yolo_loaded": bool(self._worker.stats.get("yolo_loaded")),
                "yolo_provider": self._worker.stats.get("yolo_provider"),
                "yolo_conf": float(self._cfg.vision_yolo_conf),
                "last_yolo_out_shape": self._worker.stats.get("last_yolo_out_shape"),
                "detector_classes": list(self._worker.cfg.detector_classes),
                "prefer_detections": bool(self._cfg.vision_prefer_detections),
                "dense_fallback_patches": int(self._cfg.vision_dense_fallback_patches),
                "last_detections": 0 if last is None else len(last.detections),
                "last_detection_obs": int(self._last_detection_obs),
                "last_detection_boxes": int(self._last_detection_boxes),
                # Boxes located on one surface vs skipped (no observation).
                "box_depth": {
                    **{k: int(v) for k, v in self._box_depth_counts.items()},
                    "last": dict(self._last_box_depth),
                    "by_label": {
                        label: dict(counts)
                        for label, counts in sorted(self._box_label_counts.items())
                    },
                },
                "depth_audit": {
                    "removes": self._audit_removes(),
                    "frames": int(self._audit_frames),
                    "pose_suspect_frames": int(self._audit_pose_suspect_frames),
                    "flagged_ids": self._landmarks.seen_through_ids(),
                    "removed_total": int(lm["seen_through_removed"]),
                    "last": (
                        None
                        if self._last_audit is None
                        else dict(self._last_audit["summary"])
                    ),
                },
                "capture_pose": {
                    "stale_frames_dropped": int(self._stale_frames_dropped),
                    "grab_time_fallbacks": int(self._grab_pose_fallbacks),
                },
                "last_yolo_max_conf": self._worker.stats.get("last_yolo_max_conf"),
                "input_size": int(self._worker.cfg.input_short_side),
                "require_descriptor_match": bool(
                    self._landmarks.require_descriptor_match
                ),
                "intrinsics_source": dict(self._intrinsics_source_by_cam),
                "writeback": {
                    "attempts": self._writeback_attempts,
                    "applied": self._writeback_applied,
                    "last": self._last_writeback,
                },
                "landmark_loop": {
                    "active": self._landmark_loop_enabled(),
                    "attempts": self._loop_attempts,
                    "applied": self._loop_applied,
                    "last": self._last_landmark_loop,
                },
                "paint_gate": {
                    "last_discs": int(self._last_paint_discs),
                    "labels": list(self._worker.cfg.dynamic_labels),
                },
                "place": {
                    "enabled": bool(self._cfg.vision_place_enabled),
                    "retrieval": bool(self._cfg.vision_place_retrieval),
                    "retrieval_active": self._place_retrieval_enabled(),
                    **self._place.status_dict(),
                    "last_query": self._last_place_query,
                    "retrieval_attempts": self._place_retrieval_attempts,
                    "retrieval_applied": self._place_retrieval_applied,
                    "last_retrieval": self._last_place_retrieval,
                    "seed_attempts": self._place_seed_attempts,
                    "seed_applied": self._place_seed_applied,
                    "last_seed": self._last_place_seed,
                },
            }

    def _vision_apply_allowed(self) -> bool:
        """Pose corrections only while localizing — mapping only builds maps."""
        return (
            str(self._cfg.mode) == MODE_LOCALIZING
            and not bool(self._cfg.vision_survey_only)
            and self._apply_pose_correction is not None
            and not bool(self._worker.cfg.synthetic)
        )

    def _writeback_enabled(self) -> bool:
        return bool(self._cfg.vision_pose_writeback) and self._vision_apply_allowed()

    def _landmark_loop_enabled(self) -> bool:
        """Re-observed landmarks may add a pose-graph edge while mapping."""
        return (
            str(self._cfg.mode) == MODE_MAPPING
            and not bool(self._cfg.vision_survey_only)
            and self._add_landmark_loop is not None
            and not bool(self._worker.cfg.synthetic)
        )

    def _audit_removes(self) -> bool:
        """The depth audit may delete landmarks only while the map is being built.

        While localizing the pose is the thing in doubt: a wrong pose makes
        every landmark look seen through, so the audit only reports.
        """
        return str(self._cfg.mode) == MODE_MAPPING or bool(
            self._cfg.vision_survey_only
        )

    def _pose_at_grab(self, frame: VisionFrame) -> Optional[conv.Pose2D]:
        """Pose stamp taken right after the grab, before inference.

        ``None`` drops the frame: its capture pose is unknown and it is too old
        for the current pose to stand in.
        """
        t = float(frame.captured_at)
        pose = self._get_pose_at(t)
        if pose is not None:
            return pose
        if time.monotonic() - t > _MAX_UNSTAMPED_FRAME_AGE_S:
            self._stale_frames_dropped += 1
            return None
        return self._get_pose()

    def _capture_pose(
        self, frame: VisionFrame, grab_pose: Optional[conv.Pose2D] = None
    ) -> conv.Pose2D:
        """Robot pose when ``frame`` was captured.

        The lookup after inference interpolates inside the odometry history,
        so it is preferred. If the frame has aged out of that history, use the
        pose stamped at grab time rather than the current pose.
        """
        pose = self._get_pose_at(float(frame.captured_at))
        if pose is not None:
            return pose
        if grab_pose is not None:
            self._grab_pose_fallbacks += 1
            return grab_pose
        # Synchronous path (tests / sim): no inference latency to undo.
        return self._get_pose()

    def ingest_frame(self, frame: VisionFrame, *, mapping: bool = True) -> VisionResult:
        """Synchronous path for tests / sim: process one frame and maybe survey."""
        result = self._worker.process(frame)
        with self._lock:
            self._last_result = result
            self._frames += 1
        if self._cfg.vision_place_enabled:
            self._place_tick(frame, result, allow_keyframe=bool(mapping))
        if mapping:
            self._survey(frame, result)
        return result

    async def _run(self) -> None:
        period = 1.0 / max(0.1, float(self._cfg.vision_rate_hz))
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self._tick_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._last_error = repr(exc)
                self._log(f"vision tick error: {exc}")
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, period - elapsed))

    def _cameras_this_tick(self) -> List[LidarConfig]:
        cams = self._cfg.vision_lidar_configs()
        if not cams:
            return []
        if not self._cfg.vision_round_robin or len(cams) == 1:
            return cams
        idx = self._rr_index % len(cams)
        self._rr_index += 1
        return [cams[idx]]

    async def _tick_once(self) -> None:
        for lidar in self._cameras_this_tick():
            name = lidar.name
            if name in self._inflight:
                continue
            cam = self._cameras.get(name)
            if cam is None:
                continue
            self._inflight.add(name)
            try:
                frame = await self._grab_frame(cam, lidar)
                if frame is None:
                    continue
                grab_pose = self._pose_at_grab(frame)
                if grab_pose is None:
                    continue
                result = await asyncio.to_thread(self._worker.process, frame)
                with self._lock:
                    self._last_result = result
                    self._frames += 1
                    self._last_frame_age_s = max(
                        0.0, time.monotonic() - float(frame.captured_at)
                    )
                    self._last_error = None
                # Always survey while mapping, or whenever survey_only / writeback
                # needs a live landmark map (localizing with writeback on).
                do_survey = (
                    self._cfg.mode == MODE_MAPPING
                    or self._cfg.vision_survey_only
                    or self._writeback_enabled()
                )
                # Place VLAD: keyframe while mapping/surveying; query always when on.
                allow_place_kf = (
                    self._cfg.mode == MODE_MAPPING or self._cfg.vision_survey_only
                )
                pose = self._capture_pose(frame, grab_pose)
                if self._cfg.vision_place_enabled:
                    await asyncio.to_thread(
                        self._place_tick,
                        frame,
                        result,
                        allow_keyframe=allow_place_kf,
                        pose=pose,
                    )
                if do_survey:
                    await asyncio.to_thread(self._survey, frame, result, pose=pose)
            finally:
                self._inflight.discard(name)

    async def _resolve_intrinsics(
        self, cam, lidar: LidarConfig, *, width: int, height: int
    ) -> CameraIntrinsics:
        """Prefer live Camera.get_properties(); fall back to lidar FOV config."""
        name = lidar.name
        cached = self._intrinsics_by_cam.get(name)
        if cached is not None and cached.width == width and cached.height == height:
            return cached
        # Try (or retry on resolution change). A failed probe stores None.
        need_probe = name not in self._intrinsics_by_cam or (
            cached is not None
            and (cached.width != width or cached.height != height)
        )
        if need_probe:
            getter = getattr(cam, "get_properties", None)
            if callable(getter):
                try:
                    props = await getter(timeout=2.0)
                    K = intrinsics_from_camera_properties(
                        props, image_width=width, image_height=height
                    )
                    if K is not None:
                        self._intrinsics_by_cam[name] = K
                        self._intrinsics_source_by_cam[name] = "camera_properties"
                        self._log(
                            f"vision: {name} intrinsics from camera properties "
                            f"fx={K.fx:.1f} fy={K.fy:.1f} {K.width}x{K.height}"
                        )
                        return K
                except Exception as exc:  # noqa: BLE001
                    self._log(f"vision: {name} get_properties failed: {exc}")
            self._intrinsics_by_cam[name] = None
        K = intrinsics_from_size(
            width,
            height,
            fov_deg=float(lidar.fov_deg),
            vfov_deg=float(lidar.vfov_deg),
        )
        self._intrinsics_source_by_cam[name] = "fov_config"
        return K

    async def _grab_frame(self, cam, lidar: LidarConfig) -> Optional[VisionFrame]:
        """Pull color+depth via get_images; return None if unavailable."""
        # Stamp pose to mid-grab. Using the *start* time while spinning places
        # map_xyz with a stale yaw vs the RGB/depth content, so overlays slide
        # even on opaque walls (not only glass).
        t0 = time.monotonic()
        try:
            images, _meta = await cam.get_images(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"get_images({lidar.name}): {exc}"
            return None
        t1 = time.monotonic()
        captured_at = 0.5 * (t0 + t1)
        if not images:
            return None
        color_blob, depth_blob = pick_named_images(images)
        if color_blob is None or depth_blob is None:
            summary = []
            for im in images:
                n = getattr(im, "name", "?")
                m = getattr(im, "mime_type", getattr(im, "mimeType", "?"))
                d = getattr(im, "data", b"")
                summary.append(f"{n}:{m}:{len(d) if d is not None else 0}B")
            self._last_error = (
                f"{lidar.name}: need color+depth in get_images "
                f"(got [{', '.join(summary)}])"
            )
            return None
        try:
            color = decode_color_image(color_blob["data"], color_blob["mime"])
            depth = decode_depth_image(
                depth_blob["data"],
                depth_blob["mime"],
                named_image=depth_blob.get("raw"),
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = (
                f"decode({lidar.name}): color={color_blob['name']}/"
                f"{color_blob['mime']} depth={depth_blob['name']}/"
                f"{depth_blob['mime']}: {exc}"
            )
            return None
        if color.shape[:2] != depth.shape[:2]:
            # Resize depth with nearest neighbor to color size.
            depth = _resize_nearest(depth, color.shape[0], color.shape[1])
        K = await self._resolve_intrinsics(
            cam, lidar, width=color.shape[1], height=color.shape[0]
        )
        self._last_camera_geom = {
            "camera_name": lidar.name,
            "mount_x": float(lidar.x),
            "mount_y": float(lidar.y),
            "mount_z": float(lidar.z),
            "mount_theta": float(lidar.theta),
            "mount_pitch": float(lidar.pitch),
            "mount_roll": float(lidar.roll),
            "fov_deg": float(lidar.fov_deg),
            "vfov_deg": float(lidar.vfov_deg),
            "intrinsics_source": self._intrinsics_source_by_cam.get(
                lidar.name, "unknown"
            ),
            "fx": float(K.fx),
            "fy": float(K.fy),
            "cx": float(K.cx),
            "cy": float(K.cy),
            "width": int(K.width),
            "height": int(K.height),
            "max_range": float(lidar.max_range),
        }
        return vision_frame_from_arrays(
            camera_name=lidar.name,
            color_rgb=color,
            depth_m=depth,
            stamp=time.time(),
            captured_at=captured_at,
            lidar=lidar,
            intrinsics=K,
        )

    def _place_tick(
        self,
        frame: VisionFrame,
        result: VisionResult,
        *,
        allow_keyframe: bool,
        pose: Optional[conv.Pose2D] = None,
    ) -> None:
        """Build / query the VLAD place index from live patch tokens."""
        descs = descriptors_from_patches(result.patches, max_patches=64)
        if descs is None:
            return
        if pose is None:
            pose = self._capture_pose(frame)

        with self._lock:
            if allow_keyframe and self._should_add_place_keyframe(pose):
                kid = int(self._place_next_id)
                self._place_next_id += 1
                self._place.add_keyframe(kid, pose, descs)
                self._place_last_pose = pose
                # Fit once we have enough travel keyframes.
                if (
                    not self._place.fitted
                    and len(self._place.keyframe_ids)
                    >= int(self._cfg.vision_place_min_keyframes_to_fit)
                ):
                    ok = self._place.fit(
                        n_clusters=int(self._cfg.vision_place_vocab_size),
                        pca_dim=int(self._cfg.vision_place_pca_dim),
                    )
                    self._place_fit_attempted = True
                    if ok:
                        self._log(
                            f"vision place: fitted vocab "
                            f"k={self._cfg.vision_place_vocab_size} "
                            f"pca={self._cfg.vision_place_pca_dim} "
                            f"keyframes={len(self._place.keyframe_ids)}"
                        )

            hits = self._place.query(descs, top_k=3)
            if hits:
                kid, hit_pose, score = hits[0]
                jump_m = math.hypot(hit_pose.x - pose.x, hit_pose.y - pose.y)
                jump_deg = abs(
                    math.degrees(conv.normalize_angle(hit_pose.theta - pose.theta))
                )
                self._last_place_query = {
                    "keyframe_id": int(kid),
                    "score": round(float(score), 3),
                    "pose": {
                        "x": round(float(hit_pose.x), 3),
                        "y": round(float(hit_pose.y), 3),
                        "theta_deg": round(math.degrees(float(hit_pose.theta)), 1),
                    },
                    "jump_m": round(float(jump_m), 3),
                    "jump_deg": round(float(jump_deg), 1),
                    "hits": len(hits),
                    "fitted": bool(self._place.fitted),
                    "keyframes": len(self._place.keyframe_ids),
                }
            else:
                self._last_place_query = {
                    "status": "no_hit",
                    "fitted": bool(self._place.fitted),
                    "keyframes": len(self._place.keyframe_ids),
                }

        # Attempt path handles tracking skip so status shows skipped_tracking.
        if (
            bool(self._cfg.vision_place_enabled)
            and bool(self._cfg.vision_place_retrieval)
            and self._vision_apply_allowed()
            and self._place.keyframe_vlad is not None
            and len(self._place.keyframe_ids) > 0
        ):
            self._try_place_retrieval(descs, pose)

    def _should_add_place_keyframe(self, pose: conv.Pose2D) -> bool:
        last = self._place_last_pose
        if last is None:
            return True
        gap_m = math.hypot(pose.x - last.x, pose.y - last.y)
        gap_deg = abs(math.degrees(conv.normalize_angle(pose.theta - last.theta)))
        return gap_m >= float(self._cfg.vision_place_keyframe_min_gap_m) or gap_deg >= float(
            self._cfg.vision_place_keyframe_min_gap_deg
        )

    def _localization_lost(self) -> bool:
        """True when SLAM says lost/unknown — the only time place may apply."""
        fn = self._is_localization_lost
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False

    def _place_retrieval_enabled(self) -> bool:
        return (
            bool(self._cfg.vision_place_enabled)
            and bool(self._cfg.vision_place_retrieval)
            and self._vision_apply_allowed()
            and self._localization_lost()
            and self._place.keyframe_vlad is not None
            and len(self._place.keyframe_ids) > 0
        )

    def _place_seed_enabled(self) -> bool:
        """Lost pose may move when one place guess has 3 agreeing landmarks."""
        return self._place_retrieval_enabled() and bool(self._cfg.vision_pose_writeback)

    def _try_place_retrieval(
        self, descs: np.ndarray, pose: conv.Pose2D
    ) -> None:
        """Soft coarse relocalization from top VLAD hit (force=False).

        Place is meter-scale and ambiguous — apply only when lost/unknown, never
        as continuous drift correction while tracking.
        """
        apply = self._apply_pose_correction
        if apply is None:
            return
        if not self._localization_lost():
            self._last_place_retrieval = {"status": "skipped_tracking"}
            return
        now = time.monotonic()
        if now - self._last_place_retrieval_mono < float(
            self._cfg.vision_place_cooldown_s
        ):
            return
        self._last_place_retrieval_mono = now
        self._place_retrieval_attempts += 1

        with self._lock:
            hits = self._place.query(descs, top_k=3)
        if not hits:
            self._last_place_retrieval = {"status": "no_hit"}
            return
        kid, hit_pose, score = hits[0]
        jump_m = math.hypot(hit_pose.x - pose.x, hit_pose.y - pose.y)
        jump_deg = abs(
            math.degrees(conv.normalize_angle(hit_pose.theta - pose.theta))
        )
        margin = float(self._cfg.vision_place_min_score_margin)
        second = float(hits[1][2]) if len(hits) > 1 else None
        # Ignore tiny / no-op corrections (already near the place).
        if jump_m < 0.25 and jump_deg < 5.0:
            self._last_place_retrieval = {
                "status": "skipped_small",
                "keyframe_id": int(kid),
                "score": round(float(score), 3),
                "jump_m": round(float(jump_m), 3),
                "jump_deg": round(float(jump_deg), 1),
            }
            return
        if second is not None and (float(score) - second) < margin:
            self._last_place_retrieval = {
                "status": "rejected_ambiguous",
                "keyframe_id": int(kid),
                "score": round(float(score), 3),
                "second_score": round(second, 3),
                "margin": round(margin, 3),
                "jump_m": round(float(jump_m), 3),
                "jump_deg": round(float(jump_deg), 1),
            }
            return
        # Lost: current pose is untrusted — no relative jump cap. Large
        # teleports still need a stronger match than the floor score.
        if jump_m >= float(self._cfg.vision_place_large_jump_m) and float(
            score
        ) < float(self._cfg.vision_place_large_jump_min_score):
            self._last_place_retrieval = {
                "status": "rejected_weak_for_jump",
                "keyframe_id": int(kid),
                "score": round(float(score), 3),
                "need_score": float(self._cfg.vision_place_large_jump_min_score),
                "jump_m": round(float(jump_m), 3),
                "jump_deg": round(float(jump_deg), 1),
            }
            return
        # Landmarks get the vote when there are enough of them. A place hit
        # alone is what walked the pose down a corridor the scan would not lock.
        with self._lock:
            confirmed = len(self._landmarks.confirmed())
        if self._place_seed_enabled() and confirmed >= int(
            self._cfg.vision_pose_writeback_min_inliers
        ):
            self._last_place_retrieval = {
                "status": "deferred_to_landmarks",
                "keyframe_id": int(kid),
                "score": round(float(score), 3),
                "jump_m": round(float(jump_m), 3),
                "jump_deg": round(float(jump_deg), 1),
            }
            return
        try:
            result = apply(hit_pose, score=float(score), force=False)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"place_retrieval: {exc}"
            self._last_place_retrieval = {"status": "error", "error": repr(exc)}
            return
        applied = bool(result.get("applied"))
        if applied:
            self._place_retrieval_applied += 1
        self._last_place_retrieval = {
            "status": "applied" if applied else str(result.get("status", "not_applied")),
            "keyframe_id": int(kid),
            "score": round(float(score), 3),
            "jump_m": round(float(jump_m), 3),
            "jump_deg": round(float(jump_deg), 1),
            "shift_m": result.get("shift_m"),
            "shift_deg": result.get("shift_deg"),
            "slam_status": result.get("status"),
        }
        self._log(
            f"vision place retrieval: {self._last_place_retrieval['status']} "
            f"kf={kid} score={score:.2f} jump={jump_m:.2f}m/{jump_deg:.1f}deg"
        )

    def _try_place_seeded_fix(
        self,
        frame: VisionFrame,
        result: VisionResult,
        obs: List[PatchObservation],
        pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
    ) -> bool:
        """While lost, fit landmarks from each tied place guess.

        The guess is not applied. Landmarks that project into the image from
        the guess are paired by appearance; the fit is the geometry check
        (3 inliers, lost residual, jump measured from the guess). Two guesses
        that fit different poses are refused, including a guess on a later
        frame that disagrees with the pose this recovery already published.
        Returns True only when a correction was applied, so writeback does
        not run from the old pose on this frame.
        """
        apply = self._apply_pose_correction
        if apply is None or not self._localization_lost():
            self._last_place_seed = {"status": "skipped_tracking"}
            return False
        now = time.monotonic()
        if now - self._last_place_seed_mono < float(self._cfg.vision_place_cooldown_s):
            return False
        descs = descriptors_from_patches(result.patches, max_patches=64)
        if descs is None:
            self._last_place_seed = {"status": "no_hit"}
            return False
        min_inliers = int(self._cfg.vision_pose_writeback_min_inliers)
        margin = float(self._cfg.vision_place_min_score_margin)
        prepared: List[Tuple[int, conv.Pose2D, float, list]] = []
        with self._lock:
            hits = self._place.query(descs, top_k=_PLACE_SEED_TOP_K)
            cosine_thr = float(self._landmarks.cosine_threshold)
            if hits:
                top = float(hits[0][2])
                chosen = [hits[0]]
                for hit in hits[1:]:
                    if top - float(hit[2]) < margin:
                        chosen.append(hit)
                for kid, hit_pose, score in chosen:
                    visible = self._landmarks.image_visible_set(
                        hit_pose,
                        pose_of_node,
                        intrinsics=frame.intrinsics,
                        mount_x=float(frame.mount_x),
                        mount_y=float(frame.mount_y),
                        mount_z=float(frame.mount_z),
                        mount_theta=float(frame.mount_theta),
                        mount_pitch=float(frame.mount_pitch),
                        mount_roll=float(frame.mount_roll),
                        max_range_m=float(self._cfg.landmark_fix_max_range_m),
                    )
                    prepared.append((int(kid), hit_pose, float(score), visible))
        self._place_seed_attempts += 1
        if not prepared:
            self._last_place_seed = {"status": "no_hit"}
            return False

        accepted = []
        tried = []
        for kid, hit_pose, score, visible in prepared:
            if len(visible) < min_inliers:
                tried.append(
                    {
                        "keyframe_id": kid,
                        "score": round(score, 3),
                        "reason": "few_visible",
                        "visible": len(visible),
                    }
                )
                continue
            attempt = estimate_landmark_fix(
                obs,
                visible,
                robot_pose=hit_pose,
                cosine_threshold=cosine_thr,
                min_inliers=min_inliers,
                max_jump_m=float(self._cfg.landmark_fix_max_jump_m),
                residual_m=max(
                    float(self._cfg.landmark_fix_residual_m), _LOST_FIT_RESIDUAL_M
                ),
                max_reproj_m=None,
            )
            fix = attempt.fix
            if fix is None:
                tried.append(
                    {
                        "keyframe_id": kid,
                        "score": round(score, 3),
                        "reason": attempt.reason or "rejected_fit",
                        "visible": len(visible),
                        "pairs": int(attempt.pairs),
                    }
                )
                continue
            accepted.append(
                {
                    "keyframe_id": kid,
                    "place_score": score,
                    "hit_pose": hit_pose,
                    "fix": fix,
                    "paired_ids": [int(x) for x in attempt.paired_ids],
                }
            )
        if not accepted:
            self._last_place_seed = {"status": "no_fit", "tried": tried}
            return False
        poses = [item["fix"].fit.pose for item in accepted]
        if seeded_poses_conflict(
            poses, sep_m=_PLACE_SEED_AGREE_M, sep_deg=_PLACE_SEED_AGREE_DEG
        ):
            self._last_place_seed = {
                "status": "rejected_ambiguous_fits",
                "fits": [
                    {
                        "keyframe_id": int(item["keyframe_id"]),
                        "inliers": int(item["fix"].fit.inliers.sum()),
                        "x": round(float(item["fix"].fit.pose.x), 3),
                        "y": round(float(item["fix"].fit.pose.y), 3),
                    }
                    for item in accepted
                ],
            }
            self._log(
                "vision place seed: rejected_ambiguous_fits "
                + " ".join(f"kf={item['keyframe_id']}" for item in accepted)
            )
            return False
        best = max(
            accepted,
            key=lambda item: (
                int(item["fix"].fit.inliers.sum()),
                float(item["place_score"]),
            ),
        )
        fix = best["fix"]
        fitted = fix.fit.pose
        jump_m = math.hypot(fitted.x - pose.x, fitted.y - pose.y)
        jump_deg = abs(math.degrees(conv.normalize_angle(fitted.theta - pose.theta)))
        # A few centimetres around the lost pose is writeback's job.
        if jump_m < 0.25 and jump_deg < 5.0:
            self._last_place_seed = {
                "status": "skipped_small",
                "keyframe_id": int(best["keyframe_id"]),
                "inliers": int(fix.fit.inliers.sum()),
                "jump_m": round(jump_m, 3),
                "jump_deg": round(jump_deg, 1),
            }
            return False
        # The cart has driven off the pose we published, so that memory no
        # longer describes where we are. While we are still there, a fit
        # that lands somewhere else is the twin we already refused to bounce to.
        held = self._last_seed_pose
        if held is not None and seeded_poses_conflict(
            [held, pose], sep_m=_PLACE_SEED_AGREE_M, sep_deg=_PLACE_SEED_AGREE_DEG
        ):
            self._last_seed_pose = None
            held = None
        if held is not None and seeded_poses_conflict(
            [held, fitted], sep_m=_PLACE_SEED_AGREE_M, sep_deg=_PLACE_SEED_AGREE_DEG
        ):
            self._last_place_seed = {
                "status": "rejected_disagrees_with_last",
                "keyframe_id": int(best["keyframe_id"]),
                "inliers": int(fix.fit.inliers.sum()),
                "jump_m": round(jump_m, 3),
                "jump_deg": round(jump_deg, 1),
                "held_x": round(float(held.x), 3),
                "held_y": round(float(held.y), 3),
            }
            self._log(
                "vision place seed: rejected_disagrees_with_last "
                f"kf={best['keyframe_id']} jump={jump_m:.2f}m/{jump_deg:.1f}deg"
            )
            return False
        try:
            applied_result = apply(fitted, score=float(fix.score), force=False)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"place_seed: {exc}"
            self._last_place_seed = {"status": "error", "error": repr(exc)}
            return False
        applied = bool(applied_result.get("applied"))
        self._last_place_seed = {
            "status": "applied" if applied else str(applied_result.get("status", "not_applied")),
            "keyframe_id": int(best["keyframe_id"]),
            "place_score": round(float(best["place_score"]), 3),
            "inliers": int(fix.fit.inliers.sum()),
            "pairs": int(fix.n_pairs),
            "paired_ids": list(best["paired_ids"]),
            "jump_m": round(jump_m, 3),
            "jump_deg": round(jump_deg, 1),
            "shift_m": applied_result.get("shift_m"),
            "shift_deg": applied_result.get("shift_deg"),
            "slam_status": applied_result.get("status"),
        }
        if applied:
            self._place_seed_applied += 1
            self._last_place_seed_mono = now
            self._last_seed_pose = conv.Pose2D(fitted.x, fitted.y, fitted.theta)
        self._log(
            f"vision place seed: {self._last_place_seed['status']} "
            f"kf={best['keyframe_id']} inliers={int(fix.fit.inliers.sum())} "
            f"jump={jump_m:.2f}m/{jump_deg:.1f}deg"
        )
        return applied

    def _survey(
        self,
        frame: VisionFrame,
        result: VisionResult,
        *,
        pose: Optional[conv.Pose2D] = None,
    ) -> None:
        """Associate patches into the landmark map; optionally soft-correct pose.

        Prefer detector boxes (YOLO-World / COCO): pool DINOv2 tokens inside each
        allowed box into one landmark observation. Dense texture patches remain as
        a capped fallback when the detector is quiet or missing.

        ``pose`` is the robot pose at capture (looked up when omitted).
        """
        self._publish_paint_discs(frame, result)
        if pose is None:
            pose = self._capture_pose(frame)
        try:
            anchor_node, anchor_pose, pose_of_node = self._get_anchor()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"anchor: {exc}"
            return
        view_xy = (float(pose.x), float(pose.y), float(pose.theta))
        if anchor_node < 0:
            # No keyframe yet — park candidates on a virtual node 0 at current pose.
            anchor_node = 0
            anchor_pose = pose
            pose_of_node = {0: pose}

        # Check what is already stored before adding to it, even on a frame
        # with nothing to add: a landmark in empty air has no box to match.
        self._audit_landmarks(frame, pose, pose_of_node)

        z_min = float(self._cfg.vision_z_min_m)
        z_max = float(self._cfg.vision_z_max_m)
        prefer = bool(self._cfg.vision_prefer_detections)
        det_obs: List[PatchObservation] = []
        box_stats: Dict[str, int] = {}
        outcomes: List[dict] = []
        if prefer and result.detections and result.patches:
            det_obs = observations_from_detections(
                frame,
                result.patches,
                result.detections,
                landmark_labels=self._cfg.vision_landmark_labels,
                z_min_m=z_min,
                z_max_m=z_max,
                min_patches=1,
                stats=box_stats,
                outcomes=outcomes,
            )
        box_stats["used"] = len(det_obs)
        # One note per detection, in order (empty when boxes were not used).
        boxes: List[dict] = []
        if len(outcomes) == len(result.detections or []):
            for det, note in zip(result.detections, outcomes):
                boxes.append(
                    {
                        "label": str(det.label),
                        "conf": round(float(det.conf), 2),
                        "outcome": str(note["outcome"]),
                        "text": describe_box_outcome(
                            note["outcome"], note["depth_m"]
                        ),
                        "depth_m": (
                            None
                            if note["depth_m"] is None
                            else round(float(note["depth_m"]), 2)
                        ),
                        "landmark_id": None,
                        "landmark": None,
                    }
                )
        with self._lock:
            self._last_box_depth = dict(box_stats)
            for key, n in box_stats.items():
                self._box_depth_counts[key] = (
                    int(self._box_depth_counts.get(key, 0)) + int(n)
                )
            for box in boxes:
                per_label = self._box_label_counts.setdefault(box["label"], {})
                per_label[box["outcome"]] = int(per_label.get(box["outcome"], 0)) + 1
            self._last_boxes = {"result": result, "boxes": boxes}
        self._last_detection_obs = len(det_obs)
        self._last_detection_boxes = len(result.detections or [])

        # Dense patches beside live YOLO boxes are the main source of ID
        # inflation (parallel candidates for the same painting). When detector
        # observations exist, skip dense entirely unless explicitly configured.
        dense_cap = int(self._cfg.vision_max_patches_per_frame)
        if prefer and (det_obs or result.detections):
            dense_cap = int(self._cfg.vision_dense_fallback_patches)
        dense_obs: List[PatchObservation] = []
        if dense_cap > 0:
            dense_tokens = _tokens_outside_detections(
                result.patches,
                result.detections if prefer else (),
                landmark_labels=self._cfg.vision_landmark_labels if prefer else (),
            )
            dense_obs = backproject_patches(
                frame,
                dense_tokens,
                z_min_m=z_min,
                z_max_m=z_max,
            )
            dense_obs = subsample_observations(dense_obs, dense_cap)
        obs = list(det_obs) + list(dense_obs)
        # A box past the create range can still recognize a landmark that
        # already exists. It is not added to ``obs``, so it is not stored.
        match_det = det_obs
        fix_range = float(self._cfg.landmark_fix_max_range_m)
        if (
            prefer
            and result.detections
            and result.patches
            and fix_range > float(BOX_MAX_RANGE_M)
        ):
            match_det = observations_from_detections(
                frame,
                result.patches,
                result.detections,
                landmark_labels=self._cfg.vision_landmark_labels,
                z_min_m=z_min,
                z_max_m=z_max,
                min_patches=1,
                max_range_m=fix_range,
            )
        # Extra dense patches for writeback pairing only (not surveyed).
        wb_dense: List[PatchObservation] = []
        if prefer and result.patches and _WRITEBACK_DENSE_MATCH_CAP > 0:
            wb_tokens = _tokens_outside_detections(
                result.patches,
                result.detections or (),
                landmark_labels=self._cfg.vision_landmark_labels,
            )
            wb_dense = backproject_patches(
                frame,
                wb_tokens,
                z_min_m=z_min,
                z_max_m=z_max,
            )
            wb_dense = subsample_observations(wb_dense, _WRITEBACK_DENSE_MATCH_CAP)
        writeback_obs = list(match_det) + list(wb_dense)
        if not obs and not writeback_obs:
            return

        n_det = len(det_obs)
        n_new = 0
        # Detector observation index → the box it came from.
        box_of_obs = {
            int(note["obs_index"]): box
            for box, note in zip(boxes, outcomes)
            if note.get("obs_index") is not None
        }
        with self._lock:
            for i, o in enumerate(obs):
                # Map-frame point from capture pose.
                c, s = float(np.cos(pose.theta)), float(np.sin(pose.theta))
                pb = o.point_base
                map_xyz = np.array(
                    [
                        pose.x + c * float(pb[0]) - s * float(pb[1]),
                        pose.y + s * float(pb[0]) + c * float(pb[1]),
                        float(pb[2]),
                    ],
                    dtype=float,
                )
                n_map = np.array(
                    [
                        c * float(o.normal_base[0]) - s * float(o.normal_base[1]),
                        s * float(o.normal_base[0]) + c * float(o.normal_base[1]),
                        float(o.normal_base[2]),
                    ],
                    dtype=float,
                )
                lm = self._landmarks.observe_with_poses(
                    map_xyz=map_xyz,
                    descriptor=o.token.descriptor,
                    normal_map=n_map,
                    anchor_node=int(anchor_node),
                    anchor_pose=anchor_pose,
                    pose_of_node=pose_of_node,
                    view_xy=view_xy,
                    from_detector=i < n_det,
                    extent_m=getattr(o, "extent_m", None),
                )
                if i < n_det and i in box_of_obs:
                    box_of_obs[i]["landmark_id"] = int(lm.landmark_id)
                n_new += 1
            # Collapse detector twins that both made it to confirmed/candidate
            # on the same object (e.g. DC1 + DC12 stacked on one painting).
            self._landmarks.merge_overlapping_detector_landmarks()
            self._observations += n_new
            # Name each used box after the landmark it fed. One folded into a
            # twin just now no longer exists; the box then stays unnamed.
            for box in box_of_obs.values():
                lm = self._landmarks.get(box["landmark_id"])
                if lm is None:
                    box["landmark_id"] = None
                    continue
                box["landmark"] = display_label(lm)
                box["text"] = f"={box['landmark']}"

        if self._landmark_loop_enabled() and writeback_obs:
            self._try_landmark_loop(frame, writeback_obs, pose, pose_of_node)
        seeded = False
        if self._place_seed_enabled() and writeback_obs:
            seeded = self._try_place_seeded_fix(
                frame, result, writeback_obs, pose, pose_of_node
            )
        if self._writeback_enabled() and writeback_obs and not seeded:
            self._try_pose_writeback(frame, writeback_obs, pose, pose_of_node)

    def _publish_paint_discs(self, frame: VisionFrame, result: VisionResult) -> None:
        """Hand person (and other mover) discs to SLAM so the scan skips them."""
        note = self._note_paint_discs
        if note is None:
            return
        labels = self._worker.cfg.dynamic_labels
        discs = discs_from_detections(frame, result.detections or [], labels)
        with self._lock:
            self._last_paint_discs = len(discs)
        try:
            note(discs, float(frame.captured_at))
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"paint_gate: {exc}"

    def _try_landmark_loop(
        self,
        frame: VisionFrame,
        obs: List[PatchObservation],
        pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
    ) -> None:
        """While mapping, a fit to one old keyframe's landmarks becomes a loop edge."""
        add = self._add_landmark_loop
        travel_of = self._get_node_travel
        if add is None or travel_of is None:
            return
        now = time.monotonic()
        if now - self._last_loop_mono < _LANDMARK_LOOP_COOLDOWN_S:
            return
        min_inliers = int(self._cfg.vision_pose_writeback_min_inliers)
        with self._lock:
            confirmed = len(self._landmarks.confirmed())
            if confirmed < min_inliers:
                self._last_landmark_loop = {
                    "status": "skipped_few_confirmed",
                    "confirmed": confirmed,
                }
                return
            visible = self._landmarks.image_visible_set(
                pose,
                pose_of_node,
                intrinsics=frame.intrinsics,
                mount_x=float(frame.mount_x),
                mount_y=float(frame.mount_y),
                mount_z=float(frame.mount_z),
                mount_theta=float(frame.mount_theta),
                mount_pitch=float(frame.mount_pitch),
                mount_roll=float(frame.mount_roll),
                max_range_m=float(self._cfg.landmark_fix_max_range_m),
            )
            cosine_thr = float(self._landmarks.cosine_threshold)
            anchor_of = {
                int(lm.landmark_id): int(lm.anchor_node) for lm, _xyz in visible
            }
        if len(visible) < min_inliers:
            self._last_landmark_loop = {
                "status": "few_pairs",
                "visible": len(visible),
            }
            return
        attempt = estimate_landmark_fix(
            obs,
            visible,
            robot_pose=pose,
            cosine_threshold=cosine_thr,
            min_inliers=min_inliers,
            max_jump_m=float(self._cfg.landmark_fix_max_jump_m),
            residual_m=float(self._cfg.landmark_fix_residual_m),
        )
        fix = attempt.fix
        if fix is None:
            self._last_landmark_loop = {
                "status": "rejected_fit",
                "reason": attempt.reason or "rejected_fit",
                "pairs": int(attempt.pairs),
                "visible": len(visible),
            }
            return
        anchor = dominant_loop_anchor(
            attempt.paired_ids,
            fix.fit.inliers,
            anchor_of,
            min_inliers=min_inliers,
        )
        if anchor is None:
            self._last_landmark_loop = {
                "status": "split_anchors",
                "pairs": int(fix.n_pairs),
                "paired_ids": [int(x) for x in attempt.paired_ids],
            }
            return
        group = [
            (lm, xyz)
            for lm, xyz in visible
            if int(lm.anchor_node) == int(anchor)
        ]
        if len(group) < min_inliers:
            self._last_landmark_loop = {
                "status": "split_anchors",
                "anchor": int(anchor),
            }
            return
        if len(group) != len(visible):
            attempt = estimate_landmark_fix(
                obs,
                group,
                robot_pose=pose,
                cosine_threshold=cosine_thr,
                min_inliers=min_inliers,
                max_jump_m=float(self._cfg.landmark_fix_max_jump_m),
                residual_m=float(self._cfg.landmark_fix_residual_m),
            )
            fix = attempt.fix
            if fix is None:
                self._last_landmark_loop = {
                    "status": "rejected_fit",
                    "reason": attempt.reason or "rejected_fit",
                    "anchor": int(anchor),
                    "pairs": int(attempt.pairs),
                }
                return
        try:
            travel = travel_of() or {}
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"landmark_loop: {exc}"
            self._last_landmark_loop = {"status": "error", "error": repr(exc)}
            return
        if int(anchor) not in travel or not travel:
            self._last_landmark_loop = {
                "status": "skipped_anchor",
                "anchor": int(anchor),
            }
            return
        cur = max(int(k) for k in travel)
        gap = float(travel[cur]) - float(travel[int(anchor)])
        min_gap = float(self._cfg.builtin_loop_min_travel_m)
        if gap < min_gap or int(anchor) >= cur:
            self._last_landmark_loop = {
                "status": "skipped_near",
                "anchor": int(anchor),
                "gap_m": round(gap, 2),
            }
            return
        self._loop_attempts += 1
        self._last_loop_mono = now
        try:
            result = add(fix.fit.pose, int(anchor), score=fix.score)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"landmark_loop: {exc}"
            self._last_landmark_loop = {"status": "error", "error": repr(exc)}
            return
        applied = bool(result.get("applied"))
        if applied:
            self._loop_applied += 1
        self._last_landmark_loop = {
            "status": str(result.get("status", "applied" if applied else "not_applied")),
            "anchor": int(anchor),
            "inliers": int(fix.fit.inliers.sum()),
            "jump_m": round(fix.jump_m, 3),
            "jump_deg": round(fix.jump_deg, 2),
            "gap_m": round(gap, 2),
            "applied": applied,
        }
        self._log(
            f"vision landmark loop: {self._last_landmark_loop['status']} "
            f"anchor={int(anchor)} jump={fix.jump_m:.2f}m/{fix.jump_deg:.1f}deg "
            f"inliers={int(fix.fit.inliers.sum())}"
        )

    def _audit_landmarks(
        self,
        frame: VisionFrame,
        pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
    ) -> None:
        """Compare stored landmarks with this frame's depth; drop air landmarks."""
        removes = self._audit_removes()
        with self._lock:
            audit = self._landmarks.audit_against_depth(
                frame.depth_m,
                robot_pose=pose,
                pose_of_node=pose_of_node,
                intrinsics=frame.intrinsics,
                mount_x=float(frame.mount_x),
                mount_y=float(frame.mount_y),
                mount_z=float(frame.mount_z),
                mount_theta=float(frame.mount_theta),
                mount_pitch=float(frame.mount_pitch),
                mount_roll=float(frame.mount_roll),
                remove=removes,
            )
            rows = audit["rows"]
            summary: Dict[str, Any] = {
                "in_view": len(rows),
                # Most landmarks seen through at once: blamed on the pose, so
                # this frame counted no strikes and removed nothing.
                "pose_suspect": bool(audit["pose_suspect"]),
            }
            for row in rows:
                verdict = str(row["verdict"])
                summary[verdict] = int(summary.get(verdict, 0)) + 1
            removed = [row for row in rows if row["removed"]]
            summary["removed_ids"] = [int(row["id"]) for row in removed]
            self._audit_frames += 1
            if audit["pose_suspect"]:
                self._audit_pose_suspect_frames += 1
            self._audit_recent_removed = (
                self._audit_recent_removed + removed
            )[-_AUDIT_RECENT_REMOVED:]
            # Worst first, so a truncated list keeps what matters.
            rows.sort(
                key=lambda row: (
                    0 if row["verdict"] == AUDIT_SEEN_THROUGH else 1,
                    -int(row["strikes"]),
                    int(row["id"]),
                )
            )
            self._last_audit = {
                "mono": time.monotonic(),
                "camera": str(frame.camera_name),
                "pose": {
                    "x": round(float(pose.x), 3),
                    "y": round(float(pose.y), 3),
                    "theta_deg": round(math.degrees(float(pose.theta)), 1),
                },
                "summary": summary,
                "rows": rows[:_AUDIT_MAX_ROWS],
                "truncated": len(rows) > _AUDIT_MAX_ROWS,
            }
        for row in removed:
            self._log(
                f"vision: dropped landmark {row['label']} — camera sees "
                f"{row['measured_depth_m']} m where it was stored at "
                f"{row['stored_depth_m']} m"
            )

    def _try_pose_writeback(
        self,
        frame: VisionFrame,
        obs: List[PatchObservation],
        pose: conv.Pose2D,
        pose_of_node: Dict[int, conv.Pose2D],
    ) -> None:
        """Soft landmark fix → apply_map_pose_correction (never force)."""
        apply = self._apply_pose_correction
        if apply is None:
            return
        now = time.monotonic()
        cooldown = float(self._cfg.vision_pose_writeback_cooldown_s)
        if now - self._last_writeback_mono < cooldown:
            return
        self._last_writeback_mono = now
        self._writeback_attempts += 1

        with self._lock:
            confirmed = len(self._landmarks.confirmed())
            if confirmed < int(self._cfg.vision_pose_writeback_min_inliers):
                self._last_writeback = {
                    "status": "skipped_few_confirmed",
                    "confirmed": confirmed,
                }
                return
            # Match pool = image-projected confirmed (same cull as landmark-camera).
            # Keep frustum count as a diagnostic for the old facing filter.
            visible_frustum = self._landmarks.visible_set(
                pose,
                pose_of_node,
                camera_xy=(float(frame.mount_x), float(frame.mount_y)),
                camera_yaw=float(frame.mount_theta),
                fov_deg=87.0,
                max_range_m=float(self._cfg.landmark_fix_max_range_m),
            )
            visible = self._landmarks.image_visible_set(
                pose,
                pose_of_node,
                intrinsics=frame.intrinsics,
                mount_x=float(frame.mount_x),
                mount_y=float(frame.mount_y),
                mount_z=float(frame.mount_z),
                mount_theta=float(frame.mount_theta),
                mount_pitch=float(frame.mount_pitch),
                mount_roll=float(frame.mount_roll),
                max_range_m=float(self._cfg.landmark_fix_max_range_m),
            )
            cosine_thr = float(self._landmarks.cosine_threshold)

        min_inliers = int(self._cfg.vision_pose_writeback_min_inliers)
        visible_ids = [int(lm.landmark_id) for lm, _ in visible]
        residual_m = float(self._cfg.landmark_fix_residual_m)
        if self._localization_lost():
            residual_m = max(residual_m, _LOST_FIT_RESIDUAL_M)
        attempt = estimate_landmark_fix(
            obs,
            visible,
            robot_pose=pose,
            cosine_threshold=cosine_thr,
            min_inliers=min_inliers,
            max_jump_m=float(self._cfg.landmark_fix_max_jump_m),
            residual_m=residual_m,
        )
        fix = attempt.fix
        if fix is None:
            pairs = int(attempt.pairs)
            diag = diagnose_landmark_match(
                obs,
                visible,
                robot_pose=pose,
                cosine_threshold=cosine_thr,
            )
            reason = attempt.reason or "rejected_fit"
            if reason == "few_pairs":
                if int(diag.get("n_obs") or 0) == 0:
                    reason = "no_live_obs"
                elif int(diag.get("n_geom_ok") or 0) == 0:
                    reason = "geom_far"  # live pts not near any visible landmark
                elif int(diag.get("n_cos_ok") or 0) == 0:
                    reason = "weak_cosine"  # nearby but descriptors disagree
                else:
                    reason = "few_pairs"
            paired_ids = [int(x) for x in (attempt.paired_ids or [])]
            self._last_writeback_match = {
                "visible_ids": list(visible_ids),
                "paired_ids": paired_ids,
            }
            self._last_writeback = {
                "status": "rejected_fit",
                "reason": reason,
                "pairs": pairs,
                "min_inliers": min_inliers,
                "visible": len(visible),
                "visible_image": len(visible),
                "visible_frustum": len(visible_frustum),
                "confirmed": confirmed,
                "visible_ids": list(visible_ids),
                "paired_ids": paired_ids,
                "match": diag,
            }
            if attempt.jump_m is not None:
                self._last_writeback["jump_m"] = round(float(attempt.jump_m), 3)
            if attempt.jump_deg is not None:
                self._last_writeback["jump_deg"] = round(float(attempt.jump_deg), 2)
            return
        # Ignore tiny corrections (noise).
        if fix.jump_m < 0.04 and fix.jump_deg < 1.0:
            paired_ids = [int(x) for x in (attempt.paired_ids or [])]
            self._last_writeback_match = {
                "visible_ids": list(visible_ids),
                "paired_ids": paired_ids,
            }
            self._last_writeback = {
                "status": "skipped_small",
                "jump_m": round(fix.jump_m, 3),
                "jump_deg": round(fix.jump_deg, 2),
                "score": round(fix.score, 3),
                "inliers": int(fix.fit.inliers.sum()),
                "pairs": fix.n_pairs,
                "visible_ids": list(visible_ids),
                "paired_ids": paired_ids,
            }
            return
        try:
            result = apply(fix.fit.pose, score=fix.score, force=False)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"writeback: {exc}"
            self._last_writeback = {"status": "error", "error": repr(exc)}
            return
        applied = bool(result.get("applied"))
        if applied:
            self._writeback_applied += 1
        paired_ids = [int(x) for x in (attempt.paired_ids or [])]
        self._last_writeback_match = {
            "visible_ids": list(visible_ids),
            "paired_ids": paired_ids,
        }
        self._last_writeback = {
            "status": "applied" if applied else str(result.get("status", "not_applied")),
            "jump_m": round(fix.jump_m, 3),
            "jump_deg": round(fix.jump_deg, 2),
            "score": round(fix.score, 3),
            "residual_m": round(float(fix.fit.residual_rmse_m), 4),
            "inliers": int(fix.fit.inliers.sum()),
            "pairs": fix.n_pairs,
            "visible": len(visible),
            "visible_image": len(visible),
            "visible_frustum": len(visible_frustum),
            "visible_ids": list(visible_ids),
            "paired_ids": paired_ids,
            "shift_m": result.get("shift_m"),
            "shift_deg": result.get("shift_deg"),
            "slam_status": result.get("status"),
        }
        self._log(
            f"vision writeback: {self._last_writeback['status']} "
            f"jump={fix.jump_m:.2f}m/{fix.jump_deg:.1f}deg "
            f"score={fix.score:.2f} inliers={int(fix.fit.inliers.sum())}"
        )


def seeded_poses_conflict(
    poses: Sequence[conv.Pose2D], *, sep_m: float, sep_deg: float
) -> bool:
    """True when two fitted poses are different places."""
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            a, b = poses[i], poses[j]
            if math.hypot(a.x - b.x, a.y - b.y) > float(sep_m):
                return True
            turn = abs(math.degrees(conv.normalize_angle(a.theta - b.theta)))
            if turn > float(sep_deg):
                return True
    return False


def dominant_loop_anchor(
    paired_ids: Sequence[int],
    inliers,
    anchor_of: Dict[int, int],
    *,
    min_inliers: int,
) -> Optional[int]:
    """Keyframe that owns at least ``min_inliers`` of the fit, or None.

    A loop edge has to come from landmarks stored on one older node. Pairs
    split across several keyframes do not vote together.
    """
    mask = np.asarray(inliers, dtype=bool).reshape(-1)
    counts: Dict[int, int] = {}
    for i, lid in enumerate(paired_ids):
        if i >= int(mask.size) or not bool(mask[i]):
            continue
        anchor = anchor_of.get(int(lid))
        if anchor is None or int(anchor) < 0:
            continue
        counts[int(anchor)] = counts.get(int(anchor), 0) + 1
    if not counts:
        return None
    anchor, n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    if n < int(min_inliers):
        return None
    return int(anchor)


def _resolve_detector_classes(
    yolo_path: str,
    *,
    configured: Sequence[str],
) -> Tuple[str, ...]:
    """Config list wins; else sibling ``.classes.json`` (YOLO-World indoor default)."""
    configured_clean = tuple(
        str(x).strip() for x in configured if str(x).strip()
    )
    if configured_clean:
        return configured_clean
    if not yolo_path:
        return ()
    sidecar = Path(yolo_path).with_suffix(".classes.json")
    # Also accept the committed name next to a renamed onnx copy.
    if not sidecar.is_file():
        pkg = Path(__file__).resolve().parent / "models" / (
            Path(yolo_path).stem + ".classes.json"
        )
        if pkg.is_file():
            sidecar = pkg
    if not sidecar.is_file() and Path(yolo_path).name == _DEFAULT_YOLO_WORLD_ONNX:
        sidecar = (
            Path(__file__).resolve().parent
            / "models"
            / "yolov8s-worldv2-indoor.classes.json"
        )
    if not sidecar.is_file():
        return ()
    try:
        raw = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(x).strip() for x in raw if str(x).strip())


def _tokens_outside_detections(
    patches: Sequence[PatchToken],
    detections: Sequence[Detection],
    *,
    landmark_labels: Sequence[str],
) -> List[PatchToken]:
    """Drop dense tokens that already sit inside a landmark detector box."""
    if not detections or not landmark_labels:
        return list(patches)
    from .detect_landmarks import label_allowed

    allowed = {str(x).strip().lower() for x in landmark_labels if str(x).strip()}
    boxes = []
    for det in detections:
        if not label_allowed(det.label, allowed):
            continue
        x0, x1 = sorted((float(det.x0), float(det.x1)))
        y0, y1 = sorted((float(det.y0), float(det.y1)))
        boxes.append((x0, y0, x1, y1))
    if not boxes:
        return list(patches)
    out: List[PatchToken] = []
    for tok in patches:
        u, v = float(tok.u), float(tok.v)
        if any(x0 <= u <= x1 and y0 <= v <= y1 for x0, y0, x1, y1 in boxes):
            continue
        out.append(tok)
    return out


def _resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == new_h and w == new_w:
        return img
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[ys][:, xs]


def _resolve_model_path(raw: str) -> str:
    """Absolute path, cwd-relative, or package ``semantic/models/`` relative."""
    text = (raw or "").strip()
    if not text:
        return ""
    p = Path(text).expanduser()
    if p.is_file():
        return str(p.resolve())
    if not p.is_absolute():
        pkg = Path(__file__).resolve().parent / "models" / text
        if pkg.is_file():
            return str(pkg.resolve())
        cwd = Path.cwd() / text
        if cwd.is_file():
            return str(cwd.resolve())
    # Keep the configured string so the worker surfaces FileNotFoundError.
    return str(p)
