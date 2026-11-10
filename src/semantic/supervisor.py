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
from .detect_landmarks import observations_from_detections
from .landmarks import EmbeddingLandmarkMap
from .localize import (
    diagnose_landmark_match,
    estimate_landmark_fix,
)
from .patches import PatchObservation, backproject_patches, subsample_observations
from .persist import (
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

# Dense patches used only for writeback pairing (never surveyed) so detector-only
# survey does not starve the rigid fit of inliers.
_WRITEBACK_DENSE_MATCH_CAP = 24

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
        logger=None,
    ):
        self._cfg = cfg
        self._cameras = dict(cameras or {})
        self._get_pose = get_pose
        self._get_pose_at = get_pose_at
        self._get_anchor = get_anchor
        self._apply_pose_correction = apply_pose_correction
        self._is_localization_lost = is_localization_lost
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
        self._last_detection_obs = 0
        self._last_detection_boxes = 0
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
        with self._lock:
            last = self._last_result
            geom = self._last_camera_geom
        if last is None or not last.detections:
            return [], None
        size = None
        if isinstance(geom, dict):
            gw = int(geom.get("width") or 0)
            gh = int(geom.get("height") or 0)
            if gw > 0 and gh > 0:
                size = (gw, gh)
        return list(last.detections), size

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

    def reset_landmarks(self) -> None:
        """Drop the in-memory landmark map (e.g. after ``clear_map``)."""
        synthetic = bool(self._worker.cfg.synthetic)
        with self._lock:
            self._landmarks = EmbeddingLandmarkMap(
                associate_radius_m=float(self._cfg.vision_associate_radius_m),
                cell_m=float(self._cfg.vision_cell_m),
                require_descriptor_match=not synthetic,
                demote_passes=5 if synthetic else 3,
            )
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
            self._observations = 0
            self._frames = 0
            self._last_result = None
            self._last_error = None

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
                "place": {
                    "enabled": bool(self._cfg.vision_place_enabled),
                    "retrieval": bool(self._cfg.vision_place_retrieval),
                    "retrieval_active": self._place_retrieval_enabled(),
                    **self._place.status_dict(),
                    "last_query": self._last_place_query,
                    "retrieval_attempts": self._place_retrieval_attempts,
                    "retrieval_applied": self._place_retrieval_applied,
                    "last_retrieval": self._last_place_retrieval,
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
                if self._cfg.vision_place_enabled:
                    await asyncio.to_thread(
                        self._place_tick,
                        frame,
                        result,
                        allow_keyframe=allow_place_kf,
                    )
                if do_survey:
                    await asyncio.to_thread(self._survey, frame, result)
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
    ) -> None:
        """Build / query the VLAD place index from live patch tokens."""
        descs = descriptors_from_patches(result.patches, max_patches=64)
        if descs is None:
            return
        pose = self._get_pose_at(float(frame.captured_at)) or self._get_pose()

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

    def _survey(self, frame: VisionFrame, result: VisionResult) -> None:
        """Associate patches into the landmark map; optionally soft-correct pose.

        Prefer detector boxes (YOLO-World / COCO): pool DINOv2 tokens inside each
        allowed box into one landmark observation. Dense texture patches remain as
        a capped fallback when the detector is quiet or missing.
        """
        z_min = float(self._cfg.vision_z_min_m)
        z_max = float(self._cfg.vision_z_max_m)
        prefer = bool(self._cfg.vision_prefer_detections)
        det_obs: List[PatchObservation] = []
        if prefer and result.detections and result.patches:
            det_obs = observations_from_detections(
                frame,
                result.patches,
                result.detections,
                landmark_labels=self._cfg.vision_landmark_labels,
                z_min_m=z_min,
                z_max_m=z_max,
                min_patches=1,
            )
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
        writeback_obs = list(det_obs) + list(wb_dense)
        if not obs and not writeback_obs:
            return
        try:
            anchor_node, anchor_pose, pose_of_node = self._get_anchor()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"anchor: {exc}"
            return
        # Pose at capture for multi-view promotion (XY + yaw); fall back to current.
        pose = self._get_pose_at(float(frame.captured_at)) or self._get_pose()
        view_xy = (float(pose.x), float(pose.y), float(pose.theta))
        if anchor_node < 0:
            # No keyframe yet — park candidates on a virtual node 0 at current pose.
            anchor_node = 0
            anchor_pose = pose
            pose_of_node = {0: pose}

        n_det = len(det_obs)
        n_new = 0
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
                self._landmarks.observe_with_poses(
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
                n_new += 1
            # Collapse detector twins that both made it to confirmed/candidate
            # on the same object (e.g. DC1 + DC12 stacked on one painting).
            self._landmarks.merge_overlapping_detector_landmarks()
            self._observations += n_new

        if self._writeback_enabled() and writeback_obs:
            self._try_pose_writeback(frame, writeback_obs, pose, pose_of_node)

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
        attempt = estimate_landmark_fix(
            obs,
            visible,
            robot_pose=pose,
            cosine_threshold=cosine_thr,
            min_inliers=min_inliers,
            max_jump_m=float(self._cfg.landmark_fix_max_jump_m),
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
            self._last_writeback = {
                "status": "rejected_fit",
                "reason": reason,
                "pairs": pairs,
                "min_inliers": min_inliers,
                "visible": len(visible),
                "visible_image": len(visible),
                "visible_frustum": len(visible_frustum),
                "confirmed": confirmed,
                "match": diag,
            }
            if attempt.jump_m is not None:
                self._last_writeback["jump_m"] = round(float(attempt.jump_m), 3)
            if attempt.jump_deg is not None:
                self._last_writeback["jump_deg"] = round(float(attempt.jump_deg), 2)
            return
        # Ignore tiny corrections (noise).
        if fix.jump_m < 0.04 and fix.jump_deg < 1.0:
            self._last_writeback = {
                "status": "skipped_small",
                "jump_m": round(fix.jump_m, 3),
                "jump_deg": round(fix.jump_deg, 2),
                "score": round(fix.score, 3),
                "inliers": int(fix.fit.inliers.sum()),
                "pairs": fix.n_pairs,
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
            "shift_m": result.get("shift_m"),
            "shift_deg": result.get("shift_deg"),
            "slam_status": result.get("status"),
        }
        self._log(
            f"vision writeback: {self._last_writeback['status']} "
            f"jump={fix.jump_m:.2f}m/{fix.jump_deg:.1f}deg "
            f"score={fix.score:.2f} inliers={int(fix.fit.inliers.sum())}"
        )


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
