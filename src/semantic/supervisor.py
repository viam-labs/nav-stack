"""Background vision loop: grab RGB-D, store recovery keyframes, recover a lost pose."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MODE_LOCALIZING, MODE_MAPPING, LidarConfig, SlamConfig
from ..geom import conversions as conv
from .decode import decode_color_image, decode_depth_image, pick_named_images
from .frames import (
    intrinsics_from_camera_properties,
    intrinsics_from_size,
    vision_frame_from_arrays,
)
from .keyframes import KeyframeIndex, keyframe_gap_reached, stack_tokens, token_points
from .paint_mask import discs_from_detections
from .persist import load_place_index, save_place_index
from .place import PlaceIndex, descriptors_from_patches
from .relocalize import LidarSearch, recover_pose
from .types import CameraIntrinsics, VisionFrame, VisionResult
from .worker import VisionWorker, WorkerConfig

PoseProvider = Callable[[], conv.Pose2D]
PoseAtProvider = Callable[[float], Optional[conv.Pose2D]]
AnchorProvider = Callable[[], Tuple[int, conv.Pose2D, Dict[int, conv.Pose2D]]]
# Same signature as BuiltinSlamHost.apply_map_pose_correction (force always False).
PoseCorrection = Callable[..., Dict[str, Any]]
# True when SLAM localization_state is lost/unknown (place apply only then).
LocLostProvider = Callable[[], bool]
# ``(discs, captured_at)`` → engine.note_paint_discs.
PaintDiscNote = Callable[..., None]
# ``(pose)`` → True when SLAM took the recovered pose.
Relocalize = Callable[[conv.Pose2D], bool]

# A frame whose capture-time pose cannot be looked up is still usable while it
# is this fresh (no odometry history yet); older than this it is dropped.
_MAX_UNSTAMPED_FRAME_AGE_S = 1.0

# Fetched by setup.sh / run.sh next to this package (same pattern as DINOv2).
_DEFAULT_YOLO_WORLD_ONNX = "yolov8s-worldv2-indoor.onnx"


class VisionSupervisor:
    """Owns the worker and the recovery keyframes; never blocks the SLAM tick."""

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
        note_paint_discs: Optional[PaintDiscNote] = None,
        lidar_search: Optional[LidarSearch] = None,
        apply_relocalization: Optional[Relocalize] = None,
        is_base_stopped: Optional[Callable[[], bool]] = None,
        logger=None,
    ):
        self._cfg = cfg
        self._cameras = dict(cameras or {})
        self._get_pose = get_pose
        self._get_pose_at = get_pose_at
        self._get_anchor = get_anchor
        self._apply_pose_correction = apply_pose_correction
        self._is_localization_lost = is_localization_lost
        self._note_paint_discs = note_paint_discs
        self._lidar_search = lidar_search
        self._apply_relocalization = apply_relocalization
        self._is_base_stopped = is_base_stopped
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
        self._last_result: Optional[VisionResult] = None
        self._last_frame_age_s: Optional[float] = None
        self._last_error: Optional[str] = None
        # Per-camera K from Camera.get_properties (None = not tried / failed).
        self._intrinsics_by_cam: Dict[str, Optional[CameraIntrinsics]] = {}
        self._intrinsics_source_by_cam: Dict[str, str] = {}
        self._frames = 0
        self._inflight: set = set()
        self._rr_index = 0
        self._last_paint_discs = 0
        # Frames dropped / stamped at grab time for lack of a capture pose.
        self._stale_frames_dropped = 0
        self._grab_pose_fallbacks = 0
        self._place = PlaceIndex(cosine_threshold=float(cfg.vision_place_min_score))
        self._place_next_id = 1
        self._place_last_pose: Optional[conv.Pose2D] = None
        self._place_fit_attempted = False
        self._last_place_query: Optional[dict] = None
        self._place_retrieval_attempts = 0
        self._place_retrieval_applied = 0
        self._last_place_retrieval: Optional[dict] = None
        self._last_place_retrieval_mono = 0.0
        # Camera keyframes for lost-pose recovery (see relocalize.py).
        self._keyframes = KeyframeIndex()
        self._keyframe_next_id = 1
        self._keyframe_last_pose: Optional[conv.Pose2D] = None
        # Keyframe id → (pose-graph node, pose relative to that node), for
        # keyframes stored this session, so a loop closure moves them too.
        self._keyframe_anchor: Dict[int, Tuple[int, conv.Pose2D]] = {}
        self._recovery_attempts = 0
        self._recovery_applied = 0
        self._last_recovery: Optional[dict] = None
        self._last_recovery_mono = 0.0

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
        )
        self._worker = VisionWorker(worker_cfg)
        # If ONNX failed to load the worker flips to synthetic, and synthetic
        # tokens are never stored as keyframes.
        if self._worker.cfg.synthetic and not want_synthetic:
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

    def load_from_map_dir(self, map_dir) -> bool:
        place = load_place_index(map_dir)
        keyframes = KeyframeIndex.load(map_dir)
        ok = False
        with self._lock:
            if keyframes is not None:
                self._keyframes = keyframes
                self._keyframe_next_id = max(keyframes.keyframe_ids, default=0) + 1
                self._keyframe_anchor = {}
                ok = True
            if place is not None:
                place.cosine_threshold = float(self._cfg.vision_place_min_score)
                self._place = place
                self._place_next_id = (
                    max(self._place.keyframe_ids) + 1 if self._place.keyframe_ids else 1
                )
                self._place_fit_attempted = bool(place.fitted)
                ok = True
        if place is not None:
            self._log(
                f"vision: loaded place index keyframes={len(place.keyframe_ids)} "
                f"fitted={place.fitted} from {map_dir}"
            )
        if keyframes is not None:
            self._log(
                f"vision: loaded {len(keyframes)} recovery keyframes from {map_dir}"
            )
        return ok

    def save_to_map_dir(self, map_dir) -> Optional[str]:
        """Write the keyframes (and place index) next to the map; keyframe file path."""
        with self._lock:
            place_path = save_place_index(map_dir, self._place)
            keyframes_path = self._keyframes.save(map_dir)
        if place_path is not None:
            self._log(f"vision: saved place index → {place_path}")
        if keyframes_path is not None:
            self._log(
                f"vision: saved {len(self._keyframes)} recovery keyframes → {keyframes_path}"
            )
        return str(keyframes_path) if keyframes_path is not None else None

    def reset(self) -> None:
        """Drop everything stored for the map (e.g. after ``clear_map``)."""
        with self._lock:
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
            self._keyframes = KeyframeIndex()
            self._keyframe_next_id = 1
            self._keyframe_last_pose = None
            self._keyframe_anchor = {}
            self._recovery_attempts = 0
            self._recovery_applied = 0
            self._last_recovery = None
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
            f"recovery={self._recovery_enabled()} "
            f"round_robin={self._cfg.vision_round_robin} "
            f"provider={self._worker.stats.get('provider')} "
            f"dinov2={self._dinov2_path or 'none'} "
            f"yolo={self._yolo_path or 'none'} "
            f"yolo_loaded={yolo_loaded} "
            f"yolo_conf={self._cfg.vision_yolo_conf}"
        )
        if not yolo_loaded:
            self._log(
                "vision: YOLO detector not loaded, so people are not masked "
                "out of the map. Ensure setup/run fetched "
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
            self._log(
                f"vision: DINOv2 reload failed ({worker.stats.get('last_error')}); "
                "falling back to synthetic"
            )

    def stop(self) -> None:
        self._stop_loop()
        self._worker.close()

    def status(self) -> dict:
        with self._lock:
            last = self._last_result
            return {
                "enabled": bool(self._cfg.vision_enabled),
                "mode": str(self._cfg.mode),
                "survey_only": bool(self._cfg.vision_survey_only),
                "cameras": list(self._cfg.vision_cameras),
                "provider": self._worker.stats.get("provider"),
                "frames": self._frames,
                "last_infer_ms": None if last is None else float(last.infer_ms),
                "last_frame_age_s": self._last_frame_age_s,
                "last_error": self._last_error,
                "worker": dict(self._worker.stats),
                "round_robin": bool(self._cfg.vision_round_robin),
                "dinov2_onnx": self._dinov2_path or None,
                "yolo_onnx": self._yolo_path or None,
                "yolo_loaded": bool(self._worker.stats.get("yolo_loaded")),
                "yolo_provider": self._worker.stats.get("yolo_provider"),
                "yolo_conf": float(self._cfg.vision_yolo_conf),
                "last_yolo_out_shape": self._worker.stats.get("last_yolo_out_shape"),
                "detector_classes": list(self._worker.cfg.detector_classes),
                # Boxes of the masked classes in the latest frame.
                "last_detections": 0 if last is None else len(last.detections),
                "capture_pose": {
                    "stale_frames_dropped": int(self._stale_frames_dropped),
                    "grab_time_fallbacks": int(self._grab_pose_fallbacks),
                },
                "last_yolo_max_conf": self._worker.stats.get("last_yolo_max_conf"),
                "input_size": int(self._worker.cfg.input_short_side),
                "intrinsics_source": dict(self._intrinsics_source_by_cam),
                "paint_gate": {
                    "last_discs": int(self._last_paint_discs),
                    "labels": list(self._worker.cfg.dynamic_labels),
                },
                "recovery": {
                    "enabled": self._recovery_enabled(),
                    "active": self._recovery_active(),
                    **self._keyframes.status_dict(),
                    "attempts": self._recovery_attempts,
                    "applied": self._recovery_applied,
                    "last": self._last_recovery,
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
        """Synchronous path for tests / replay: process one frame.

        ``mapping`` masks people out of the scan and stores keyframes, as the
        loop does while the map is being built.
        """
        result = self._worker.process(frame)
        with self._lock:
            self._last_result = result
            self._frames += 1
        if mapping:
            self._publish_paint_discs(frame, result)
        if self._cfg.vision_place_enabled:
            self._place_tick(frame, result, allow_keyframe=bool(mapping))
        self._keyframe_tick(frame, result, allow_keyframe=bool(mapping))
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
            if not self._tick_needed():
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
                # While the map is being built: mask people out of the scan
                # and store keyframes. Otherwise a frame is only here to
                # recover a lost pose (or to query the place index).
                building = (
                    self._cfg.mode == MODE_MAPPING or self._cfg.vision_survey_only
                )
                if building:
                    # First: the mask only applies to scans taken within a
                    # fraction of a second of this frame.
                    self._publish_paint_discs(frame, result)
                pose = self._capture_pose(frame, grab_pose)
                if self._cfg.vision_place_enabled:
                    await asyncio.to_thread(
                        self._place_tick,
                        frame,
                        result,
                        allow_keyframe=building,
                        pose=pose,
                    )
                await asyncio.to_thread(
                    self._keyframe_tick,
                    frame,
                    result,
                    allow_keyframe=building,
                    pose=pose,
                )
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
            cached is not None and (cached.width != width or cached.height != height)
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
        return vision_frame_from_arrays(
            camera_name=lidar.name,
            color_rgb=color,
            depth_m=depth,
            stamp=time.time(),
            captured_at=captured_at,
            lidar=lidar,
            intrinsics=K,
        )

    def _recovery_enabled(self) -> bool:
        """Keyframes are stored and may be used (real DINOv2 tokens only)."""
        return bool(self._cfg.vision_recovery) and not bool(self._worker.cfg.synthetic)

    def _recovery_active(self) -> bool:
        """True while a lost pose may be recovered from the camera and lidar."""
        return (
            self._recovery_enabled()
            and str(self._cfg.mode) == MODE_LOCALIZING
            and not bool(self._cfg.vision_survey_only)
            and self._lidar_search is not None
            and self._apply_relocalization is not None
            and len(self._keyframes) > 0
            and self._localization_lost()
        )

    def _tick_needed(self) -> bool:
        """False when nothing would use a frame, so none is grabbed or inferred.

        While localizing with the pose tracking, vision has no job: keyframes
        are only stored while mapping and recovery only runs when lost.
        """
        if str(self._cfg.mode) == MODE_MAPPING or bool(self._cfg.vision_survey_only):
            return True
        if bool(self._cfg.vision_place_enabled) and bool(
            self._cfg.vision_place_retrieval
        ):
            return True
        return self._recovery_active()

    def _keyframe_tick(
        self,
        frame: VisionFrame,
        result: VisionResult,
        *,
        allow_keyframe: bool,
        pose: Optional[conv.Pose2D] = None,
    ) -> None:
        """Store a recovery keyframe while mapping; try a recovery while lost."""
        if not self._recovery_enabled() or not result.grid_patches:
            return
        if not allow_keyframe:
            if self._recovery_active():
                self._try_recover(frame, result)
            return
        if pose is None:
            pose = self._capture_pose(frame)
        if not keyframe_gap_reached(
            pose,
            self._keyframe_last_pose,
            gap_m=float(self._cfg.vision_place_keyframe_min_gap_m),
            gap_deg=float(self._cfg.vision_place_keyframe_min_gap_deg),
        ):
            return
        tokens, uv = stack_tokens(result.grid_patches)
        points = token_points(frame, uv)
        anchor = None
        try:
            node, node_pose, _poses = self._get_anchor()
            if int(node) >= 0:
                anchor = (
                    int(node),
                    conv.compose_poses(conv.invert_pose(node_pose), pose),
                )
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"keyframe anchor: {exc}"
        with self._lock:
            keyframe_id = int(self._keyframe_next_id)
            self._keyframe_next_id += 1
            self._keyframes.add(keyframe_id, pose, tokens, points)
            self._keyframe_last_pose = pose
            if anchor is not None:
                self._keyframe_anchor[keyframe_id] = anchor

    def _try_recover(self, frame: VisionFrame, result: VisionResult) -> None:
        """Lost and standing still: ask the camera where, let the lidar decide."""
        search = self._lidar_search
        apply = self._apply_relocalization
        if search is None or apply is None:
            return
        now = time.monotonic()
        if now - self._last_recovery_mono < float(self._cfg.vision_recovery_cooldown_s):
            return
        # The scan is read after the camera frame. They describe one pose
        # only while the base is still.
        stopped = self._is_base_stopped
        if stopped is not None:
            try:
                still = bool(stopped())
            except Exception:  # noqa: BLE001
                still = False
            if not still:
                self._last_recovery = {"status": "skipped_moving"}
                return
        self._last_recovery_mono = now
        self._recovery_attempts += 1
        tokens, uv = stack_tokens(result.grid_patches)
        points = token_points(frame, uv)
        try:
            recovery = recover_pose(self._keyframes, tokens, points, search)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"recovery: {exc}"
            self._last_recovery = {"status": "error", "error": repr(exc)}
            return
        report = recovery.as_dict()
        if recovery.accepted:
            try:
                applied = bool(apply(recovery.pose))
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"recovery apply: {exc}"
                applied = False
            report["applied"] = applied
            if applied:
                self._recovery_applied += 1
        self._last_recovery = report
        if recovery.accepted or recovery.status == "rival_place":
            self._log(
                f"vision recovery: {recovery.status} "
                f"keyframe={report.get('keyframe_id')} pose={report.get('pose')}"
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
                if not self._place.fitted and len(self._place.keyframe_ids) >= int(
                    self._cfg.vision_place_min_keyframes_to_fit
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
        return gap_m >= float(
            self._cfg.vision_place_keyframe_min_gap_m
        ) or gap_deg >= float(self._cfg.vision_place_keyframe_min_gap_deg)

    def follow_anchor_motion(
        self,
        old_poses: Dict[int, conv.Pose2D],
        new_poses: Dict[int, conv.Pose2D],
    ) -> None:
        """Pose-graph update: each keyframe moves with its node."""
        del old_poses
        with self._lock:
            for keyframe_id, (node, rel) in self._keyframe_anchor.items():
                node_pose = new_poses.get(node)
                if node_pose is not None:
                    self._keyframes.move(
                        keyframe_id, conv.compose_poses(node_pose, rel)
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

    def _try_place_retrieval(self, descs: np.ndarray, pose: conv.Pose2D) -> None:
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
        jump_deg = abs(math.degrees(conv.normalize_angle(hit_pose.theta - pose.theta)))
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
            "status": "applied"
            if applied
            else str(result.get("status", "not_applied")),
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


def _resolve_detector_classes(
    yolo_path: str,
    *,
    configured: Sequence[str],
) -> Tuple[str, ...]:
    """Config list wins; else sibling ``.classes.json`` (YOLO-World indoor default)."""
    configured_clean = tuple(str(x).strip() for x in configured if str(x).strip())
    if configured_clean:
        return configured_clean
    if not yolo_path:
        return ()
    sidecar = Path(yolo_path).with_suffix(".classes.json")
    # Also accept the committed name next to a renamed onnx copy.
    if not sidecar.is_file():
        pkg = (
            Path(__file__).resolve().parent
            / "models"
            / (Path(yolo_path).stem + ".classes.json")
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
