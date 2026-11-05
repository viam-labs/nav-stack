"""Background vision loop: grab RGB-D, infer, survey landmarks (no pose writeback)."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import LidarConfig, MODE_MAPPING, SlamConfig
from ..geom import conversions as conv
from .decode import decode_color_image, decode_depth_image, pick_named_images
from .frames import intrinsics_from_size, vision_frame_from_arrays
from .landmarks import EmbeddingLandmarkMap
from .patches import backproject_patches
from .persist import load_landmarks, save_landmarks
from .types import VisionFrame, VisionResult
from .worker import VisionWorker, WorkerConfig


PoseProvider = Callable[[], conv.Pose2D]
PoseAtProvider = Callable[[float], Optional[conv.Pose2D]]
AnchorProvider = Callable[[], Tuple[int, conv.Pose2D, Dict[int, conv.Pose2D]]]


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
        logger=None,
    ):
        self._cfg = cfg
        self._cameras = dict(cameras or {})
        self._get_pose = get_pose
        self._get_pose_at = get_pose_at
        self._get_anchor = get_anchor
        self._logger = logger
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._landmarks = EmbeddingLandmarkMap()
        self._last_result: Optional[VisionResult] = None
        self._last_frame_age_s: Optional[float] = None
        self._last_error: Optional[str] = None
        self._frames = 0
        self._observations = 0
        self._inflight: set = set()

        worker_cfg = WorkerConfig(
            dinov2_onnx=str(cfg.vision_dinov2_onnx or ""),
            yolo_onnx=str(cfg.vision_yolo_onnx or ""),
            synthetic=bool(cfg.vision_synthetic) or not bool(cfg.vision_dinov2_onnx),
            cpu_threads=2,
            input_short_side=224 if not cfg.vision_dinov2_onnx else 448,
        )
        self._worker = VisionWorker(worker_cfg)

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

    def load_from_map_dir(self, map_dir) -> bool:
        loaded = load_landmarks(map_dir)
        if loaded is None:
            return False
        with self._lock:
            self._landmarks = loaded
        self._log(f"vision: loaded {len(loaded)} landmarks from {map_dir}")
        return True

    def save_to_map_dir(self, map_dir) -> Optional[str]:
        with self._lock:
            path = save_landmarks(map_dir, self._landmarks)
        return str(path) if path is not None else None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._cfg.vision_enabled:
            return
        self.stop()
        self._stop = asyncio.Event()
        self._task = loop.create_task(self._run(), name="vision-supervisor")
        self._log(
            f"vision: started cameras={list(self._cfg.vision_cameras)} "
            f"rate={self._cfg.vision_rate_hz}Hz "
            f"survey_only={self._cfg.vision_survey_only} "
            f"provider={self._worker.stats.get('provider')}"
        )

    def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            self._stop.set()
            task.cancel()
        self._worker.close()

    def status(self) -> dict:
        with self._lock:
            lm = self._landmarks.status_dict()
            last = self._last_result
            return {
                "enabled": bool(self._cfg.vision_enabled),
                "survey_only": bool(self._cfg.vision_survey_only),
                "cameras": list(self._cfg.vision_cameras),
                "provider": self._worker.stats.get("provider"),
                "frames": self._frames,
                "observations": self._observations,
                "landmarks": lm,
                "last_infer_ms": None if last is None else float(last.infer_ms),
                "last_frame_age_s": self._last_frame_age_s,
                "last_error": self._last_error,
                "worker": dict(self._worker.stats),
            }

    def ingest_frame(self, frame: VisionFrame, *, mapping: bool = True) -> VisionResult:
        """Synchronous path for tests / sim: process one frame and maybe survey."""
        result = self._worker.process(frame)
        with self._lock:
            self._last_result = result
            self._frames += 1
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

    async def _tick_once(self) -> None:
        for lidar in self._cfg.vision_lidar_configs():
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
                if self._cfg.mode == MODE_MAPPING or self._cfg.vision_survey_only:
                    await asyncio.to_thread(self._survey, frame, result)
            finally:
                self._inflight.discard(name)

    async def _grab_frame(self, cam, lidar: LidarConfig) -> Optional[VisionFrame]:
        """Pull color+depth via get_images; return None if unavailable."""
        received = time.monotonic()
        try:
            images, _meta = await cam.get_images(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"get_images({lidar.name}): {exc}"
            return None
        if not images:
            return None
        color_blob, depth_blob = pick_named_images(images)
        if color_blob is None or depth_blob is None:
            self._last_error = f"{lidar.name}: need color+depth in get_images"
            return None
        try:
            color = decode_color_image(color_blob["data"], color_blob["mime"])
            depth = decode_depth_image(depth_blob["data"], depth_blob["mime"])
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"decode({lidar.name}): {exc}"
            return None
        if color.shape[:2] != depth.shape[:2]:
            # Resize depth with nearest neighbor to color size.
            depth = _resize_nearest(depth, color.shape[0], color.shape[1])
        K = intrinsics_from_size(
            color.shape[1], color.shape[0], fov_deg=float(lidar.fov_deg)
        )
        return vision_frame_from_arrays(
            camera_name=lidar.name,
            color_rgb=color,
            depth_m=depth,
            stamp=time.time(),
            captured_at=received,
            lidar=lidar,
            intrinsics=K,
        )

    def _survey(self, frame: VisionFrame, result: VisionResult) -> None:
        """Associate patches into the landmark map (no pose writeback)."""
        obs = backproject_patches(frame, result.patches)
        if not obs:
            return
        try:
            anchor_node, anchor_pose, pose_of_node = self._get_anchor()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"anchor: {exc}"
            return
        # Pose at capture for view_xy; fall back to current.
        pose = self._get_pose_at(float(frame.captured_at)) or self._get_pose()
        view_xy = (float(pose.x), float(pose.y))
        if anchor_node < 0:
            # No keyframe yet — park candidates on a virtual node 0 at current pose.
            anchor_node = 0
            anchor_pose = pose
            pose_of_node = {0: pose}

        n_new = 0
        with self._lock:
            for o in obs:
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
                )
                n_new += 1
            self._observations += n_new


def _resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == new_h and w == new_w:
        return img
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[ys][:, xs]
