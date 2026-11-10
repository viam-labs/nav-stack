"""Landmark overlay camera: ``viam-labs:nav-stack:vision-camera``.

Pulls a live RGB frame from a source camera and draws embedding-landmark
boxes from the in-process SLAM vision supervisor (confirmed / candidates).
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from ..config import VisionCameraConfig
from ..runtime import get_slam, get_slam_service
from ..semantic.decode import decode_color_image, pick_named_images
from ..semantic.frames import intrinsics_from_size
from ..semantic.types import CameraIntrinsics
from ..viz.landmarks_view import (
    LandmarkOverlayOptions,
    collect_landmark_map_points,
    placeholder_landmarks_png,
    render_landmarks_overlay,
)

LOGGER = getLogger(__name__)


class VisionCamera(Camera):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "vision-camera"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[VisionCameraConfig] = None
        self._source: Optional[Camera] = None
        self._opts = LandmarkOverlayOptions()
        self._stable_candidates_only = True
        self._last_error: Optional[str] = None
        self._last_drawn = 0
        self._last_detections_drawn = 0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        cfg = VisionCameraConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = VisionCameraConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._opts = LandmarkOverlayOptions(
            show_confirmed=cfg.show_confirmed,
            show_candidates=cfg.show_candidates,
            show_moved=cfg.show_moved,
            show_detections=cfg.show_detections,
            show_labels=cfg.show_labels,
            max_landmarks=cfg.max_landmarks,
            box_m=cfg.box_m,
        )
        self._stable_candidates_only = bool(cfg.stable_candidates_only)
        source = dependencies.get(Camera.get_resource_name(cfg.source))
        if source is None:
            raise RuntimeError(f"vision-camera source {cfg.source!r} missing")
        self._source = source  # type: ignore[assignment]
        LOGGER.info(
            f"nav-stack vision-camera '{self.name}' overlaying landmarks from "
            f"slam={cfg.slam!r} on source={cfg.source!r}"
        )

    async def _grab_rgb(self) -> np.ndarray:
        """Pull a color frame from the source camera as HxWx3 uint8."""
        cam = self._source
        if cam is None:
            raise RuntimeError("vision-camera: source not configured")
        # Prefer get_images (RealSense color+depth); fall back to get_image.
        if hasattr(cam, "get_images"):
            try:
                images, _meta = await cam.get_images(timeout=2.0)
                if images:
                    color_blob, _depth = pick_named_images(images)
                    if color_blob is not None:
                        return decode_color_image(
                            color_blob["data"], color_blob["mime"]
                        )
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"get_images: {exc}"
        img = await cam.get_image(mime_type="", timeout=2.0)
        data = getattr(img, "data", None) or bytes(img)
        mime = str(
            getattr(img, "mime_type", None)
            or getattr(img, "mimeType", None)
            or "image/jpeg"
        )
        return decode_color_image(data, mime)

    def _mount_and_fov(self, source_name: str, vision=None):
        """Mount/FOV for overlay — prefer last survey geometry when available."""
        geom = getattr(vision, "_last_camera_geom", None) if vision is not None else None
        if isinstance(geom, dict) and (
            not source_name
            or str(geom.get("camera_name") or "") in ("", source_name)
        ):
            return (
                float(geom.get("mount_x", 0.0)),
                float(geom.get("mount_y", 0.0)),
                float(geom.get("mount_z", 0.0)),
                float(geom.get("mount_theta", 0.0)),
                float(geom.get("mount_pitch", 0.0)),
                float(geom.get("mount_roll", 0.0)),
                float(geom.get("fov_deg", 87.0)),
                float(geom.get("vfov_deg", geom.get("fov_deg", 58.0))),
                geom,
            )
        runtime = get_slam(self._cfg.slam) if self._cfg else None
        slam_cfg = getattr(runtime, "slam_cfg", None) if runtime else None
        if slam_cfg is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 87.0, 58.0, None
        by_name = {lidar.name: lidar for lidar in getattr(slam_cfg, "lidars", [])}
        lidar = by_name.get(source_name)
        if lidar is None:
            for name in getattr(slam_cfg, "vision_cameras", None) or []:
                lidar = by_name.get(name)
                if lidar is not None:
                    break
        if lidar is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 87.0, 58.0, None
        return (
            float(lidar.x),
            float(lidar.y),
            float(lidar.z),
            float(lidar.theta),
            float(lidar.pitch),
            float(lidar.roll),
            float(lidar.fov_deg),
            float(lidar.vfov_deg),
            None,
        )

    async def _render(self) -> bytes:
        cfg = self._cfg
        if cfg is None or self._source is None:
            return placeholder_landmarks_png("vision-camera: not configured")

        svc = get_slam_service(cfg.slam)
        if svc is None:
            return placeholder_landmarks_png(
                f"vision-camera: slam {cfg.slam!r} not running"
            )
        vision = getattr(svc, "_vision", None)
        engine = getattr(svc, "_engine", None)
        if vision is None:
            return placeholder_landmarks_png(
                "vision-camera: vision not enabled on slam "
                "(set vision_enabled / vision_cameras)"
            )
        if engine is None:
            return placeholder_landmarks_png("vision-camera: slam engine missing")

        # Stamp pose to the RGB grab time. Reading pose *after* a slow
        # get_images during a spin projects landmarks with a newer yaw than
        # the image — boxes slide on the painting even when map_xyz is fixed.
        try:
            t0 = time.monotonic()
            color = await self._grab_rgb()
            t1 = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            self._last_error = repr(exc)
            return placeholder_landmarks_png(f"vision-camera: RGB grab failed: {exc}")

        h, w = color.shape[:2]
        mx, my, mz, mth, mpitch, mroll, fov, vfov, geom = self._mount_and_fov(
            cfg.source, vision=vision
        )
        if (
            isinstance(geom, dict)
            and int(geom.get("width") or 0) == w
            and int(geom.get("height") or 0) == h
        ):
            K = CameraIntrinsics(
                fx=float(geom["fx"]),
                fy=float(geom["fy"]),
                cx=float(geom["cx"]),
                cy=float(geom["cy"]),
                width=w,
                height=h,
            )
        else:
            K = intrinsics_from_size(w, h, fov_deg=fov, vfov_deg=vfov)
        try:
            t_img = 0.5 * (t0 + t1)
            pose = None
            get_at = getattr(engine, "get_pose_at", None)
            if callable(get_at):
                pose = get_at(t_img)
            if pose is None:
                pose = engine.get_pose()
            _idx, _ap, pose_of_node = engine.vision_anchor()
            if not pose_of_node:
                pose_of_node = {0: pose}
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"pose: {exc}"
            return placeholder_landmarks_png(f"vision-camera: pose failed: {exc}")

        landmarks = collect_landmark_map_points(
            vision.landmarks,
            pose_of_node,
            include_candidates=bool(self._opts.show_candidates),
            include_moved=bool(self._opts.show_moved),
            stable_candidates_only=bool(self._stable_candidates_only),
        )
        detections = []
        det_size = None
        yolo_loaded = None
        worker_stats = getattr(getattr(vision, "_worker", None), "stats", None)
        if isinstance(worker_stats, dict):
            yolo_loaded = bool(worker_stats.get("yolo_loaded"))
        if self._opts.show_detections and hasattr(vision, "last_detections"):
            try:
                detections, det_size = vision.last_detections(camera_name=cfg.source)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"detections: {exc}"
                detections, det_size = [], None
        wb_vis: list = []
        wb_pair: list = []
        if hasattr(vision, "writeback_match_overlay"):
            try:
                wb = vision.writeback_match_overlay()
                wb_vis = list(wb.get("visible_ids") or [])
                wb_pair = list(wb.get("paired_ids") or [])
            except Exception:  # noqa: BLE001
                wb_vis, wb_pair = [], []
        seen_through: list = []
        if hasattr(vision, "seen_through_ids"):
            try:
                seen_through = list(vision.seen_through_ids())
            except Exception:  # noqa: BLE001
                seen_through = []
        png = await asyncio.to_thread(
            render_landmarks_overlay,
            color,
            landmarks,
            robot_pose=pose,
            intrinsics=K,
            mount_x=mx,
            mount_y=my,
            mount_z=mz,
            mount_theta=mth,
            mount_pitch=mpitch,
            mount_roll=mroll,
            options=self._opts,
            detections=detections,
            detection_frame_size=det_size,
            yolo_loaded=yolo_loaded,
            writeback_visible_ids=wb_vis,
            writeback_paired_ids=wb_pair,
            seen_through_ids=seen_through,
        )
        self._last_error = None
        self._last_drawn = min(len(landmarks), int(self._opts.max_landmarks))
        self._last_detections_drawn = len(detections)
        return png

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra=None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ViamImage:
        png = await self._render()
        return ViamImage(png, CameraMimeType.PNG)

    async def get_images(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[list[NamedImage], ResponseMetadata]:
        png = await self._render()
        return [NamedImage(self.name, png, CameraMimeType.PNG)], ResponseMetadata()

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        raise NotImplementedError("vision-camera renders images only, not point clouds")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        return Camera.Properties(supports_pcd=False, mime_types=[CameraMimeType.PNG])

    async def do_command(
        self,
        command: Mapping[str, object],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, object]:
        cfg = self._cfg
        return {
            "configured": cfg is not None,
            "slam": None if cfg is None else cfg.slam,
            "source": None if cfg is None else cfg.source,
            "last_error": self._last_error,
            "last_drawn": self._last_drawn,
            "last_detections_drawn": self._last_detections_drawn,
            "show_candidates": self._opts.show_candidates,
            "show_confirmed": self._opts.show_confirmed,
            "show_detections": self._opts.show_detections,
        }


Registry.register_resource_creator(
    Camera.API,
    VisionCamera.MODEL,
    ResourceCreatorRegistration(VisionCamera.new, VisionCamera.validate_config),
)
