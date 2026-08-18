"""Background PCD republisher: ``viam-labs:nav-stack:shm-pointcloud``.

Wraps an existing Viam camera (e.g. ``viam:lidar:rplidar``) and copies each
``get_point_cloud`` frame into a POSIX shm ring that SLAM/Nav2 can read without
blocking the ROS scan timer on gRPC.

Until the *source* camera writes shm itself, this is the fallback for Livox /
depth cams: gRPC still happens, but on a dedicated thread. For Slamtec A1/A3/S1
prefer ``viam-labs:nav-stack:rplidar``, which talks UART and writes shm directly.

Wire format matches ``viam-shared-memory-test`` (see :mod:`~..ros.pcshm`).
"""
from __future__ import annotations

import asyncio
import threading
from typing import ClassVar, Mapping, Optional, Sequence

from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from ..ros import pcshm

LOGGER = getLogger(__name__)


class ShmPointCloud(Camera):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "shm-pointcloud"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._source: Optional[Camera] = None
        self._source_name: Optional[str] = None
        self._shm: Optional[pcshm.Writer] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._latest = b""
        self._produce_hz = 10.0
        self._timeout_s = 2.0
        self._writes = 0
        self._errors = 0
        self._last_error: Optional[str] = None
        self._last_bytes = 0

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
        attrs = struct_to_dict(config.attributes)
        source = str(attrs.get("source") or attrs.get("camera") or "")
        if not source:
            raise ValueError("shm-pointcloud requires attributes.source")
        if not str(attrs.get("shm_name") or "").strip():
            raise ValueError("shm-pointcloud requires attributes.shm_name")
        return [source], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self.close_sync()
        attrs = struct_to_dict(config.attributes)
        source_name = str(attrs.get("source") or attrs.get("camera") or "")
        shm_name = str(attrs.get("shm_name") or "").strip()
        region = int(attrs.get("shm_region_size", pcshm.DEFAULT_REGION_SIZE))
        self._produce_hz = float(attrs.get("produce_hz", 10.0))
        self._timeout_s = max(float(attrs.get("timeout_s", 2.0)), 0.5)
        self._source_name = source_name
        source = dependencies.get(Camera.get_resource_name(source_name))
        if source is None:
            raise RuntimeError(f"shm-pointcloud source camera {source_name!r} missing")
        self._source = source  # type: ignore[assignment]
        self._shm = pcshm.open_writer(shm_name, region)
        self._stop = threading.Event()
        self._writes = 0
        self._errors = 0
        self._last_error = None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        if self._produce_hz > 0:
            self._thread = threading.Thread(
                target=self._produce_loop, name=f"{self.name}-shm", daemon=True
            )
            self._thread.start()
        LOGGER.info(
            "shm-pointcloud %r publishing %s -> shm %s at %.1f Hz",
            self.name,
            source_name,
            pcshm.normalize_name(shm_name),
            self._produce_hz,
        )

    def _produce_loop(self) -> None:
        period = 1.0 / max(self._produce_hz, 0.1)
        while not self._stop.wait(period):
            try:
                self._publish_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self._errors += 1
                self._last_error = repr(exc)
                LOGGER.warning("shm-pointcloud %r produce failed: %s", self.name, exc)

    def _publish_once(self) -> None:
        source = self._source
        loop = self._loop
        shm = self._shm
        if source is None or loop is None or shm is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            source.get_point_cloud(timeout=self._timeout_s), loop
        )
        data = fut.result(timeout=self._timeout_s + 0.5)
        raw = data[0] if isinstance(data, tuple) else data
        if not raw:
            raise RuntimeError("empty point cloud")
        shm.write(raw)
        with self._lock:
            self._latest = raw
        self._writes += 1
        self._last_bytes = len(raw)
        self._last_error = None

    async def get_images(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError(
            "shm-pointcloud does not support get_images; use get_point_cloud"
        )

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        if self._produce_hz > 0:
            with self._lock:
                pcd = self._latest
            if not pcd:
                self._publish_once()
                with self._lock:
                    pcd = self._latest
            return pcd, "pointcloud/pcd"
        self._publish_once()
        with self._lock:
            return self._latest, "pointcloud/pcd"

    async def get_properties(self, *, timeout=None, **kwargs) -> Camera.Properties:
        return Camera.Properties(supports_pcd=True, mime_types=["pointcloud/pcd"])

    async def do_command(
        self, command: Mapping[str, object], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, object]:
        return {
            "source": self._source_name,
            "shm_name": self._shm.name if self._shm is not None else None,
            "produce_hz": self._produce_hz,
            "writes": self._writes,
            "errors": self._errors,
            "last_bytes": self._last_bytes,
            "last_error": self._last_error,
            "latest_bytes": len(self._latest),
        }

    def close_sync(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        self._source = None

    async def close(self):
        self.close_sync()


Registry.register_resource_creator(
    Camera.API,
    ShmPointCloud.MODEL,
    ResourceCreatorRegistration(ShmPointCloud.new, ShmPointCloud.validate_config),
)
