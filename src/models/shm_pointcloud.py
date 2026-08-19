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
import time
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
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
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
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                f"shm-pointcloud {self.name!r} producer thread still running after close_sync"
            )
        self._stop.clear()
        self._shm = pcshm.open_writer(shm_name, region)
        self._writes = 0
        self._errors = 0
        self._last_error = None
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

    def _join_timeout_s(self) -> float:
        period = 1.0 / max(self._produce_hz, 0.1) if self._produce_hz > 0 else 0.0
        return self._timeout_s + period + 2.0

    def _produce_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        period = 1.0 / max(self._produce_hz, 0.1)
        try:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._publish_once_async())
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc)
            while not self._stop.is_set():
                if self._stop.wait(period):
                    break
                try:
                    loop.run_until_complete(self._publish_once_async())
                except Exception as exc:  # noqa: BLE001
                    self._record_error(exc)
        finally:
            self._worker_loop = None
            loop.close()

    def _record_error(self, exc: BaseException) -> None:
        self._errors += 1
        self._last_error = repr(exc)
        LOGGER.warning("shm-pointcloud %r produce failed: %s", self.name, exc)

    async def _publish_once_async(self) -> None:
        if self._stop.is_set():
            return
        source = self._source
        shm = self._shm
        if source is None or shm is None:
            return
        data = await source.get_point_cloud(timeout=self._timeout_s)
        if self._stop.is_set():
            return
        raw = data[0] if isinstance(data, tuple) else data
        if not raw:
            raise RuntimeError("empty point cloud")
        with self._write_lock:
            if self._stop.is_set() or shm is not self._shm:
                return
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
        wait_s = float(timeout) if timeout is not None else self._timeout_s
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                pcd = self._latest
            if pcd:
                return pcd, "pointcloud/pcd"
            if self._produce_hz > 0:
                await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            await self._publish_once_async()
            with self._lock:
                pcd = self._latest
            if pcd:
                return pcd, "pointcloud/pcd"
            break
        raise RuntimeError("shm-pointcloud has no frame yet")

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
        self._source = None
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout_s())
        if thread is not None and thread.is_alive():
            LOGGER.error(
                "shm-pointcloud %r producer thread did not stop within %.1fs",
                self.name,
                self._join_timeout_s(),
            )
            return
        self._thread = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    async def close(self):
        self.close_sync()


Registry.register_resource_creator(
    Camera.API,
    ShmPointCloud.MODEL,
    ResourceCreatorRegistration(ShmPointCloud.new, ShmPointCloud.validate_config),
)
