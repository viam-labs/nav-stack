"""In-process RPLIDAR camera that publishes PCD to POSIX shm.

``viam-labs:nav-stack:rplidar`` talks the Slamtec UART protocol directly (no
``viam:lidar:rplidar`` gRPC hop) and writes each revolution into the same shm
layout as ``viam-shared-memory-test``.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np
from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from ..lidar.rplidar_protocol import RPLidarError, scan_to_xyz_m
from ..lidar.rplidar_serial import RPLidarSerial
from ..lidar.serial_ports import list_candidate_serial_ports
from ..ros import conversions as conv
from ..ros import pcshm

LOGGER = getLogger(__name__)


def _shm_name_for(component_name: str, explicit: Optional[str]) -> str:
    if explicit:
        return pcshm.normalize_name(explicit)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "", component_name) or "lidar"
    return pcshm.normalize_name(f"viam-pc-{slug}")


class RPLidarShm(Camera):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "rplidar")

    def __init__(self, name: str):
        super().__init__(name)
        self._device: Optional[RPLidarSerial] = None
        self._shm: Optional[pcshm.Writer] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._stall_abort = threading.Event()
        self._scan_epoch = 0
        self._scan_loop_progress_wall: Optional[float] = None
        self._last_thread_restart_wall = 0.0
        self._thread_restarts = 0
        self._latest = b""
        self._scans = 0
        self._errors = 0
        self._reconnects = 0
        self._last_error: Optional[str] = None
        self._last_points = 0
        self._last_publish_wall: Optional[float] = None
        self._shm_name: Optional[str] = None
        self._serial_path: Optional[str] = None
        self._min_range_mm = 0.0
        self._warmup_scans = 5
        self._info: dict = {}
        self._serial_autodetect = False
        self._baudrate: Optional[int] = None
        self._timeout_s = 2.0
        self._motor_warmup_s = 1.0
        self._reset_settle_s = 0.5
        self._reconnect_backoff_s = 1.0
        self._max_reconnect_backoff_s = 15.0
        self._max_publish_gap_s = 5.0
        self._stall_thread: Optional[threading.Thread] = None

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
        serial_path = str(attrs.get("serial_path") or "").strip()
        autodetect = bool(attrs.get("serial_autodetect", False))
        if not serial_path and not autodetect:
            raise ValueError(
                "rplidar requires attributes.serial_path or serial_autodetect=true"
            )
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        _ = dependencies
        self.close_sync()
        attrs = struct_to_dict(config.attributes)
        serial_path = str(attrs.get("serial_path") or "").strip()
        self._serial_autodetect = bool(attrs.get("serial_autodetect", False))
        baud = attrs.get("baud_rate") or attrs.get("baudrate")
        self._baudrate = int(baud) if baud else None
        self._timeout_s = float(attrs.get("timeout_s", 2.0))
        self._min_range_mm = float(attrs.get("min_range_mm", 0.0))
        self._warmup_scans = int(attrs.get("warmup_scans", 5))
        region = int(attrs.get("shm_region_size", pcshm.DEFAULT_REGION_SIZE))
        explicit = str(attrs.get("shm_name") or "").strip() or None
        self._shm_name = _shm_name_for(self.name, explicit)
        self._motor_warmup_s = float(attrs.get("motor_warmup_s", 1.0))
        self._reset_settle_s = float(attrs.get("reset_settle_s", 0.5))
        self._reconnect_backoff_s = float(attrs.get("reconnect_backoff_s", 1.0))
        self._max_reconnect_backoff_s = float(attrs.get("max_reconnect_backoff_s", 15.0))
        self._max_publish_gap_s = float(attrs.get("max_publish_gap_s", 5.0))
        self._serial_path = serial_path or None
        self._stop = threading.Event()
        self._stall_abort.clear()
        self._scan_epoch += 1
        self._scan_loop_progress_wall = None
        self._last_thread_restart_wall = 0.0
        self._thread_restarts = 0
        self._scans = 0
        self._errors = 0
        self._reconnects = 0
        self._last_error = None
        self._open_device()
        self._shm = pcshm.open_writer(self._shm_name, region)
        self._start_scan_thread()
        if self._max_publish_gap_s > 0:
            self._stall_thread = threading.Thread(
                target=self._stall_watchdog,
                name=f"{self.name}-rplidar-stall",
                daemon=True,
            )
            self._stall_thread.start()
        LOGGER.info(
            "nav-stack rplidar %r serial=%s baud=%s model=%s shm=%s",
            self.name,
            self._serial_path,
            self._device.baudrate if self._device else None,
            self._info.get("model"),
            self._shm_name,
        )

    def _start_scan_thread(self) -> None:
        self._thread = threading.Thread(
            target=self._scan_loop, name=f"{self.name}-rplidar", daemon=True
        )
        self._thread.start()

    def _restart_scan_thread(self, reason: str) -> None:
        if self._stop.is_set():
            return
        now = time.monotonic()
        if now - self._last_thread_restart_wall < self._max_publish_gap_s:
            return
        self._last_thread_restart_wall = now
        self._thread_restarts += 1
        self._last_error = reason
        LOGGER.error("rplidar %r restarting scan thread (#%d): %s", self.name, self._thread_restarts, reason)
        self._scan_epoch += 1
        self._stall_abort.set()
        self._close_device()
        old = self._thread
        if old is not None and old.is_alive() and old is not threading.current_thread():
            old.join(timeout=self._timeout_s + self._motor_warmup_s + 3.0)
        self._stall_abort.clear()
        self._start_scan_thread()

    def _open_device(self) -> None:
        with self._device_lock:
            self._open_device_unlocked()

    def _open_device_unlocked(self) -> None:
        if self._serial_path:
            dev = RPLidarSerial(
                self._serial_path,
                baudrate=self._baudrate,
                timeout_s=self._timeout_s,
                motor_warmup_s=self._motor_warmup_s,
                reset_settle_s=self._reset_settle_s,
            )
            dev.open()
        elif self._serial_autodetect:
            ports = list_candidate_serial_ports()
            dev = RPLidarSerial.open_first_working(
                ports,
                baudrate=self._baudrate,
                timeout_s=self._timeout_s,
                motor_warmup_s=self._motor_warmup_s,
                reset_settle_s=self._reset_settle_s,
            )
            self._serial_path = dev.port
            LOGGER.info("nav-stack rplidar autodetected serial=%s", self._serial_path)
        else:
            raise ValueError("rplidar requires serial_path or serial_autodetect")
        self._device = dev
        self._info = dict(dev.info)

    def _close_device(self) -> None:
        with self._device_lock:
            device = self._device
            if device is None:
                return
            try:
                device.stop()
            except Exception:
                pass
            try:
                device.close()
            except Exception:
                pass
            self._device = None

    def _publish_scan(self, measurements) -> None:
        xyz = scan_to_xyz_m(measurements, min_range_mm=self._min_range_mm)
        pcd = conv.points_to_pcd(
            np.asarray(xyz, dtype=float) if xyz else np.empty((0, 3))
        )
        if self._shm is not None:
            self._shm.write(pcd)
        with self._lock:
            self._latest = pcd
        self._scans += 1
        self._last_points = len(xyz)
        self._last_error = None
        self._last_publish_wall = time.monotonic()

    def _scan_loop(self) -> None:
        epoch = self._scan_epoch
        backoff = self._reconnect_backoff_s
        try:
            while not self._stop.is_set() and self._scan_epoch == epoch:
                self._scan_loop_progress_wall = time.monotonic()
                device = self._device
                if device is None:
                    if not self._try_reconnect(backoff):
                        time.sleep(backoff)
                        backoff = min(backoff * 2.0, self._max_reconnect_backoff_s)
                        continue
                    backoff = self._reconnect_backoff_s
                    device = self._device
                discarded = 0
                self._stall_abort.clear()
                try:
                    for measurements in device.iter_scans(
                        max_stall_s=self._max_publish_gap_s,
                        abort_check=self._stall_abort.is_set,
                    ):
                        self._scan_loop_progress_wall = time.monotonic()
                        if self._stop.is_set() or self._scan_epoch != epoch:
                            return
                        if discarded < self._warmup_scans:
                            discarded += 1
                            continue
                        try:
                            self._publish_scan(measurements)
                        except Exception as exc:  # noqa: BLE001
                            self._errors += 1
                            self._last_error = repr(exc)
                            LOGGER.warning("rplidar %r publish failed: %s", self.name, exc)
                except Exception as exc:  # noqa: BLE001
                    self._errors += 1
                    self._last_error = repr(exc)
                    if self._stop.is_set() or self._scan_epoch != epoch:
                        return
                    LOGGER.error("rplidar %r scan loop exited: %s", self.name, exc)
                    self._close_device()
                    self._stall_abort.clear()
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, self._max_reconnect_backoff_s)
        finally:
            if not self._stop.is_set() and self._scan_epoch == epoch:
                LOGGER.error("rplidar %r scan loop stopped unexpectedly", self.name)

    def _try_reconnect(self, backoff_s: float) -> bool:
        if self._stop.is_set():
            return False
        try:
            self._open_device()
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            self._last_error = repr(exc)
            LOGGER.warning(
                "rplidar %r reconnect failed (backoff %.1fs): %s",
                self.name,
                backoff_s,
                exc,
            )
            return False
        self._reconnects += 1
        self._last_error = None
        LOGGER.info(
            "rplidar %r reconnected on %s (attempt #%d)",
            self.name,
            self._serial_path,
            self._reconnects,
        )
        return True

    def _stall_watchdog(self) -> None:
        """Abort hung scans and restart the scan thread if publishes stay stale."""
        restart_after = max(self._max_publish_gap_s * 2.0, 10.0)
        while not self._stop.wait(1.0):
            gap = self._max_publish_gap_s
            if gap <= 0:
                continue
            last = self._last_publish_wall
            if last is None:
                continue
            age = time.monotonic() - last
            if age <= gap:
                continue
            progress = self._scan_loop_progress_wall
            progress_age = (time.monotonic() - progress) if progress is not None else age
            thread = self._thread
            thread_alive = thread is not None and thread.is_alive()
            if age >= restart_after and (
                not thread_alive or progress_age >= gap
            ):
                self._restart_scan_thread(f"publish stalled {age:.1f}s")
                continue
            if self._stall_abort.is_set():
                continue
            LOGGER.warning(
                "rplidar %r publish stalled %.1fs (> %.1fs); aborting scan loop",
                self.name,
                age,
                gap,
            )
            self._last_error = f"publish stalled {age:.1f}s"
            self._stall_abort.set()

    async def get_images(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("rplidar does not support get_images; use get_point_cloud")

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        deadline = time.monotonic() + (float(timeout) if timeout else 2.0)
        while time.monotonic() < deadline:
            with self._lock:
                pcd = self._latest
            if pcd:
                return pcd, "pointcloud/pcd"
            await asyncio.sleep(0.05)
        raise RPLidarError("rplidar has no scan yet")

    async def get_properties(self, *, timeout=None, **kwargs) -> Camera.Properties:
        return Camera.Properties(supports_pcd=True, mime_types=["pointcloud/pcd"])

    async def do_command(
        self, command: Mapping[str, object], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, object]:
        cmd = command.get("command") if isinstance(command, Mapping) else None
        if cmd == "get_laser_scan":
            raise NotImplementedError("rplidar does not support get_laser_scan")
        if cmd == "restart":
            self._restart_scan_thread("do_command restart")
            return {"status": "restarting", "thread_restarts": self._thread_restarts}
        last_pub_age_s = None
        if self._last_publish_wall is not None:
            last_pub_age_s = round(time.monotonic() - self._last_publish_wall, 3)
        progress_age_s = None
        if self._scan_loop_progress_wall is not None:
            progress_age_s = round(time.monotonic() - self._scan_loop_progress_wall, 3)
        thread = self._thread
        return {
            "serial_path": self._serial_path,
            "baudrate": self._device.baudrate if self._device else None,
            "shm_name": self._shm_name,
            "info": dict(self._info),
            "scans": self._scans,
            "errors": self._errors,
            "reconnects": self._reconnects,
            "thread_restarts": self._thread_restarts,
            "scan_thread_alive": thread is not None and thread.is_alive(),
            "scan_loop_progress_age_s": progress_age_s,
            "stall_abort_set": self._stall_abort.is_set(),
            "last_points": self._last_points,
            "last_error": self._last_error,
            "last_publish_age_s": last_pub_age_s,
            "latest_bytes": len(self._latest),
        }

    def close_sync(self) -> None:
        self._stop.set()
        self._close_device()
        for thread in (self._thread, self._stall_thread):
            if thread is not None:
                thread.join(timeout=self._timeout_s + 2.0)
        self._thread = None
        self._stall_thread = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    async def close(self):
        self.close_sync()


Registry.register_resource_creator(
    Camera.API,
    RPLidarShm.MODEL,
    ResourceCreatorRegistration(RPLidarShm.new, RPLidarShm.validate_config),
)
