"""In-process WitMotion IMU with USB autodetect + POSIX shm publish.

``viam-labs:nav-stack:wit-imu`` talks the WitMotion UART protocol (same as
``viam:wit-motion:imu-wit``) and optionally publishes samples to ``imushm`` so
builtin SLAM can read odom without a gRPC hop — mirroring ``rplidar`` + pcshm.

Autodetect listens for streaming ``0x55`` frames; silent RPLIDARs are skipped,
which avoids the CP2102 ``by-id`` collision when lidar and IMU share serial
``0001``.
"""
from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self

from viam.components.movement_sensor import MovementSensor
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.spatialmath import EulerAngles
from viam.utils import struct_to_dict

from ..imu.wit_protocol import WitError
from ..imu.wit_serial import WitSerial
from ..lidar.serial_ports import list_candidate_serial_ports
from ..ros import imushm

LOGGER = getLogger(__name__)


def _shm_name_for(component_name: str, explicit: Optional[str]) -> str:
    from ..ros import pcshm

    if explicit:
        return pcshm.normalize_name(explicit)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "", component_name) or "imu"
    return pcshm.normalize_name(f"viam-imu-{slug}")


class WitImu(MovementSensor):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "wit-imu")

    def __init__(self, name: str):
        super().__init__(name)
        self._device: Optional[WitSerial] = None
        self._shm: Optional[imushm.Writer] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._serial_path: Optional[str] = None
        self._serial_autodetect = False
        self._baudrate: Optional[int] = None
        self._shm_name: Optional[str] = None
        self._publish_hz = 50.0
        self._packets = 0
        self._errors = 0
        self._last_error: Optional[str] = None
        self._has_mag = False
        self._ax = self._ay = self._az = 0.0
        self._gx = self._gy = self._gz = 0.0
        self._roll = self._pitch = self._yaw = 0.0
        self._mx = self._my = self._mz = 0.0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        imu = cls(config.name)
        imu.reconfigure(config, dependencies)
        return imu

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        serial_path = str(attrs.get("serial_path") or "").strip()
        autodetect = bool(attrs.get("serial_autodetect", False))
        if not serial_path and not autodetect:
            raise ValueError(
                "wit-imu requires attributes.serial_path or serial_autodetect=true"
            )
        baud = attrs.get("serial_baud_rate", attrs.get("baud_rate", attrs.get("baudrate")))
        if baud is not None and int(baud) not in (0, 9600, 115200):
            raise ValueError("wit-imu serial_baud_rate must be 9600 or 115200")
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        _ = dependencies
        self.close_sync()
        attrs = struct_to_dict(config.attributes)
        serial_path = str(attrs.get("serial_path") or "").strip()
        self._serial_autodetect = bool(attrs.get("serial_autodetect", False))
        baud = attrs.get("serial_baud_rate", attrs.get("baud_rate", attrs.get("baudrate")))
        self._baudrate = int(baud) if baud else None
        if self._baudrate == 0:
            self._baudrate = None
        explicit = str(attrs.get("shm_name") or "").strip() or None
        self._shm_name = _shm_name_for(self.name, explicit)
        self._publish_hz = float(attrs.get("publish_hz", 50.0))
        self._serial_path = serial_path or None
        self._stop.clear()
        self._open_device()
        region = int(attrs.get("shm_region_size", imushm.DEFAULT_REGION_SIZE))
        self._shm = imushm.Writer(self._shm_name, region_size=region)
        self._thread = threading.Thread(
            target=self._read_loop, name=f"{self.name}-wit-imu", daemon=True
        )
        self._thread.start()
        LOGGER.info(
            "nav-stack wit-imu %r serial=%s baud=%s shm=%s",
            self.name,
            self._serial_path,
            self._device.baudrate if self._device else None,
            self._shm_name,
        )

    def _open_device(self) -> None:
        if self._serial_path:
            dev = WitSerial(self._serial_path, baudrate=self._baudrate)
            dev.open()
        elif self._serial_autodetect:
            # Protocol detect only — chip brands vary (this robot: Wit on CH340,
            # RPLIDAR on CP210). Hard chip filters caused false mismatches.
            ports = list_candidate_serial_ports(prefer_cp210=False)
            dev = WitSerial.open_first_working(ports, baudrate=self._baudrate)
            self._serial_path = dev.port
            LOGGER.info("nav-stack wit-imu autodetected serial=%s", self._serial_path)
        else:
            raise ValueError("wit-imu requires serial_path or serial_autodetect")
        self._device = dev

    def _read_loop(self) -> None:
        period = 1.0 / max(self._publish_hz, 1.0)
        while not self._stop.is_set():
            try:
                device = self._device
                if device is None:
                    time.sleep(0.1)
                    continue
                device.poll()
                s = device.sample
                with self._lock:
                    self._ax, self._ay, self._az = s.ax, s.ay, s.az
                    self._gx, self._gy, self._gz = s.gx, s.gy, s.gz
                    self._roll, self._pitch, self._yaw = s.roll, s.pitch, s.yaw
                    self._mx, self._my, self._mz = s.mx, s.my, s.mz
                    self._has_mag = s.has_mag
                    self._packets = s.packets
                if self._shm is not None:
                    self._shm.write_sample(
                        imushm.ImuShmSample(
                            ax=s.ax,
                            ay=s.ay,
                            az=s.az,
                            gx=s.gx,
                            gy=s.gy,
                            gz=s.gz,
                            roll=s.roll,
                            pitch=s.pitch,
                            yaw=s.yaw,
                            mx=s.mx,
                            my=s.my,
                            mz=s.mz,
                            has_mag=s.has_mag,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                self._errors += 1
                self._last_error = repr(exc)
                LOGGER.warning("wit-imu %r read error: %s", self.name, exc)
                time.sleep(0.2)
                continue
            self._stop.wait(period)

    def close_sync(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None

    async def close(self) -> None:
        await asyncio.to_thread(self.close_sync)

    async def get_angular_velocity(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Vector3:
        del extra, timeout, kwargs
        with self._lock:
            return Vector3(x=self._gx, y=self._gy, z=self._gz)

    async def get_linear_acceleration(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Vector3:
        del extra, timeout, kwargs
        with self._lock:
            return Vector3(x=self._ax, y=self._ay, z=self._az)

    async def get_orientation(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ):
        del extra, timeout, kwargs
        with self._lock:
            ea = EulerAngles(roll=self._roll, pitch=self._pitch, yaw=self._yaw)
        return ea.to_quaternion().to_orientation_vector().to_proto()

    async def get_compass_heading(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> float:
        del extra, timeout, kwargs
        with self._lock:
            if not self._has_mag:
                raise NotImplementedError("compass heading requires magnetometer packets")
            # π/2 - atan2(y, x) identity used by wit-motion (North = 0).
            rad = math.atan2(self._my, self._mx)
            compass = math.degrees(rad) % 360.0
            return compass

    async def get_linear_velocity(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Vector3:
        del extra, timeout, kwargs
        raise NotImplementedError("wit-imu does not provide linear velocity")

    async def get_position(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[Any, float]:
        del extra, timeout, kwargs
        raise NotImplementedError("wit-imu does not provide position")

    async def get_properties(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> MovementSensor.Properties:
        del extra, timeout, kwargs
        with self._lock:
            has_mag = self._has_mag
        return MovementSensor.Properties(
            linear_velocity_supported=False,
            angular_velocity_supported=True,
            orientation_supported=True,
            position_supported=False,
            compass_heading_supported=has_mag,
            linear_acceleration_supported=True,
        )

    async def get_accuracy(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> MovementSensor.Accuracy:
        del extra, timeout, kwargs
        # Datasheet compass accuracy for supported WitMotion models (~0.5 deg)
        # when tilt is moderate; otherwise leave NaN.
        with self._lock:
            roll, pitch = self._roll, self._pitch
        max_tilt = math.radians(45.0)
        if abs(roll) <= 1.0 and abs(pitch) <= max_tilt:
            return MovementSensor.Accuracy(compass_degrees_error=0.5)
        return MovementSensor.Accuracy(compass_degrees_error=float("nan"))

    async def get_readings(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, Any]:
        del extra, timeout, kwargs
        with self._lock:
            out: Dict[str, Any] = {
                "angular_velocity": {"x": self._gx, "y": self._gy, "z": self._gz},
                "linear_acceleration": {"x": self._ax, "y": self._ay, "z": self._az},
                "orientation": {
                    "roll": self._roll,
                    "pitch": self._pitch,
                    "yaw": self._yaw,
                },
                "serial_path": self._serial_path,
                "shm_name": self._shm_name,
                "packets": self._packets,
            }
            if self._has_mag:
                out["magnetometer"] = {"x": self._mx, "y": self._my, "z": self._mz}
            if self._last_error:
                out["last_error"] = self._last_error
            return out

    async def do_command(
        self, command: Mapping[str, Any], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, Any]:
        del timeout, kwargs
        cmd = str(command.get("command") or command.get("cmd") or "")
        if cmd in ("status", "get_status"):
            with self._lock:
                return {
                    "serial_path": self._serial_path,
                    "baudrate": self._device.baudrate if self._device else None,
                    "shm_name": self._shm_name,
                    "packets": self._packets,
                    "errors": self._errors,
                    "has_mag": self._has_mag,
                    "last_error": self._last_error,
                    "yaw_rad": self._yaw,
                }
        raise ValueError(f"unknown wit-imu command {cmd!r}")


Registry.register_resource_creator(
    MovementSensor.API,
    WitImu.MODEL,
    ResourceCreatorRegistration(WitImu.new, WitImu.validate_config),
)
