"""Serial session for a Slamtec RPLIDAR (SCAN mode)."""

from __future__ import annotations

import time
from typing import Iterator, List, Optional, Tuple

from . import rplidar_protocol as proto

try:
    import serial as _pyserial
except ImportError:  # pragma: no cover - optional until the camera is used
    _pyserial = None


Measurement = Tuple[int, float, float]  # quality, angle_deg, distance_mm


class RPLidarSerial:
    """Talks the Slamtec UART protocol; ``serial_port`` may be injected in tests."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: Optional[int] = None,
        timeout_s: float = 1.0,
        serial_port=None,
        motor_warmup_s: float = 1.0,
        reset_settle_s: float = 0.5,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.motor_warmup_s = motor_warmup_s
        self.reset_settle_s = reset_settle_s
        self._ser = serial_port
        self._owns_port = serial_port is None
        self.info: dict = {}

    def open(self) -> None:
        if self._ser is not None:
            self._handshake()
            return
        if _pyserial is None:
            raise proto.RPLidarError("pyserial is not installed")
        bauds = (self.baudrate,) if self.baudrate else proto.BAUDRATES
        last = None
        for baud in bauds:
            ser = None
            try:
                ser = _pyserial.Serial(
                    self.port,
                    baudrate=int(baud),
                    parity=_pyserial.PARITY_NONE,
                    stopbits=_pyserial.STOPBITS_ONE,
                    timeout=self.timeout_s,
                    dsrdtr=True,
                )
                self._ser = ser
                self.baudrate = int(baud)
                self._handshake()
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
                self._ser = None
        raise proto.RPLidarError(
            f"failed to open RPLIDAR on {self.port!r}: {last!r}"
        )

    def close(self) -> None:
        ser = self._ser
        if ser is None:
            return
        try:
            self.stop()
            self.stop_motor()
        except Exception:
            pass
        if self._owns_port:
            try:
                ser.close()
            except Exception:
                pass
        self._ser = None

    def _write(self, data: bytes) -> None:
        self._ser.write(data)

    def _read_exact(self, n: int) -> bytes:
        buf = self._ser.read(n)
        if len(buf) != n:
            raise proto.RPLidarError(f"short read {len(buf)}/{n}")
        return buf

    def _read_descriptor(self) -> tuple[int, bool, int]:
        """Read one response descriptor, skipping leading startup junk.

        Some units emit a few stray bytes right after reset/open (or when the
        wrong port briefly echoes noise). Be tolerant of bytes before the
        ``0xA5 0x5A`` descriptor sync instead of failing on the first 7-byte
        window.
        """
        first = self._read_exact(1)
        limit = max(64, int(self.timeout_s * 256))
        for _ in range(limit):
            if first and first[0] == proto.SYNC:
                second = self._read_exact(1)
                if second and second[0] == proto.SYNC2:
                    rest = self._read_exact(5)
                    return proto.parse_descriptor(first + second + rest)
                first = second
                continue
            first = self._read_exact(1)
        raise proto.RPLidarError("bad descriptor sync")

    def _handshake(self) -> None:
        self.stop()
        time.sleep(0.02)
        self._write(proto.command(proto.CMD_RESET))
        if self.reset_settle_s > 0:
            time.sleep(self.reset_settle_s)
        self._clear()
        self.info = self.get_info()
        status, code = self.get_health()
        if status == 2:
            raise proto.RPLidarError(f"RPLIDAR health error code={code}")

    def _clear(self) -> None:
        ser = self._ser
        read_all = getattr(ser, "read_all", None)
        if callable(read_all):
            read_all()
            return
        waiting = getattr(ser, "in_waiting", 0) or 0
        if waiting:
            ser.read(waiting)

    def get_info(self) -> dict:
        self._write(proto.command(proto.CMD_GET_INFO))
        size, single, dtype = self._read_descriptor()
        if size != proto.INFO_LEN or not single or dtype != proto.INFO_TYPE:
            raise proto.RPLidarError(
                f"unexpected info descriptor size={size} single={single} type={dtype}"
            )
        return proto.decode_info(self._read_exact(size))

    def get_health(self) -> Tuple[int, int]:
        self._write(proto.command(proto.CMD_GET_HEALTH))
        size, single, dtype = self._read_descriptor()
        if size != proto.HEALTH_LEN or not single or dtype != proto.HEALTH_TYPE:
            raise proto.RPLidarError("unexpected health descriptor")
        return proto.decode_health(self._read_exact(size))

    def start_motor(self) -> None:
        model = int(self.info.get("model") or 0)
        if model == proto.MODEL_S1:
            return
        if hasattr(self._ser, "dtr"):
            self._ser.dtr = False

    def stop_motor(self) -> None:
        model = int(self.info.get("model") or 0)
        if model == proto.MODEL_S1:
            return
        if hasattr(self._ser, "dtr"):
            self._ser.dtr = True

    def stop(self) -> None:
        self._write(proto.command(proto.CMD_STOP))
        time.sleep(0.01)
        self._clear()

    def start_scan(self) -> None:
        self.start_motor()
        if self.motor_warmup_s > 0:
            time.sleep(self.motor_warmup_s)
        self._write(proto.command(proto.CMD_SCAN))
        size, single, dtype = self._read_descriptor()
        if size != proto.NODE_LEN or single or dtype != proto.SCAN_TYPE:
            raise proto.RPLidarError(
                f"unexpected scan descriptor size={size} single={single} type={dtype}"
            )

    def iter_scans(
        self,
        *,
        min_points: int = 20,
        max_buffer_nodes: int = 2000,
    ) -> Iterator[List[Measurement]]:
        self.start_scan()
        scan: List[Measurement] = []
        while True:
            waiting = getattr(self._ser, "in_waiting", 0) or 0
            if waiting > max_buffer_nodes * proto.NODE_LEN:
                drop = waiting - (max_buffer_nodes * proto.NODE_LEN)
                drop -= drop % proto.NODE_LEN
                if drop:
                    self._ser.read(drop)
            raw = self._read_exact(proto.NODE_LEN)
            try:
                new_scan, quality, angle, dist = proto.decode_node(raw)
            except proto.RPLidarError:
                continue
            if new_scan:
                if len(scan) >= min_points:
                    yield scan
                scan = []
            if quality > 0 and dist > 0:
                scan.append((quality, angle, dist))
