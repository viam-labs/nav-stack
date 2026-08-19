"""Serial session for a Slamtec RPLIDAR (SCAN mode)."""

from __future__ import annotations

import time
from typing import Callable, Iterator, List, Optional, Tuple

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
        timeout_s: float = 2.0,
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
                ser = self._connect_serial(int(baud))
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
            f"failed to open RPLIDAR on {self.port!r} at {bauds}: {last!r}. "
            "If the lidar and IMU USB ports swapped, try the other /dev/ttyUSB* "
            "or a /dev/serial/by-id/... path."
        )

    @classmethod
    def open_first_working(
        cls,
        ports: List[str],
        *,
        baudrate: Optional[int] = None,
        timeout_s: float = 2.0,
        motor_warmup_s: float = 1.0,
        reset_settle_s: float = 0.5,
    ) -> "RPLidarSerial":
        """Try each port (and baud) until GET_INFO succeeds."""
        errors: List[str] = []
        for port in ports:
            dev = cls(
                port,
                baudrate=baudrate,
                timeout_s=timeout_s,
                motor_warmup_s=motor_warmup_s,
                reset_settle_s=reset_settle_s,
            )
            try:
                dev.open()
                return dev
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{port}: {exc!r}")
                dev.close()
        raise proto.RPLidarError(
            "no RPLIDAR responded on any candidate serial port. "
            f"Tried: {', '.join(ports)}. Errors: {'; '.join(errors)}"
        )

    def _connect_serial(self, baud: int):
        """Open the UART like viam-modules/rplidar: no DTR/DSR flow control."""
        kwargs = dict(
            baudrate=baud,
            parity=_pyserial.PARITY_NONE,
            stopbits=_pyserial.STOPBITS_ONE,
            timeout=self.timeout_s,
            dsrdtr=False,
            rtscts=False,
        )
        try:
            return _pyserial.Serial(self.port, exclusive=True, **kwargs)
        except TypeError:
            return _pyserial.Serial(self.port, **kwargs)

    def _prepare_port(self) -> None:
        ser = self._ser
        for name in ("reset_input_buffer", "reset_output_buffer"):
            fn = getattr(ser, name, None)
            if callable(fn):
                fn()
        self._clear()
        # A1 motor off (DTR high) until start_scan, matching post-connect idle state.
        if hasattr(ser, "dtr"):
            ser.dtr = True
        if hasattr(ser, "rts"):
            ser.rts = False
        time.sleep(0.05)

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
        flush = getattr(self._ser, "flush", None)
        if callable(flush):
            flush()

    def _read_exact(self, n: int, *, context: str = "") -> bytes:
        buf = self._ser.read(n)
        if len(buf) != n:
            hint = ""
            if len(buf) == 0:
                hint = (
                    " (no bytes — wrong serial_path, lidar powered off, "
                    "or another process owns the port)"
                )
            raise proto.RPLidarError(
                f"short read {len(buf)}/{n}{(' during ' + context) if context else ''}{hint}"
            )
        return buf

    def _read_descriptor(self) -> tuple[int, bool, int]:
        """Read one response descriptor, skipping leading startup junk.

        Some units emit a few stray bytes right after reset/open (or when the
        wrong port briefly echoes noise). Be tolerant of bytes before the
        ``0xA5 0x5A`` descriptor sync instead of failing on the first 7-byte
        window.
        """
        deadline = time.monotonic() + max(self.timeout_s, 1.0)
        first = self._read_exact(1, context="descriptor sync")
        while time.monotonic() < deadline:
            if first and first[0] == proto.SYNC:
                second = self._read_exact(1, context="descriptor sync")
                if second and second[0] == proto.SYNC2:
                    rest = self._read_exact(5, context="descriptor")
                    return proto.parse_descriptor(first + second + rest)
                first = second
                continue
            first = self._read_exact(1, context="descriptor sync")
        raise proto.RPLidarError("bad descriptor sync (timed out)")

    def _handshake(self) -> None:
        """Match viam-modules/rplidar: GET_INFO first, no motor/reset upfront."""
        if self._owns_port:
            self._prepare_port()
        else:
            self._clear()

        last: Optional[Exception] = None
        for attempt in (
            lambda: None,
            lambda: (self.stop(), time.sleep(0.05), self._clear()),
            lambda: (
                self._write(proto.command(proto.CMD_RESET)),
                time.sleep(self.reset_settle_s),
                self._clear(),
            ),
        ):
            try:
                attempt()
                self.info = self.get_info()
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise proto.RPLidarError(f"GET_INFO failed after stop/reset retries: {last!r}")

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
        if self._ser is None:
            return
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
        max_stall_s: float = 5.0,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> Iterator[List[Measurement]]:
        self.start_scan()
        scan: List[Measurement] = []
        last_complete = time.monotonic()
        while True:
            if abort_check is not None and abort_check():
                raise proto.RPLidarError("scan aborted")
            if max_stall_s > 0 and time.monotonic() - last_complete > max_stall_s:
                raise proto.RPLidarError(
                    f"no complete scan in {max_stall_s:.1f}s (motor or UART stalled)"
                )
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
                    last_complete = time.monotonic()
                    yield scan
                scan = []
            if quality > 0 and dist > 0:
                scan.append((quality, angle, dist))
