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
                if _port_looks_like_wit(ser):
                    raise proto.RPLidarError(
                        "port streams WitMotion IMU frames (not an RPLIDAR)"
                    )
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
            "or a /dev/serial/by-path/... path. If the UART opens but GET_INFO "
            "times out, power-cycle the lidar (USB-UART can stay up while the "
            "sensor MCU is hung/unpowered)."
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
        rounds: int = 8,
        retry_sleep_s: float = 0.5,
    ) -> "RPLidarSerial":
        """Try each port until GET_INFO succeeds.

        Retries ports that are exclusively locked (IMU still probing) and
        re-lists candidates when a USB path vanishes mid-startup.
        Soft claims from the IMU are advisory only — we still try those ports
        last so a false WitMotion claim cannot permanently hide the lidar.
        """
        from .serial_ports import (
            claim_serial_port,
            is_port_busy_error,
            is_port_missing_error,
            list_candidate_serial_ports,
            sort_unclaimed_first,
            steal_serial_port,
        )

        errors: dict[str, str] = {}
        candidates = list(ports)
        for round_i in range(max(1, rounds)):
            if round_i > 0:
                refreshed = list_candidate_serial_ports(prefer_cp210=False)
                if refreshed:
                    candidates = refreshed
            candidates = sort_unclaimed_first("lidar", candidates)
            busy_seen = False
            missing_seen = False
            for port in candidates:
                dev = cls(
                    port,
                    baudrate=baudrate,
                    timeout_s=timeout_s,
                    motor_warmup_s=motor_warmup_s,
                    reset_settle_s=reset_settle_s,
                )
                try:
                    dev.open()
                    steal_serial_port("lidar", port)
                    return dev
                except Exception as exc:  # noqa: BLE001
                    errors[port] = repr(exc)
                    if is_port_busy_error(exc):
                        busy_seen = True
                    if is_port_missing_error(exc):
                        missing_seen = True
                    try:
                        dev.close()
                    except Exception:
                        pass
            if round_i + 1 < rounds and (busy_seen or missing_seen):
                time.sleep(retry_sleep_s)
                continue
            break
        detail = "; ".join(f"{p}: {e}" for p, e in errors.items()) or "(no ports)"
        raise proto.RPLidarError(
            "no RPLIDAR responded on any candidate serial port. "
            f"Tried: {', '.join(candidates)}. Errors: {detail}. "
            "If a port is exclusively locked, the IMU may still be probing — "
            "retry, or pin serial_path / depends_on so lidar starts first. "
            "Check ls /dev/ttyUSB* /dev/serial/by-path for two devices."
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
        from .serial_ports import release_serial_port

        ser = self._ser
        if ser is None:
            release_serial_port(self.port)
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
        release_serial_port(self.port)

    def _write(self, data: bytes) -> None:
        self._ser.write(data)
        flush = getattr(self._ser, "flush", None)
        if callable(flush):
            flush()

    def _read_exact(
        self,
        n: int,
        *,
        context: str = "",
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> bytes:
        buf = bytearray()
        deadline = time.monotonic() + max(self.timeout_s, 1.0) + (n * 0.05)
        while len(buf) < n:
            if abort_check is not None and abort_check():
                raise proto.RPLidarError("scan aborted")
            chunk = self._ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
                continue
            if time.monotonic() >= deadline:
                hint = ""
                if len(buf) == 0:
                    hint = (
                        " (no bytes — wrong serial_path, lidar powered off, "
                        "or another process owns the port)"
                    )
                raise proto.RPLidarError(
                    f"short read {len(buf)}/{n}{(' during ' + context) if context else ''}{hint}"
                )
        return bytes(buf)

    def _read_descriptor(
        self, *, abort_check: Optional[Callable[[], bool]] = None
    ) -> tuple[int, bool, int]:
        """Read one response descriptor, skipping leading startup junk.

        Some units emit a few stray bytes right after reset/open (or when the
        wrong port briefly echoes noise). Be tolerant of bytes before the
        ``0xA5 0x5A`` descriptor sync instead of failing on the first 7-byte
        window.
        """
        deadline = time.monotonic() + max(self.timeout_s, 1.0)
        first = self._read_exact(1, context="descriptor sync", abort_check=abort_check)
        while time.monotonic() < deadline:
            if abort_check is not None and abort_check():
                raise proto.RPLidarError("scan aborted")
            if first and first[0] == proto.SYNC:
                second = self._read_exact(1, context="descriptor sync", abort_check=abort_check)
                if second and second[0] == proto.SYNC2:
                    rest = self._read_exact(5, context="descriptor", abort_check=abort_check)
                    return proto.parse_descriptor(first + second + rest)
                first = second
                continue
            first = self._read_exact(1, context="descriptor sync", abort_check=abort_check)
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
        # Match viam-modules/rplidar: S-series do not use DTR motor control.
        # (TOF spin is started by EXPRESS_SCAN / firmware, not USB DTR.)
        if proto.is_s_series(int(self.info.get("model") or 0)):
            return
        if hasattr(self._ser, "dtr"):
            self._ser.dtr = False

    def stop_motor(self) -> None:
        if proto.is_s_series(int(self.info.get("model") or 0)):
            return
        if hasattr(self._ser, "dtr"):
            self._ser.dtr = True

    def stop(self) -> None:
        self._write(proto.command(proto.CMD_STOP))
        time.sleep(0.01)
        self._clear()

    def _get_lidar_conf(
        self, conf_type: int, reserve: bytes = b"", *, abort_check=None
    ) -> bytes:
        self._write(
            proto.command_with_payload(
                proto.CMD_GET_LIDAR_CONF,
                proto.encode_get_lidar_conf(conf_type, reserve),
            )
        )
        size, single, dtype = self._read_descriptor(abort_check=abort_check)
        if dtype != proto.GET_LIDAR_CONF_TYPE or not single:
            raise proto.RPLidarError(
                f"unexpected lidar conf descriptor type={dtype} single={single}"
            )
        raw = self._read_exact(size, abort_check=abort_check)
        if len(raw) < 4:
            raise proto.RPLidarError("lidar conf response too short")
        reply_type = int.from_bytes(raw[:4], "little")
        if reply_type != conf_type:
            raise proto.RPLidarError(
                f"lidar conf type mismatch asked={conf_type} got={reply_type}"
            )
        return raw[4:]

    def _typical_scan_mode(self, *, abort_check=None) -> Tuple[int, int]:
        """Return ``(working_mode_id, answer_type)`` for Express typical scan."""
        # Firmware < 1.24 has no GET_LIDAR_CONF; fall back to classic Express.
        fw = self.info.get("firmware") or (0, 0)
        fw_u16 = (int(fw[0]) << 8) | int(fw[1])
        if fw_u16 < ((0x1 << 8) | 24):
            return 1, proto.CAPSULED_TYPE  # EXPRESS mode id / capsulated ans
        payload = self._get_lidar_conf(
            proto.CONF_SCAN_MODE_TYPICAL, abort_check=abort_check
        )
        if len(payload) < 2:
            raise proto.RPLidarError("typical scan mode response too short")
        mode_id = int.from_bytes(payload[:2], "little")
        ans_payload = self._get_lidar_conf(
            proto.CONF_SCAN_MODE_ANS_TYPE,
            mode_id.to_bytes(2, "little"),
            abort_check=abort_check,
        )
        if not ans_payload:
            raise proto.RPLidarError("scan mode ans type response empty")
        return mode_id, int(ans_payload[0])

    def start_scan(self, *, abort_check: Optional[Callable[[], bool]] = None) -> None:
        """Legacy 5-byte SCAN (A1/A3). Prefer ``start_express_scan`` for S2/S3."""
        self.start_motor()
        if self.motor_warmup_s > 0:
            time.sleep(self.motor_warmup_s)
        if abort_check is not None and abort_check():
            raise proto.RPLidarError("scan aborted")
        self._write(proto.command(proto.CMD_SCAN))
        size, single, dtype = self._read_descriptor(abort_check=abort_check)
        if size != proto.NODE_LEN or single or dtype != proto.SCAN_TYPE:
            raise proto.RPLidarError(
                f"unexpected scan descriptor size={size} single={single} type={dtype}"
            )

    def start_express_scan(
        self, *, abort_check: Optional[Callable[[], bool]] = None
    ) -> int:
        """Start typical Express scan (viam ``StartScan(false, true)``).

        Returns the answer type (``DENSE_CAPSULED_TYPE`` for S2/S3 DenseBoost).
        """
        self.start_motor()
        if self.motor_warmup_s > 0:
            time.sleep(self.motor_warmup_s)
        if abort_check is not None and abort_check():
            raise proto.RPLidarError("scan aborted")
        mode_id, ans_type = self._typical_scan_mode(abort_check=abort_check)
        # SDK: working_mode is mode id unless STD/EXPRESS sentinel.
        working_mode = 0 if mode_id in (0, 1) else (mode_id & 0xFF)
        self._write(
            proto.command_with_payload(
                proto.CMD_EXPRESS_SCAN,
                proto.encode_express_scan_payload(working_mode),
            )
        )
        size, single, dtype = self._read_descriptor(abort_check=abort_check)
        if single or dtype != ans_type:
            raise proto.RPLidarError(
                f"unexpected express descriptor size={size} single={single} "
                f"type=0x{dtype:02X} (expected type=0x{ans_type:02X})"
            )
        if dtype == proto.DENSE_CAPSULED_TYPE and size < proto.DENSE_CAPSULE_LEN:
            raise proto.RPLidarError(
                f"dense capsule descriptor size {size} < {proto.DENSE_CAPSULE_LEN}"
            )
        return int(dtype)

    def _read_dense_capsule(
        self, *, abort_check: Optional[Callable[[], bool]] = None
    ) -> bytes:
        """Resync and read one 84-byte dense capsule with checksum check."""
        deadline = time.monotonic() + max(self.timeout_s, 2.0)
        while time.monotonic() < deadline:
            if abort_check is not None and abort_check():
                raise proto.RPLidarError("scan aborted")
            b0 = self._read_exact(1, abort_check=abort_check)[0]
            if (b0 >> 4) != proto.EXP_SYNC_1:
                continue
            b1 = self._read_exact(1, abort_check=abort_check)[0]
            if (b1 >> 4) != proto.EXP_SYNC_2:
                continue
            rest = self._read_exact(
                proto.DENSE_CAPSULE_LEN - 2, abort_check=abort_check
            )
            raw = bytes((b0, b1)) + rest
            if proto.dense_capsule_checksum_ok(raw):
                return raw
        raise proto.RPLidarError("dense capsule sync timed out")

    def iter_scans(
        self,
        *,
        min_points: int = 20,
        max_buffer_nodes: int = 2000,
        max_stall_s: float = 5.0,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> Iterator[List[Measurement]]:
        model = int(self.info.get("model") or 0)
        # S2/S3 (and other TOF): match viam-modules/rplidar StartScan(false, true)
        # → typical Express / DenseBoost. Legacy SCAN only yields sparse/stalled
        # 5-byte nodes ("short read 1/5") on these units.
        if proto.is_tof_lidar(model) or proto.is_s_series(model):
            yield from self._iter_express_scans(
                min_points=min_points,
                max_stall_s=max_stall_s,
                abort_check=abort_check,
            )
            return

        self.start_scan(abort_check=abort_check)
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
            raw = self._read_exact(proto.NODE_LEN, abort_check=abort_check)
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

    def _iter_express_scans(
        self,
        *,
        min_points: int = 20,
        max_stall_s: float = 5.0,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> Iterator[List[Measurement]]:
        ans_type = self.start_express_scan(abort_check=abort_check)
        if ans_type != proto.DENSE_CAPSULED_TYPE:
            raise proto.RPLidarError(
                f"express scan answer type 0x{ans_type:02X} not implemented "
                f"(need DenseBoost 0x{proto.DENSE_CAPSULED_TYPE:02X} for S2/S3)"
            )
        # First capsule is often incomplete; seed then decode on each new one.
        prev = self._read_dense_capsule(abort_check=abort_check)
        scan: List[Measurement] = []
        last_complete = time.monotonic()
        while True:
            if abort_check is not None and abort_check():
                raise proto.RPLidarError("scan aborted")
            if max_stall_s > 0 and time.monotonic() - last_complete > max_stall_s:
                raise proto.RPLidarError(
                    f"no complete dense scan in {max_stall_s:.1f}s"
                )
            curr = self._read_dense_capsule(abort_check=abort_check)
            try:
                samples = proto.decode_dense_capsule_pair(prev, curr)
            except proto.RPLidarError:
                prev = curr
                continue
            prev = curr
            for quality, angle, dist, new_scan in samples:
                if new_scan:
                    if len(scan) >= min_points:
                        last_complete = time.monotonic()
                        yield scan
                    scan = []
                if quality > 0 and dist > 0:
                    scan.append((quality, angle, dist))


def _port_looks_like_wit(ser, *, listen_s: float = 0.12) -> bool:
    """True if the port is already streaming WitMotion frames.

    Idle RPLIDARs are silent; WitMotion IMUs stream continuously. Skipping
    those ports avoids multi-second GET_INFO timeouts on the IMU.
    """
    try:
        from ..imu.wit_protocol import probe_is_wit
    except Exception:  # noqa: BLE001
        return False
    return bool(probe_is_wit(ser, listen_s=listen_s, min_packets=3))
