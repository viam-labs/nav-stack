"""Serial session for a WitMotion IMU."""
from __future__ import annotations

import time
from typing import List, Optional

from .wit_protocol import BAUDRATES, WitError, WitSample, WitStreamParser, probe_is_wit

try:
    import serial as _pyserial
except ImportError:  # pragma: no cover
    _pyserial = None


class WitSerial:
    """Reads continuous WitMotion UART frames into a ``WitSample``."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: Optional[int] = None,
        timeout_s: float = 0.2,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._ser = None
        self._parser = WitStreamParser()
        self._owns_port = True

    @property
    def sample(self) -> WitSample:
        return self._parser.sample

    def open(self) -> None:
        if _pyserial is None:
            raise WitError("pyserial is not installed")
        bauds = (self.baudrate,) if self.baudrate else BAUDRATES
        last = None
        for baud in bauds:
            ser = None
            try:
                ser = self._connect(int(baud))
                if not probe_is_wit(ser, listen_s=0.7, min_packets=3):
                    raise WitError(f"no WitMotion frames at baud={baud}")
                self._ser = ser
                self.baudrate = int(baud)
                # Keep buffered packets from the probe.
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
                self._ser = None
        raise WitError(
            f"failed to open WitMotion IMU on {self.port!r} at {bauds}: {last!r}. "
            "If the lidar and IMU USB ports swapped, set serial_autodetect=true "
            "or use a unique /dev/serial/by-path/... path."
        )

    @classmethod
    def open_first_working(
        cls,
        ports: List[str],
        *,
        baudrate: Optional[int] = None,
        timeout_s: float = 0.2,
        exclude_ports: Optional[List[str]] = None,
        rounds: int = 8,
        retry_sleep_s: float = 0.5,
    ) -> "WitSerial":
        """Try each port until WitMotion frames are seen (skips silent lidars)."""
        from ..lidar.serial_ports import (
            claim_serial_port,
            is_port_busy_error,
            is_port_missing_error,
            list_candidate_serial_ports,
            sort_unclaimed_first,
        )

        skip = set(exclude_ports or [])
        errors: dict = {}
        candidates = list(ports)
        for round_i in range(max(1, rounds)):
            if round_i > 0:
                refreshed = list_candidate_serial_ports(prefer_cp210=False)
                if refreshed:
                    candidates = refreshed
            candidates = sort_unclaimed_first("imu", candidates)
            busy_seen = False
            missing_seen = False
            for port in candidates:
                if port in skip:
                    continue
                dev = cls(port, baudrate=baudrate, timeout_s=timeout_s)
                try:
                    dev.open()
                    claim_serial_port("imu", port)
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
        raise WitError(
            "no WitMotion IMU responded on any candidate serial port. "
            f"Tried: {', '.join(candidates)}. Errors: {detail}"
        )

    def _connect(self, baud: int):
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

    def poll(self) -> int:
        """Read available bytes; return packets parsed this call."""
        ser = self._ser
        if ser is None:
            return 0
        waiting = getattr(ser, "in_waiting", 0) or 0
        chunk = ser.read(max(waiting, 1) if waiting else 1)
        if not chunk and waiting == 0:
            # Brief block to avoid busy-spin when the OS has no buffered data.
            chunk = ser.read(64)
        return self._parser.feed(chunk) if chunk else 0

    def close(self) -> None:
        from ..lidar.serial_ports import release_serial_port

        ser = self._ser
        self._ser = None
        if ser is not None and self._owns_port:
            try:
                ser.close()
            except Exception:
                pass
        release_serial_port(self.port)