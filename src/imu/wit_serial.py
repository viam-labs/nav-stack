"""Serial session for a WitMotion IMU."""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from .wit_protocol import BAUDRATES, WitError, WitSample, WitStreamParser, probe_is_wit

try:
    import serial as _pyserial
except ImportError:  # pragma: no cover
    _pyserial = None

LOGGER = logging.getLogger(__name__)


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
        self._hub_soft = False

    @property
    def sample(self) -> WitSample:
        return self._parser.sample

    def open(self) -> None:
        if _pyserial is None:
            raise WitError("pyserial is not installed")
        from ..lidar.serial_ports import (
            port_chip_family,
            shares_usb_hub_with_can,
            usb_serial_open_lock,
        )

        # Never open the lidar's CP210 — that resets the shared hub / kills scan.
        if port_chip_family(self.port) == "cp210":
            raise WitError(
                f"refusing {self.port!r}: CP210 is reserved for RPLIDAR on this "
                "stack (Wit is CH340). Pin serial_path to the 1a86 by-id device."
            )

        with usb_serial_open_lock():
            self._open_unlocked()

    def _open_unlocked(self) -> None:
        from ..lidar.serial_ports import shares_usb_hub_with_can

        self._hub_soft = bool(shares_usb_hub_with_can(self.port))
        if self._hub_soft:
            LOGGER.warning(
                "WitMotion %s shares a USB hub with CAN — hub-safe open. "
                "Not probing CP210 (lidar) ports.",
                self.port,
            )
        # On a shared hub, only try the primary baud to avoid retune churn.
        if self.baudrate:
            bauds: tuple = (int(self.baudrate),)
        elif self._hub_soft:
            bauds = (115200,)
        else:
            bauds = BAUDRATES
        last = None
        ser = None
        try:
            ser = self._connect(int(bauds[0]))
            probe_s = 0.35 if self._hub_soft else 0.7
            for baud in bauds:
                try:
                    if int(getattr(ser, "baudrate", 0) or 0) != int(baud):
                        ser.baudrate = int(baud)
                    if not probe_is_wit(ser, listen_s=probe_s, min_packets=3):
                        raise WitError(f"no WitMotion frames at baud={baud}")
                    self._ser = ser
                    self.baudrate = int(baud)
                    return
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    continue
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
            "Pin /dev/serial/by-id/...1a86... (CH340), not the Silicon Labs lidar port."
        )

    @classmethod
    def open_first_working(
        cls,
        ports: List[str],
        *,
        baudrate: Optional[int] = None,
        timeout_s: float = 0.2,
        exclude_ports: Optional[List[str]] = None,
        rounds: int = 3,
        retry_sleep_s: float = 0.5,
        prefer_cp210: bool = False,
        chip: Optional[str] = None,
        include_tty_acm: bool = False,
        exclude: Optional[List[str]] = None,
    ) -> "WitSerial":
        """Try each port until WitMotion frames are seen (skips silent lidars)."""
        from ..lidar.serial_ports import (
            claim_serial_port,
            drop_claimed_by_other,
            is_port_busy_error,
            is_port_missing_error,
            list_candidate_serial_ports,
            port_chip_family,
            usb_serial_open_lock,
        )

        skip = set(exclude_ports or [])
        errors: dict = {}
        # Default: CH340 only — never probe CP210 (lidar) during IMU detect.
        chip = chip or "ch340"
        candidates = list(ports)
        list_kwargs = dict(
            prefer_cp210=prefer_cp210,
            chip=chip,
            include_tty_acm=include_tty_acm,
            exclude=exclude,
        )
        for round_i in range(max(1, rounds)):
            if round_i > 0:
                refreshed = list_candidate_serial_ports(**list_kwargs)
                if refreshed:
                    candidates = refreshed
            # Never open a port the lidar already owns (hub reset / steal).
            candidates = drop_claimed_by_other("imu", candidates)
            candidates = [
                p
                for p in candidates
                if p not in skip and port_chip_family(p) != "cp210"
            ]
            busy_seen = False
            missing_seen = False
            for port in candidates:
                dev = cls(port, baudrate=baudrate, timeout_s=timeout_s)
                try:
                    with usb_serial_open_lock():
                        dev._open_unlocked()
                    claim_serial_port("imu", port)
                    return dev
                except Exception as exc:  # noqa: BLE001
                    errors[port] = repr(exc)
                    if is_port_busy_error(exc):
                        busy_seen = True
                    if is_port_missing_error(exc):
                        missing_seen = True
                    try:
                        with usb_serial_open_lock():
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
            f"Tried: {', '.join(candidates)}. Errors: {detail}. "
            "IMU autodetect only probes CH340 (1a86), never CP210 (lidar)."
        )

    def _connect(self, baud: int):
        from ..lidar.serial_ports import is_safe_sensor_serial_port, port_chip_family

        if port_chip_family(self.port) == "cp210":
            raise WitError(f"refusing CP210 port {self.port!r} (lidar)")
        if not is_safe_sensor_serial_port(self.port):
            raise WitError(
                f"refusing to open {self.port!r}: not a known IMU UART bridge "
                "(or USB id is a CAN adapter such as OpenMoko 1d50:606f)"
            )
        kwargs = dict(
            baudrate=baud,
            parity=_pyserial.PARITY_NONE,
            stopbits=_pyserial.STOPBITS_ONE,
            timeout=self.timeout_s,
            dsrdtr=False,
            rtscts=False,
            xonxoff=False,
        )
        if self._hub_soft:
            return _pyserial.Serial(self.port, **kwargs)
        try:
            return _pyserial.Serial(self.port, exclusive=True, **kwargs)
        except TypeError:
            return _pyserial.Serial(self.port, **kwargs)

    def write_command(self, payload: bytes, *, settle_s: float = 0.15) -> None:
        """Write one ``FF AA ..`` register command and pause for the device."""
        ser = self._ser
        if ser is None:
            raise WitError("write_command on closed port")
        ser.write(payload)
        try:
            ser.flush()
        except Exception:  # noqa: BLE001
            pass
        if settle_s > 0:
            time.sleep(settle_s)

    def configure(self, algorithm: str = "keep", *, zero_yaw: bool = False) -> int:
        """Send the startup config sequence; returns number of commands sent."""
        from .wit_protocol import config_commands

        cmds = config_commands(algorithm, zero_yaw=zero_yaw)
        for cmd in cmds:
            self.write_command(cmd)
        return len(cmds)

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
        from ..lidar.serial_ports import release_serial_port, usb_serial_open_lock

        with usb_serial_open_lock():
            ser = self._ser
            self._ser = None
            if ser is not None and self._owns_port:
                try:
                    ser.close()
                except Exception:
                    pass
        release_serial_port(self.port)
