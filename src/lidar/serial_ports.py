"""Helpers for finding likely serial devices (RPLIDAR / WitMotion IMU)."""

from __future__ import annotations

import glob
import os
import re
import threading
from typing import Dict, List, Optional, Sequence


# In-process registry so lidar + IMU (same module process) don't steal each
# other's port during parallel resource startup.
_claims_lock = threading.Lock()
_claims: Dict[str, str] = {}  # realpath -> owner ("lidar" | "imu")

# by-id / path substrings that must never be opened (USB-CAN, etc.).
# Opening these with exclusive=True knocks down SocketCAN / slcand.
_EXCLUDE_SUBSTRINGS = (
    "canable",
    "cantact",
    "candle",
    "gs_usb",
    "geschmacksrichtung",  # OpenMoko CANable product string
    "slcan",
    "socketcan",
    "usb2can",
    "usb-to-can",
    "usb_to_can",
    "ixxat",
    "peak_system",
    "pcan",
    "kvaser",
    "can_usb",
    "canusb",
    "canbus",
    "can-bus",
    "lawicel",
)

# Prefer these UART bridge chips for lidar/IMU; unknown ACM gadgets are skipped.
_SENSOR_CHIP_SUBSTRINGS = (
    "silicon_labs",
    "cp210",
    "1a86",  # WCH CH340 / CH341
    "wch.cn",
    "wch_",
    "usb_serial",  # common CH340 by-id: usb-1a86_USB_Serial-...
    "ftdi",
    "ft232",
    "ft230",
    "prolific",
    "pl2303",
)


def list_candidate_serial_ports(
    *,
    prefer_cp210: bool = True,
    chip: Optional[str] = None,
    include_tty_acm: bool = False,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return serial device candidates safe to probe for lidar / IMU.

    Skips USB-CAN and other non-sensor adapters so exclusive opens cannot
    disrupt SocketCAN / slcand. Prefers ``/dev/serial/by-id`` over by-path.

    ``prefer_cp210=True``: Silicon Labs / CP210 by-id first (typical RPLIDAR).
    ``prefer_cp210=False``: non-CP210 (CH340) first (typical WitMotion).

    ``chip`` hard-filters when both adapters are present:
      - ``\"cp210\"``: only Silicon Labs / CP210 devices
      - ``\"ch340\"``: only non-CP210 USB-serial (typical CH340)
      - ``None``: no chip filter

    ``include_tty_acm``: also consider ``/dev/ttyACM*`` (off by default —
    CANable / CDC gadgets live there; RPLIDAR / Wit are almost always ttyUSB).

    ``exclude``: extra path substrings to skip (from config ``serial_exclude``).
    """
    seen: set[str] = set()
    out: List[str] = []
    extra_exclude = tuple(str(x) for x in (exclude or []) if str(x).strip())

    def add(path: str) -> None:
        if path in seen:
            return
        if _should_skip_port(path, extra_exclude=extra_exclude):
            return
        seen.add(path)
        out.append(path)

    by_path = sorted(glob.glob("/dev/serial/by-path/*"))
    by_id = sorted(glob.glob("/dev/serial/by-id/usb-*"))
    cp210 = [p for p in by_id if _is_cp210_id(p)]
    other_id = [p for p in by_id if p not in cp210]
    cp210_reals = {_realpath(p) for p in cp210}

    # Prefer by-id over by-path: by-path can linger after USB re-enumeration.
    if prefer_cp210:
        for path in cp210:
            add(path)
        for path in other_id:
            add(path)
        for path in by_path:
            add(path)
    else:
        for path in other_id:
            add(path)
        for path in cp210:
            add(path)
        for path in by_path:
            add(path)

    # Only fall back to raw tty nodes when by-id/by-path yielded nothing useful.
    # Always skip ttyACM unless explicitly enabled (CAN / CDC collision).
    if not out:
        for path in sorted(glob.glob("/dev/ttyUSB*")):
            add(path)
        if include_tty_acm:
            for path in sorted(glob.glob("/dev/ttyACM*")):
                add(path)
    elif include_tty_acm:
        for path in sorted(glob.glob("/dev/ttyACM*")):
            add(path)

    deduped = _dedupe_by_realpath(out)
    # Drop unknown gadgets when by-id naming is available (keep CH340/CP210/FTDI).
    sensorish = [p for p in deduped if _looks_like_sensor_uart(p) or not _has_by_id_name(p)]
    if sensorish:
        deduped = sensorish

    if chip == "cp210":
        filtered = [p for p in deduped if _realpath(p) in cp210_reals or _is_cp210_id(p)]
        return filtered or deduped
    if chip == "ch340":
        filtered = [
            p
            for p in deduped
            if _realpath(p) not in cp210_reals and not _is_cp210_id(p)
        ]
        return filtered or deduped
    return deduped


def normalize_exclude_list(value) -> List[str]:
    """Parse ``serial_exclude`` from Viam attrs (list, tuple, or comma string)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(normalize_exclude_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _should_skip_port(path: str, *, extra_exclude: Sequence[str] = ()) -> bool:
    lowered = path.lower()
    for token in _EXCLUDE_SUBSTRINGS:
        if token in lowered:
            return True
    for token in extra_exclude:
        if token.lower() in lowered:
            return True
    # Resolve aliases and re-check by-id siblings for the same tty.
    real = _realpath(path)
    for sibling in _by_id_aliases(real):
        sib = sibling.lower()
        for token in _EXCLUDE_SUBSTRINGS:
            if token in sib:
                return True
        for token in extra_exclude:
            if token.lower() in sib:
                return True
    return False


def _by_id_aliases(real: str) -> List[str]:
    out: List[str] = []
    for path in glob.glob("/dev/serial/by-id/usb-*"):
        try:
            if os.path.realpath(path) == real:
                out.append(path)
        except OSError:
            continue
    return out


def _has_by_id_name(path: str) -> bool:
    return "/serial/by-id/" in path.replace("\\", "/")


def _looks_like_sensor_uart(path: str) -> bool:
    """True for known USB-UART bridge naming used by RPLIDAR / Wit adapters."""
    lowered = path.lower()
    if any(token in lowered for token in _SENSOR_CHIP_SUBSTRINGS):
        return True
    real = _realpath(path)
    for sibling in _by_id_aliases(real):
        sib = sibling.lower()
        if any(token in sib for token in _SENSOR_CHIP_SUBSTRINGS):
            return True
    # by-path / raw ttyUSB with no by-id: allow (legacy setups).
    if not _by_id_aliases(real) and (
        re.search(r"/ttyUSB\d+$", real) or "/serial/by-path/" in path
    ):
        return True
    return False


def _is_cp210_id(path: str) -> bool:
    return "Silicon_Labs" in path or "CP210" in path


def _dedupe_by_realpath(paths: List[str]) -> List[str]:
    """Keep first path for each underlying device (by-id/by-path/tty aliases)."""
    seen_real: set[str] = set()
    out: List[str] = []
    for path in paths:
        try:
            real = os.path.realpath(path)
        except OSError:
            real = path
        if not os.path.exists(path):
            continue
        if real in seen_real:
            continue
        seen_real.add(real)
        out.append(path)
    return out


def _realpath(port: str) -> str:
    try:
        return os.path.realpath(port)
    except OSError:
        return port


def claim_serial_port(owner: str, port: str) -> None:
    """Record that ``owner`` owns this device (in-process)."""
    real = _realpath(port)
    with _claims_lock:
        _claims[real] = owner


def release_serial_port(port: Optional[str]) -> None:
    if not port:
        return
    real = _realpath(port)
    with _claims_lock:
        if _claims.get(real):
            _claims.pop(real, None)


def steal_serial_port(owner: str, port: str) -> None:
    """Force-claim ``port`` for ``owner`` (e.g. lidar wins over false IMU claim)."""
    claim_serial_port(owner, port)


def is_serial_claimed_by_other(owner: str, port: str) -> bool:
    real = _realpath(port)
    with _claims_lock:
        who = _claims.get(real)
    return who is not None and who != owner


def claimed_owner(port: str) -> Optional[str]:
    real = _realpath(port)
    with _claims_lock:
        return _claims.get(real)


def sort_unclaimed_first(owner: str, ports: List[str]) -> List[str]:
    """Try ports we don't think another driver owns first; still try claimed last."""
    free: List[str] = []
    taken: List[str] = []
    for port in ports:
        if is_serial_claimed_by_other(owner, port):
            taken.append(port)
        else:
            free.append(port)
    return free + taken


def is_port_busy_error(exc: BaseException) -> bool:
    """True for exclusive-lock / EBUSY failures while another driver probes."""
    errno = getattr(exc, "errno", None)
    if errno in (11, 16):  # EAGAIN / EBUSY
        return True
    args = getattr(exc, "args", ())
    if args and args[0] in (11, 16):
        return True
    msg = str(exc).lower()
    return (
        "resource temporarily unavailable" in msg
        or "could not exclusively lock" in msg
        or "device or resource busy" in msg
        or "[errno 11]" in msg
        or "[errno 16]" in msg
    )


def is_port_missing_error(exc: BaseException) -> bool:
    errno = getattr(exc, "errno", None)
    # Include EIO (5): common after USB hub reset / path swap while the
    # kernel still exposes a stale node; retry rounds often recover.
    if errno in (2, 5, 6, 19):  # ENOENT / EIO / ENXIO / ENODEV
        return True
    args = getattr(exc, "args", ())
    if args and args[0] in (2, 5, 6, 19):
        return True
    msg = str(exc).lower()
    return (
        "no such device" in msg
        or "no such file or directory" in msg
        or "input/output error" in msg
        or "[errno 19]" in msg
        or "[errno 5]" in msg
        or "[errno 2]" in msg
    )
