"""Helpers for finding likely serial devices (RPLIDAR / WitMotion IMU)."""

from __future__ import annotations

import glob
import os
import threading
from typing import Dict, List, Optional


# In-process registry so lidar + IMU (same module process) don't steal each
# other's port during parallel resource startup.
_claims_lock = threading.Lock()
_claims: Dict[str, str] = {}  # realpath -> owner ("lidar" | "imu")


def list_candidate_serial_ports(
    *,
    prefer_cp210: bool = True,
    chip: Optional[str] = None,
) -> List[str]:
    """Return serial device candidates.

    ``prefer_cp210=True``: Silicon Labs / CP210 by-id first.
    ``prefer_cp210=False``: by-path / non-CP210 first.

    ``chip`` hard-filters when both adapters are present:
      - ``\"cp210\"``: only Silicon Labs / CP210 devices (WitMotion IMU)
      - ``\"ch340\"``: only non-CP210 USB-serial (typical CH340 RPLIDAR)
      - ``None``: no chip filter
    """
    seen: set[str] = set()
    out: List[str] = []

    def add(path: str) -> None:
        if path in seen:
            return
        seen.add(path)
        out.append(path)

    by_path = sorted(glob.glob("/dev/serial/by-path/*"))
    by_id = sorted(glob.glob("/dev/serial/by-id/usb-*"))
    cp210 = [p for p in by_id if _is_cp210_id(p)]
    other_id = [p for p in by_id if p not in cp210]
    cp210_reals = {_realpath(p) for p in cp210}

    if prefer_cp210:
        for path in cp210:
            add(path)
        for path in other_id:
            add(path)
        for path in by_path:
            add(path)
    else:
        for path in by_path:
            add(path)
        for path in other_id:
            add(path)
        for path in cp210:
            add(path)

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in sorted(glob.glob(pattern)):
            add(path)

    deduped = _dedupe_by_realpath(out)
    if chip == "cp210":
        filtered = [p for p in deduped if _realpath(p) in cp210_reals or _is_cp210_id(p)]
        # Fall back to full list if no CP210 is enumerated yet (USB still probing).
        return filtered or deduped
    if chip == "ch340":
        filtered = [
            p
            for p in deduped
            if _realpath(p) not in cp210_reals and not _is_cp210_id(p)
        ]
        return filtered or deduped
    return deduped


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
    if errno in (2, 6, 19):  # ENOENT / ENXIO / ENODEV
        return True
    args = getattr(exc, "args", ())
    if args and args[0] in (2, 6, 19):
        return True
    msg = str(exc).lower()
    return (
        "no such device" in msg
        or "no such file or directory" in msg
        or "[errno 19]" in msg
        or "[errno 2]" in msg
    )
