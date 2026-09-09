"""Helpers for finding likely serial devices (RPLIDAR / WitMotion IMU)."""

from __future__ import annotations

import glob
import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Set, Tuple


LOGGER = logging.getLogger(__name__)

# In-process registry so lidar + IMU (same module process) don't steal each
# other's port during parallel resource startup.
_claims_lock = threading.Lock()
_claims: Dict[str, str] = {}  # realpath -> owner ("lidar" | "imu")

# USB VID:PID that must never be opened. Opening / resetting these (or their
# hub siblings via aggressive probes) knocks SocketCAN offline.
# 1d50:606f = OpenMoko / Geschwister Schneider / CANable / candleLight (gs_usb).
_CAN_USB_IDS: Set[Tuple[str, str]] = {
    ("1d50", "606f"),  # candleLight / CANable / Geschwister Schneider
    ("08d8", "0008"),  # IXXAT USB-to-CAN
    ("0c72", "000c"),  # PEAK PCAN-USB (common)
    ("0c72", "000d"),
}

# by-id / path substrings that must never be opened.
_EXCLUDE_SUBSTRINGS = (
    "canable",
    "cantact",
    "candle",
    "gs_usb",
    "geschwister",  # OpenMoko Geschwister Schneider CAN
    "geschmacksrichtung",
    "openmoko",
    "schneider_can",
    "1d50_606f",
    "1d50:606f",
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

# Autodetect allowlist: only these UART-bridge by-id tokens (lidar / IMU).
_ALLOW_BY_ID_SUBSTRINGS = (
    "silicon_labs",
    "cp210",
    "1a86",  # WCH CH340 / CH341
    "wch.cn",
    "wch_",
    "ftdi",
    "ft232",
    "ft230",
    "prolific",
    "pl2303",
)

# Kernel USB-serial drivers we will open. Anything else (cdc_acm, etc.) is skip.
_ALLOW_TTY_DRIVERS = frozenset(
    {
        "cp210x",
        "ch341",
        "ch343",
        "ftdi_sio",
        "pl2303",
        "usbserial",  # generic wrapper sometimes shown as parent
    }
)


def list_candidate_serial_ports(
    *,
    prefer_cp210: bool = True,
    chip: Optional[str] = None,
    include_tty_acm: bool = False,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return serial device candidates safe to probe for lidar / IMU.

    Autodetect is **allowlist-only**:
      - ``/dev/serial/by-id/usb-*`` matching known UART chips (CP210/CH340/…)
      - sysfs driver must be a USB-UART bridge (not ``cdc_acm``)
      - USB VID:PID must not be a known CAN adapter (e.g. ``1d50:606f``)

    Never probes by-path, raw ``ttyACM*``, or unknown gadgets — those are how
    Geschwister Schneider / CANable adapters get exclusive-opened or the hub
    gets reset and ``can0`` drops.
    """
    _ = include_tty_acm  # retained for config compat; ACM never auto-probed
    seen: set[str] = set()
    out: List[str] = []
    extra_exclude = tuple(str(x) for x in (exclude or []) if str(x).strip())

    def add(path: str) -> None:
        if path in seen:
            return
        if not is_safe_sensor_serial_port(path, extra_exclude=extra_exclude):
            return
        seen.add(path)
        out.append(path)

    by_id = sorted(glob.glob("/dev/serial/by-id/usb-*"))
    # Strict allowlist: only named UART bridges. No by-path / ttyUSB / ttyACM.
    allowed = [p for p in by_id if _by_id_allowlisted(p)]
    cp210 = [p for p in allowed if _is_cp210_id(p)]
    other_id = [p for p in allowed if p not in cp210]
    cp210_reals = {_realpath(p) for p in cp210}

    if prefer_cp210:
        for path in cp210:
            add(path)
        for path in other_id:
            add(path)
    else:
        for path in other_id:
            add(path)
        for path in cp210:
            add(path)

    deduped = _dedupe_by_realpath(out)

    if chip == "cp210":
        filtered = [p for p in deduped if _realpath(p) in cp210_reals or _is_cp210_id(p)]
        # Do NOT fall back to non-CP210 — that reintroduces wrong-port probes.
        return filtered
    if chip == "ch340":
        filtered = [
            p
            for p in deduped
            if _realpath(p) not in cp210_reals and not _is_cp210_id(p)
        ]
        return filtered
    return deduped


def is_safe_sensor_serial_port(
    path: str, *, extra_exclude: Sequence[str] = ()
) -> bool:
    """False for CAN adapters / unknown USB gadgets — never open these."""
    if _should_skip_port(path, extra_exclude=extra_exclude):
        return False
    if _usb_id_is_can(path):
        LOGGER.warning(
            "skipping serial port %s: USB id matches CAN adapter denylist", path
        )
        return False
    if _tty_bound_to_can_netdev(path):
        LOGGER.warning("skipping serial port %s: tied to SocketCAN netdev", path)
        return False
    driver = _tty_driver_name(path)
    if driver and driver not in _ALLOW_TTY_DRIVERS and driver != "usbserial":
        # cdc_acm / gs_usb / etc.
        if driver in ("cdc_acm", "gs_usb", "slcan", "usb_wwan"):
            LOGGER.warning(
                "skipping serial port %s: kernel driver %r is not a UART bridge",
                path,
                driver,
            )
            return False
    # Allowlist by-id name when present.
    aliases = _by_id_aliases(_realpath(path))
    if aliases and not any(_by_id_allowlisted(a) for a in aliases):
        return False
    if _has_by_id_name(path) and not _by_id_allowlisted(path):
        return False
    return True


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


def _by_id_allowlisted(path: str) -> bool:
    lowered = path.lower()
    return any(token in lowered for token in _ALLOW_BY_ID_SUBSTRINGS)


def _should_skip_port(path: str, *, extra_exclude: Sequence[str] = ()) -> bool:
    lowered = path.lower()
    for token in _EXCLUDE_SUBSTRINGS:
        if token in lowered:
            return True
    for token in extra_exclude:
        if token.lower() in lowered:
            return True
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


def _sysfs_tty_dir(path: str) -> Optional[str]:
    real = _realpath(path)
    base = os.path.basename(real)
    if not base.startswith("tty"):
        return None
    candidate = f"/sys/class/tty/{base}/device"
    if os.path.isdir(candidate) or os.path.islink(candidate):
        return candidate
    return None


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _usb_vid_pid_for_port(path: str) -> Optional[Tuple[str, str]]:
    """Walk sysfs from a tty node up to the USB device; return (vid, pid)."""
    tty_dev = _sysfs_tty_dir(path)
    if not tty_dev:
        return None
    try:
        cur = os.path.realpath(tty_dev)
    except OSError:
        return None
    for _ in range(8):
        vid = _read_text(os.path.join(cur, "idVendor")).lower()
        pid = _read_text(os.path.join(cur, "idProduct")).lower()
        if vid and pid:
            return vid, pid
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _usb_id_is_can(path: str) -> bool:
    pair = _usb_vid_pid_for_port(path)
    if pair and pair in _CAN_USB_IDS:
        return True
    # Also match by-id spelling usb-1d50_606f_...
    lowered = path.lower()
    if "1d50_606f" in lowered or "1d50:606f" in lowered:
        return True
    for sibling in _by_id_aliases(_realpath(path)):
        s = sibling.lower()
        if "1d50_606f" in s or "1d50:606f" in s or "openmoko" in s:
            return True
    return False


def _tty_driver_name(path: str) -> Optional[str]:
    tty_dev = _sysfs_tty_dir(path)
    if not tty_dev:
        return None
    driver_link = os.path.join(tty_dev, "driver")
    try:
        if os.path.islink(driver_link) or os.path.exists(driver_link):
            return os.path.basename(os.path.realpath(driver_link))
    except OSError:
        return None
    # usb-serial child: .../ttyUSB0/device -> ../.. may be the usb-serial iface
    try:
        parent = os.path.realpath(os.path.join(tty_dev, ".."))
        driver_link = os.path.join(parent, "driver")
        if os.path.exists(driver_link):
            return os.path.basename(os.path.realpath(driver_link))
    except OSError:
        pass
    return None


def _tty_bound_to_can_netdev(path: str) -> bool:
    """True if this USB device also hosts a SocketCAN netdev (gs_usb etc.)."""
    pair = _usb_vid_pid_for_port(path)
    tty_dev = _sysfs_tty_dir(path)
    usb_roots: List[str] = []
    if tty_dev:
        try:
            cur = os.path.realpath(tty_dev)
            for _ in range(8):
                if os.path.exists(os.path.join(cur, "idVendor")):
                    usb_roots.append(cur)
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
        except OSError:
            pass

    for net in glob.glob("/sys/class/net/can*"):
        try:
            net_dev = os.path.realpath(os.path.join(net, "device"))
        except OSError:
            continue
        for root in usb_roots:
            if net_dev == root or net_dev.startswith(root + os.sep):
                return True
            if root.startswith(net_dev + os.sep):
                return True
        if pair:
            vid = _read_text(os.path.join(net_dev, "idVendor")).lower()
            pid = _read_text(os.path.join(net_dev, "idProduct")).lower()
            # Walk up for idVendor on netdev's USB parent
            cur = net_dev
            for _ in range(6):
                vid = _read_text(os.path.join(cur, "idVendor")).lower() or vid
                pid = _read_text(os.path.join(cur, "idProduct")).lower() or pid
                if vid and pid:
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
            if vid and pid and (vid, pid) == pair:
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
