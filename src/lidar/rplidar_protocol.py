"""Minimal Slamtec RPLIDAR UART protocol (SCAN / INFO / HEALTH / STOP).

Compatible with A1/A2/A3-class devices on 115200 or 256000 baud. Frame layout
follows the public Slamtec interface protocol; polar→XYZ matches the Viam
rplidar module (180° about Y so +X is the flipped lidar heading).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

SYNC = 0xA5
SYNC2 = 0x5A
CMD_STOP = 0x25
CMD_RESET = 0x40
CMD_SCAN = 0x20
CMD_GET_INFO = 0x50
CMD_GET_HEALTH = 0x52

INFO_LEN = 20
HEALTH_LEN = 3
NODE_LEN = 5
INFO_TYPE = 0x04
HEALTH_TYPE = 0x06
SCAN_TYPE = 0x81

# Viam rplidar model bytes (rplidar.go rplidarModelByteMap).
MODEL_A1 = 24
MODEL_A3 = 49
MODEL_S1 = 97

BAUDRATES = (256000, 115200)


class RPLidarError(RuntimeError):
    pass


def command(cmd: int) -> bytes:
    return bytes((SYNC, cmd))


def descriptor(size: int, single: bool, dtype: int) -> bytes:
    send_mode = 0 if single else 1
    packed = (size & 0x3FFFFFFF) | (send_mode << 30)
    return bytes((SYNC, SYNC2)) + packed.to_bytes(4, "little") + bytes((dtype,))


def parse_descriptor(raw: bytes) -> Tuple[int, bool, int]:
    if len(raw) != 7:
        raise RPLidarError(f"descriptor length {len(raw)}")
    if raw[0] != SYNC or raw[1] != SYNC2:
        raise RPLidarError("bad descriptor sync")
    packed = int.from_bytes(raw[2:6], "little")
    size = packed & 0x3FFFFFFF
    single = (packed >> 30) == 0
    return size, single, raw[6]


def encode_node(
    *,
    new_scan: bool,
    quality: int,
    angle_deg: float,
    distance_mm: float,
) -> bytes:
    """Build one 5-byte SCAN measurement (for tests / fakes)."""
    sync = 1 if new_scan else 0
    inv = 0 if new_scan else 1
    b0 = ((int(quality) & 0x3F) << 2) | (inv << 1) | sync
    angle_q6 = int(round(float(angle_deg) * 64.0)) & 0x7FFF
    b1 = ((angle_q6 & 0x7F) << 1) | 1
    b2 = (angle_q6 >> 7) & 0xFF
    dist_q2 = max(0, int(round(float(distance_mm) * 4.0))) & 0xFFFF
    b3 = dist_q2 & 0xFF
    b4 = (dist_q2 >> 8) & 0xFF
    return bytes((b0, b1, b2, b3, b4))


def decode_node(raw: bytes) -> Tuple[bool, int, float, float]:
    if len(raw) != NODE_LEN:
        raise RPLidarError(f"node length {len(raw)}")
    new_scan = bool(raw[0] & 0x01)
    inversed = bool((raw[0] >> 1) & 0x01)
    if new_scan == inversed:
        raise RPLidarError("scan sync flags mismatch")
    if raw[1] & 0x01 != 1:
        raise RPLidarError("angle check bit")
    quality = raw[0] >> 2
    angle_deg = ((raw[1] >> 1) + (raw[2] << 7)) / 64.0
    distance_mm = (raw[3] + (raw[4] << 8)) / 4.0
    return new_scan, quality, angle_deg, distance_mm


def decode_info(raw: bytes) -> dict:
    if len(raw) != INFO_LEN:
        raise RPLidarError(f"info length {len(raw)}")
    return {
        "model": raw[0],
        "firmware": (raw[2], raw[1]),
        "hardware": raw[3],
        "serial": raw[4:].hex().upper(),
    }


def decode_health(raw: bytes) -> Tuple[int, int]:
    if len(raw) != HEALTH_LEN:
        raise RPLidarError(f"health length {len(raw)}")
    status = raw[0]
    code = raw[1] + (raw[2] << 8)
    return status, code


def polar_to_xyz_m(angle_deg: float, distance_mm: float) -> Tuple[float, float, float]:
    """Sensor-frame meters, matching Viam rplidar ``pointFrom`` (X flipped)."""
    dist_m = float(distance_mm) / 1000.0
    yaw = math.radians(float(angle_deg))
    return (-math.cos(yaw) * dist_m, math.sin(yaw) * dist_m, 0.0)


def scan_to_xyz_m(
    measurements: Sequence[Tuple[int, float, float]],
    *,
    min_range_mm: float = 0.0,
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for quality, angle_deg, distance_mm in measurements:
        if quality <= 0 or distance_mm <= 0:
            continue
        if distance_mm < min_range_mm:
            continue
        out.append(polar_to_xyz_m(angle_deg, distance_mm))
    return out


def encode_info(*, model: int = MODEL_A1, firmware=(1, 0), hardware=0, serial: Optional[bytes] = None) -> bytes:
    ser = (serial or b"\x00" * 16)[:16].ljust(16, b"\x00")
    return bytes((model, firmware[1], firmware[0], hardware)) + ser


def encode_health(status: int = 0, error_code: int = 0) -> bytes:
    return bytes((status, error_code & 0xFF, (error_code >> 8) & 0xFF))


def encode_scan_stream(scans: Iterable[Sequence[Tuple[int, float, float]]]) -> bytes:
    """Concatenate SCAN nodes; each scan starts with ``new_scan=True``."""
    buf = bytearray()
    for meas in scans:
        first = True
        for quality, angle, dist in meas:
            buf += encode_node(
                new_scan=first, quality=quality, angle_deg=angle, distance_mm=dist
            )
            first = False
    return bytes(buf)
