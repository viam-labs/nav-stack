"""Minimal Slamtec RPLIDAR UART protocol (SCAN / INFO / HEALTH / STOP / EXPRESS).

A1/A3 use legacy 5-byte SCAN at 115200/256000. S2/S3 (1M baud) use typical
Express / DenseBoost capsules — same path as viam-modules/rplidar
``StartScan(false, true)``.
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
CMD_EXPRESS_SCAN = 0x82
CMD_GET_LIDAR_CONF = 0x84
CMD_HQ_MOTOR_SPEED_CTRL = 0xA8

INFO_LEN = 20
HEALTH_LEN = 3
NODE_LEN = 5
DENSE_CAPSULE_LEN = 84
INFO_TYPE = 0x04
HEALTH_TYPE = 0x06
SCAN_TYPE = 0x81
GET_LIDAR_CONF_TYPE = 0x20
DENSE_CAPSULED_TYPE = 0x85
CAPSULED_TYPE = 0x82

CONF_SCAN_MODE_TYPICAL = 0x0000007C
CONF_SCAN_MODE_ANS_TYPE = 0x00000075
DEFAULT_TOF_RPM = 600
TOF_MIN_MAJOR_ID = 5  # model>>4 > 5 → TOF (S1/S2/S3)

EXP_SYNC_1 = 0xA
EXP_SYNC_2 = 0x5
EXP_SYNC_BIT = 1 << 15

# Viam rplidar model bytes (rplidar.go rplidarModelByteMap).
# Encoding is (major<<4)|submodel: A1=0x18, A3=0x31, S1=0x61, S2=0x71, S3=0x81.
MODEL_A1 = 24
MODEL_A3 = 49
MODEL_S1 = 97
MODEL_S2 = 113
MODEL_S3 = 129

MODEL_NAMES = {
    MODEL_A1: "A1",
    MODEL_A3: "A3",
    MODEL_S1: "S1",
    MODEL_S2: "S2",
    MODEL_S3: "S3",
}

# Probe order matches viam-modules/rplidar: 1M (S2/S3), 256k (A3/S1), 115200 (A1).
BAUDRATES = (1000000, 256000, 115200)


class RPLidarError(RuntimeError):
    pass


def is_s_series(model: int) -> bool:
    """S1/S2/S3 family — use Express/DenseBoost (not legacy SCAN / DTR motor)."""
    return int(model) in (MODEL_S1, MODEL_S2, MODEL_S3)


def is_tof_lidar(model: int) -> bool:
    """Slamtec SDK: ``(model >> 4) > 5`` marks TOF units (S1/S2/S3)."""
    return (int(model) >> 4) > TOF_MIN_MAJOR_ID


def model_name(model: int) -> str:
    return MODEL_NAMES.get(int(model), f"unknown(0x{int(model):02X})")


def command(cmd: int) -> bytes:
    return bytes((SYNC, cmd))


def command_with_payload(cmd: int, payload: bytes) -> bytes:
    """Request with payload (EXPRESS_SCAN, GET_LIDAR_CONF, HQ motor, …)."""
    payload = bytes(payload)
    size = len(payload)
    if size > 255:
        raise RPLidarError(f"payload too large ({size})")
    flagged = cmd | 0x80
    checksum = SYNC ^ flagged ^ size
    for b in payload:
        checksum ^= b
    return bytes((SYNC, flagged, size)) + payload + bytes((checksum & 0xFF,))


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


def encode_hq_motor_rpm(rpm: int = DEFAULT_TOF_RPM) -> bytes:
    return int(rpm).to_bytes(2, "little")


def encode_express_scan_payload(working_mode: int = 0, working_flags: int = 0) -> bytes:
    return bytes((int(working_mode) & 0xFF,)) + int(working_flags).to_bytes(
        2, "little"
    ) + (0).to_bytes(2, "little")


def encode_get_lidar_conf(conf_type: int, reserve: bytes = b"") -> bytes:
    payload = int(conf_type).to_bytes(4, "little") + (reserve + bytes(32))[:32]
    return payload


def dense_capsule_checksum_ok(raw: bytes) -> bool:
    if len(raw) != DENSE_CAPSULE_LEN:
        return False
    if (raw[0] >> 4) != EXP_SYNC_1 or (raw[1] >> 4) != EXP_SYNC_2:
        return False
    recv = (raw[0] & 0x0F) | ((raw[1] & 0x0F) << 4)
    checksum = 0
    for b in raw[2:]:
        checksum ^= b
    return checksum == recv


def decode_dense_capsule_pair(
    prev: bytes, curr: bytes
) -> List[Tuple[int, float, float, bool]]:
    """Unpack one dense capsule using the previous capsule's start angle.

    Returns ``(quality, angle_deg, distance_mm, new_scan)`` samples (40).
    Mirrors Slamtec ``_dense_capsuleToNormal``.
    """
    if len(prev) != DENSE_CAPSULE_LEN or len(curr) != DENSE_CAPSULE_LEN:
        raise RPLidarError("dense capsule length")
    curr_start_q8 = (int.from_bytes(curr[2:4], "little") & 0x7FFF) << 2
    prev_start_q8 = (int.from_bytes(prev[2:4], "little") & 0x7FFF) << 2
    diff_q8 = curr_start_q8 - prev_start_q8
    if prev_start_q8 > curr_start_q8:
        diff_q8 += 360 << 8
    angle_inc_q16 = (diff_q8 << 8) // 40
    current_angle_q16 = prev_start_q8 << 8
    out: List[Tuple[int, float, float, bool]] = []
    for pos in range(40):
        dist = int.from_bytes(prev[4 + pos * 2 : 6 + pos * 2], "little")
        angle_q6 = current_angle_q16 >> 10
        new_scan = ((current_angle_q16 + angle_inc_q16) % (360 << 16)) < angle_inc_q16
        current_angle_q16 += angle_inc_q16
        if angle_q6 < 0:
            angle_q6 += 360 << 6
        if angle_q6 >= (360 << 6):
            angle_q6 -= 360 << 6
        quality = 0x2F if dist else 0
        out.append((quality, angle_q6 / 64.0, float(dist), bool(new_scan)))
    return out
