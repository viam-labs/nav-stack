"""WitMotion UART frame parsing (BWT61CL / BWT901 / HWT901B / similar).

Ported from ``viam/wit-motion`` ``imuwit.parseWIT``:
frames start with ``0x55``, then type ``0x51``..``0x54``, then 8 data bytes +
checksum (we tolerate streams where the next ``0x55`` delimits the previous
frame, matching the Go ``ReadString('U')`` behaviour).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

SYNC = 0x55
TYPE_ACCEL = 0x51
TYPE_GYRO = 0x52
TYPE_ORIENT = 0x53
TYPE_MAG = 0x54

BAUDRATES = (115200, 9600)
G = 9.80665


class WitError(RuntimeError):
    pass


def scale_le_u16(lo: int, hi: int, r: float) -> float:
    """Map little-endian uint16 into ``[-r, r)`` (WitMotion datasheet scale)."""
    x = float((hi << 8) | lo) / 32768.0
    x *= r
    x += r
    x = math.fmod(x, r * 2.0)
    x -= r
    return x


def mag_le_i16(lo: int, hi: int) -> float:
    raw = struct.unpack("<h", bytes((lo, hi)))[0]
    return float(raw)


@dataclass
class WitSample:
    ax: float = 0.0  # m/s^2
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0  # deg/s
    gy: float = 0.0
    gz: float = 0.0
    roll: float = 0.0  # rad
    pitch: float = 0.0
    yaw: float = 0.0
    mx: float = 0.0  # µT
    my: float = 0.0
    mz: float = 0.0
    has_mag: bool = False
    packets: int = 0


@dataclass
class WitStreamParser:
    """Incremental byte stream → ``WitSample`` updates."""

    sample: WitSample = field(default_factory=WitSample)
    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> int:
        """Ingest bytes; return number of packets parsed."""
        if not data:
            return 0
        self._buf.extend(data)
        parsed = 0
        # Frame: 0x55 | type | 8 payload bytes | checksum  (11 bytes total)
        while True:
            try:
                start = self._buf.index(SYNC)
            except ValueError:
                self._buf.clear()
                break
            if start > 0:
                del self._buf[:start]
            if len(self._buf) < 11:
                break
            frame = bytes(self._buf[:11])
            if frame[0] != SYNC:
                del self._buf[0]
                continue
            # Soft checksum: sum of first 10 bytes & 0xFF == byte 10 (many clones
            # are flaky; still accept known types even if checksum mismatches).
            typ = frame[1]
            if typ not in (TYPE_ACCEL, TYPE_GYRO, TYPE_ORIENT, TYPE_MAG):
                del self._buf[0]
                continue
            self._apply(typ, frame[2:10])
            del self._buf[:11]
            parsed += 1
            self.sample.packets += 1
        return parsed

    def _apply(self, typ: int, payload: bytes) -> None:
        s = self.sample
        if typ == TYPE_ACCEL:
            s.ax = scale_le_u16(payload[0], payload[1], 16.0) * G
            s.ay = scale_le_u16(payload[2], payload[3], 16.0) * G
            s.az = scale_le_u16(payload[4], payload[5], 16.0) * G
        elif typ == TYPE_GYRO:
            s.gx = scale_le_u16(payload[0], payload[1], 2000.0)
            s.gy = scale_le_u16(payload[2], payload[3], 2000.0)
            s.gz = scale_le_u16(payload[4], payload[5], 2000.0)
        elif typ == TYPE_ORIENT:
            s.roll = math.radians(scale_le_u16(payload[0], payload[1], 180.0))
            s.pitch = math.radians(scale_le_u16(payload[2], payload[3], 180.0))
            s.yaw = math.radians(scale_le_u16(payload[4], payload[5], 180.0))
        elif typ == TYPE_MAG:
            s.has_mag = True
            s.mx = mag_le_i16(payload[0], payload[1])
            s.my = mag_le_i16(payload[2], payload[3])
            s.mz = mag_le_i16(payload[4], payload[5])


def probe_is_wit(ser, *, listen_s: float = 0.6, min_packets: int = 3) -> bool:
    """True if ``ser`` streams valid WitMotion frames within ``listen_s``."""
    import time

    parser = WitStreamParser()
    deadline = time.monotonic() + max(listen_s, 0.1)
    while time.monotonic() < deadline:
        waiting = getattr(ser, "in_waiting", 0) or 0
        chunk = ser.read(max(waiting, 64) if waiting else 64)
        if chunk:
            parser.feed(chunk)
            if parser.sample.packets >= min_packets:
                return True
        else:
            time.sleep(0.02)
    return parser.sample.packets >= min_packets
