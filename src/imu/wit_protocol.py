"""WitMotion UART frame parsing (BWT61CL / BWT901 / HWT901B / similar).

Framing and scaling match ``viam-modules/wit-motion`` ``imuwit``:

* Sync on ``0x55`` (``'U'``) the same way Go ``bufio.ReadString('U')`` does:
  each accepted line is ``type | 8 payload bytes | checksum | 0x55`` (11 bytes),
  with ``line[0]`` = packet type (no leading sync in the parse buffer).
* ``scale(lo, hi, r)`` is the same unsigned remap into ``[-r, r)``.
* Angle packets (``0x53``) convert degrees → radians; gyro (``0x52``) stays deg/s
  for the Viam ``GetAngularVelocity`` API (same as the Go module).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Optional

SYNC = 0x55  # 'U' — same delimiter as wit-motion ReadString('U')
TYPE_ACCEL = 0x51
TYPE_GYRO = 0x52
TYPE_ORIENT = 0x53
TYPE_MAG = 0x54

BAUDRATES = (115200, 9600)
G = 9.80665

# --- Host → device config (WitMotion register protocol, FF AA REG LO HI) ---
CMD_UNLOCK = bytes((0xFF, 0xAA, 0x69, 0x88, 0xB5))
REG_SAVE = 0x00
REG_CALSW = 0x01
REG_AXIS6 = 0x24  # 0 = 9-axis (mag-fused yaw), 1 = 6-axis (gyro-integrated yaw)
CALSW_ZERO_YAW = 0x04  # "Z-axis angle to zero" (6-axis mode only)
ALGORITHMS = ("6axis", "9axis", "keep")


def cmd_set_register(reg: int, value: int) -> bytes:
    """``FF AA reg lo hi`` register write."""
    value &= 0xFFFF
    return bytes((0xFF, 0xAA, reg & 0xFF, value & 0xFF, (value >> 8) & 0xFF))


def config_commands(algorithm: str = "keep", *, zero_yaw: bool = False) -> list[bytes]:
    """Startup command sequence (each needs ~100-200 ms spacing on the wire).

    Nothing is saved to flash (no ``REG_SAVE``) so the device reverts on power
    cycle; the sequence is re-sent every time the component starts.
    """
    if algorithm not in ALGORITHMS:
        raise WitError(f"unknown Wit algorithm {algorithm!r}; use one of {ALGORITHMS}")
    out: list[bytes] = []
    if algorithm != "keep" or zero_yaw:
        out.append(CMD_UNLOCK)
    if algorithm == "6axis":
        out.append(cmd_set_register(REG_AXIS6, 1))
    elif algorithm == "9axis":
        out.append(cmd_set_register(REG_AXIS6, 0))
    if zero_yaw:
        out.append(cmd_set_register(REG_CALSW, CALSW_ZERO_YAW))
    return out


class WitError(RuntimeError):
    pass


def scale_le_u16(lo: int, hi: int, r: float) -> float:
    """Map little-endian uint16 into ``[-r, r)`` — identical to wit-motion ``scale``."""
    x = float((int(hi) << 8) | int(lo)) / 32768.0  # 0 -> 2
    x *= r  # 0 -> 2r
    x += r
    x = math.fmod(x, r * 2.0)
    x -= r
    return x


def mag_le_i16(lo: int, hi: int) -> float:
    """Signed magnetometer count — wit-motion ``convertMagByteToTesla``."""
    raw = struct.unpack("<h", bytes((lo & 0xFF, hi & 0xFF)))[0]
    return float(raw)


@dataclass
class WitSample:
    ax: float = 0.0  # m/s^2
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0  # deg/s (Viam / wit-motion AngularVelocity)
    gy: float = 0.0
    gz: float = 0.0
    roll: float = 0.0  # rad (wit-motion EulerAngles)
    pitch: float = 0.0
    yaw: float = 0.0
    mx: float = 0.0  # µT
    my: float = 0.0
    mz: float = 0.0
    has_mag: bool = False
    packets: int = 0
    bad_readings: int = 0
    # Raw decode trail for scale verification (0x53 / 0x52 packets).
    yaw_raw_u16: int = 0  # (YawH<<8)|YawL straight off the wire
    yaw_deg_decoded: float = 0.0  # scale(yaw_raw_u16, 180) before deg→rad
    gz_raw_u16: int = 0
    angle_packets: int = 0
    gyro_packets: int = 0


@dataclass
class WitStreamParser:
    """Incremental byte stream → ``WitSample`` (wit-motion ReadString framing)."""

    sample: WitSample = field(default_factory=WitSample)
    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> int:
        """Ingest bytes; return number of 11-byte frames accepted this call."""
        if not data:
            return 0
        self._buf.extend(data)
        parsed = 0
        # Mirror Go: line, err := portReader.ReadString('U') with len(line)==11.
        # Stream: ... [type][8 data][cs][U] [type][8 data][cs][U] ...
        # after an initial sync U is consumed as a short (len≠11) discard.
        while True:
            try:
                end = self._buf.index(SYNC)
            except ValueError:
                # Keep a little tail so a split frame can complete next feed.
                if len(self._buf) > 64:
                    del self._buf[:-64]
                break
            line = bytes(self._buf[: end + 1])
            del self._buf[: end + 1]
            if len(line) != 11:
                self.sample.bad_readings += 1
                continue
            if self._parse_go_line(line):
                parsed += 1
                self.sample.packets += 1
            else:
                self.sample.bad_readings += 1
        return parsed

    def _parse_go_line(self, line: bytes) -> bool:
        """Parse one wit-motion ``parseWIT`` line (``line[0]`` = type)."""
        typ = line[0]
        if typ not in (TYPE_ACCEL, TYPE_GYRO, TYPE_ORIENT, TYPE_MAG):
            return False
        # line[1:9] payload, line[9] checksum, line[10] == SYNC
        s = self.sample
        if typ == TYPE_GYRO:
            s.gx = scale_le_u16(line[1], line[2], 2000.0)
            s.gy = scale_le_u16(line[3], line[4], 2000.0)
            s.gz = scale_le_u16(line[5], line[6], 2000.0)
            s.gz_raw_u16 = (line[6] << 8) | line[5]
            s.gyro_packets += 1
        elif typ == TYPE_ORIENT:
            # Fused Euler angles: raw16 * 180 / 32768 degrees (wraps at ±180).
            s.roll = math.radians(scale_le_u16(line[1], line[2], 180.0))
            s.pitch = math.radians(scale_le_u16(line[3], line[4], 180.0))
            yaw_deg = scale_le_u16(line[5], line[6], 180.0)
            s.yaw = math.radians(yaw_deg)
            s.yaw_raw_u16 = (line[6] << 8) | line[5]
            s.yaw_deg_decoded = yaw_deg
            s.angle_packets += 1
        elif typ == TYPE_ACCEL:
            s.ax = scale_le_u16(line[1], line[2], 16.0) * G
            s.ay = scale_le_u16(line[3], line[4], 16.0) * G
            s.az = scale_le_u16(line[5], line[6], 16.0) * G
        elif typ == TYPE_MAG:
            s.has_mag = True
            s.mx = mag_le_i16(line[1], line[2])
            s.my = mag_le_i16(line[3], line[4])
            s.mz = mag_le_i16(line[5], line[6])
        return True


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
