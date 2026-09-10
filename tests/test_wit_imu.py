"""Tests for WitMotion protocol, IMU shm, and wit-imu helpers."""
from __future__ import annotations

import math
import struct

import pytest

from src.imu.wit_protocol import (
    CMD_UNLOCK,
    TYPE_ACCEL,
    TYPE_GYRO,
    TYPE_ORIENT,
    WitError,
    WitStreamParser,
    config_commands,
    scale_le_u16,
)
from src.ros import imushm


def _frame(typ: int, values_le: bytes) -> bytes:
    """One wit-motion ``ReadString('U')`` line: type|8data|cs|0x55 (11 bytes)."""
    assert len(values_le) == 8
    wire = bytes([0x55, typ]) + values_le
    checksum = sum(wire) & 0xFF
    return bytes([typ]) + values_le + bytes([checksum, 0x55])


def _u16_pair(value: float, r: float) -> bytes:
    """Inverse of scale_le_u16 for test payloads (approximate)."""
    # scale maps uint16/32768 * r into [-r,r) after wrap — use mid-scale 0.
    # For 0: x=0 -> uint such that after transform ~0. Use 0x0000.
    if abs(value) < 1e-9:
        return b"\x00\x00"
    # Encode: want result ≈ value. From scale: x = u/32768 * r; then wrap.
    # Simplified: u = (value / r) * 32768 for value in [-r, r).
    u = int(round((value / r) * 32768.0)) & 0xFFFF
    return struct.pack("<H", u)


def test_scale_le_u16_zero():
    assert scale_le_u16(0, 0, 180.0) == pytest.approx(0.0)


def test_parser_accepts_classic_wire_stream_like_device():
    """Device wire is 0x55|type|data|cs; Go ReadString consumes sync then 11-byte lines."""
    parser = WitStreamParser()
    yaw = struct.pack("<h", 16384)  # +90°
    payload = b"\x00\x00\x00\x00" + yaw + b"\x00\x00"
    # Two classic frames back-to-back (second sync terminates the first Go line).
    wire = b"".join(
        [
            bytes([0x55, TYPE_ORIENT]) + payload + bytes([
                sum(bytes([0x55, TYPE_ORIENT]) + payload) & 0xFF
            ]),
            bytes([0x55, TYPE_GYRO]) + (b"\x00" * 8) + bytes([
                sum(bytes([0x55, TYPE_GYRO]) + b"\x00" * 8) & 0xFF
            ]),
        ]
    )
    n = parser.feed(wire)
    assert n >= 1
    assert parser.sample.yaw == pytest.approx(math.radians(90.0))


def test_parser_orient_yaw_90_matches_wit_motion_scale():
    """Official Wit scale (/32768*180) — same as viam-modules/wit-motion."""
    parser = WitStreamParser()
    # Signed int16 for +90°: 90/180*32768 = 16384.
    yaw = struct.pack("<h", 16384)
    payload = b"\x00\x00\x00\x00" + yaw + b"\x00\x00"
    n = parser.feed(_frame(TYPE_ORIENT, payload))
    assert n == 1
    assert parser.sample.yaw == pytest.approx(math.radians(90.0))


def test_parser_accel_gyro_orient():
    parser = WitStreamParser()
    # Three frames: accel, gyro, orient around zero.
    stream = b"".join(
        [
            _frame(TYPE_ACCEL, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
            _frame(TYPE_GYRO, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
            _frame(TYPE_ORIENT, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
        ]
    )
    n = parser.feed(stream)
    assert n == 3
    assert parser.sample.packets == 3
    assert parser.sample.ax == pytest.approx(0.0)
    assert parser.sample.gx == pytest.approx(0.0)
    assert parser.sample.yaw == pytest.approx(0.0)


def test_imushm_pack_unpack():
    payload = imushm.pack_sample(
        ax=1.0,
        ay=2.0,
        az=9.8,
        gx=0.1,
        gy=0.2,
        gz=3.0,
        roll=0.01,
        pitch=0.02,
        yaw=1.23,
        has_mag=True,
        mx=10.0,
        my=11.0,
        mz=12.0,
    )
    got = imushm.unpack_sample(payload, timestamp_ns=123)
    assert got.ax == pytest.approx(1.0)
    assert got.gz == pytest.approx(3.0)
    assert got.yaw == pytest.approx(1.23)
    assert got.has_mag is True
    assert got.mx == pytest.approx(10.0)
    assert got.timestamp_ns == 123


@pytest.mark.skipif(
    __import__("sys").platform != "linux",
    reason="POSIX shm writer zero-fill differs on macOS CI",
)
def test_imushm_roundtrip():
    name = "/viam-imu-test-navstack"
    w = imushm.Writer(name, region_size=4096)
    try:
        sample = imushm.ImuShmSample(
            ax=1.0,
            ay=2.0,
            az=9.8,
            gx=0.1,
            gy=0.2,
            gz=3.0,
            roll=0.01,
            pitch=0.02,
            yaw=1.23,
            has_mag=False,
        )
        w.write_sample(sample)
        r = imushm.Reader(name, region_size=4096)
        try:
            got = r.read_latest(max_age_s=2.0)
            assert got is not None
            assert got.ax == pytest.approx(1.0)
            assert got.gz == pytest.approx(3.0)
            assert got.yaw == pytest.approx(1.23)
        finally:
            r.close()
    finally:
        w.close()


def test_serial_ports_prefer_by_path_for_imu():
    from src.lidar.serial_ports import list_candidate_serial_ports

    # Just ensure both modes return lists (may be empty on macOS CI).
    a = list_candidate_serial_ports(prefer_cp210=True)
    b = list_candidate_serial_ports(prefer_cp210=False)
    assert isinstance(a, list)
    assert isinstance(b, list)


def test_parser_yaw_decode_matches_datasheet_and_exposes_raw():
    """Datasheet: Yaw = ((YawH<<8)|YawL)/32768*180 deg. 169.14 -> 178.86 must
    decode 1:1 (no extra scaling), and the raw u16 is exposed for verification."""
    parser = WitStreamParser()
    for deg in (169.14, 178.86, -102.31, 90.0):
        payload = b"\x00\x00\x00\x00" + _u16_pair(deg, 180.0) + b"\x00\x00"
        assert parser.feed(_frame(TYPE_ORIENT, payload)) == 1
        s = parser.sample
        assert s.yaw_deg_decoded == pytest.approx(deg, abs=0.01)
        assert math.degrees(s.yaw) == pytest.approx(deg, abs=0.01)
        assert s.yaw_raw_u16 == int(round(deg / 180.0 * 32768.0)) & 0xFFFF
    assert parser.sample.angle_packets == 4
    # A 90 deg physical turn must move decoded yaw 90 deg.
    a = scale_le_u16(*_u16_pair(169.14, 180.0), 180.0)
    b = scale_le_u16(*_u16_pair(169.14 + 90.0 - 360.0, 180.0), 180.0)
    assert ((b - a + 180.0) % 360.0) - 180.0 == pytest.approx(90.0, abs=0.01)


def test_parser_gyro_decode_raw_and_counts():
    parser = WitStreamParser()
    payload = b"\x00\x00\x00\x00" + _u16_pair(45.0, 2000.0) + b"\x00\x00"
    parser.feed(_frame(TYPE_GYRO, payload))
    s = parser.sample
    assert s.gz == pytest.approx(45.0, abs=0.1)
    assert s.gz_raw_u16 == int(round(45.0 / 2000.0 * 32768.0))
    assert s.gyro_packets == 1


def test_config_commands_wit_register_protocol():
    assert config_commands("keep") == []
    assert config_commands("6axis") == [CMD_UNLOCK, bytes.fromhex("ffaa240100")]
    assert config_commands("9axis") == [CMD_UNLOCK, bytes.fromhex("ffaa240000")]
    assert config_commands("6axis", zero_yaw=True) == [
        CMD_UNLOCK,
        bytes.fromhex("ffaa240100"),
        bytes.fromhex("ffaa010400"),
    ]
    assert config_commands("keep", zero_yaw=True) == [CMD_UNLOCK, bytes.fromhex("ffaa010400")]
    # Never saves to flash.
    assert all(c != bytes.fromhex("ffaa000000") for c in config_commands("6axis", zero_yaw=True))
    with pytest.raises(WitError):
        config_commands("3axis")


def test_wit_serial_configure_writes_sequence():
    from src.imu.wit_serial import WitSerial

    class _Port:
        def __init__(self):
            self.writes = []

        def write(self, b):
            self.writes.append(bytes(b))

        def flush(self):
            pass

    dev = WitSerial("/dev/null")
    dev._ser = _Port()  # noqa: SLF001
    n = dev.configure("6axis", zero_yaw=False)
    assert n == 2
    assert dev._ser.writes == [CMD_UNLOCK, bytes.fromhex("ffaa240100")]  # noqa: SLF001
