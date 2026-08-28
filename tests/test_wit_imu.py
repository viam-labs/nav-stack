"""Tests for WitMotion protocol, IMU shm, and wit-imu helpers."""
from __future__ import annotations

import math
import struct

import pytest

from src.imu.wit_protocol import (
    TYPE_ACCEL,
    TYPE_GYRO,
    TYPE_ORIENT,
    WitStreamParser,
    scale_le_u16,
)
from src.ros import imushm


def _frame(typ: int, values_le: bytes) -> bytes:
    assert len(values_le) == 8
    body = bytes([0x55, typ]) + values_le
    checksum = sum(body) & 0xFF
    return body + bytes([checksum])


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
