"""Double-buffered POSIX shm for WitMotion / IMU samples.

Same slot header as ``pcshm`` (seq / timestamp_ns / nbytes / pad) so tooling
can share the ring machinery; payload is a fixed binary sample, not PCD.

    payload (little-endian):
      flags          u32   # bit0 = has_mag
      ax ay az       3xf32  # m/s^2
      gx gy gz       3xf32  # deg/s
      roll pitch yaw 3xf32  # rad
      mx my mz       3xf32  # µT
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional

from . import pcshm

SAMPLE_FORMAT = "<Iffffffffffff"
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)
FLAG_HAS_MAG = 1
DEFAULT_REGION_SIZE = 4096  # 2 × 2 KiB slots


@dataclass
class ImuShmSample:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    roll: float
    pitch: float
    yaw: float
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    has_mag: bool = False
    timestamp_ns: int = 0


def pack_sample(
    *,
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    roll: float,
    pitch: float,
    yaw: float,
    mx: float = 0.0,
    my: float = 0.0,
    mz: float = 0.0,
    has_mag: bool = False,
) -> bytes:
    flags = FLAG_HAS_MAG if has_mag else 0
    return struct.pack(
        SAMPLE_FORMAT,
        flags,
        float(ax),
        float(ay),
        float(az),
        float(gx),
        float(gy),
        float(gz),
        float(roll),
        float(pitch),
        float(yaw),
        float(mx),
        float(my),
        float(mz),
    )


def unpack_sample(payload: bytes, *, timestamp_ns: int = 0) -> ImuShmSample:
    if len(payload) < SAMPLE_SIZE:
        raise ValueError(f"imu shm payload too short: {len(payload)}")
    (
        flags,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        roll,
        pitch,
        yaw,
        mx,
        my,
        mz,
    ) = struct.unpack_from(SAMPLE_FORMAT, payload, 0)
    return ImuShmSample(
        ax=ax,
        ay=ay,
        az=az,
        gx=gx,
        gy=gy,
        gz=gz,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        mx=mx,
        my=my,
        mz=mz,
        has_mag=bool(flags & FLAG_HAS_MAG),
        timestamp_ns=int(timestamp_ns),
    )


class Writer:
    def __init__(self, name: str, region_size: int = DEFAULT_REGION_SIZE):
        self._w = pcshm.Writer(name, region_size=region_size)
        self.name = self._w.name

    def write_sample(self, sample: ImuShmSample, timestamp_ns: Optional[int] = None) -> None:
        payload = pack_sample(
            ax=sample.ax,
            ay=sample.ay,
            az=sample.az,
            gx=sample.gx,
            gy=sample.gy,
            gz=sample.gz,
            roll=sample.roll,
            pitch=sample.pitch,
            yaw=sample.yaw,
            mx=sample.mx,
            my=sample.my,
            mz=sample.mz,
            has_mag=sample.has_mag,
        )
        self._w.write(payload, timestamp_ns=timestamp_ns)

    def close(self) -> None:
        self._w.close()


class Reader:
    def __init__(self, name: str, region_size: int = DEFAULT_REGION_SIZE):
        self._r = pcshm.Reader(name, region_size=region_size)
        self.name = self._r.name

    def read_latest(self, *, max_age_s: float = 0.5) -> Optional[ImuShmSample]:
        try:
            payload, ts_ns = self._r.read()
        except (pcshm.NoFrameError, pcshm.TornReadError, FileNotFoundError):
            return None
        age = (time.time_ns() - ts_ns) / 1e9
        if max_age_s > 0 and age > max_age_s:
            return None
        return unpack_sample(payload, timestamp_ns=ts_ns)

    def close(self) -> None:
        self._r.close()
