"""Double-buffered POSIX shared-memory ring for PCD frames.

Wire format matches ``viam-shared-memory-test`` (Go ``internal/pcshm`` and
Python ``pcshm``) so a lidar module that publishes that layout can feed
nav-stack without gRPC ``GetPointCloud``.

    region = slot0 | slot1
    each slot:
      seq          uint64  # odd while writing, even when complete; 0 = empty
      timestamp_ns uint64
      nbytes       uint32
      pad          uint32
      payload      [slotSize-24] bytes  # raw PCD bytes

Names are POSIX shm_open names such as ``/viam-pc-lidar``.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

HEADER_FORMAT = "<QQII"  # seq, timestamp_ns, nbytes, pad
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
NUM_SLOTS = 2
DEFAULT_REGION_SIZE = 2 * 1024 * 1024


class NoFrameError(RuntimeError):
    pass


class TornReadError(RuntimeError):
    pass


def normalize_name(name: str) -> str:
    name = name.strip()
    if name.startswith("/"):
        return name
    return "/" + name


def object_identity(name: str) -> Optional[int]:
    """Return a Linux ``/dev/shm`` inode when the object exists, else ``None``."""
    if sys.platform != "linux":
        return None
    path = f"/dev/shm/{_python_shm_name(name)}"
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


def _python_shm_name(name: str) -> str:
    # multiprocessing.SharedMemory prepends "/" on POSIX. Passing a name that
    # already has a slash would become "//foo" and miss the Go shm_open object.
    return normalize_name(name).lstrip("/")


def _try_unlink(name: str) -> None:
    try:
        existing = shared_memory.SharedMemory(name=_python_shm_name(name))
    except FileNotFoundError:
        return
    try:
        existing.close()
        existing.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        try:
            existing.close()
        except Exception:
            pass


def _open_shm(name: str, create: bool, size: int) -> shared_memory.SharedMemory:
    kwargs = {"name": name, "create": create}
    if create:
        kwargs["size"] = size
    try:
        return shared_memory.SharedMemory(**kwargs, track=False)
    except TypeError:
        return shared_memory.SharedMemory(**kwargs)


class Writer:
    def __init__(self, name: str, region_size: int = DEFAULT_REGION_SIZE):
        if region_size % NUM_SLOTS != 0:
            raise ValueError(f"region size {region_size} not divisible by {NUM_SLOTS}")
        self.name = normalize_name(name)
        self.region_size = region_size
        self.slot_size = region_size // NUM_SLOTS
        self._next = 0
        py_name = _python_shm_name(self.name)
        _try_unlink(py_name)
        self._shm = _open_shm(py_name, create=True, size=region_size)
        self._buf = self._shm.buf
        self._buf[:] = b"\x00" * region_size

    def max_payload(self) -> int:
        return self.slot_size - HEADER_SIZE

    def write(self, payload: bytes, timestamp_ns: Optional[int] = None) -> None:
        if len(payload) > self.max_payload():
            raise ValueError(f"payload {len(payload)} exceeds max {self.max_payload()}")
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        slot = self._next
        self._next = 1 - slot
        off = slot * self.slot_size
        seq = struct.unpack_from("<Q", self._buf, off)[0]
        if seq % 2 == 0:
            seq += 1
        else:
            seq += 2
        struct.pack_into("<Q", self._buf, off, seq)
        n = len(payload)
        self._buf[off + HEADER_SIZE : off + HEADER_SIZE + n] = payload
        struct.pack_into("<QII", self._buf, off + 8, timestamp_ns, n, 0)
        struct.pack_into("<Q", self._buf, off, seq + 1)

    def close(self) -> None:
        if self._shm is None:
            return
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass
        self._shm = None
        self._buf = None

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Reader:
    def __init__(self, name: str, region_size: int = DEFAULT_REGION_SIZE):
        if region_size % NUM_SLOTS != 0:
            raise ValueError(f"region size {region_size} not divisible by {NUM_SLOTS}")
        self.name = normalize_name(name)
        self.region_size = region_size
        self.slot_size = region_size // NUM_SLOTS
        self._shm = _open_shm(_python_shm_name(self.name), create=False, size=region_size)
        self._buf = self._shm.buf

    def read(self) -> Tuple[bytes, int]:
        for _ in range(64):
            best_slot = -1
            best_seq = 0
            for s in range(NUM_SLOTS):
                off = s * self.slot_size
                seq = struct.unpack_from("<Q", self._buf, off)[0]
                if seq != 0 and seq % 2 == 0 and seq >= best_seq:
                    best_seq = seq
                    best_slot = s
            if best_slot < 0:
                raise NoFrameError("pcshm: no complete frame")
            off = best_slot * self.slot_size
            _, ts_ns, nbytes, _ = struct.unpack_from(HEADER_FORMAT, self._buf, off)
            if nbytes > self.slot_size - HEADER_SIZE:
                raise RuntimeError(f"pcshm: corrupt nbytes {nbytes}")
            payload = bytes(self._buf[off + HEADER_SIZE : off + HEADER_SIZE + nbytes])
            seq2 = struct.unpack_from("<Q", self._buf, off)[0]
            if seq2 == best_seq:
                return payload, ts_ns
        raise TornReadError("pcshm: torn read after retries")

    def close(self) -> None:
        if self._shm is None:
            return
        try:
            self._shm.close()
        except Exception:
            pass
        self._shm = None
        self._buf = None

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_writer(name: str, region_size: int = DEFAULT_REGION_SIZE) -> Writer:
    return Writer(name, region_size)


def open_reader(name: str, region_size: int = DEFAULT_REGION_SIZE) -> Reader:
    return Reader(name, region_size)
