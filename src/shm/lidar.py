"""Latest-PCD client over POSIX shm (see :mod:`.pcshm`).

Keeps one mapped reader per ``shm_name`` so the ROS scan timer is a memcpy,
not ``shm_open``. Writer restarts (unlink + recreate) are detected and the
reader is reopened.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from . import pcshm


@dataclass
class ShmReadStats:
    hits: int = 0
    misses: int = 0
    stale_hits: int = 0
    fallbacks: int = 0
    errors: int = 0
    remaps: int = 0
    last_bytes: int = 0
    last_age_s: Optional[float] = None
    last_error: Optional[str] = None
    opened: bool = False

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_hits": self.stale_hits,
            "grpc_fallbacks": self.fallbacks,
            "errors": self.errors,
            "remaps": self.remaps,
            "last_bytes": self.last_bytes,
            "last_age_s": self.last_age_s,
            "last_error": self.last_error,
            "opened": self.opened,
        }


@dataclass
class _Slot:
    name: str
    region_size: int
    reader: Optional[pcshm.Reader] = None
    stats: ShmReadStats = field(default_factory=ShmReadStats)
    _logged_open: bool = False
    _logged_miss: bool = False
    _shm_object_id: Optional[int] = None


class ShmPointCloudClient:
    """Thread-safe cache of shm readers keyed by POSIX name."""

    def __init__(self, logger=None):
        self._logger = logger
        self._lock = threading.Lock()
        self._slots: Dict[str, _Slot] = {}

    def _log(self, level: str, msg: str, *args) -> None:
        log = self._logger
        if log is None:
            return
        fn = getattr(log, level, None)
        if fn is not None:
            fn(msg, *args)

    def close(self) -> None:
        with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            if slot.reader is not None:
                try:
                    slot.reader.close()
                except Exception:
                    pass

    def status(self) -> Dict[str, dict]:
        with self._lock:
            return {name: slot.stats.to_dict() for name, slot in self._slots.items()}

    def note_fallback(self, shm_name: str) -> None:
        key = pcshm.normalize_name(shm_name)
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot(name=key, region_size=pcshm.DEFAULT_REGION_SIZE)
                self._slots[key] = slot
            slot.stats.fallbacks += 1

    def _get_slot(self, key: str, region_size: int) -> _Slot:
        slot = self._slots.get(key)
        if slot is None or slot.region_size != region_size:
            if slot is not None:
                self._invalidate_reader(slot)
            slot = _Slot(name=key, region_size=region_size)
            self._slots[key] = slot
        return slot

    def _invalidate_reader(self, slot: _Slot) -> None:
        if slot.reader is not None:
            try:
                slot.reader.close()
            except Exception:
                pass
            slot.reader = None
            slot.stats.opened = False
        slot._shm_object_id = None

    def _mapping_stale(self, slot: _Slot) -> bool:
        """True when the writer unlinked/recreated the POSIX object."""
        if slot.reader is None:
            return False
        current = pcshm.object_identity(slot.name)
        if current is None:
            return slot._shm_object_id is not None
        if slot._shm_object_id is None:
            return False
        return current != slot._shm_object_id

    def _open_reader(self, slot: _Slot) -> Optional[pcshm.Reader]:
        try:
            reader = pcshm.open_reader(slot.name, slot.region_size)
        except FileNotFoundError:
            with self._lock:
                slot.stats.misses += 1
                slot.stats.last_error = "shm not created yet"
                slot.stats.opened = False
                if not slot._logged_miss:
                    slot._logged_miss = True
                    self._log(
                        "info",
                        "lidar shm %s not present yet; will retry",
                        slot.name,
                    )
            return None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                slot.stats.errors += 1
                slot.stats.last_error = repr(exc)
            return None
        with self._lock:
            slot.reader = reader
            slot._shm_object_id = pcshm.object_identity(slot.name)
            slot.stats.opened = True
            if not slot._logged_open:
                slot._logged_open = True
                self._log("info", "lidar reading shm %s", slot.name)
        return reader

    def _remap_reader(self, slot: _Slot, *, reason: str) -> None:
        with self._lock:
            slot.stats.remaps += 1
            slot.stats.last_error = reason
        self._invalidate_reader(slot)

    def try_read(
        self,
        shm_name: str,
        region_size: int = pcshm.DEFAULT_REGION_SIZE,
        *,
        max_age_s: Optional[float] = None,
    ) -> Optional[Tuple[bytes, float]]:
        """Return ``(pcd_bytes, age_s)`` or ``None`` if no complete frame yet."""
        key = pcshm.normalize_name(shm_name)
        with self._lock:
            slot = self._get_slot(key, region_size)

        for attempt in range(2):
            reader = slot.reader
            if reader is None:
                reader = self._open_reader(slot)
                if reader is None:
                    return None
            elif self._mapping_stale(slot):
                self._remap_reader(
                    slot, reason="writer recreated shm object (identity changed)"
                )
                continue

            try:
                payload, ts_ns = reader.read()
            except pcshm.NoFrameError:
                with self._lock:
                    slot.stats.misses += 1
                    slot.stats.last_error = "no complete frame"
                if attempt == 0:
                    self._remap_reader(
                        slot,
                        reason="no complete frame (remap after writer restart?)",
                    )
                    continue
                return None
            except (pcshm.TornReadError, Exception) as exc:
                with self._lock:
                    slot.stats.errors += 1
                    slot.stats.last_error = repr(exc)
                self._invalidate_reader(slot)
                if attempt == 0:
                    continue
                return None

            age_s = 0.0
            if ts_ns:
                age_s = max(0.0, (time.time_ns() - int(ts_ns)) / 1e9)
            if max_age_s is not None and max_age_s > 0.0 and age_s > max_age_s:
                with self._lock:
                    slot.stats.stale_hits += 1
                    slot.stats.last_age_s = age_s
                    slot.stats.last_error = (
                        f"frame too old ({age_s:.2f}s > max_age_s={max_age_s:.2f}s)"
                    )
                if attempt == 0:
                    self._remap_reader(
                        slot,
                        reason="stale frame (remap after writer restart?)",
                    )
                    continue
                return None

            with self._lock:
                slot.stats.hits += 1
                slot.stats.last_bytes = len(payload)
                slot.stats.last_age_s = age_s
                slot.stats.last_error = None
            return payload, age_s

        return None
