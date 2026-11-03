"""Run the scan-to-map matcher in a dedicated subprocess.

``global_localize_scan`` / ``score_pose_with_rays`` / ``choose_yaw_or_flip``
are Python-loop heavy. Running them in a thread pool still holds the module
GIL for seconds, which starves everything else in-process: the nav control
tick (drive timeouts), the SLAM tick (odom read gaps → pose lag), and the
gRPC event loop itself. A subprocess puts that work on another core and out
of the GIL entirely.

Design:

* One long-lived worker (``spawn`` start method — safe with threads + gRPC).
* The occupancy map is shipped once per content hash and cached in the
  worker; each call sends only the token + scan + parameters.
* Callers get a ``concurrent.futures.Future`` (or ``await`` via ``run``).
* Any subprocess failure (spawn error, broken pool, pickling) falls back to
  running the function in-process so localization never stops working.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import multiprocessing
import threading
import time
from typing import Any, Callable, Dict, Optional

from . import global_localize as gl
from .global_localize import OccupancyMap

_FUNCS: Dict[str, Callable[..., Any]] = {
    "global_localize_scan": gl.global_localize_scan,
    "score_pose_with_rays": gl.score_pose_with_rays,
    "choose_yaw_or_flip": gl.choose_yaw_or_flip,
}

# ---- worker side ---------------------------------------------------------

_MAP_CACHE: Dict[str, OccupancyMap] = {}
_MAP_CACHE_MAX = 3


class MapNotCached(Exception):
    """Worker does not hold the requested map token; caller must resend."""


def _worker_call(
    name: str,
    map_token: str,
    occ_map: Optional[OccupancyMap],
    args: tuple,
    kwargs: dict,
):
    if occ_map is not None:
        if len(_MAP_CACHE) >= _MAP_CACHE_MAX:
            _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        _MAP_CACHE[map_token] = occ_map
    cached = _MAP_CACHE.get(map_token)
    if cached is None:
        raise MapNotCached(map_token)
    return _FUNCS[name](cached, *args, **kwargs)


def _worker_ping() -> bool:
    return True


# ---- caller side ---------------------------------------------------------


def map_token(occ_map: OccupancyMap) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(occ_map.grid.tobytes())
    h.update(
        f"{occ_map.grid.shape}|{occ_map.resolution:.6f}|"
        f"{occ_map.origin_x:.6f}|{occ_map.origin_y:.6f}".encode()
    )
    return h.hexdigest()


class LocalizeWorker:
    """Subprocess-backed matcher with in-process fallback."""

    def __init__(self, *, enabled: bool = True, logger=None, call_timeout_s: float = 90.0):
        self._enabled = bool(enabled)
        self._logger = logger
        self._call_timeout_s = float(call_timeout_s)
        self._lock = threading.Lock()
        self._pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._sent_tokens: set[str] = set()
        self._fallback_logged = False
        self.stats: Dict[str, Any] = {
            "enabled": self._enabled,
            "subprocess_calls": 0,
            "fallback_calls": 0,
            "pool_restarts": 0,
            "last_call_s": None,
            "last_error": None,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- lifecycle ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self._logger is None:
            return
        try:
            self._logger.info(msg)
        except Exception:  # noqa: BLE001
            try:
                self._logger(msg)
            except Exception:  # noqa: BLE001
                pass

    def _ensure_pool(self) -> Optional[concurrent.futures.ProcessPoolExecutor]:
        if not self._enabled:
            return None
        with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                ctx = multiprocessing.get_context("spawn")
                self._pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=ctx
                )
                self._sent_tokens = set()
                self._log("localize worker: subprocess pool started")
            except Exception as exc:  # noqa: BLE001
                self.stats["last_error"] = f"spawn: {exc}"
                self._enabled = False
                self.stats["enabled"] = False
                self._log(f"localize worker: cannot start subprocess ({exc}); in-process")
                return None
            return self._pool

    def _kill_pool(self, reason: str) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
            self._sent_tokens = set()
        if pool is None:
            return
        self.stats["pool_restarts"] += 1
        self.stats["last_error"] = reason
        self._log(f"localize worker: restarting subprocess ({reason})")
        try:
            for proc in list(getattr(pool, "_processes", {}).values()):
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass

    def warm(self) -> None:
        """Spawn the worker ahead of the first real call (import cost ~0.5-1 s)."""
        pool = self._ensure_pool()
        if pool is None:
            return
        try:
            pool.submit(_worker_ping)
        except Exception as exc:  # noqa: BLE001
            self._kill_pool(f"warm: {exc}")

    def close(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    # -- calls -------------------------------------------------------------
    def call(self, name: str, occ_map: OccupancyMap, *args, **kwargs):
        """Blocking call. Prefers the subprocess; falls back in-process."""
        fn = _FUNCS[name]
        pool = self._ensure_pool()
        if pool is None:
            return self._fallback(fn, occ_map, args, kwargs)

        token = map_token(occ_map)
        t0 = time.monotonic()
        for attempt in range(2):
            send_map = token not in self._sent_tokens
            try:
                fut = pool.submit(
                    _worker_call,
                    name,
                    token,
                    occ_map if send_map else None,
                    args,
                    kwargs,
                )
                result = fut.result(timeout=self._call_timeout_s)
            except MapNotCached:
                # Worker restarted or evicted; resend on the next attempt.
                self._sent_tokens.discard(token)
                continue
            except concurrent.futures.TimeoutError:
                self._kill_pool(f"{name} exceeded {self._call_timeout_s:.0f}s")
                return self._fallback(fn, occ_map, args, kwargs)
            except concurrent.futures.process.BrokenProcessPool as exc:
                self._kill_pool(f"broken pool: {exc}")
                pool = self._ensure_pool()
                if pool is None or attempt == 1:
                    return self._fallback(fn, occ_map, args, kwargs)
                continue
            except Exception as exc:  # noqa: BLE001
                # Matcher raised (e.g. no returns) — same exception in-process.
                if isinstance(exc, (ValueError, RuntimeError)):
                    raise
                self._kill_pool(f"{name}: {exc}")
                return self._fallback(fn, occ_map, args, kwargs)
            self._sent_tokens.add(token)
            self.stats["subprocess_calls"] += 1
            self.stats["last_call_s"] = round(time.monotonic() - t0, 3)
            return result
        return self._fallback(fn, occ_map, args, kwargs)

    def _fallback(self, fn, occ_map, args, kwargs):
        if not self._fallback_logged:
            self._fallback_logged = True
            self._log("localize worker: running matcher in-process (fallback)")
        self.stats["fallback_calls"] += 1
        t0 = time.monotonic()
        try:
            return fn(occ_map, *args, **kwargs)
        finally:
            self.stats["last_call_s"] = round(time.monotonic() - t0, 3)

    async def run(self, name: str, occ_map: OccupancyMap, *args, **kwargs):
        """Awaitable wrapper; the blocking wait happens on a helper thread."""
        return await asyncio.to_thread(self.call, name, occ_map, *args, **kwargs)
