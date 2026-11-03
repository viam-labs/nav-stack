"""Per-phase tick timers land in progress + control_loop."""

from __future__ import annotations

from src.nav.control.sense_tick import PHASE_MS_KEYS, elapsed_ms
from src.nav.control.tick_output import _PROGRESS_KEYS


def test_phase_ms_keys_whitelisted_for_status():
    for key in PHASE_MS_KEYS:
        assert key in _PROGRESS_KEYS


def test_elapsed_ms_non_negative():
    from time import perf_counter

    started = perf_counter()
    ms = elapsed_ms(started)
    assert ms >= 0.0
    assert isinstance(ms, float)


def test_control_stats_includes_phase_ms():
    from src.nav.control.supervisor import NavSupervisor

    # Minimal stub: only the fields control_stats / _note_phase_ms need.
    class _Stub:
        _local_update_period_s = 0.2
        _local_costmap_updates = 0
        _global_costmap_rebuilds = 0
        _control_ticks = 0
        _control_overruns = 0
        _control_tick_last_s = None
        _control_tick_ema_s = None
        _control_period_s = 0.1
        _phase_ms_last: dict = {}
        _global_cache_lock = __import__("threading").Lock()
        _global_cache_generation = None

        control_stats = NavSupervisor.control_stats
        _note_phase_ms = NavSupervisor._note_phase_ms

    stub = _Stub()
    assert stub.control_stats()["phase_ms"] is None
    stub._note_phase_ms(
        {
            "sense_ms": 1.2,
            "costmap_ms": 3.4,
            "guard_ms": 0.5,
            "dwa_ms": 8.0,
            "gate_ms": 0.3,
        }
    )
    phase = stub.control_stats()["phase_ms"]
    assert phase == {
        "sense_ms": 1.2,
        "costmap_ms": 3.4,
        "guard_ms": 0.5,
        "dwa_ms": 8.0,
        "gate_ms": 0.3,
    }
