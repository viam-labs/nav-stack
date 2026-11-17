"""Put the vision runtime in place the first time vision is enabled.

Vision is off by default, so setup installs none of it. When a config turns
``vision_enabled`` on, the SLAM service calls :func:`prepare` on a worker
thread before it starts the vision supervisor:

- ONNX Runtime is pip-installed into the module's environment if missing.
- The default DINOv2 and YOLO-World weights are downloaded if missing (for
  YOLO-World, also if the file was baked from an older class list).

Each step is a subprocess with a time limit, so a stalled download or package
index cannot hold the module. A failed step is reported, never raised, and
the service runs :func:`prepare` again by itself (after ``RETRY_FIRST_S``,
doubling up to ``RETRY_MAX_S``) until every step has succeeded. Nothing has
to be run by hand: a robot that was offline gets vision once it is online.

Vision starts as soon as ONNX Runtime and DINOv2 are in place
(:func:`essentials_ready`). The detector only feeds the person mask, so it
may arrive later and is loaded into the running supervisor.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements-vision.txt"
# Used when the OpenVINO build in the requirements has no wheel for this host.
CPU_RUNTIME = "onnxruntime>=1.17.0"
DINOV2_FETCH = ROOT / "tools" / "fetch_dinov2_onnx.py"
YOLO_WORLD_FETCH = ROOT / "tools" / "fetch_yolo_world_onnx.py"
PIP_TIMEOUT_S = 900.0
# About 140 MB in all. Generous, because this never runs on the event loop.
FETCH_TIMEOUT_S = 1800.0
# After a failed attempt the service tries again by itself: first after
# RETRY_FIRST_S, then doubling, with waits capped at RETRY_MAX_S.
RETRY_FIRST_S = 30.0
RETRY_MAX_S = 600.0
# Steps vision cannot run without. The detector is not one of them.
ESSENTIAL_STEPS = ("onnxruntime", "dinov2")
# How often a running step is checked for its time limit and for a cancel.
_POLL_S = 0.25

# ``(command, timeout_s)`` → exit code and the last line it printed.
Runner = Callable[[Sequence[str], float], Tuple[int, str]]
Log = Optional[Callable[[str], None]]

# One install / download at a time, however many services ask.
_lock = threading.Lock()


def uses_models(cfg) -> bool:
    """True when this config runs the real models (enabled and not synthetic)."""
    return bool(cfg.vision_enabled) and not bool(cfg.vision_synthetic)


def essentials_ready(report: Dict[str, object]) -> bool:
    """True when nothing vision needs in order to start has failed."""
    steps = report.get("steps") or {}
    return not any(
        str(steps.get(name, "")).startswith("failed") for name in ESSENTIAL_STEPS
    )


def step_ok(report: Dict[str, object], name: str) -> bool:
    """True when ``name`` ran in this report and did not fail."""
    outcome = str((report.get("steps") or {}).get(name, ""))
    return bool(outcome) and not outcome.startswith("failed")


def runtime_installed() -> bool:
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except (ImportError, ValueError):
        return False


def _run(
    cmd: Sequence[str],
    timeout_s: float,
    cancel: Optional[threading.Event] = None,
) -> Tuple[int, str]:
    """Run one step; stop it when it overruns or ``cancel`` is set."""
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return 127, str(exc)
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            out, _ = proc.communicate(timeout=_POLL_S)
            break
        except subprocess.TimeoutExpired:
            cancelled = cancel is not None and cancel.is_set()
            if not cancelled and time.monotonic() < deadline:
                continue
            proc.kill()
            proc.communicate()
            if cancelled:
                return 130, "cancelled"
            return 124, f"timed out after {timeout_s:.0f} s"
    lines = [line.strip() for line in (out or "").splitlines() if line.strip()]
    return int(proc.returncode), (lines[-1] if lines else "")


def _say(log: Log, msg: str) -> None:
    if log is None:
        return
    try:
        log(msg)
    except Exception:  # noqa: BLE001
        pass


def _ensure_runtime(log: Log, run: Runner, installed: Callable[[], bool]) -> str:
    if installed():
        return "present"
    _say(log, "vision: installing ONNX Runtime (first use)")
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    code, tail = 1, f"{REQUIREMENTS.name} is not in this install"
    if REQUIREMENTS.is_file():
        code, tail = run([*pip, "-r", str(REQUIREMENTS)], PIP_TIMEOUT_S)
    if code != 0:
        code, tail = run([*pip, CPU_RUNTIME], PIP_TIMEOUT_S)
    importlib.invalidate_caches()
    if code == 0 and installed():
        _say(log, "vision: ONNX Runtime installed")
        return "installed"
    reason = tail or f"pip exit {code}"
    _say(log, f"vision: ONNX Runtime not installed ({reason})")
    return f"failed: {reason}"


def _fetch(script: Path, what: str, log: Log, run: Runner) -> str:
    if not script.is_file():
        return f"failed: {script.name} is not in this install"
    # --no-export: on a robot a failed download must not start a torch export.
    code, tail = run([sys.executable, str(script), "--no-export"], FETCH_TIMEOUT_S)
    if code == 0:
        if tail.startswith("saved"):
            _say(log, f"vision: {what} weights downloaded")
            return "fetched"
        return "present"
    reason = tail or f"exit {code}"
    _say(log, f"vision: {what} weights not fetched ({reason})")
    return f"failed: {reason}"


def prepare(
    cfg,
    log: Log = None,
    *,
    cancel: Optional[threading.Event] = None,
    run: Optional[Runner] = None,
    installed: Callable[[], bool] = runtime_installed,
) -> Dict[str, object]:
    """Install and download what ``cfg`` needs. Blocks: call it off the event loop.

    Returns ``{"state": "ready" | "failed" | "cancelled", "steps": {name:
    outcome}}`` where an outcome is ``present``, ``installed``, ``fetched`` or
    ``failed: why``. A model path set in the config is the operator's file
    and is not fetched. Setting ``cancel`` stops the running step and skips
    the rest (the service is closing or being reconfigured).
    """
    steps: Dict[str, str] = {}
    if not uses_models(cfg):
        return {"state": "ready", "steps": steps}
    if run is None:

        def run(cmd: Sequence[str], timeout_s: float) -> Tuple[int, str]:
            return _run(cmd, timeout_s, cancel)

    def stopped() -> bool:
        return cancel is not None and cancel.is_set()

    with _lock:
        if not stopped():
            steps["onnxruntime"] = _ensure_runtime(log, run, installed)
        if not stopped() and not str(cfg.vision_dinov2_onnx or "").strip():
            steps["dinov2"] = _fetch(DINOV2_FETCH, "DINOv2", log, run)
        if not stopped() and not str(cfg.vision_yolo_onnx or "").strip():
            steps["yolo_world"] = _fetch(YOLO_WORLD_FETCH, "YOLO-World", log, run)
    if stopped():
        return {"state": "cancelled", "steps": steps}
    failed = any(outcome.startswith("failed") for outcome in steps.values())
    return {"state": "failed" if failed else "ready", "steps": steps}
