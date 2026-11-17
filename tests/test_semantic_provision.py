"""The vision runtime is installed and downloaded only when vision is enabled."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import SlamConfig
from src.semantic import provision
from tools import fetch_dinov2_onnx, fetch_yolo_world_onnx


def _cfg(**extra) -> SlamConfig:
    d = {
        "base": "b",
        "lidars": [
            {"name": "front"},
            {
                "name": "depth-cam",
                "obstacles_only": True,
                "cloud_frame": "camera_optical",
                "scan_source": "point_cloud",
            },
        ],
    }
    d.update(extra)
    return SlamConfig.from_dict(d)


def _vision_cfg(**extra) -> SlamConfig:
    return _cfg(vision_enabled=True, vision_cameras=["depth-cam"], **extra)


class _Runner:
    """Stands in for the subprocess runner; answers by what the command is."""

    def __init__(self, **answers) -> None:
        self.commands: list = []
        self._answers = answers

    def __call__(self, cmd, timeout_s):
        cmd = [str(part) for part in cmd]
        self.commands.append(cmd)
        if "pip" in cmd:
            key = "pip_cpu" if provision.CPU_RUNTIME in cmd else "pip"
        elif cmd[1].endswith("fetch_dinov2_onnx.py"):
            key = "dinov2"
        else:
            key = "yolo_world"
        return self._answers.get(key, (0, "already present: x"))

    def ran(self, key: str) -> bool:
        names = {
            "pip": "requirements-vision.txt",
            "pip_cpu": provision.CPU_RUNTIME,
            "dinov2": "fetch_dinov2_onnx.py",
            "yolo_world": "fetch_yolo_world_onnx.py",
        }
        return any(
            any(part.endswith(names[key]) for part in cmd) for cmd in self.commands
        )


def test_nothing_is_prepared_while_vision_is_off_or_synthetic():
    run = _Runner()
    assert provision.uses_models(_cfg()) is False
    assert provision.prepare(_cfg(), run=run) == {"state": "ready", "steps": {}}
    synthetic = _vision_cfg(vision_synthetic=True)
    assert provision.uses_models(synthetic) is False
    assert provision.prepare(synthetic, run=run) == {"state": "ready", "steps": {}}
    assert run.commands == []


def test_first_use_installs_the_runtime_and_fetches_the_weights():
    state = {"installed": False}
    run = _Runner(pip=(0, ""), dinov2=(0, "saved /m/dinov2.onnx (88.3 MB)"))
    real = run.__call__

    def running(cmd, timeout_s):
        out = real(cmd, timeout_s)
        if "pip" in [str(p) for p in cmd]:
            state["installed"] = True
        return out

    log: list = []
    report = provision.prepare(
        _vision_cfg(), log.append, run=running, installed=lambda: state["installed"]
    )
    assert report == {
        "state": "ready",
        "steps": {
            "onnxruntime": "installed",
            "dinov2": "fetched",
            "yolo_world": "present",
        },
    }
    assert run.ran("pip") and not run.ran("pip_cpu")
    # A failed download on a robot must not turn into a torch export.
    fetches = [cmd for cmd in run.commands if "pip" not in cmd]
    assert len(fetches) == 2 and all(cmd[-1] == "--no-export" for cmd in fetches)
    assert any("installing ONNX Runtime" in line for line in log)


def test_runtime_already_installed_is_left_alone():
    run = _Runner()
    report = provision.prepare(_vision_cfg(), run=run, installed=lambda: True)
    assert report["steps"]["onnxruntime"] == "present" and report["state"] == "ready"
    assert not run.ran("pip") and not run.ran("pip_cpu")


def test_the_cpu_build_is_tried_when_the_openvino_build_does_not_install():
    state = {"installed": False}
    run = _Runner(pip=(1, "ERROR: no matching distribution"), pip_cpu=(0, ""))
    real = run.__call__

    def running(cmd, timeout_s):
        out = real(cmd, timeout_s)
        state["installed"] = provision.CPU_RUNTIME in [str(p) for p in cmd]
        return out

    report = provision.prepare(
        _vision_cfg(), run=running, installed=lambda: state["installed"]
    )
    assert report["steps"]["onnxruntime"] == "installed"
    assert run.ran("pip") and run.ran("pip_cpu")


def test_a_failed_step_is_reported_and_the_others_still_run():
    run = _Runner(
        pip=(1, "no network"),
        pip_cpu=(1, "no network"),
        dinov2=(1, "download failed (timed out)"),
    )
    report = provision.prepare(_vision_cfg(), run=run, installed=lambda: False)
    assert report["state"] == "failed"
    assert report["steps"]["onnxruntime"] == "failed: no network"
    assert report["steps"]["dinov2"] == "failed: download failed (timed out)"
    assert report["steps"]["yolo_world"] == "present"


def test_a_model_path_set_in_the_config_is_not_fetched():
    run = _Runner()
    cfg = _vision_cfg(
        vision_dinov2_onnx="/models/mine.onnx", vision_yolo_onnx="/models/yolo.onnx"
    )
    report = provision.prepare(cfg, run=run, installed=lambda: True)
    assert report == {"state": "ready", "steps": {"onnxruntime": "present"}}
    assert run.commands == []


def test_a_step_that_hangs_is_cut_off_and_reports_its_last_line():
    hang = [sys.executable, "-c", "import time; time.sleep(30)"]
    code, tail = provision._run(hang, 0.5)  # noqa: SLF001
    assert code != 0 and "timed out" in tail
    chatty = [sys.executable, "-c", "print('downloading x'); print('saved x (1.0 MB)')"]
    assert provision._run(chatty, 30.0) == (0, "saved x (1.0 MB)")  # noqa: SLF001


def test_a_cancel_stops_the_running_step_and_skips_the_rest():
    cancel = threading.Event()
    hang = [sys.executable, "-c", "import time; time.sleep(30)"]
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    assert provision._run(hang, 30.0, cancel) == (130, "cancelled")  # noqa: SLF001
    assert time.monotonic() - started < 5.0

    run = _Runner()
    report = provision.prepare(
        _vision_cfg(), cancel=cancel, run=run, installed=lambda: True
    )
    assert report == {"state": "cancelled", "steps": {}} and run.commands == []


def test_vision_can_start_once_the_runtime_and_dinov2_are_in_place():
    waiting = {"steps": {"onnxruntime": "present", "dinov2": "failed: no route"}}
    assert provision.essentials_ready(waiting) is False
    no_mask_yet = {
        "steps": {
            "onnxruntime": "installed",
            "dinov2": "fetched",
            "yolo_world": "failed: 404",
        }
    }
    assert provision.essentials_ready(no_mask_yet) is True
    assert provision.step_ok(no_mask_yet, "yolo_world") is False
    assert provision.step_ok(no_mask_yet, "dinov2") is True
    # A model path from the config has no step of its own.
    assert provision.essentials_ready({"steps": {"onnxruntime": "present"}}) is True
    assert (
        provision.step_ok({"steps": {"onnxruntime": "present"}}, "yolo_world") is False
    )


# --- The fetchers the prepare step runs ----------------------------------------


def _no_export(monkeypatch, tool) -> list:
    calls: list = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    return calls


def test_dinov2_download_lands_complete_or_not_at_all(tmp_path: Path, monkeypatch):
    exports = _no_export(monkeypatch, fetch_dinov2_onnx)
    good = tmp_path / "release.onnx"
    good.write_bytes(b"x" * 1_200_000)
    out = tmp_path / "models" / "dinov2.onnx"
    assert (
        fetch_dinov2_onnx.main(
            ["--out", str(out), "--url", good.as_uri(), "--no-export"]
        )
        == 0
    )
    assert out.stat().st_size == 1_200_000
    assert list(out.parent.glob("*.partial")) == []
    # Already there: no second download.
    assert (
        fetch_dinov2_onnx.main(
            ["--out", str(out), "--url", "file:///nowhere", "--no-export"]
        )
        == 0
    )

    truncated = tmp_path / "truncated.onnx"
    truncated.write_bytes(b"x" * 10)
    other = tmp_path / "models" / "other.onnx"
    assert (
        fetch_dinov2_onnx.main(
            ["--out", str(other), "--url", truncated.as_uri(), "--no-export"]
        )
        == 1
    )
    assert (
        fetch_dinov2_onnx.main(
            ["--out", str(other), "--url", "file:///nowhere", "--no-export"]
        )
        == 1
    )
    assert not other.exists() and list(other.parent.glob("*.partial")) == []
    assert exports == []


def test_dinov2_export_fallback_is_only_for_a_manual_run(tmp_path: Path, monkeypatch):
    exports = _no_export(monkeypatch, fetch_dinov2_onnx)
    out = tmp_path / "dinov2.onnx"
    assert fetch_dinov2_onnx.main(["--out", str(out), "--url", "file:///nowhere"]) == 0
    assert len(exports) == 1 and "export_dinov2_onnx.py" in exports[0][1]


def test_yolo_world_failed_download_does_not_export_when_asked_not_to(
    tmp_path: Path, monkeypatch
):
    exports = _no_export(monkeypatch, fetch_yolo_world_onnx)
    out = tmp_path / "yolo.onnx"
    assert (
        fetch_yolo_world_onnx.main(
            ["--out", str(out), "--url", "file:///nowhere", "--no-export"]
        )
        == 1
    )
    assert not out.exists() and exports == []
    assert (
        fetch_yolo_world_onnx.main(["--out", str(out), "--url", "file:///nowhere"]) == 0
    )
    assert len(exports) == 1


# --- The SLAM service: when it prepares, and when it starts vision -------------


class _FakeVision:
    """What ``_launch_vision`` would put in ``slam._vision``."""

    def __init__(self) -> None:
        self.detector_loads = 0

    def load_detector(self) -> bool:
        self.detector_loads += 1
        return True

    def status(self) -> dict:
        return {"enabled": True, "provider": "test"}

    def stop(self) -> None:
        pass


def _slam(cfg: SlamConfig, monkeypatch):
    from src.models.slam import SlamService

    slam = SlamService("slam")
    slam._cfg = cfg  # noqa: SLF001
    slam._engine = MagicMock()  # noqa: SLF001
    launched: list = []

    def launch(loop):
        slam._vision = _FakeVision()  # noqa: SLF001
        launched.append(slam._vision)  # noqa: SLF001

    monkeypatch.setattr(slam, "_launch_vision", launch)
    # Retries are seconds apart in service; a test does not wait that long.
    monkeypatch.setattr(provision, "RETRY_FIRST_S", 0.01)
    monkeypatch.setattr(provision, "RETRY_MAX_S", 0.02)
    return slam, launched


def _scripted_prepare(monkeypatch, reports: list) -> list:
    """``provision.prepare`` answers with each report in turn; returns the call log."""
    calls: list = []

    def prepare(cfg, log=None, *, cancel=None):
        calls.append(threading.current_thread())
        return reports[min(len(calls), len(reports)) - 1]

    monkeypatch.setattr(provision, "prepare", prepare)
    return calls


def _status(slam) -> dict:
    return slam._cmd_vision_status({})  # noqa: SLF001


def test_vision_off_prepares_nothing_and_loads_nothing(monkeypatch):
    def unexpected(*_a, **_k):
        raise AssertionError("vision is off: nothing should be prepared")

    monkeypatch.setattr(provision, "prepare", unexpected)
    slam, launched = _slam(_cfg(), monkeypatch)

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001

    asyncio.run(go())
    assert launched == [] and slam._vision_prepare_task is None  # noqa: SLF001
    assert _status(slam) == {"enabled": False, "reason": "vision not running"}


def test_enabling_vision_prepares_off_the_loop_and_then_starts(monkeypatch):
    steps = {"onnxruntime": "installed", "dinov2": "fetched", "yolo_world": "fetched"}
    calls = _scripted_prepare(monkeypatch, [{"state": "ready", "steps": steps}])
    slam, launched = _slam(_vision_cfg(), monkeypatch)

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001
        assert launched == []
        assert _status(slam) == {
            "enabled": True,
            "reason": "preparing",
            "runtime": {"state": "preparing"},
        }
        await slam._vision_prepare_task  # noqa: SLF001
        return threading.current_thread()

    loop_thread = asyncio.run(go())
    assert len(launched) == 1 and len(calls) == 1 and calls[0] is not loop_thread
    assert _status(slam)["runtime"] == {"state": "ready", "steps": steps, "attempts": 1}
    assert launched[0].detector_loads == 0  # the supervisor loaded it at start


def test_a_failed_download_is_tried_again_without_anyone_asking(monkeypatch):
    offline = {
        "state": "failed",
        "steps": {"onnxruntime": "present", "dinov2": "failed: no route to host"},
    }
    online = {
        "state": "ready",
        "steps": {"onnxruntime": "present", "dinov2": "fetched"},
    }
    calls = _scripted_prepare(monkeypatch, [offline, offline, online])
    slam, launched = _slam(_vision_cfg(), monkeypatch)
    seen: list = []

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001
        while len(calls) < 2:
            await asyncio.sleep(0.005)
        seen.append((len(launched), _status(slam)))
        await slam._vision_prepare_task  # noqa: SLF001

    asyncio.run(go())
    # Without DINOv2 there is nothing to start: it waits, and says why.
    started, waiting = seen[0]
    assert started == 0 and waiting["reason"] == "preparing"
    assert waiting["runtime"]["state"] == "retrying"
    assert waiting["runtime"]["steps"]["dinov2"].startswith("failed")
    assert "retry_in_s" in waiting["runtime"]
    assert len(calls) == 3 and len(launched) == 1
    assert _status(slam)["runtime"]["state"] == "ready"
    assert _status(slam)["runtime"]["attempts"] == 3


def test_vision_starts_without_the_person_mask_and_loads_it_when_it_arrives(
    monkeypatch,
):
    base = {"onnxruntime": "present", "dinov2": "present"}
    no_detector = {
        "state": "failed",
        "steps": {**base, "yolo_world": "failed: HTTP 503"},
    }
    complete = {"state": "ready", "steps": {**base, "yolo_world": "fetched"}}
    calls = _scripted_prepare(monkeypatch, [no_detector, complete])
    slam, launched = _slam(_vision_cfg(), monkeypatch)

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001
        await slam._vision_prepare_task  # noqa: SLF001

    asyncio.run(go())
    # One supervisor throughout, so the keyframes it stored meanwhile are kept.
    assert len(calls) == 2 and len(launched) == 1
    assert launched[0].detector_loads == 1


def test_a_reconfigure_during_the_prepare_stops_it_and_starts_nothing(monkeypatch):
    cancels: list = []

    def prepare(cfg, log=None, *, cancel=None):
        cancels.append(cancel)
        cancel.wait(5.0)
        return {"state": "cancelled", "steps": {}}

    monkeypatch.setattr(provision, "prepare", prepare)
    slam, launched = _slam(_vision_cfg(), monkeypatch)

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001
        while not cancels:
            await asyncio.sleep(0.005)
        slam._stop_vision()  # what a reconfigure or close does first  # noqa: SLF001
        await asyncio.sleep(0.05)

    started = time.monotonic()
    asyncio.run(go())
    # The step in flight was told to stop, so nothing waited out its 5 s.
    assert cancels[0].is_set() and time.monotonic() - started < 3.0
    assert launched == []
    assert _status(slam) == {"enabled": False, "reason": "vision not running"}


def test_synthetic_vision_starts_at_once(monkeypatch):
    monkeypatch.setattr(provision, "prepare", pytest.fail)
    slam, launched = _slam(_vision_cfg(vision_synthetic=True), monkeypatch)

    async def go():
        slam._start_vision(asyncio.get_running_loop())  # noqa: SLF001

    asyncio.run(go())
    assert len(launched) == 1 and slam._vision_prepare_task is None  # noqa: SLF001


# --- The supervisor: a detector that arrives after it started ------------------


def test_detector_weights_that_arrive_late_are_loaded_into_the_running_supervisor(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    from src.geom.conversions import Pose2D
    from src.semantic import onnx_session
    from src.semantic.supervisor import VisionSupervisor

    onnx = tmp_path / "detector.onnx"
    pose = Pose2D(0.0, 0.0, 0.0)
    vision = VisionSupervisor(
        cfg=_vision_cfg(vision_synthetic=True, vision_yolo_onnx=str(onnx)),
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
    )
    # Synthetic tokens never get a detector.
    assert vision.load_detector() is False
    # As a supervisor running the real DINOv2 would be.
    vision._worker.cfg.synthetic = False  # noqa: SLF001
    assert vision.load_detector() is False  # the file is not there yet
    assert vision.status()["yolo_loaded"] is False

    loaded: list = []

    def make_session(path, **kwargs):
        loaded.append(path)
        return object(), SimpleNamespace(provider="CPUExecutionProvider")

    monkeypatch.setattr(onnx_session, "make_session", make_session)
    onnx.write_bytes(b"weights")
    (tmp_path / "detector.classes.json").write_text('["door", "person"]')
    assert vision.load_detector() is True
    status = vision.status()
    assert loaded == [str(onnx)] and status["yolo_loaded"] is True
    assert status["detector_classes"] == ["door", "person"]
    assert status["yolo_onnx"] == str(onnx)
