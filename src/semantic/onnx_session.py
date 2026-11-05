"""ONNX Runtime session helpers with OpenVINO / CUDA / CPU provider order."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class SessionInfo:
    path: str
    providers: List[str]
    provider: str


def available_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return []
    return list(ort.get_available_providers())


def preferred_providers(
    *,
    prefer_openvino: bool = True,
    prefer_cuda: bool = True,
    cpu_threads: int = 2,
) -> List[str]:
    """Provider list in try order. OpenVINO GPU first on Intel NUCs."""
    del cpu_threads  # applied in session options
    avail = set(available_providers())
    ordered: List[str] = []
    if prefer_openvino and "OpenVINOExecutionProvider" in avail:
        ordered.append("OpenVINOExecutionProvider")
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        ordered.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in avail:
        ordered.append("CPUExecutionProvider")
    return ordered


def make_session(
    model_path: str,
    *,
    providers: Optional[Sequence[str]] = None,
    cpu_threads: int = 2,
) -> tuple[Any, SessionInfo]:
    """Create an InferenceSession. Raises if onnxruntime or the file is missing."""
    import onnxruntime as ort  # type: ignore

    path = os.path.abspath(model_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(cpu_threads))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = list(providers) if providers is not None else preferred_providers()
    if not prov:
        prov = ["CPUExecutionProvider"]
    # Try providers in order; fall back if a provider fails to load.
    last_err: Optional[BaseException] = None
    session = None
    used: List[str] = []
    for i in range(len(prov)):
        try_list = prov[i:]
        try:
            session = ort.InferenceSession(path, sess_options=opts, providers=try_list)
            used = list(session.get_providers())
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            session = None
    if session is None:
        raise RuntimeError(f"failed to load ONNX session for {path}: {last_err}")
    info = SessionInfo(path=path, providers=used, provider=used[0] if used else "none")
    return session, info


def openvino_gpu_available() -> bool:
    return "OpenVINOExecutionProvider" in available_providers()
