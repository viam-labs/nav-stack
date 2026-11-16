"""Cache DINOv2 patch tokens per frame so descriptor experiments re-run fast.

Inference is the slow part (a few frames per second on a laptop CPU). Every
in-image token of every frame is stored once as float16; place descriptors
and frame-to-frame matching are then computed from the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.semantic.patches import select_survey_tokens
from src.semantic.types import PatchToken
from src.semantic.worker import VisionWorker, WorkerConfig, _letterbox

from .lilocbench import LilocSequence

DEFAULT_DINOV2 = (
    Path(__file__).resolve().parents[2]
    / "src" / "semantic" / "models" / "dinov2_vits14_patches_448.onnx"
)
# Tokens the runtime keeps per frame (``VisionWorker.process``).
RUNTIME_KEEP = 64


@dataclass
class FrameFeatures:
    """Tokens for every frame of one sequence.

    ``desc`` is ``(frames, tokens, dim)`` float16, L2-normalised. ``uv`` are
    the token centres in source pixels and ``grid`` their (row, col); both
    are the same for every frame. ``runtime_keep[f]`` lists the token indices
    the runtime would have kept for frame ``f`` (texture-first, -1 padded).
    """

    desc: np.ndarray
    uv: np.ndarray
    grid: np.ndarray
    runtime_keep: np.ndarray

    def tokens(self, frame: int) -> np.ndarray:
        return np.asarray(self.desc[frame], dtype=np.float32)

    def runtime_tokens(self, frame: int) -> np.ndarray:
        idx = self.runtime_keep[frame]
        return self.tokens(frame)[idx[idx >= 0]]


def make_worker(dinov2_onnx: Path = DEFAULT_DINOV2, *, input_size: int = 448) -> VisionWorker:
    worker = VisionWorker(
        WorkerConfig(
            dinov2_onnx=str(dinov2_onnx),
            input_short_side=int(input_size),
            cpu_threads=4,
        )
    )
    worker._load_models()  # noqa: SLF001
    if worker.cfg.synthetic or worker._dino is None:  # noqa: SLF001
        raise RuntimeError(
            f"DINOv2 did not load from {dinov2_onnx}: {worker.stats.get('last_error')}"
        )
    return worker


def frame_tokens(worker: VisionWorker, color: np.ndarray) -> List[PatchToken]:
    """All in-image tokens for one frame (no texture cull)."""
    size = int(worker.cfg.input_short_side)
    canvas, scale, pad_x, pad_y = _letterbox(color, size)
    tokens, _place = worker._dino_tokens(  # noqa: SLF001
        canvas, scale=scale, pad_x=pad_x, pad_y=pad_y
    )
    return tokens


def ensure_features(
    seq: LilocSequence,
    cache_root: Path,
    *,
    worker: Optional[VisionWorker] = None,
    log=print,
) -> FrameFeatures:
    """Load the token cache for ``seq``, computing it first when missing."""
    folder = Path(cache_root).expanduser() / seq.name / f"camera_{seq.camera}"
    meta_path = folder / "meta.json"
    desc_path = folder / "tokens_f16.npy"
    n = len(seq)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("frames", -1)) == n and meta.get("first_stamp") == seq.stamps[0]:
            return FrameFeatures(
                desc=np.load(desc_path, mmap_mode="r"),
                uv=np.load(folder / "uv.npy"),
                grid=np.load(folder / "grid.npy"),
                runtime_keep=np.load(folder / "runtime_keep.npy"),
            )
    if worker is None:
        worker = make_worker()
    folder.mkdir(parents=True, exist_ok=True)
    desc = None
    keep = np.full((n, RUNTIME_KEEP), -1, dtype=np.int16)
    uv = grid = None
    for i in range(n):
        color = seq.color(i)
        tokens = frame_tokens(worker, color)
        if desc is None:
            dim = int(tokens[0].descriptor.shape[0])
            desc = np.lib.format.open_memmap(
                desc_path, mode="w+", dtype=np.float16, shape=(n, len(tokens), dim)
            )
            uv = np.array([[t.u, t.v] for t in tokens], dtype=np.float32)
            grid = np.array([[t.row, t.col] for t in tokens], dtype=np.int16)
        desc[i] = np.stack([t.descriptor for t in tokens]).astype(np.float16)
        index_of = {(t.row, t.col): j for j, t in enumerate(tokens)}
        kept = select_survey_tokens(color, tokens, min_texture=8.0, max_keep=RUNTIME_KEEP)
        for j, tok in enumerate(kept[:RUNTIME_KEEP]):
            keep[i, j] = index_of[(tok.row, tok.col)]
        if (i + 1) % 100 == 0 or i + 1 == n:
            log(f"  {seq.name}: tokens {i + 1}/{n}")
    desc.flush()
    np.save(folder / "uv.npy", uv)
    np.save(folder / "grid.npy", grid)
    np.save(folder / "runtime_keep.npy", keep)
    meta_path.write_text(json.dumps({"frames": n, "first_stamp": seq.stamps[0]}))
    return FrameFeatures(
        desc=np.load(desc_path, mmap_mode="r"), uv=uv, grid=grid, runtime_keep=keep
    )
