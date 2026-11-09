"""Vision worker: DINOv2 (+ optional YOLO) inference, latest-frame only."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .detect_landmarks import tokens_inside_detections
from .patches import patch_grid, select_survey_tokens
from .place import l2_normalize
from .types import Detection, PatchToken, VisionFrame, VisionResult

# Standard COCO-80 order (YOLOv8 class ids). Used when detector_classes is empty.
COCO80_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass
class WorkerConfig:
    dinov2_onnx: str = ""
    yolo_onnx: str = ""
    input_short_side: int = 224  # bench-friendly; plan target is 448 wide
    patch: int = 14
    cpu_threads: int = 2
    # When True (or no dinov2 path), emit synthetic descriptors for benches/tests.
    synthetic: bool = False
    yolo_input: int = 640
    yolo_conf: float = 0.15
    yolo_nms_iou: float = 0.5
    # Class-id → name for the loaded ONNX. Empty → COCO-80 (standard YOLOv8).
    # YOLO-World exports must pass the same prompt list used at export time.
    detector_classes: Tuple[str, ...] = ()
    # Paint-gate / dynamic obstacle labels (always kept when detected).
    dynamic_labels: Tuple[str, ...] = ("person", "bicycle", "car", "motorcycle")
    # When non-empty, also keep these labels (landmark proposals). Empty = keep all.
    landmark_labels: Tuple[str, ...] = field(default_factory=tuple)


class VisionWorker:
    """In-process vision inference. Subprocess isolation can wrap this later."""

    def __init__(self, cfg: Optional[WorkerConfig] = None):
        self.cfg = cfg or WorkerConfig()
        self._dino = None
        self._dino_info = None
        self._yolo = None
        self._yolo_info = None
        self._provider = "synthetic" if self.cfg.synthetic else "none"
        self.stats = {
            "frames": 0,
            "infer_ms_ema": None,
            "provider": self._provider,
            "last_error": None,
            "yolo_loaded": False,
            "yolo_provider": None,
            "last_yolo_raw": 0,
            "last_yolo_kept": 0,
            "last_yolo_max_conf": None,
            "last_yolo_out_shape": None,
        }
        if not self.cfg.synthetic and self.cfg.dinov2_onnx:
            self._load_models()

    def _load_models(self) -> None:
        from .onnx_session import make_session

        try:
            self._dino, self._dino_info = make_session(
                self.cfg.dinov2_onnx, cpu_threads=self.cfg.cpu_threads
            )
            self._provider = self._dino_info.provider
            self.stats["provider"] = self._provider
        except Exception as exc:  # noqa: BLE001
            self.stats["last_error"] = f"dinov2: {exc}"
            self.cfg.synthetic = True
            self._provider = "synthetic"
            self.stats["provider"] = self._provider
            return
        if self.cfg.yolo_onnx:
            try:
                # Keep YOLO-World on CPU FP32. OpenVINO GPU FP16 (used for
                # DINOv2) collapses open-vocab class scores to ~0 on Meerkat.
                self._yolo, self._yolo_info = make_session(
                    self.cfg.yolo_onnx,
                    cpu_threads=self.cfg.cpu_threads,
                    prefer_openvino=False,
                    prefer_cuda=False,
                )
                self.stats["yolo_loaded"] = True
                self.stats["yolo_provider"] = self._yolo_info.provider
            except Exception as exc:  # noqa: BLE001
                self.stats["last_error"] = f"yolo: {exc}"
                self._yolo = None
                self.stats["yolo_loaded"] = False
                self.stats["yolo_provider"] = None
        else:
            self.stats["yolo_loaded"] = False
            self.stats["yolo_provider"] = None

    def close(self) -> None:
        self._dino = None
        self._yolo = None
        self.stats["yolo_loaded"] = False

    def process(self, frame: VisionFrame) -> VisionResult:
        t0 = time.perf_counter()
        color = np.asarray(frame.color_rgb)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"color_rgb must be HxWx3, got {color.shape}")
        h, w = color.shape[:2]

        if self.cfg.synthetic or self._dino is None:
            # Aspect-preserving resize is fine for synthetic RGB-stat descriptors.
            short = int(self.cfg.input_short_side)
            scale = short / float(min(h, w))
            new_h = max(self.cfg.patch, int(round(h * scale)))
            new_w = max(self.cfg.patch, int(round(w * scale)))
            new_h = max(self.cfg.patch, (new_h // self.cfg.patch) * self.cfg.patch)
            new_w = max(self.cfg.patch, (new_w // self.cfg.patch) * self.cfg.patch)
            resized = _resize_rgb(color, new_h, new_w)
            tokens, place = self._synthetic_tokens(
                resized, scale_u=w / new_w, scale_v=h / new_h
            )
            provider = "synthetic"
        else:
            # Static ONNX exports require a fixed square; letterbox and map UVs
            # back into the original image for depth back-projection.
            size = int(self.cfg.input_short_side)
            if size % self.cfg.patch != 0:
                size = max(self.cfg.patch, (size // self.cfg.patch) * self.cfg.patch)
            canvas, scale, pad_x, pad_y = _letterbox(color, size)
            tokens, place = self._dino_tokens(
                canvas, scale=scale, pad_x=pad_x, pad_y=pad_y
            )
            provider = self._provider

        detections: List[Detection] = []
        if self._yolo is not None:
            detections = self._run_yolo(color)

        # Texture-first survey select (paintings/edges over flat walls).
        # Always keep tokens that fall inside YOLO boxes so detector pooling
        # is not starved by the max_keep=64 cull.
        # Skip in synthetic mode — RGB-stat tokens need the full grid.
        if tokens and not self.cfg.synthetic:
            selected = select_survey_tokens(
                color, tokens, min_texture=8.0, max_keep=64
            )
            if detections:
                in_box = tokens_inside_detections(tokens, detections)
                seen = {(t.row, t.col, round(float(t.u), 1), round(float(t.v), 1)) for t in selected}
                merged = list(selected)
                for t in in_box:
                    key = (t.row, t.col, round(float(t.u), 1), round(float(t.v), 1))
                    if key not in seen:
                        seen.add(key)
                        merged.append(t)
                tokens = merged
            else:
                tokens = selected

        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.stats["frames"] += 1
        ema = self.stats["infer_ms_ema"]
        self.stats["infer_ms_ema"] = (
            infer_ms if ema is None else 0.8 * float(ema) + 0.2 * infer_ms
        )
        self.stats["provider"] = provider
        return VisionResult(
            camera_name=frame.camera_name,
            stamp=frame.stamp,
            captured_at=frame.captured_at,
            patches=tokens,
            detections=detections,
            place_descriptor=place,
            infer_ms=infer_ms,
            provider=provider,
        )

    def _synthetic_tokens(
        self, resized: np.ndarray, *, scale_u: float, scale_v: float
    ) -> Tuple[List[PatchToken], np.ndarray]:
        """Deterministic descriptors from local RGB stats (no ONNX)."""
        h, w = resized.shape[:2]
        us, vs = patch_grid(h, w, patch=self.cfg.patch)
        tokens: List[PatchToken] = []
        descs: List[np.ndarray] = []
        half = self.cfg.patch // 2
        for i, (u, v) in enumerate(zip(us, vs)):
            ui, vi = int(u), int(v)
            x0, x1 = max(0, ui - half), min(w, ui + half)
            y0, y1 = max(0, vi - half), min(h, vi + half)
            win = resized[y0:y1, x0:x1].astype(np.float32)
            # 48-D: mean/std RGB + coarse 4x4 mean luminance histogram-ish.
            mean = win.mean(axis=(0, 1))
            std = win.std(axis=(0, 1))
            gray = win.mean(axis=2)
            gh, gw = gray.shape
            # 4x4 block means.
            blocks = []
            for br in range(4):
                for bc in range(4):
                    r0 = br * gh // 4
                    r1 = (br + 1) * gh // 4
                    c0 = bc * gw // 4
                    c1 = (bc + 1) * gw // 4
                    blocks.append(float(gray[r0:r1, c0:c1].mean()) if r1 > r0 and c1 > c0 else 0.0)
            vec = np.concatenate([mean, std, np.asarray(blocks, dtype=np.float32)])
            # Bake patch location into the descriptor so synthetic benches keep
            # spatially distinct tokens after the distinctiveness filter.
            loc = np.array(
                [ui / max(1, w), vi / max(1, h), (ui + vi) / max(1, w + h)],
                dtype=np.float32,
            )
            rng = np.random.default_rng(
                abs(hash((ui, vi, int(mean[0] * 10), int(mean[1] * 10)))) % (2**32)
            )
            need = 128 - len(vec) - len(loc)
            extra = rng.standard_normal(max(0, need)).astype(np.float32) * 0.05
            desc = l2_normalize(
                np.concatenate([vec.astype(np.float32), loc * 3.0, extra])
            )
            tokens.append(
                PatchToken(
                    row=vi // self.cfg.patch,
                    col=ui // self.cfg.patch,
                    u=float(u) * scale_u,
                    v=float(v) * scale_v,
                    descriptor=desc.reshape(-1),
                )
            )
            descs.append(desc.reshape(-1))
        place = l2_normalize(np.mean(np.stack(descs, axis=0), axis=0))
        return tokens, place

    def _dino_tokens(
        self,
        letterboxed: np.ndarray,
        *,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> Tuple[List[PatchToken], np.ndarray]:
        """Run DINOv2 ONNX on a square letterboxed RGB; map UVs to source image.

        ONNX export layouts vary; we accept:
        - output (1, N, D) patch tokens
        - output (1, D, H, W) spatial map
        """
        assert self._dino is not None
        img = letterboxed.astype(np.float32) / 255.0
        # ImageNet normalize.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        nchw = np.transpose(img, (2, 0, 1))[None, ...]
        inputs = self._dino.get_inputs()
        input_name = inputs[0].name
        outs = self._dino.run(None, {input_name: nchw})
        feat = np.asarray(outs[0])
        tokens: List[PatchToken] = []
        inv = 1.0 / max(1e-9, float(scale))

        def _to_source(u_net: float, v_net: float) -> Tuple[float, float]:
            # Undo letterbox pad, then undo resize scale → original pixels.
            return (float(u_net) - float(pad_x)) * inv, (
                float(v_net) - float(pad_y)
            ) * inv

        if feat.ndim == 3:
            # (1, N, D) — drop CLS if N = 1 + H*W
            patches = feat[0]
            h, w = letterboxed.shape[:2]
            grid_h = h // self.cfg.patch
            grid_w = w // self.cfg.patch
            expected = grid_h * grid_w
            if patches.shape[0] == expected + 1:
                patches = patches[1:]
            patches = l2_normalize(patches)
            for i in range(min(len(patches), expected)):
                r, c = divmod(i, grid_w) if grid_w else (0, 0)
                u_net = (c + 0.5) * self.cfg.patch
                v_net = (r + 0.5) * self.cfg.patch
                # Skip patches whose centers fall in the letterbox padding.
                if (
                    u_net < pad_x
                    or v_net < pad_y
                    or u_net >= (w - pad_x)
                    or v_net >= (h - pad_y)
                ):
                    continue
                u, v = _to_source(u_net, v_net)
                tokens.append(
                    PatchToken(
                        row=r,
                        col=c,
                        u=u,
                        v=v,
                        descriptor=patches[i].astype(np.float32),
                    )
                )
            place = l2_normalize(patches.mean(axis=0))
            return tokens, place
        if feat.ndim == 4:
            # (1, D, Gh, Gw)
            _, dim, gh, gw = feat.shape
            flat = feat[0].reshape(dim, -1).T
            flat = l2_normalize(flat)
            for i in range(flat.shape[0]):
                r, c = divmod(i, gw)
                u_net = (c + 0.5) * (letterboxed.shape[1] / gw)
                v_net = (r + 0.5) * (letterboxed.shape[0] / gh)
                if (
                    u_net < pad_x
                    or v_net < pad_y
                    or u_net >= (letterboxed.shape[1] - pad_x)
                    or v_net >= (letterboxed.shape[0] - pad_y)
                ):
                    continue
                u, v = _to_source(u_net, v_net)
                tokens.append(
                    PatchToken(
                        row=r,
                        col=c,
                        u=u,
                        v=v,
                        descriptor=flat[i].astype(np.float32),
                    )
                )
            place = l2_normalize(flat.mean(axis=0))
            return tokens, place
        raise RuntimeError(f"unexpected DINOv2 output shape {feat.shape}")

    def _class_names(self) -> Tuple[str, ...]:
        if self.cfg.detector_classes:
            return tuple(self.cfg.detector_classes)
        return COCO80_CLASSES

    def _keep_detection_label(self, label: str) -> bool:
        """Keep dynamics always; landmark labels when configured; else all."""
        lab = str(label).strip().lower()
        if lab in {x.lower() for x in self.cfg.dynamic_labels}:
            return True
        if not self.cfg.landmark_labels:
            return True
        allowed = {str(x).strip().lower() for x in self.cfg.landmark_labels}
        if lab in allowed:
            return True
        for a in allowed:
            if lab in a or a in lab:
                return True
        return False

    def _run_yolo(self, color: np.ndarray) -> List[Detection]:
        """Minimal YOLOv8 / YOLO-World ONNX postprocess (xyxy + conf + cls)."""
        assert self._yolo is not None
        size = int(self.cfg.yolo_input)
        img, scale, pad_x, pad_y = _letterbox(color, size)
        blob = img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        name = self._yolo.get_inputs()[0].name
        out = np.asarray(self._yolo.run(None, {name: blob})[0])
        self.stats["last_yolo_out_shape"] = list(out.shape)
        pred = _yolo_predictions(out)
        if pred is None or pred.shape[1] < 5:
            self.stats["last_yolo_raw"] = 0
            self.stats["last_yolo_kept"] = 0
            self.stats["last_yolo_max_conf"] = None
            return []
        # pred: (N, 4+nc) = cx,cy,w,h, scores...
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        # Always record peak score (even when nothing clears the gate).
        peak = float(confs.max()) if len(confs) else 0.0
        self.stats["last_yolo_max_conf"] = peak
        keep = confs >= float(self.cfg.yolo_conf)
        detections: List[Detection] = []
        names = self._class_names()
        raw_kept = 0
        for i in np.where(keep)[0]:
            cid = int(cls_ids[i])
            label = names[cid] if 0 <= cid < len(names) else str(cid)
            conf_i = float(confs[i])
            raw_kept += 1
            if not self._keep_detection_label(label):
                continue
            cx, cy, bw, bh = boxes[i]
            # Undo letterbox to original pixels.
            x0 = (cx - bw / 2.0 - pad_x) / scale
            y0 = (cy - bh / 2.0 - pad_y) / scale
            x1 = (cx + bw / 2.0 - pad_x) / scale
            y1 = (cy + bh / 2.0 - pad_y) / scale
            detections.append(
                Detection(
                    label=label,
                    conf=conf_i,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                )
            )
        detections = _nms_xyxy(detections, iou_thr=float(self.cfg.yolo_nms_iou))
        self.stats["last_yolo_raw"] = int(raw_kept)
        self.stats["last_yolo_kept"] = len(detections)
        return detections


def _resize_rgb(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize (no Pillow dependency required)."""
    h, w = img.shape[:2]
    if h == new_h and w == new_w:
        return img
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[ys][:, xs]


def _yolo_predictions(out: np.ndarray) -> Optional[np.ndarray]:
    """Normalize YOLO ONNX output to ``(N, 4+nc)`` rows."""
    out = np.asarray(out)
    if out.ndim == 3:
        # (1, 4+nc, N) or (1, N, 4+nc)
        if out.shape[1] < out.shape[2]:
            return out[0].T
        return out[0]
    if out.ndim == 2:
        # Some EPs drop the batch dim: (4+nc, N) or (N, 4+nc)
        if out.shape[0] < out.shape[1]:
            return out.T
        return out
    return None


def _letterbox(
    img: np.ndarray, size: int
) -> Tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    scale = float(size) / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = _resize_rgb(img, nh, nw)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, float(pad_x), float(pad_y)


def _nms_xyxy(dets: List[Detection], *, iou_thr: float = 0.5) -> List[Detection]:
    """Greedy per-class NMS on axis-aligned boxes."""
    if len(dets) <= 1 or iou_thr <= 0.0:
        return dets
    by_label: dict = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)
    out: List[Detection] = []
    for group in by_label.values():
        group = sorted(group, key=lambda d: float(d.conf), reverse=True)
        while group:
            best = group.pop(0)
            out.append(best)
            kept = []
            for d in group:
                if _iou_xyxy(best, d) < iou_thr:
                    kept.append(d)
            group = kept
    out.sort(key=lambda d: float(d.conf), reverse=True)
    return out


def _iou_xyxy(a: Detection, b: Detection) -> float:
    ax0, ax1 = sorted((float(a.x0), float(a.x1)))
    ay0, ay1 = sorted((float(a.y0), float(a.y1)))
    bx0, bx1 = sorted((float(b.x0), float(b.x1)))
    by0, by1 = sorted((float(b.y0), float(b.y1)))
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0
