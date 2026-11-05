"""Decode Viam camera image bytes into RGB uint8 and depth meters."""

from __future__ import annotations

import io
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def decode_color_image(data: bytes, mime_type: str = "") -> np.ndarray:
    """Return HxWx3 uint8 RGB."""
    if Image is None:
        raise RuntimeError("Pillow is required to decode camera images")
    mime = (mime_type or "").lower()
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"unexpected color image shape {arr.shape} mime={mime}")
    return arr[:, :, :3].copy()


def decode_depth_image(
    data: bytes,
    mime_type: str = "",
    *,
    depth_scale: float = 0.001,
) -> np.ndarray:
    """Return HxW float32 depth in meters (0 = invalid).

    Accepts 16-bit PNG (millimetres by default) or float32 raw dump.
    """
    mime = (mime_type or "").lower()
    if "float" in mime or mime.endswith("f32"):
        # Raw little-endian float32 HxW — size unknown without metadata.
        raise ValueError("raw float32 depth requires explicit shape; use PNG")
    if Image is None:
        raise RuntimeError("Pillow is required to decode depth images")
    img = Image.open(io.BytesIO(data))
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint16 or arr.dtype == np.int32:
        depth = arr.astype(np.float32) * float(depth_scale)
    elif arr.dtype == np.uint8:
        # Some stacks encode depth as 8-bit; treat as centimetres.
        depth = arr.astype(np.float32) * 0.01
    elif np.issubdtype(arr.dtype, np.floating):
        depth = arr.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth dtype {arr.dtype}")
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def pick_named_images(
    images: Sequence,
) -> Tuple[Optional[object], Optional[object]]:
    """From a Viam ``get_images`` list, pick (color, depth) NamedImage-like objs.

    Prefers names/mimes containing ``color``/``rgb`` and ``depth``.
    """
    color = None
    depth = None
    for im in images:
        name = str(getattr(im, "name", "") or "").lower()
        mime = str(getattr(im, "mime_type", "") or getattr(im, "mimeType", "") or "").lower()
        data_attr = getattr(im, "data", None)
        if data_attr is None and hasattr(im, "image"):
            # Some SDK versions nest ViamImage.
            data_attr = getattr(im.image, "data", None)
            mime = mime or str(getattr(im.image, "mime_type", "") or "").lower()
        blob = {
            "name": name,
            "mime": mime,
            "data": data_attr if isinstance(data_attr, (bytes, bytearray)) else None,
            "raw": im,
        }
        if blob["data"] is None:
            continue
        is_depth = "depth" in name or "depth" in mime or "dep" in mime
        is_color = (
            "color" in name
            or "rgb" in name
            or "jpeg" in mime
            or "jpg" in mime
            or "png" in mime
            and not is_depth
        )
        if is_depth and depth is None:
            depth = blob
        elif is_color and color is None:
            color = blob
    # Fallback: first non-depth as color, first depth-ish as depth.
    if color is None or depth is None:
        for im in images:
            name = str(getattr(im, "name", "") or "").lower()
            mime = str(getattr(im, "mime_type", "") or "").lower()
            data = getattr(im, "data", None)
            if not isinstance(data, (bytes, bytearray)):
                continue
            blob = {"name": name, "mime": mime, "data": bytes(data), "raw": im}
            if depth is None and ("depth" in name or "depth" in mime):
                depth = blob
            elif color is None:
                color = blob
    return color, depth
