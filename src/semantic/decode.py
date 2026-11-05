"""Decode Viam camera image bytes into RGB uint8 and depth meters."""

from __future__ import annotations

import io
import struct
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

# Viam raw depth: magic "DEPTHMAP" + be64 width + be64 height + be uint16 mm.
_VIAM_DEPTH_MAGIC = b"DEPTHMAP"
_VIAM_DEPTH_MIME = "image/vnd.viam.dep"
_VIAM_RGBA_MIME = "image/vnd.viam.rgba"


def _as_bytes(data: Any) -> bytes:
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, bytes):
        return data
    raise TypeError(f"image data must be bytes, got {type(data).__name__}")


def decode_color_image(data: bytes, mime_type: str = "") -> np.ndarray:
    """Return HxWx3 uint8 RGB."""
    raw = _as_bytes(data)
    mime = (mime_type or "").lower()
    if _VIAM_RGBA_MIME in mime or mime.endswith("viam.rgba"):
        return _decode_viam_rgba(raw)
    if Image is None:
        raise RuntimeError("Pillow is required to decode camera images")
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cannot decode color image mime={mime!r} bytes={len(raw)}: {exc}"
        ) from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"unexpected color image shape {arr.shape} mime={mime}")
    return arr[:, :, :3].copy()


def _decode_viam_rgba(data: bytes) -> np.ndarray:
    """``image/vnd.viam.rgba``: be64 width, be64 height, then RGBA8888 pixels."""
    if len(data) < 16:
        raise ValueError(f"viam.rgba too short ({len(data)} bytes)")
    width = int.from_bytes(data[0:8], "big")
    height = int.from_bytes(data[8:16], "big")
    need = 16 + width * height * 4
    if len(data) < need:
        raise ValueError(
            f"viam.rgba size mismatch: {width}x{height} needs {need}, got {len(data)}"
        )
    rgba = np.frombuffer(data, dtype=np.uint8, offset=16, count=width * height * 4)
    return rgba.reshape((height, width, 4))[:, :, :3].copy()


def decode_viam_raw_depth(data: bytes, *, depth_scale: float = 0.001) -> np.ndarray:
    """Decode ``image/vnd.viam.dep`` to HxW float32 meters (0 = invalid)."""
    raw = _as_bytes(data)
    if len(raw) < 24 or raw[:8] != _VIAM_DEPTH_MAGIC:
        raise ValueError(
            "not a Viam raw depth map (expected DEPTHMAP header); "
            f"got magic={raw[:8]!r} len={len(raw)}"
        )
    width = int.from_bytes(raw[8:16], "big")
    height = int.from_bytes(raw[16:24], "big")
    need = 24 + width * height * 2
    if len(raw) < need:
        raise ValueError(
            f"viam.dep size mismatch: {width}x{height} needs {need}, got {len(raw)}"
        )
    # Big-endian uint16 millimetres.
    mm = np.frombuffer(raw, dtype=">u2", offset=24, count=width * height)
    depth = mm.astype(np.float32).reshape((height, width)) * float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def decode_depth_image(
    data: bytes,
    mime_type: str = "",
    *,
    depth_scale: float = 0.001,
    named_image: Any = None,
) -> np.ndarray:
    """Return HxW float32 depth in meters (0 = invalid).

    Handles ``image/vnd.viam.dep`` (RealSense via Viam), 16-bit PNG, and the
    SDK helper ``NamedImage.bytes_to_depth_array`` when present.
    """
    raw = _as_bytes(data)
    mime = (mime_type or "").lower()

    if named_image is not None and hasattr(named_image, "bytes_to_depth_array"):
        try:
            arr = np.asarray(named_image.bytes_to_depth_array(), dtype=np.uint16)
            depth = arr.astype(np.float32) * float(depth_scale)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth < 0.0] = 0.0
            return depth
        except Exception:  # noqa: BLE001
            pass

    if (
        _VIAM_DEPTH_MIME in mime
        or "viam.dep" in mime
        or raw[:8] == _VIAM_DEPTH_MAGIC
    ):
        return decode_viam_raw_depth(raw, depth_scale=depth_scale)

    if "float" in mime or mime.endswith("f32"):
        raise ValueError("raw float32 depth requires explicit shape; use viam.dep")

    if Image is None:
        raise RuntimeError("Pillow is required to decode depth images")
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cannot decode depth image mime={mime!r} bytes={len(raw)} "
            f"magic={raw[:8]!r}: {exc}"
        ) from exc
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint16 or arr.dtype == np.int32:
        depth = arr.astype(np.float32) * float(depth_scale)
    elif arr.dtype == np.uint8:
        depth = arr.astype(np.float32) * 0.01
    elif np.issubdtype(arr.dtype, np.floating):
        depth = arr.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth dtype {arr.dtype}")
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def _image_fields(im: Any) -> Tuple[str, str, Optional[bytes], Any]:
    """Extract (name, mime, data_bytes, raw_obj) from a NamedImage-like value."""
    name = str(getattr(im, "name", "") or "").lower()
    mime = str(
        getattr(im, "mime_type", None) or getattr(im, "mimeType", None) or ""
    ).lower()
    data_attr = getattr(im, "data", None)
    raw_obj = im
    if data_attr is None and hasattr(im, "image"):
        nested = im.image
        raw_obj = nested
        data_attr = getattr(nested, "data", None)
        mime = mime or str(getattr(nested, "mime_type", "") or "").lower()
    data = _as_bytes(data_attr) if isinstance(data_attr, (bytes, bytearray, memoryview)) else None
    return name, mime, data, raw_obj


def pick_named_images(
    images: Sequence,
) -> Tuple[Optional[dict], Optional[dict]]:
    """From a Viam ``get_images`` list, pick (color, depth) blobs.

    Prefers names/mimes containing ``color``/``rgb``/``jpeg`` and
    ``depth`` / ``image/vnd.viam.dep``.
    """
    color = None
    depth = None
    for im in images:
        name, mime, data, raw_obj = _image_fields(im)
        if data is None:
            continue
        blob = {"name": name, "mime": mime, "data": data, "raw": raw_obj}
        is_depth = (
            "depth" in name
            or "depth" in mime
            or "viam.dep" in mime
            or data[:8] == _VIAM_DEPTH_MAGIC
        )
        is_color = (
            "color" in name
            or "rgb" in name
            or "jpeg" in mime
            or "jpg" in mime
            or "viam.rgba" in mime
            or (("png" in mime) and not is_depth)
        )
        if is_depth and depth is None:
            depth = blob
        elif is_color and color is None:
            color = blob
    if color is None or depth is None:
        for im in images:
            name, mime, data, raw_obj = _image_fields(im)
            if data is None:
                continue
            blob = {"name": name, "mime": mime, "data": data, "raw": raw_obj}
            is_depth = (
                "depth" in name
                or "depth" in mime
                or "viam.dep" in mime
                or data[:8] == _VIAM_DEPTH_MAGIC
            )
            if depth is None and is_depth:
                depth = blob
            elif color is None and not is_depth:
                color = blob
    return color, depth
