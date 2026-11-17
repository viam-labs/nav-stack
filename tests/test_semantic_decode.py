"""Viam depth / color mime decoding."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from src.semantic.decode import (
    decode_color_image,
    decode_depth_image,
    decode_viam_raw_depth,
    pick_named_images,
)


def _viam_dep(width: int, height: int, mm: int = 1500) -> bytes:
    header = b"DEPTHMAP" + struct.pack(">QQ", width, height)
    # Big-endian uint16 pixels.
    pix = np.full((height, width), mm, dtype=">u2")
    return header + pix.tobytes()


def test_decode_viam_raw_depth_meters():
    raw = _viam_dep(4, 3, mm=2000)
    depth = decode_viam_raw_depth(raw)
    assert depth.shape == (3, 4)
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(2.0)


def test_decode_depth_image_by_mime():
    raw = _viam_dep(2, 2, mm=1000)
    depth = decode_depth_image(raw, "image/vnd.viam.dep")
    assert depth.shape == (2, 2)
    assert float(depth.mean()) == pytest.approx(1.0)


def test_decode_depth_image_by_magic_without_mime():
    raw = _viam_dep(2, 2, mm=500)
    depth = decode_depth_image(raw, "")
    assert depth[0, 0] == pytest.approx(0.5)


def test_pick_named_images_prefers_viam_dep():
    class Img:
        def __init__(self, name, mime, data):
            self.name = name
            self.mime_type = mime
            self.data = data

    color = Img("color", "image/jpeg", b"\xff\xd8\xff" + b"\x00" * 20)
    depth = Img("depth", "image/vnd.viam.dep", _viam_dep(2, 2))
    c, d = pick_named_images([color, depth])
    assert c is not None and d is not None
    assert d["mime"] == "image/vnd.viam.dep"
    assert decode_depth_image(d["data"], d["mime"]).shape == (2, 2)


def test_decode_color_rejects_depth_bytes():
    with pytest.raises(ValueError, match="cannot decode color"):
        decode_color_image(_viam_dep(2, 2), "image/jpeg")
