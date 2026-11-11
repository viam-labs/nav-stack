"""YOLO-World fetch replaces an onnx baked from an older class list."""

from __future__ import annotations

from pathlib import Path

from tools.fetch_yolo_world_onnx import (
    class_list_rev,
    onnx_is_current,
    write_stamp,
)


def test_missing_stamp_is_not_the_current_class_list(tmp_path: Path):
    classes = tmp_path / "model.classes.json"
    classes.write_text('["painting", "person"]\n')
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x" * (5_000_001))
    assert class_list_rev(classes)
    assert onnx_is_current(onnx, classes) is False
    write_stamp(onnx, classes)
    assert onnx_is_current(onnx, classes) is True
    classes.write_text('["painting", "person", "door"]\n')
    assert onnx_is_current(onnx, classes) is False
