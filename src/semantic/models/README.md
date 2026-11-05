# Vision ONNX models

Runtime inference uses ONNX Runtime (OpenVINO GPU preferred on Intel NUCs).
Weights are **not** committed to git.

## Suggested files

| Role | Example path | Notes |
| --- | --- | --- |
| DINOv2 ViT-S/14 | `dinov2_vits14.onnx` | Export from `facebookresearch/dinov2`; input NCHW ImageNet-normalized |
| YOLOv8n | `yolov8n.onnx` | Paint-gate only (`ultralytics` export) |
| Site head (phase 7) | `head.onnx` | Optional; replaces per-map PCA |

Point `SlamConfig.vision_dinov2_onnx` / `vision_yolo_onnx` at absolute paths on the robot.

## Meerkat GPU runtime

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo
clinfo -l   # should list Intel Arc Graphics
pip install onnxruntime-openvino   # or onnxruntime if CPU-only
```

## Bench

```bash
python tools/bench_vision.py                  # synthetic descriptors
python tools/bench_vision.py --dinov2 /path/to/dinov2_vits14.onnx
```
