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

## Survey density / nav load (rc2+)

Defaults assume a density pilot, not pose writeback:

| Attribute | Default | Why |
| --- | --- | --- |
| `vision_rate_hz` | `1.0` | Leave CPU for SLAM/nav |
| `vision_round_robin` | `true` | One camera per tick when multiple are listed |
| `vision_z_min_m` / `vision_z_max_m` | `0.25` / `2.0` | Drop floor/ceiling |
| `vision_max_patches_per_frame` | `64` | Cap associations per frame |
| `vision_associate_radius_m` / `vision_cell_m` | `0.25` / `0.30` | Merge nearby patches |

Wipe `embedding_landmarks.npz` (or `clear_map`) before re-surveying after upgrading. Prefer a single camera in `vision_cameras` while measuring density.

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
