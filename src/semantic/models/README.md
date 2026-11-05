# Vision ONNX models

Runtime inference uses ONNX Runtime (OpenVINO GPU preferred on Intel NUCs).
Weights are **not** committed to git and are excluded from `module.tar.gz`.

## Suggested files

| Role | Example path | Notes |
| --- | --- | --- |
| DINOv2 ViT-S/14 patch tokens | `dinov2_vits14_patches_448.onnx` | Square 448 input → `(1, 1024, 384)` |
| YOLOv8n | `yolov8n.onnx` | Paint-gate only (`ultralytics` export) |
| Site head (phase 7) | `head.onnx` | Optional; replaces per-map PCA |

## Fetch / export on the robot

```bash
# After pinning 1.2.0-rc3+ and restarting once (so tools/ is present):
cd /path/to/nav-stack-module   # or wherever the module unpacked
NAV_STACK_VISION=1 ./setup.sh   # installs onnxruntime-openvino
python tools/fetch_dinov2_onnx.py
# → src/semantic/models/dinov2_vits14_patches_448.onnx

# Or export from torch yourself (needs network + torch):
python tools/export_dinov2_onnx.py --size 448
```

Point config at the file (absolute path is safest):

```json
"vision_enabled": true,
"vision_survey_only": true,
"vision_cameras": ["camera-top"],
"vision_dinov2_onnx": "/absolute/path/to/dinov2_vits14_patches_448.onnx",
"vision_input_size": 448
```

Omit `vision_synthetic` (or set `false`). `vision_status.provider` should show
`OpenVINOExecutionProvider` (or `CPUExecutionProvider`), **not** `synthetic`.

Relative names like `dinov2_vits14_patches_448.onnx` resolve against
`src/semantic/models/` next to the running module.

## Survey density / nav load (rc2+)

Defaults assume a density pilot, not pose writeback:

| Attribute | Default | Why |
| --- | --- | --- |
| `vision_rate_hz` | `1.0` | Leave CPU for SLAM/nav |
| `vision_round_robin` | `true` | One camera per tick when multiple are listed |
| `vision_z_min_m` / `vision_z_max_m` | `0.25` / `2.0` | Drop floor/ceiling |
| `vision_max_patches_per_frame` | `64` | Cap associations per frame |
| `vision_associate_radius_m` / `vision_cell_m` | `0.25` / `0.30` | Merge nearby patches |
| `vision_input_size` | `448` | Must match the ONNX export |

Wipe `embedding_landmarks.npz` (or `clear_map`) before re-surveying after
switching synthetic → DINOv2. Prefer a single camera in `vision_cameras`.

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
python tools/bench_vision.py --dinov2 src/semantic/models/dinov2_vits14_patches_448.onnx --short-side 448
```
