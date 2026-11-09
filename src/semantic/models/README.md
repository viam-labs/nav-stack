# Vision ONNX models

Runtime inference uses ONNX Runtime (OpenVINO GPU preferred on Intel NUCs).
Weights are **not** committed to git and are excluded from `module.tar.gz`.

## Suggested files

| Role | Example path | Notes |
| --- | --- | --- |
| DINOv2 ViT-S/14 patch tokens | `dinov2_vits14_patches_448.onnx` | Square 448 input → `(1, 1024, 384)` |
| YOLOv8n | `yolov8n.onnx` | Paint-gate only (`ultralytics` export) |
| Site head (phase 7) | `head.onnx` | Optional; replaces per-map PCA |

## Auto-install

`setup.sh` (and `run.sh` on upgrade) install `requirements-vision.txt` and
fetch `dinov2_vits14_patches_448.onnx` when missing. No manual
`NAV_STACK_VISION` flag.

To re-fetch / export by hand:

```bash
python tools/fetch_dinov2_onnx.py
python tools/export_dinov2_onnx.py --size 448   # needs torch + network
```

Point config at the file (relative name resolves under `src/semantic/models/`):

```json
"vision_enabled": true,
"vision_survey_only": true,
"vision_cameras": ["camera-top"],
"vision_dinov2_onnx": "dinov2_vits14_patches_448.onnx",
"vision_input_size": 448
```

Omit `vision_synthetic` (or set `false`). `vision_status.provider` should show
`OpenVINOExecutionProvider` (or `CPUExecutionProvider`), **not** `synthetic`.

## Pose writeback (opt-in)

After a successful density survey, enable soft landmark corrections:

```json
"vision_survey_only": false,
"vision_pose_writeback": true
```

Uses confirmed landmarks → RANSAC rigid fit → `apply_map_pose_correction`
(`force=False`, lidar revisit path). Synthetic / survey-only never write pose.
Watch `vision_status.writeback` (`attempts` / `applied` / `last`).

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
