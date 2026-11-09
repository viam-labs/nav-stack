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
Intrinsics are auto-read from `Camera.get_properties()` when the camera
publishes them; see `vision_status.intrinsics_source`.

## Landmark overlay camera

Add a ``viam-labs:nav-stack:vision-camera`` component to stream RGB with
boxes (green=confirmed, yellow=candidate):

```json
{
  "name": "vision-landmarks",
  "model": "viam-labs:nav-stack:vision-camera",
  "attributes": {
    "slam": "slam",
    "source": "camera-top",
    "show_candidates": true,
    "show_confirmed": true
  },
  "depends_on": ["slam", "camera-top"]
}
```

``source`` should be the same RealSense as ``vision_cameras``. Open it in the
Viam camera stream view.

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

Promotion defaults (code, not config): ≥3 obs, ≥2 views with ≥0.5 m **or**
≥25° yaw, position std ≤ **0.12 m**. Patch selection is **texture-first**
(local RGB contrast) so painting interiors survive; distinctiveness is only a
soft bonus (hard distinctiveness alone dropped coherent artwork).

**Windows / outdoor through glass:** RGB may latch onto a far building while
depth returns the glass plane — the box looks right from the first viewpoint
then slides onto the window frame when you turn. Depth support checks reject
mixed/holey neighborhoods, and ``vision-camera`` hides unstable single-view
candidates by default (`stable_candidates_only: true`). Prefer indoor opaque
surfaces (paintings, posters, furniture) for survey.

**Intrinsics / FOV:** prefer live `Camera.get_properties().intrinsic_parameters`
(auto-scaled to the frame size). `vision_status.intrinsics_source` is
`camera_properties` when that works, else `fov_config` (lidar `fov_deg` /
`vfov_deg` fallback — set those only if properties are empty).

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
