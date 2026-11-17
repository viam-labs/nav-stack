# Vision ONNX models

Runtime inference uses ONNX Runtime (OpenVINO GPU preferred on Intel NUCs).
Weights are **not** committed to git and are excluded from `module.tar.gz`.

## Suggested files

| Role | Example path | Notes |
| --- | --- | --- |
| DINOv2 ViT-S/14 patch tokens | `dinov2_vits14_patches_448.onnx` | Square 448 input → `(1, 1024, 384)` |
| YOLO-Worldv2 (indoor) | `yolov8s-worldv2-indoor.onnx` | Boxes for people and other movers, masked out of the scan while mapping |
| YOLOv8n (COCO) | `yolov8n.onnx` | Works too: it has `person` |
| Site head (phase 7) | `head.onnx` | Optional; replaces per-map PCA |

## Auto-install

`setup.sh` (and `run.sh` on upgrade) install `requirements-vision.txt` and
fetch `dinov2_vits14_patches_448.onnx` when missing. No manual
`NAV_STACK_VISION` flag.

To re-fetch / export by hand:

```bash
python tools/fetch_dinov2_onnx.py
python tools/fetch_yolo_world_onnx.py
# Optional local rebuilds (torch / ultralytics):
python tools/export_dinov2_onnx.py --size 448
python tools/export_yolo_world_onnx.py
```

`setup.sh` / `run.sh` fetch both ONNX files automatically when missing.
You do **not** need to set `vision_dinov2_onnx` / `vision_yolo_onnx` for the
defaults — empty config resolves to:

- `dinov2_vits14_patches_448.onnx`
- `yolov8s-worldv2-indoor.onnx` (+ class list from
  `yolov8s-worldv2-indoor.classes.json`)

```json
"vision_enabled": true,
"vision_cameras": ["camera-top"],
"vision_input_size": 448
```

The detector is only used to find people (and bicycles, cars, motorcycles);
boxes with any other label are dropped. Override `vision_yolo_onnx` /
`vision_detector_classes` only for a custom detector (e.g. COCO YOLOv8 —
leave classes empty to use COCO-80).

Omit `vision_synthetic` (or set `false`). `vision_status.provider` should show
`OpenVINOExecutionProvider` (or `CPUExecutionProvider`), **not** `synthetic`.
Intrinsics are auto-read from `Camera.get_properties()` when the camera
publishes them; see `vision_status.intrinsics_source`. Check
`vision_status.yolo_loaded` / `yolo_onnx` if nobody is ever masked.

`vision_status.capture_pose` counts frames whose capture-time pose had to
come from the grab-time stamp (`grab_time_fallbacks`) or that were dropped
because no capture pose was available and they were over 1 s old
(`stale_frames_dropped`).

## What vision does (defaults — no mode knobs)

With `vision_enabled: true` the defaults are:

- **mapping**: store a recovery keyframe every 0.75 m or 25°
  (`vision_keyframes.npz` next to the map: the frame's pose, one pooled
  descriptor, and every patch with its descriptor and 3D point; about 0.2 MB
  per keyframe). A keyframe hangs from its pose-graph node, so a loop closure
  moves it with the map. `save_map` writes the file and `clear_map` removes
  it.
- **mapping paint**: a detection labelled person, bicycle, car, or motorcycle
  blanks the lidar returns inside that box for 0.2 s, so a passer-by is not
  stored as a wall. Furniture and this robot are not blanked.
- **localizing, pose tracking**: vision does nothing. No frame is grabbed and
  no model runs.
- **localizing, lost or unknown, base standing still**: lost-pose recovery
  (`vision_recovery`, see below).

Kill switch: `"vision_survey_only": true` (store keyframes, never move the
pose).

### Lost-pose recovery (`src/semantic/relocalize.py`)

The camera proposes, the lidar decides:

1. The live frame's pooled descriptor shortlists 30 keyframes; the shortlist
   is reordered by how many patches match one-to-one, and the top 5 are kept.
2. For each, the matched patches that have depth on both sides are fitted to
   one camera pose (at least 8 within 15 cm).
3. The scan search runs around each camera pose (0.6 m, ±20°).
4. The pose is placed only when the scan match passes the usual floors
   (score ≥ 0.35, ray error ≤ 1.0 m), the lidar and camera poses agree within
   0.25 m and 6°, and no other guess more than 1 m away scores within 0.10.

`vision_status.recovery` reports `keyframes`, `attempts`, `applied`, and
`last` (`accepted`, `no_agreement`, `rival_place`, `skipped_moving`, with the
score of every guess).

These rules were chosen on the LILocBench office dataset, treating each
sampled frame as a kidnapped robot (right = within 0.5 m and 15°):

| Method | Static run | Crowded run | Rearranged-furniture run |
| --- | --- | --- | --- |
| Whole-map lidar search, before its repair | not measured | 2% placed, 0% wrong | 2% placed, 0% wrong |
| Whole-map lidar search, repaired | 99% placed, 0% wrong | 14% placed, 0% wrong | 27% placed, 1% wrong |
| Camera recovery as shipped | 74% placed, 2% wrong | 26% placed, 0% wrong | 38% placed, 2% wrong |
| Lidar search given a hint 1 m and 30° off | 100% placed, 0% wrong | 80% placed, 0% wrong | 79% placed, 6% wrong |

Rates are per attempt; a lost robot tries again every
`vision_recovery_cooldown_s`. One office, 187 / 50 / 137 frames: treat the
percentages as rough.

What the table says:

- The lidar search had two faults, both repaired in
  `src/nav/localize/global_localize.py`: a coarse grid wider than the match
  tolerance dropped the true pose before the fine search, and a correct
  answer was flagged `ambiguous` 83-93% of the time because "second best" was
  a neighbouring sample of the same peak. Repaired, it needs no camera in a
  scene that matches the map.
- Where people or moved furniture change the scene, a wrong place often
  out-scores the true one for the lidar, and the camera adds 11-12 points.
- The last row is the ceiling for any hint-based method: a better place guess
  could still roughly double the changed-scene numbers.
- Most wrong placements were in the hallway, where look-alike doors and two
  parallel walls fool both sensors.

### Off by default

`vision_place_enabled` / `vision_place_retrieval` (VLAD place index and its
teleport: while lost, the top keyframe's pose is applied with no scan check).
Its top guess was the true place for 26-60% of frames, against 66-71% for the
keyframe patch match that recovery uses. Setting both `true` restores it,
including `vision_status.place`.

### Removed: object landmarks

Earlier builds also stored one landmark per detector box (a pooled descriptor
and a 3D point, hung from a pose-graph node) and used them to nudge the pose
while localizing (`vision_pose_writeback`) and to add loop edges while
mapping. That layer is gone, with its depth audit, the `list_landmarks` /
`landmark_audit` / `clear_landmarks` commands and the `vision-camera` overlay
model. Its settings are ignored if a robot config still has them. What was
measured on LILocBench before removing it:

- **Loop edges almost never formed.** 0, 1 and 3 edges in runs of 211 m,
  212 m and 42 m, against about 200 lidar loop closures per 210 m. In the
  first 420 s of the mapping run (786 frames), 302 frames had 3 or more
  confirmed landmarks projecting into the image. About a third of those
  paired 3: half the landmarks "in view" were hidden behind something, and
  the detector found a median 2 of 5 again. 37 fits passed, and all 37 mixed
  landmarks from several keyframes, which the loop rule refused.
- **Relaxing that rule did not help.** Letting neighbouring keyframes count
  as one visit offered 2-3 edges in the whole run. With lidar loop closure on
  the result was unchanged (0.18 m / 0.14 m against 0.19 m / 0.14 m). With it
  off, two 3-landmark edges with about 5° of heading error took the live
  error from 0.42 m to 2.81 m (14 m at worst). One run, experimental rule.
- **Writeback set the pose with no scan check.** While localizing,
  `apply_map_pose_correction` takes the pose as given, and a pose fitted from
  the camera alone was wrong about half the time, with nothing in the fit to
  show it.

The code is in git history up to commit `7125b4a`. Using the camera to
*propose* a loop closure (recognise an old keyframe, let the scan match
compute the edge) would reuse the recovery keyframes; it has not been built.

## Measuring vision offline

```bash
python tools/fetch_lilocbench.py     # ~2 GB subset into ~/datasets/lilocbench
python tools/eval_vision.py place    # is the true place among the top guesses?
python tools/eval_vision.py reloc    # kidnapped-robot recovery, lidar vs camera+lidar
python tools/eval_vision.py mapping  # replay a run through SLAM, score the trajectory
```

LILocBench (Uni Bonn) has a 2D lidar, RealSense RGB-D, odometry and
ground-truth poses for one mapping run and later runs with people and
rearranged furniture. It is licensed CC BY-NC-SA 4.0 and is not part of this
repo. `reloc --place-methods runtime` runs `recover_pose` itself; other
method names compare alternatives (see `--help`). DINOv2 tokens are cached
under `~/datasets/lilocbench/cache`, so re-runs take seconds to minutes.

### Mapping replay

`place` and `reloc` judge one frame at a time. `mapping` feeds a run's
odometry and scans through the SLAM engine in order, on a clock that follows
the recording, with the vision supervisor alongside when asked, and compares
the published pose and the final keyframes with ground truth. The engine is
stepped as its own threads would step it, with nothing in the background.
Detector boxes are cached after the first vision replay of a run.

Three runs, each mapped from scratch. Error against ground truth after
aligning frames, RMSE of the live pose / of the final keyframes:

| Run | Wheel odometry | Lidar loop closure off | Lidar only | + person mask, result 0.3 s late | + person mask, result at once |
| --- | --- | --- | --- | --- | --- |
| mapping (211 m, static) | 0.89 m / - | 0.42 m / 0.43 m | 0.20 m / 0.14 m | 0.19 m / 0.13 m | 0.17 m / 0.13 m |
| static_0 (212 m, static) | 0.69 m / - | 0.42 m / 0.40 m | 0.08 m / 0.07 m | 0.08 m / 0.06 m | 0.09 m / 0.07 m |
| dynamics_0 (42 m, crowded) | 0.21 m / - | not run | 0.09 m / 0.09 m | 0.09 m / 0.09 m | 0.16 m / 0.15 m |

- **Lidar loop closure is what holds the map together**: about 190-210
  closures per 210 m, taking the live error from 0.42 m to 0.08-0.20 m.
- **Person masking only applies when the result is immediate**, and then it
  is not a reliable gain. The mask covers scans within 0.2 s of the camera
  frame, so a result 0.3 s late never applies. Applied at once it improved
  one static run (0.20 m to 0.17 m), slightly worsened the other (0.08 m to
  0.09 m) and clearly worsened the crowded one (0.09 m to 0.16 m), where a
  person was in 260 of 299 frames. It did keep them out of the map there
  (occupied cells down by a third). Why the pose gets worse is not yet
  explained.
- **Differences under 1 cm are noise.** The late-result column changes
  nothing the engine sees, so it should equal lidar alone. It does on the
  short run and is within 1 cm on the two long ones.

`--variants` covers `odometry`, `lidar_no_loops`, `lidar`, `vision` (result
applied at once) and `vision_late` (0.3 s after the frame).

The replay does not run the service-level tasks (mapping revisit, periodic
relocalization), and the dataset robot's wheel odometry is not the cart's.

## Nav load

| Attribute | Default | Why |
| --- | --- | --- |
| `vision_rate_hz` | `2.0` | One camera. Two RealSense cameras faster than this starved nav |
| `vision_round_robin` | `true` | One camera per tick when multiple are listed |
| `vision_input_size` | `448` | Must match the ONNX export |

While localizing with the pose tracking, no frame is grabbed and no model
runs. Prefer a single camera in `vision_cameras`.

**Intrinsics / FOV:** prefer live `Camera.get_properties().intrinsic_parameters`
(auto-scaled to the frame size). `vision_status.intrinsics_source` is
`camera_properties` when that works, else `fov_config` (lidar `fov_deg` /
`vfov_deg` fallback — set those only if properties are empty).

Keyframes are never stored from the synthetic provider. A map made before
DINOv2 was loaded needs another mapping pass to collect them.

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
