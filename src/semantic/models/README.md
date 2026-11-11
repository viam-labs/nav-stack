# Vision ONNX models

Runtime inference uses ONNX Runtime (OpenVINO GPU preferred on Intel NUCs).
Weights are **not** committed to git and are excluded from `module.tar.gz`.

## Suggested files

| Role | Example path | Notes |
| --- | --- | --- |
| DINOv2 ViT-S/14 patch tokens | `dinov2_vits14_patches_448.onnx` | Square 448 input → `(1, 1024, 384)` |
| YOLO-Worldv2 (indoor) | `yolov8s-worldv2-indoor.onnx` | Open-vocab boxes → pooled DINO landmarks |
| YOLOv8n (COCO) | `yolov8n.onnx` | Works too: `tv`/`clock`/`book`/… as interim |
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
"vision_input_size": 448,
"vision_prefer_detections": true
```


Survey prefers detector boxes: each allowed YOLO-World box pools DINOv2 patch
tokens into one landmark (much stabler re-ID than dense 14×14 tokens). Dense
texture patches remain as a small fallback (`vision_dense_fallback_patches`,
default 16). Override `vision_yolo_onnx` / `vision_detector_classes` only for
a custom detector (e.g. COCO YOLOv8 — leave classes empty to use COCO-80).
`vision_landmark_labels` filters which detections become landmarks.

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
    "show_confirmed": true,
    "show_detections": true
  },
  "depends_on": ["slam", "camera-top"]
}
```

``source`` should be the same RealSense as ``vision_cameras``. Open it in the
Viam camera stream view. Overlay colors: **green** confirmed, **yellow**
candidate, **cyan** / **grey** latest YOLO-World boxes (see below).
**Orange ring + V** = in the latest writeback match pool (image-visible
confirmed); **magenta ring + P** = unique pair used for the pose fit.
HUD ``wb=V/P`` are those counts. Plain greens without a ring are
on the map overlay but not in the last navigation match (e.g. candidate,
out of range, or not paired). HUD ``yolo=OFF`` means the detector ONNX
is not loaded; ``yolo=3/10`` means 3 of the 10 latest boxes fed a landmark.
Set ``show_detections: false`` to hide the detector layer. Check
``vision_status.yolo_loaded`` / ``yolo_onnx`` if boxes never appear.

Each YOLO box says what became of it in the last surveyed frame:

- **cyan**, labelled with a landmark (``poster =DC23``): the box was located
  and matched to (or created) that landmark. A thin line joins the box centre
  to the landmark's square, so a box that fed a landmark stored somewhere
  else is visible as a long line.
- **grey**, thin, labelled with a reason: the box made no observation.
  ``far 9.6m`` / ``near 0.3m`` = outside the 0.4–4.5 m box range,
  ``mixed depth`` = no single surface, ``no depth`` = too little valid depth,
  ``height`` = outside the z gate, ``not a landmark class`` = label not in
  ``vision_landmark_labels``.

A plain cyan box with a confidence (``poster 0.27``) has not been surveyed
yet.

Every landmark is drawn as a small square at the projection of its own
stored point, labelled with its ID and stored depth (``DC124 3.7m``). It is
never drawn as the YOLO rectangle: a landmark stored in front of or behind
an object still projects inside that object's box, so a filled box made a
wrong position look right. The square scales with stored depth, so a
landmark stored too near draws larger than the object behind it. A **red
square with a cross** is a landmark the camera currently sees past (see the
depth audit below); HUD ``air=N`` counts them.

## Depth audit (`landmark_audit`)

A landmark is a claim that a surface exists at its stored point. Each survey
frame, every landmark that projects into the image has its stored depth
compared with the depth image in a ~10 cm window:

| Verdict | Meaning |
| --- | --- |
| `agrees` | ≥15% of the window reads the stored depth (±max(0.20 m, 8%)) |
| `seen_through` | ≥80% of the window reads beyond stored + max(0.40 m, 15%), and those readings look like a surface |
| `occluded` | most of the window is nearer than the stored point |
| `no_depth` | under 25% of the window has valid depth |
| `unclear` | none of the above, including far readings that are steep both across and down the image — stereo mismatches on patterned or glossy objects, not a surface |

Three `seen_through` frames in a row flag the landmark; an `agrees` frame
clears the count; the other verdicts leave it unchanged. While **mapping**
(or `vision_survey_only`) a flagged landmark, candidate or confirmed, is
deleted and logged (`vision: dropped landmark DC125 …`), so the real object
can be re-minted cleanly. A landmark with 30 or more observations is deleted
only when the camera reads at least 1.2 m past the stored depth; a smaller
gap stays flagged (red X) instead of throwing away a well-seen object. While
**localizing** the audit only reports: the
pose is what is in doubt there, and a wrong pose makes every landmark look
seen through. For the same reason a frame where at least 4 landmarks are
seen through and they outnumber the agreeing ones is marked `pose_suspect`
and changes nothing.

`{"command": "landmark_audit"}` on the SLAM service returns the latest frame's
rows (`id`, `label`, `status`, `u`, `v`, `stored_depth_m`, `measured_depth_m`,
`verdict`, `strikes`, `removed`; worst first, at most 200), a `summary` of
verdict counts, `flagged_ids`, `removed_total` and `recent_removed`.
`vision_status.depth_audit` carries the counts. Landmarks stored up to 6.5 m
from the camera are audited.

The reply also lists every detector box of that frame under `boxes`
(`label`, `conf`, `outcome`, `text`, `depth_m`, `landmark`): the same
information the overlay draws, so "YOLO sees it but there is no landmark"
can be answered without the image.

`{"command": "clear_landmarks"}` drops every landmark and deletes the active
map's `embedding_landmarks.npz`; the occupancy map and the place index are
kept. Use it before re-surveying.

## Where a detector box's point comes from

The point is read from raw depth in the central 60% of the box. The pixels
are grouped by depth and the largest group within one surface width
(2 × max(0.12 m, 6% of depth)) is the object. The box becomes an observation
only when that group is unambiguous; otherwise it is skipped and counted in
`vision_status.box_depth`:

| Count | Why the box made no observation |
| --- | --- |
| `skipped_sparse_depth` | under 30% of the box core has valid depth |
| `skipped_mixed_depth` | no surface holds 60% of the valid pixels (object + background, or an occluder) |
| `skipped_range` | the surface is nearer than 0.4 m or beyond 4.5 m, or the box holds only scattered depth beyond 4.5 m |
| `skipped_height` | the point is outside `vision_z_min_m` / `vision_z_max_m` |

`used` counts boxes that did. `box_depth.by_label` splits the same counts
per YOLO label (plus `not_landmark_label`), which shows which classes rarely
produce a usable surface.

The box range is 0.4–4.5 m. It was tried out to 6 m, and a poster at 5.5 m
repeated within one depth step from one pose, but landmarks stored at
5.5–6.0 m were deleted on the next viewpoint, where the camera read 9–12 m.
A box past 4.5 m is labelled `far …m`. A detector landmark confirms after 8
observations from more than one view, and only when its position has
scattered by 12 cm or less. Pose writeback pairs confirmed landmarks out to
`landmark_fix_max_range_m` (6 m) and accepts a fit that agrees within
`landmark_fix_residual_m` (15 cm). It still needs 3 pairs.

A box mostly filled by something standing in
front of the object still lands on that nearer thing: it is a real surface,
so the audit will not flag it.

`vision_status.capture_pose` counts frames whose capture-time pose had to
come from the grab-time stamp (`grab_time_fallbacks`) or that were dropped
because no capture pose was available and they were over 1 s old
(`stale_frames_dropped`).

## Pose corrections (defaults — no mode knobs)

With `vision_enabled: true` the defaults are:

- **mapping**: build landmarks + VLAD place index. Re-seeing at least 3
  confirmed landmarks that all belong to one keyframe at least
  `builtin_loop_min_travel_m` (6 m of path) behind the robot adds a pose-graph
  loop edge. The edge is dropped if it disagrees with the rest of the graph.
  Small agreements (under 8 cm and 2°) add nothing. This is separate from
  localizing writeback, which still does not run while mapping.
- **mapping paint**: a detection labelled person, bicycle, car, or motorcycle
  blanks the lidar returns inside that box for 0.2 s, so a passer-by is not
  stored as a wall. Furniture and this robot are not blanked. The indoor
  ONNX from v1.2.0-rc49 includes `person` (appended, so older class ids stay
  put). Startup replaces an onnx that was fetched for an older class list.
- **localizing + tracking/degraded**: landmark writeback only (fine)
- **localizing + lost/unknown**: place retrieval (coarse relocalization), then
  landmark writeback — both via `apply_map_pose_correction(force=False)`

You should not need to flip `vision_pose_writeback` / `vision_place_retrieval`
when switching modes. Check `vision_status.pose_writeback_active` and
`place.retrieval_active` (true only while localizing **and** lost/unknown).

Place is meter-scale and ambiguous — it recovers when SLAM reports lost, it
does not nudge pose while tracking. Landmark writeback handles fine drift.

Writeback needs ≥3 landmark pairs (2-point SE(2) fits are unstable). Matching may use a small dense patch set that is **not** surveyed, so detector-only mapping still gets enough inliers. Rejections report `few_pairs` / `rejected_residual` / `rejected_jump` (no catch-all).

Kill switch if needed: `"vision_survey_only": true` (build only, never apply).

Watch `vision_status.writeback`, `vision_status.landmark_loop`,
`vision_status.paint_gate`, and `vision_status.place`.

Writeback `visible` / `visible_image` counts confirmed landmarks that project into the camera image (same cull as landmark-camera greens). `visible_frustum` is the older facing-filtered count for comparison.

## Survey density / nav load (rc2+)

Defaults assume a density pilot, not pose writeback:

| Attribute | Default | Why |
| --- | --- | --- |
| `vision_rate_hz` | `1.0` | Leave CPU for SLAM/nav |
| `vision_round_robin` | `true` | One camera per tick when multiple are listed |
| `vision_z_min_m` / `vision_z_max_m` | `0.25` / `2.0` | Drop floor/ceiling |
| `vision_max_patches_per_frame` | `64` | Cap dense associations per frame |
| `vision_prefer_detections` | `true` | Pool DINO inside YOLO boxes first |
| `vision_dense_fallback_patches` | `0` | Dense only when YOLO is quiet (avoids duplicate IDs) |
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

Run `clear_landmarks` (or `clear_map`, which also wipes the occupancy map)
before re-surveying after switching synthetic → DINOv2. Prefer a single camera in `vision_cameras`.

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
