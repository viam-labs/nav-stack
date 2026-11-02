"""Named hard-won literals shared across supervisor / controller / escape.

Config knobs (``BuiltinNavConfig``) stay out of here — these are policy
numbers discovered in the field, not operator settings. Deadband / crawl
floors that the base sanitizer already names live in ``base_envelope``;
import those rather than duplicating.
"""
from __future__ import annotations

import math

# --- Corridor open-heading motion (supervisor / final_gate / block_policy) ---
# A few degrees is "closed" while translating; pursuit can finish the rest.
CORRIDOR_ALIGN_RAD = math.radians(3.0)
# Opening this far off the nose: arc-drive would sweep into the wall — spin.
CORRIDOR_SPIN_RAD = math.radians(20.0)
# Opening must clear at least this far to count as a usable corridor.
CORRIDOR_MIN_TRAVEL_M = 0.8
# Search only headings within ±90° of the nose.
CORRIDOR_MAX_YAW_RAD = math.radians(90.0)
# Slow crawl while corridor-aligned so the footprint guard can still veto.
CORRIDOR_DRIVE_MPS = 0.15

# Heading must improve by this much to refresh the final-spin give-up clock.
YAW_ALIGN_PROGRESS_RAD = math.radians(6.0)

# Failed replans while nose-blocked before we escalate / fail the goal.
BLOCKED_REPLAN_FAIL_LIMIT = 8

# --- Escape rear-clear / spin margins ---
# Lidar rear must clear past the body by this much (on top of robot radius).
REAR_CLEAR_PAST_BODY_M = 0.08
# Absolute floor for rear-clear even on tiny footprints.
REAR_CLEAR_FLOOR_M = 0.25
# Rectangle free-space that counts as "rear open" for a reverse.
RECTANGLE_REAR_MIN_FREE_M = 0.12
# Extra spin-disc padding beyond spin_radius so corners do not graze.
SPIN_CLEAR_MARGIN_M = 0.05
# Soft reverse that counts as "we moved" before preferring a turn instead.
UNSTICK_REVERSE_NUDGE_M = 0.05

# --- Stop-bubble / footprint padding (supervisor ObstacleConfig) ---
# Lidar stop must clear the bumper by this beyond nose_offset.
NOSE_STOP_MARGIN_M = 0.05
# Half-width pad so shoulder grazes count as "ahead" without sealing doorways.
AHEAD_HALF_WIDTH_PAD_M = 0.12

# Cap translate yaw under the sanitizer's SPIN_W_CAP so arcs stay arcs.
TRANSLATE_YAW_CAP_RAD_S = 0.22
