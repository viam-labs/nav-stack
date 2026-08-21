#!/usr/bin/env bash
# Diagnose the external-SLAM /map bridge: is Nav2 getting a usable occupancy grid,
# and does it cover the robot? (the "Robot is out of bounds of the costmap" check).
#
# Usage:
#   scripts/check_map.sh [map_topic] [map_frame] [base_frame]
# Defaults: /map  map  base_link
# For a shadow/compare setup: scripts/check_map.sh /map_ext map_ext base_link
set -euo pipefail

MAP_TOPIC="${1:-/map}"
MAP_FRAME="${2:-map}"
BASE_FRAME="${3:-base_link}"

# Match the module's ROS env so we see its topics/TF. ROS setup.bash references
# optional unset vars (e.g. AMENT_TRACE_SETUP_FILES); disable `set -u` while
# sourcing or it aborts (same guard as run.sh's source_ros_setup).
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

MAP_TOPIC="$MAP_TOPIC" MAP_FRAME="$MAP_FRAME" BASE_FRAME="$BASE_FRAME" python3 - <<'PY'
import os
import sys
import time

import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener

MAP_TOPIC = os.environ["MAP_TOPIC"]
MAP_FRAME = os.environ["MAP_FRAME"]
BASE_FRAME = os.environ["BASE_FRAME"]

rclpy.init()
node = rclpy.create_node("navstack_map_check")

# Match the adapter's latched publisher (transient-local) or we get nothing.
qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
latest = {"msg": None}
node.create_subscription(
    OccupancyGrid, MAP_TOPIC, lambda m: latest.__setitem__("msg", m), qos
)
tf_buffer = Buffer()
TransformListener(tf_buffer, node)

print(f"listening on {MAP_TOPIC} ({MAP_FRAME} -> {BASE_FRAME}) ...")
end = time.time() + 6.0
while time.time() < end and latest["msg"] is None:
    rclpy.spin_once(node, timeout_sec=0.2)

msg = latest["msg"]
if msg is None:
    print(f"\nFAIL: no message on {MAP_TOPIC} within 6s.")
    print("  -> adapter not publishing /map (get_grid empty/failed?), or QoS/discovery mismatch.")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1)

# Robot pose in the map frame (best-effort; a few spins to fill the TF buffer).
robot = None
end2 = time.time() + 3.0
while time.time() < end2 and robot is None:
    rclpy.spin_once(node, timeout_sec=0.1)
    try:
        t = tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, Time())
        robot = (t.transform.translation.x, t.transform.translation.y)
    except Exception:
        pass

res = msg.info.resolution
w, h = msg.info.width, msg.info.height
ox = msg.info.origin.position.x
oy = msg.info.origin.position.y
x_max, y_max = ox + w * res, oy + h * res

try:
    import numpy as np

    a = np.asarray(msg.data, dtype=np.int16)
    unknown = int((a < 0).sum())
    free = int((a == 0).sum())
    occ = int((a > 0).sum())
except Exception:
    unknown = sum(1 for v in msg.data if v < 0)
    free = sum(1 for v in msg.data if v == 0)
    occ = sum(1 for v in msg.data if v > 0)

total = len(msg.data)
print("\n=== /map ===")
print(f"  size         : {w} x {h} cells @ {res:.4f} m  ({w * res:.2f} x {h * res:.2f} m)")
print(f"  origin       : ({ox:.3f}, {oy:.3f})")
print(f"  extent       : x [{ox:.3f} .. {x_max:.3f}]  y [{oy:.3f} .. {y_max:.3f}]")
print(f"  cells        : {total}  (expected {w * h})")
print(f"  unknown(-1)  : {unknown}  ({100.0 * unknown / max(total,1):.1f}%)")
print(f"  free(0)      : {free}")
print(f"  occupied(>0) : {occ}")

problems = []
if total != w * h:
    problems.append(f"data length {total} != w*h {w*h} (grid decode wrong)")
if w == 0 or h == 0:
    problems.append("zero-size grid (get_grid returned empty)")
if total and unknown == total:
    problems.append("grid is 100% unknown (rtabmap has no map, or all cells -1)")
if res <= 0:
    problems.append(f"nonsense resolution {res}")

print("\n=== robot ===")
if robot is None:
    print(f"  pose         : UNAVAILABLE ({MAP_FRAME} -> {BASE_FRAME} TF not found)")
    problems.append(f"no {MAP_FRAME}->{BASE_FRAME} TF (adapter map->odom or bridge odom->base_link down)")
else:
    rx, ry = robot
    inside = (ox <= rx <= x_max) and (oy <= ry <= y_max)
    print(f"  pose         : ({rx:.3f}, {ry:.3f}) in {MAP_FRAME}")
    print(f"  in map bounds: {'YES' if inside else 'NO'}")
    if not inside:
        problems.append(
            f"robot ({rx:.3f},{ry:.3f}) OUTSIDE map extent "
            f"-> 'Robot is out of bounds of the costmap'"
        )

print("\n=== verdict ===")
if problems:
    for p in problems:
        print(f"  FAIL: {p}")
    rc = 1
else:
    print("  OK: /map is valid and covers the robot.")
    rc = 0

node.destroy_node()
rclpy.shutdown()
sys.exit(rc)
PY
