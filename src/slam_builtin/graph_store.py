"""Save / load the mapping pose graph next to ``map.pgm``.

The occupancy image alone is a flat picture: once saved, nothing can move its
walls. Keeping the keyframes (node poses, their scans, the graph edges, the
legacy base layer and manual edits) lets a later session reload them as graph
nodes so loop closure and ``repair_map`` can re-solve and re-render old
geometry. The file is tied to the image it was saved with by a hash; if the
image changes without it (older module, external edit) it is ignored.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..geom import conversions as conv
from .pose_graph import GraphEdge, GraphNode, GraphScan
from .types import LogOddsGrid

KEYFRAMES_FILENAME = "keyframes.npz"
# v1: edit_xyr (N,3) for clear/mark disks. v2: edit_params (N,4) also holds
# clear_box as (x_min, y_min, x_max, y_max); disks pad the 4th column with 0.
FORMAT_VERSION = 2
_SUPPORTED_VERSIONS = frozenset({1, 2})

_EDIT_KINDS = ("clear", "mark", "clear_box")

# Disk erase/paint: (kind, x, y, radius). Box erase: (kind, x0, y0, x1, y1).
GridEdit = Union[
    Tuple[str, float, float, float],
    Tuple[str, float, float, float, float],
]


@dataclass
class SavedGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    last_seq: int
    resolution: float
    base: Optional[LogOddsGrid] = None
    edits: List[GridEdit] = field(default_factory=list)


def keyframes_path(map_dir: Path) -> Path:
    return Path(map_dir) / KEYFRAMES_FILENAME


def file_sha1(path: Path) -> Optional[str]:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _edit_params_row(edit: GridEdit) -> List[float]:
    kind = edit[0]
    if kind == "clear_box":
        if len(edit) != 5:
            raise ValueError("clear_box edit needs x_min,y_min,x_max,y_max")
        return [float(edit[1]), float(edit[2]), float(edit[3]), float(edit[4])]
    if len(edit) < 4:
        raise ValueError(f"{kind} edit needs x,y,radius")
    return [float(edit[1]), float(edit[2]), float(edit[3]), 0.0]


def _decode_edit(kind_i: int, row: Sequence[float], *, version: int) -> GridEdit:
    kind = _EDIT_KINDS[int(kind_i)] if int(kind_i) < len(_EDIT_KINDS) else None
    if kind is None:
        raise ValueError(f"unknown edit kind {kind_i}")
    if kind == "clear_box":
        if version < 2 or len(row) < 4:
            raise ValueError("clear_box requires format v2")
        return (kind, float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    return (kind, float(row[0]), float(row[1]), float(row[2]))


def save_graph(
    map_dir: Path,
    *,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    last_seq: int,
    resolution: float,
    image_sha1: str,
    base: Optional[LogOddsGrid] = None,
    edits: Optional[List[GridEdit]] = None,
) -> Path:
    """Atomically write ``keyframes.npz`` for the image hashed ``image_sha1``."""
    scans = [(k, s) for k, n in enumerate(nodes) for s in n.scans]
    lengths = np.array([s.ranges.size for _, s in scans], dtype=np.int32)
    flat = (
        np.concatenate([np.asarray(s.ranges, dtype=np.float16) for _, s in scans])
        if scans
        else np.zeros(0, dtype=np.float16)
    )
    edit_list = list(edits or [])
    arrays = {
        "version": np.array(FORMAT_VERSION, dtype=np.int32),
        "image_sha1": np.array(image_sha1),
        "resolution": np.array(float(resolution)),
        "last_seq": np.array(int(last_seq), dtype=np.int64),
        "node_pose": np.array(
            [[n.pose.x, n.pose.y, n.pose.theta] for n in nodes], dtype=np.float64
        ).reshape(-1, 3),
        "node_travel": np.array([n.travel_m for n in nodes], dtype=np.float64),
        "node_inserts": np.array([n.inserts for n in nodes], dtype=np.int32),
        "scan_node": np.array([k for k, _ in scans], dtype=np.int32),
        "scan_seq": np.array([s.seq for _, s in scans], dtype=np.int64),
        "scan_rel": np.array(
            [[s.rel.x, s.rel.y, s.rel.theta] for _, s in scans], dtype=np.float64
        ).reshape(-1, 3),
        "scan_band": np.array([s.band for _, s in scans], dtype=np.int16),
        "scan_geom": np.array(
            [
                [s.angle_min, s.angle_increment, s.range_min, s.range_max]
                for _, s in scans
            ],
            dtype=np.float64,
        ).reshape(-1, 4),
        "scan_len": lengths,
        "scan_ranges": flat,
        "edge_ij": np.array([[e.i, e.j] for e in edges], dtype=np.int32).reshape(-1, 2),
        "edge_rel": np.array(
            [[e.rel.x, e.rel.y, e.rel.theta] for e in edges], dtype=np.float64
        ).reshape(-1, 3),
        "edge_info": np.array([e.info for e in edges], dtype=np.float64).reshape(
            -1, 3, 3
        ),
        "edge_loop": np.array([e.loop for e in edges], dtype=bool),
        "edit_kind": np.array(
            [_EDIT_KINDS.index(k) for k, *_ in edit_list], dtype=np.int8
        ),
        "edit_params": np.array(
            [_edit_params_row(e) for e in edit_list], dtype=np.float64
        ).reshape(-1, 4),
    }
    if base is not None:
        arrays["base_log_odds"] = np.asarray(base.log_odds, dtype=np.float16)
        arrays["base_origin"] = np.array(
            [base.origin_x, base.origin_y, base.resolution], dtype=np.float64
        )
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    path = keyframes_path(map_dir)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


def load_graph(
    map_dir: Path,
    *,
    image_sha1: Optional[str],
    resolution: float,
) -> Optional[SavedGraph]:
    """Keyframes saved with this exact image, or ``None`` (missing / stale / bad)."""
    path = keyframes_path(map_dir)
    if image_sha1 is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            version = int(z["version"])
            if version not in _SUPPORTED_VERSIONS:
                return None
            if str(z["image_sha1"]) != image_sha1:
                return None
            if abs(float(z["resolution"]) - float(resolution)) > 1e-9:
                return None
            return _decode(z, version=version)
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _decode(z, *, version: int) -> SavedGraph:
    node_pose = z["node_pose"]
    nodes = [
        GraphNode(
            pose=conv.Pose2D(float(p[0]), float(p[1]), float(p[2])),
            travel_m=float(t),
            scans=[],
            inserts=int(k),
        )
        for p, t, k in zip(node_pose, z["node_travel"], z["node_inserts"])
    ]
    flat = z["scan_ranges"]
    offset = 0
    for k, seq, rel, band, geom, length in zip(
        z["scan_node"],
        z["scan_seq"],
        z["scan_rel"],
        z["scan_band"],
        z["scan_geom"],
        z["scan_len"],
    ):
        ranges = np.asarray(flat[offset : offset + int(length)], dtype=np.float32)
        offset += int(length)
        nodes[int(k)].scans.append(
            GraphScan(
                seq=int(seq),
                rel=conv.Pose2D(float(rel[0]), float(rel[1]), float(rel[2])),
                ranges=ranges,
                angle_min=float(geom[0]),
                angle_increment=float(geom[1]),
                range_min=float(geom[2]),
                range_max=float(geom[3]),
                band=int(band),
            )
        )
    n = len(nodes)
    edges = []
    for (i, j), rel, info, loop in zip(
        z["edge_ij"], z["edge_rel"], z["edge_info"], z["edge_loop"]
    ):
        if int(j) >= n or int(i) >= n:
            raise ValueError("edge references a missing node")
        edges.append(
            GraphEdge(
                int(i),
                int(j),
                conv.Pose2D(float(rel[0]), float(rel[1]), float(rel[2])),
                np.array(info, dtype=float),
                loop=bool(loop),
            )
        )
    if version >= 2 and "edit_params" in z.files:
        rows = z["edit_params"]
    else:
        rows = z["edit_xyr"]
    edits = [
        _decode_edit(int(kind), row, version=version)
        for kind, row in zip(z["edit_kind"], rows)
    ]
    base = None
    if "base_log_odds" in z.files:
        ox, oy, res = (float(v) for v in z["base_origin"])
        base = LogOddsGrid(
            log_odds=np.asarray(z["base_log_odds"], dtype=np.float32),
            resolution=res,
            origin_x=ox,
            origin_y=oy,
        )
    return SavedGraph(
        nodes=nodes,
        edges=edges,
        last_seq=int(z["last_seq"]),
        resolution=float(z["resolution"]),
        base=base,
        edits=edits,
    )
