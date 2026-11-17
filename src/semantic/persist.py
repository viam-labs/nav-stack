"""Save / load the place index next to a map directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..geom import conversions as conv
from .place import PlaceIndex

PLACE_INDEX_FILENAME = "place_index.npz"


def place_index_path(map_dir: Path) -> Path:
    return Path(map_dir) / PLACE_INDEX_FILENAME


def save_place_index(map_dir: Path, index: PlaceIndex) -> Optional[Path]:
    """Persist fitted (or mean-pool) keyframe VLAD index next to the map."""
    if not index.keyframe_ids or index.keyframe_vlad is None:
        path = place_index_path(map_dir)
        if path.is_file():
            path.unlink()
        return None
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    path = place_index_path(map_dir)
    n = len(index.keyframe_ids)
    poses = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(index.keyframe_poses[:n]):
        poses[i] = (float(p.x), float(p.y), float(p.theta))
    payload = {
        "keyframe_ids": np.asarray(index.keyframe_ids[:n], dtype=np.int32),
        "keyframe_poses": poses,
        "keyframe_vlad": np.asarray(index.keyframe_vlad[:n], dtype=np.float32),
        "cosine_threshold": np.float32(index.cosine_threshold),
    }
    if index.pca_components is not None and index.pca_mean is not None:
        payload["pca_components"] = np.asarray(index.pca_components, dtype=np.float32)
        payload["pca_mean"] = np.asarray(index.pca_mean, dtype=np.float32)
    if index.vocabulary is not None and index.vocabulary.size:
        payload["vocabulary"] = np.asarray(index.vocabulary, dtype=np.float32)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(str(tmp), **payload)
    tmp.replace(path)
    return path


def load_place_index(map_dir: Path) -> Optional[PlaceIndex]:
    path = place_index_path(map_dir)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001
        return None
    ids = [int(x) for x in data["keyframe_ids"]]
    poses_arr = data["keyframe_poses"]
    poses = [
        conv.Pose2D(
            float(poses_arr[i, 0]), float(poses_arr[i, 1]), float(poses_arr[i, 2])
        )
        for i in range(len(ids))
    ]
    index = PlaceIndex(
        keyframe_ids=ids,
        keyframe_poses=poses,
        keyframe_vlad=np.asarray(data["keyframe_vlad"], dtype=np.float32),
        cosine_threshold=float(data["cosine_threshold"])
        if "cosine_threshold" in data
        else 0.4,
        pca_components=(
            np.asarray(data["pca_components"], dtype=np.float32)
            if "pca_components" in data
            else None
        ),
        pca_mean=(
            np.asarray(data["pca_mean"], dtype=np.float32)
            if "pca_mean" in data
            else None
        ),
        vocabulary=(
            np.asarray(data["vocabulary"], dtype=np.float32)
            if "vocabulary" in data
            else None
        ),
    )
    return index
