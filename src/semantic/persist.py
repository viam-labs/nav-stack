"""Save / load embedding landmarks next to a map directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..geom import conversions as conv
from .landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from .place import PlaceIndex
from .types import LandmarkStatus

LANDMARKS_FILENAME = "embedding_landmarks.npz"
PLACE_INDEX_FILENAME = "place_index.npz"


def landmarks_path(map_dir: Path) -> Path:
    return Path(map_dir) / LANDMARKS_FILENAME


def save_landmarks(map_dir: Path, store: EmbeddingLandmarkMap) -> Optional[Path]:
    """Atomic write of confirmed + candidate landmarks. Returns path or None."""
    lms = [lm for lm in store.all_landmarks() if lm.status != LandmarkStatus.MOVED]
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    path = landmarks_path(map_dir)
    if not lms:
        if path.is_file():
            path.unlink()
        return None

    max_desc = max(len(lm.descriptors) for lm in lms)
    dim = int(lms[0].descriptors[0].shape[0]) if lms[0].descriptors else 0
    n = len(lms)
    ids = np.zeros(n, dtype=np.int32)
    anchors = np.zeros(n, dtype=np.int32)
    rel = np.zeros((n, 3), dtype=np.float32)
    map_xyz = np.zeros((n, 3), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    obs = np.zeros(n, dtype=np.int32)
    stds = np.zeros(n, dtype=np.float32)
    status = np.zeros(n, dtype=np.int8)
    n_desc = np.zeros(n, dtype=np.int8)
    from_detector = np.zeros(n, dtype=np.int8)
    extent_m = np.full(n, -1.0, dtype=np.float32)
    descs = np.zeros((n, max_desc, dim), dtype=np.float16) if dim else np.zeros((n, 0, 0))

    status_code = {
        LandmarkStatus.CANDIDATE: 0,
        LandmarkStatus.CONFIRMED: 1,
        LandmarkStatus.MOVED: 2,
    }
    for i, lm in enumerate(lms):
        ids[i] = lm.landmark_id
        anchors[i] = lm.anchor_node
        rel[i] = lm.rel_xyz.astype(np.float32)
        if lm.map_xyz is not None:
            map_xyz[i] = np.asarray(lm.map_xyz, dtype=np.float32).reshape(3)
        else:
            map_xyz[i] = lm.rel_xyz.astype(np.float32)
        normals[i] = lm.normal_rel.astype(np.float32)
        obs[i] = lm.observations
        stds[i] = float(lm.position_std_m) if np.isfinite(lm.position_std_m) else -1.0
        status[i] = status_code.get(lm.status, 0)
        n_desc[i] = len(lm.descriptors)
        from_detector[i] = 1 if bool(lm.from_detector) else 0
        if lm.extent_m is not None and float(lm.extent_m) > 0:
            extent_m[i] = float(lm.extent_m)
        for j, d in enumerate(lm.descriptors[:max_desc]):
            descs[i, j] = d.astype(np.float16)

    # numpy always appends ``.npz`` unless the path already ends with it.
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        ids=ids,
        anchors=anchors,
        rel=rel,
        map_xyz=map_xyz,
        normals=normals,
        observations=obs,
        stds=stds,
        status=status,
        n_desc=n_desc,
        from_detector=from_detector,
        extent_m=extent_m,
        descriptors=descs,
        next_id=np.int32(store._next_id),  # noqa: SLF001
        cosine_threshold=np.float32(store.cosine_threshold),
    )
    tmp.replace(path)
    return path


def load_landmarks(map_dir: Path) -> Optional[EmbeddingLandmarkMap]:
    path = landmarks_path(map_dir)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001
        return None
    store = EmbeddingLandmarkMap(
        cosine_threshold=float(data["cosine_threshold"])
        if "cosine_threshold" in data
        else 0.55
    )
    ids = data["ids"]
    anchors = data["anchors"]
    rel = data["rel"]
    normals = data["normals"]
    obs = data["observations"]
    stds = data["stds"]
    status = data["status"]
    n_desc = data["n_desc"]
    descs = data["descriptors"]
    code_status = {
        0: LandmarkStatus.CANDIDATE,
        1: LandmarkStatus.CONFIRMED,
        2: LandmarkStatus.MOVED,
    }
    for i in range(len(ids)):
        nd = int(n_desc[i])
        descriptors = [
            descs[i, j].astype(np.float32) for j in range(nd) if descs.shape[1] > j
        ]
        std = float(stds[i])
        if "map_xyz" in data:
            mxyz = data["map_xyz"][i].astype(float)
        else:
            mxyz = rel[i].astype(float)
        fd = False
        if "from_detector" in data:
            fd = bool(int(data["from_detector"][i]))
        ext = None
        if "extent_m" in data:
            ev = float(data["extent_m"][i])
            if ev > 0:
                ext = ev
        lm = EmbeddingLandmark(
            landmark_id=int(ids[i]),
            anchor_node=int(anchors[i]),
            rel_xyz=rel[i].astype(float),
            normal_rel=normals[i].astype(float),
            map_xyz=mxyz,
            descriptors=descriptors,
            observations=int(obs[i]),
            position_std_m=std if std >= 0.0 else float("inf"),
            status=code_status.get(int(status[i]), LandmarkStatus.CANDIDATE),
            from_detector=fd,
            extent_m=ext,
        )
        store._landmarks[lm.landmark_id] = lm  # noqa: SLF001
        store._index_add(lm, lm.map_xyz)  # noqa: SLF001
    if "next_id" in data:
        store._next_id = max(int(data["next_id"]), max(ids) + 1 if len(ids) else 1)  # noqa: SLF001
    else:
        store._next_id = (int(ids.max()) + 1) if len(ids) else 1  # noqa: SLF001
    return store


def pose_table_from_arrays(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray
) -> dict:
    """Helper for tests: node index → Pose2D."""
    return {
        i: conv.Pose2D(float(xs[i]), float(ys[i]), float(thetas[i]))
        for i in range(len(xs))
    }


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
        conv.Pose2D(float(poses_arr[i, 0]), float(poses_arr[i, 1]), float(poses_arr[i, 2]))
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
