"""Place recognition: build an index from a mapping run, query later runs.

Answers "how often is one of the top-K place guesses close enough to the true
pose for a lidar search around it to succeed?" for several ways of turning a
frame's patch tokens into one descriptor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.geom import conversions as conv
from src.semantic.keyframes import mutual_matches
from src.semantic.place import PlaceIndex, l2_normalize

from .features import FrameFeatures
from .lilocbench import LilocSequence

# Runtime defaults (``SlamConfig``).
KEYFRAME_GAP_M = 0.75
KEYFRAME_GAP_DEG = 25.0
VOCAB_SIZE = 32
PCA_DIM = 128
MIN_KEYFRAMES_TO_FIT = 8


def select_keyframes(
    poses: Sequence[conv.Pose2D],
    *,
    gap_m: float = KEYFRAME_GAP_M,
    gap_deg: float = KEYFRAME_GAP_DEG,
) -> List[int]:
    """Frame indices the runtime would keep (0.75 m or 25° since the last)."""
    keep: List[int] = []
    last: Optional[conv.Pose2D] = None
    for i, pose in enumerate(poses):
        if last is not None:
            moved = math.hypot(pose.x - last.x, pose.y - last.y)
            turned = abs(math.degrees(conv.normalize_angle(pose.theta - last.theta)))
            if moved < gap_m and turned < gap_deg:
                continue
        keep.append(i)
        last = pose
    return keep


@dataclass
class PlaceMethod:
    """A descriptor: ``build`` sees the keyframes, ``describe`` one frame."""

    name: str
    note: str
    build: Callable[[FrameFeatures, Sequence[int]], np.ndarray]
    describe: Callable[[FrameFeatures, int], np.ndarray]


def _runtime_vlad(fit_at_end: bool, all_tokens: bool) -> PlaceMethod:
    """The shipped ``PlaceIndex`` (PCA + k-means VLAD)."""
    state: Dict[str, PlaceIndex] = {}

    def tokens_of(feats: FrameFeatures, i: int) -> np.ndarray:
        return feats.tokens(i) if all_tokens else feats.runtime_tokens(i)

    def build(feats: FrameFeatures, keyframes: Sequence[int]) -> np.ndarray:
        index = PlaceIndex()
        for n, i in enumerate(keyframes):
            index.add_keyframe(n, conv.Pose2D(0.0, 0.0, 0.0), tokens_of(feats, i))
            if not fit_at_end and not index.fitted and n + 1 >= MIN_KEYFRAMES_TO_FIT:
                index.fit(n_clusters=VOCAB_SIZE, pca_dim=PCA_DIM)
        if fit_at_end:
            index.fit(n_clusters=VOCAB_SIZE, pca_dim=PCA_DIM)
        state["index"] = index
        return np.asarray(index.keyframe_vlad, dtype=np.float32)

    def describe(feats: FrameFeatures, i: int) -> np.ndarray:
        return state["index"].vlad(tokens_of(feats, i))

    if not fit_at_end and not all_tokens:
        name, note = "shipped", "VLAD, 64 textured tokens, vocabulary fitted on first 8 keyframes"
    elif fit_at_end and not all_tokens:
        name, note = "vlad_refit", "VLAD, 64 textured tokens, vocabulary fitted on all keyframes"
    else:
        name, note = "vlad_refit_all", "VLAD, all 768 tokens, vocabulary fitted on all keyframes"
    return PlaceMethod(name, note, build, describe)


def _mean_pool(center: bool) -> PlaceMethod:
    """Average of all tokens; no fitting. ``center`` removes the map's mean token."""
    state: Dict[str, np.ndarray] = {}

    def describe(feats: FrameFeatures, i: int) -> np.ndarray:
        pooled = feats.tokens(i).mean(axis=0)
        if center:
            pooled = pooled - state["mean"]
        return l2_normalize(pooled)

    def build(feats: FrameFeatures, keyframes: Sequence[int]) -> np.ndarray:
        if center:
            state["mean"] = np.mean(
                [feats.tokens(i).mean(axis=0) for i in keyframes], axis=0
            )
        return np.stack([describe(feats, i) for i in keyframes])

    if center:
        return PlaceMethod(
            "mean_centered", "mean of all tokens minus the map-wide mean", build, describe
        )
    return PlaceMethod("mean", "mean of all tokens, nothing fitted", build, describe)


def methods() -> List[PlaceMethod]:
    return [
        _runtime_vlad(fit_at_end=False, all_tokens=False),
        _runtime_vlad(fit_at_end=True, all_tokens=False),
        _runtime_vlad(fit_at_end=True, all_tokens=True),
        _mean_pool(center=False),
        _mean_pool(center=True),
    ]


@dataclass
class PlaceRanking:
    """Keyframes ranked for each query frame."""

    method: str
    order: np.ndarray  # (queries, top) keyframe ranks, best first
    scores: np.ndarray  # (queries, top) cosine similarity


def rank_queries(
    method: PlaceMethod,
    map_feats: FrameFeatures,
    keyframes: Sequence[int],
    query_feats: FrameFeatures,
    query_frames: Sequence[int],
    *,
    top: int = 10,
) -> PlaceRanking:
    bank = l2_normalize(method.build(map_feats, keyframes))
    queries = l2_normalize(
        np.stack([method.describe(query_feats, i) for i in query_frames])
    )
    sims = queries @ bank.T
    top = min(int(top), sims.shape[1])
    order = np.argsort(-sims, axis=1)[:, :top]
    return PlaceRanking(method.name, order, np.take_along_axis(sims, order, axis=1))


PATCH_SUFFIX = "+patches"
PATCH_SHORTLIST = 30


def rerank_by_patches(
    ranking: PlaceRanking,
    map_feats: FrameFeatures,
    keyframes: Sequence[int],
    query_feats: FrameFeatures,
    query_frames: Sequence[int],
) -> PlaceRanking:
    """Reorder each query's shortlist by its count of mutually matched patches."""
    cache: Dict[int, np.ndarray] = {}
    order = np.empty_like(ranking.order)
    scores = np.empty(ranking.order.shape, dtype=np.float32)
    for n, frame in enumerate(query_frames):
        q_tokens = query_feats.tokens(frame)
        counts = []
        for k in ranking.order[n]:
            k = int(k)
            if k not in cache:
                cache[k] = map_feats.tokens(keyframes[k])
            counts.append(len(mutual_matches(q_tokens, cache[k])))
        counts = np.asarray(counts, dtype=np.float32)
        by_count = np.argsort(-counts, kind="stable")
        order[n] = ranking.order[n][by_count]
        scores[n] = counts[by_count]
    return PlaceRanking(ranking.method + PATCH_SUFFIX, order, scores)


def rank_named(
    name: str,
    map_feats: FrameFeatures,
    keyframes: Sequence[int],
    query_feats: FrameFeatures,
    query_frames: Sequence[int],
    *,
    top: int = 10,
) -> PlaceRanking:
    """Rank by method name; ``<name>+patches`` adds the patch re-rank."""
    base = name[: -len(PATCH_SUFFIX)] if name.endswith(PATCH_SUFFIX) else name
    by_name = {m.name: m for m in methods()}
    if base not in by_name:
        raise ValueError(f"unknown place method {name!r}; have {sorted(by_name)}")
    if base == name:
        return rank_queries(by_name[base], map_feats, keyframes, query_feats, query_frames, top=top)
    shortlist = rank_queries(
        by_name[base], map_feats, keyframes, query_feats, query_frames,
        top=max(int(top), PATCH_SHORTLIST),
    )
    ranked = rerank_by_patches(shortlist, map_feats, keyframes, query_feats, query_frames)
    return PlaceRanking(name, ranked.order[:, :top], ranked.scores[:, :top])


def method_notes() -> Dict[str, str]:
    notes = {m.name: m.note for m in methods()}
    for base in ("shipped", "mean"):
        notes[base + PATCH_SUFFIX] = (
            f"{base} shortlist of {PATCH_SHORTLIST}, reordered by mutually matched patches"
        )
    return notes


def pose_errors(
    keyframe_poses: Sequence[conv.Pose2D], query_poses: Sequence[conv.Pose2D]
) -> Tuple[np.ndarray, np.ndarray]:
    """``(queries, keyframes)`` distance in metres and heading gap in degrees."""
    kf = np.array([[p.x, p.y, p.theta] for p in keyframe_poses])
    q = np.array([[p.x, p.y, p.theta] for p in query_poses])
    dist = np.hypot(q[:, None, 0] - kf[None, :, 0], q[:, None, 1] - kf[None, :, 1])
    dyaw = q[:, None, 2] - kf[None, :, 2]
    dyaw = np.abs(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw))))
    return dist, dyaw


def recall_at_k(
    ranking: PlaceRanking,
    dist: np.ndarray,
    dyaw: np.ndarray,
    *,
    ks: Sequence[int] = (1, 3, 5),
    max_dist_m: float = 1.5,
    max_yaw_deg: Optional[float] = None,
) -> Dict[str, float]:
    """Share of queries with a close keyframe in the top K.

    ``possible`` is the share that have any close keyframe at all: the ceiling
    for every method, set by where the mapping run went and which way it faced.
    """
    good = dist <= max_dist_m
    if max_yaw_deg is not None:
        good &= dyaw <= max_yaw_deg
    hit = np.take_along_axis(good, ranking.order, axis=1)
    out = {"possible": float(good.any(axis=1).mean())}
    for k in ks:
        out[f"top{k}"] = float(hit[:, :k].any(axis=1).mean())
    return out


def evaluate_place(
    mapping: LilocSequence,
    map_feats: FrameFeatures,
    query: LilocSequence,
    query_feats: FrameFeatures,
    *,
    query_stride: int = 1,
    max_dist_m: float = 1.5,
    max_yaw_deg: float = 45.0,
) -> Dict[str, dict]:
    keyframes = select_keyframes(mapping.poses)
    kf_poses = [mapping.poses[i] for i in keyframes]
    q_frames = list(range(0, len(query), max(1, int(query_stride))))
    q_poses = [query.poses[i] for i in q_frames]
    dist, dyaw = pose_errors(kf_poses, q_poses)
    out: Dict[str, dict] = {}
    for name, note in method_notes().items():
        ranking = rank_named(name, map_feats, keyframes, query_feats, q_frames)
        margin = ranking.scores[:, 0] - ranking.scores[:, 1]
        top1_ok = np.take_along_axis(
            (dist <= max_dist_m) & (dyaw <= max_yaw_deg), ranking.order[:, :1], axis=1
        )[:, 0]
        out[name] = {
            "note": note,
            "keyframes": len(keyframes),
            "queries": len(q_frames),
            "any_heading": recall_at_k(ranking, dist, dyaw, max_dist_m=max_dist_m),
            "same_heading": recall_at_k(
                ranking, dist, dyaw, max_dist_m=max_dist_m, max_yaw_deg=max_yaw_deg
            ),
            "top1_score_right": float(ranking.scores[top1_ok, 0].mean()) if top1_ok.any() else None,
            "top1_score_wrong": float(ranking.scores[~top1_ok, 0].mean()) if (~top1_ok).any() else None,
            "top1_margin_right": float(margin[top1_ok].mean()) if top1_ok.any() else None,
            "top1_margin_wrong": float(margin[~top1_ok].mean()) if (~top1_ok).any() else None,
        }
    return out
