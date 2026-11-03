"""Localization cycles extracted from the SLAM Viam resource.

Owns refine / recover / startup-global / mapping-revisit cycles. The Viam
``SlamService`` remains the resource adapter and schedules these via the host.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Optional

import numpy as np
from viam.logging import getLogger
from viam.utils import ValueTypes

from ..config import MODE_LOCALIZING, MODE_MAPPING, SlamConfig
from ..geom import conversions as conv
from ..nav.localize.global_localize import GlobalLocalizeResult
from ..nav.localize.pose_jump_gate import (
    candidate_beats_previous,
    stuck_refine_decision,
)

LOGGER = getLogger(__name__)


class LocalizeSupervisor:
    """Refine / recover / global-localize cycles for one SlamService host."""

    def __init__(self, host: Any) -> None:
        self.host = host

    def _publish_revisit_check(self, result: Mapping[str, ValueTypes]) -> dict:
        self.host._last_revisit_check.clear()
        self.host._last_revisit_check.update(result)
        return dict(self.host._last_revisit_check)

    async def _flip_current_map_yaw(self) -> Mapping[str, ValueTypes]:
        """Reverse map-frame heading in place (corridor 180° recovery)."""
        mgr = self.host._manager
        if mgr is None:
            return self.host._publish_revisit_check({"status": "unconfigured"})
        current = mgr.get_pose_in_map()
        if current is None:
            return self.host._publish_revisit_check(
                {"status": "skipped", "reason": "no_pose_in_map"}
            )
        flipped = conv.Pose2D(
            current.x, current.y, conv.normalize_angle(current.theta + math.pi)
        )
        applied = await asyncio.to_thread(mgr.apply_map_pose_correction, flipped)
        out: dict = {
            "status": "corrected" if applied.get("applied") else "correction_failed",
            "flip_yaw_only": True,
            "yaw_flipped": True,
            "pose": {"x": flipped.x, "y": flipped.y, "theta": flipped.theta},
            "correction": applied,
            "corrected": bool(applied.get("applied")),
        }
        if out["corrected"]:
            LOGGER.info(
                "mapping revisit: flipped map yaw in place (%.1f -> %.1f deg)",
                math.degrees(current.theta),
                math.degrees(flipped.theta),
            )
        return self.host._publish_revisit_check(out)

    def _revisit_match_quality(
        self, result, cfg: SlamConfig
    ) -> tuple[bool, float, Optional[float]]:
        score = float(result.score)
        ray_mae = float(result.ray_mae_m) if math.isfinite(result.ray_mae_m) else None
        good = score >= cfg.mapping_revisit_min_score and (
            ray_mae is None or ray_mae <= cfg.mapping_revisit_max_ray_mae_m
        )
        return good, score, ray_mae

    async def _mapping_revisit_cycle(
        self,
        *,
        apply_override: Optional[bool] = None,
        yaw_flip: bool = False,
    ) -> Mapping[str, ValueTypes]:
        """One revisit check: tiered scan-to-live-map match, odom shift on drift.

        Tier 1 searches ``mapping_revisit_search_radius_m`` around the current
        pose; tier 2 widens to ``mapping_revisit_wide_radius_m``; tier 3 (full
        map) runs only when enabled and must clear a stricter score gate —
        self-similar offices produce convincing wrong corridors at map scale.

        After the best 2D match, ``choose_yaw_or_flip`` breaks corridor 180°
        ambiguity. Pass ``yaw_flip=True`` to take the opposite heading of that
        auto choice (for when XY is right but facing is wrong).
        """
        cfg = self.host._cfg
        mgr = self.host._manager
        if cfg is None or mgr is None:
            return self.host._publish_revisit_check({"status": "unconfigured"})
        if cfg.mode != MODE_MAPPING:
            return self.host._publish_revisit_check(
                {"status": "skipped", "reason": "not_mapping"}
            )
        node = mgr.node
        if node is None:
            return self.host._publish_revisit_check(
                {"status": "skipped", "reason": "bridge_not_started"}
            )

        # Motion gate: map_when_still + odom TF shift must not land mid-hop.
        # Continuous-scan backends (builtin) can correct while driving; still
        # skip fast spins where the scan is motion-distorted / match is weak.
        try:
            bridge_status = node.slam_bridge_status()
        except Exception:  # noqa: BLE001
            bridge_status = {}
        vel = bridge_status.get("odom_velocity") or {}
        linear_speed = math.hypot(float(vel.get("vx", 0.0)), float(vel.get("vy", 0.0)))
        yaw_rate = abs(float(vel.get("vtheta", 0.0)))
        moving = (
            linear_speed >= cfg.map_when_still_linear_speed_m_s
            or yaw_rate >= cfg.map_when_still_yaw_rate_rad_s
        )
        spinning_fast = yaw_rate >= float(cfg.mapping_revisit_max_yaw_rate_rad_s)
        if apply_override is not True:
            if spinning_fast:
                return self.host._publish_revisit_check(
                    {
                        "status": "skipped",
                        "reason": "spinning",
                        "yaw_rate_rad_s": round(yaw_rate, 3),
                    }
                )
            if moving and not cfg.mapping_revisit_while_moving:
                return self.host._publish_revisit_check(
                    {"status": "skipped", "reason": "moving"}
                )

        current = mgr.get_pose_in_map()
        if current is None:
            return self.host._publish_revisit_check(
                {"status": "skipped", "reason": "no_pose_in_map"}
            )

        try:
            occ_map, _ = self.host._load_active_occupancy_map("live")
        except Exception as exc:  # noqa: BLE001 - live map not ready yet
            return self.host._publish_revisit_check(
                {"status": "skipped", "reason": f"no_live_map: {exc}"}
            )
        library = self.host._get_slice_library(cfg)
        scan, band_points = await self.host._read_merged_scan_and_bands(
            library.bands if library is not None else None
        )

        async def _match(radius_m: float, yaw_window_deg: float, full_map: bool):
            return await self.host._localizer.run(
                "global_localize_scan",
                occ_map,
                scan,
                hint=None if full_map else current,
                full_map=full_map,
                search_radius_m=radius_m,
                local_yaw_window_deg=yaw_window_deg,
                coarse_position_step_m=0.6 if full_map else 0.4,
                coarse_yaw_step_deg=18.0 if full_map else 12.0,
            )

        tiers_tried = []
        match_mode = "local"
        result = await _match(cfg.mapping_revisit_search_radius_m, 90.0, False)
        good, score, ray_mae = self.host._revisit_match_quality(result, cfg)
        tiers_tried.append({"tier": "local", "score": round(score, 3)})

        if not good:
            match_mode = "wide"
            wide = await _match(cfg.mapping_revisit_wide_radius_m, 180.0, False)
            wide_good, wide_score, wide_ray_mae = self.host._revisit_match_quality(
                wide, cfg
            )
            tiers_tried.append({"tier": "wide", "score": round(wide_score, 3)})
            if wide_good or wide_score > score:
                result, good, score, ray_mae = wide, wide_good, wide_score, wide_ray_mae

        if not good and cfg.mapping_revisit_full_map_fallback:
            match_mode = "full_map"
            full = await _match(0.0, 360.0, True)
            _, full_score, full_ray_mae = self.host._revisit_match_quality(full, cfg)
            tiers_tried.append({"tier": "full_map", "score": round(full_score, 3)})
            # Full map needs the stricter gate regardless of ray MAE outcome.
            if full_score >= cfg.mapping_revisit_full_map_min_score and (
                full_ray_mae is None
                or full_ray_mae <= cfg.mapping_revisit_max_ray_mae_m
            ):
                result, good, score, ray_mae = full, True, full_score, full_ray_mae

        kf_info = None
        # Pause keyframes: when occupancy match is weak, try stored stop views
        # (2D + height slices) — covers returning at a different pause angle.
        if (
            not good
            and cfg.mapping_revisit_keyframes
            and self.host._pause_keyframes is not None
            and len(self.host._pause_keyframes) > 0
        ):
            store = self.host._pause_keyframes

            def _kf_match(radius: Optional[float]):
                with self.host._keyframe_lock:
                    return store.match(
                        scan,
                        band_points,
                        hint=current,
                        search_radius_m=radius,
                    )

            kf = await asyncio.to_thread(_kf_match, cfg.mapping_revisit_wide_radius_m)
            scope = "near"
            if kf is None or kf.score < cfg.mapping_revisit_keyframe_min_score:
                kf_all = await asyncio.to_thread(_kf_match, None)
                scope = "all"
                if kf_all is not None and (kf is None or kf_all.score > kf.score):
                    kf = kf_all
            if kf is not None:
                tiers_tried.append(
                    {
                        "tier": f"keyframe_{scope}",
                        "score": round(kf.score, 3),
                        "keyframes": kf.keyframes_considered,
                    }
                )
                kf_info = {
                    "score": round(kf.score, 3),
                    "primary_hit_rate": round(kf.primary_hit_rate, 3),
                    "slice_hit_rate": (
                        None
                        if kf.slice_hit_rate is None
                        else round(kf.slice_hit_rate, 3)
                    ),
                    "keyframe_index": kf.keyframe_index,
                    "scope": scope,
                }
                if kf.score >= cfg.mapping_revisit_keyframe_min_score:
                    # Ray-align against the live map at the keyframe pose so
                    # the usual MAE gate still applies.
                    yaw_probe = await self.host._localizer.run(
                        "choose_yaw_or_flip",
                        occ_map,
                        scan,
                        kf.pose,
                        reference_theta=current.theta,
                    )
                    result = GlobalLocalizeResult(
                        pose=yaw_probe.pose,
                        score=kf.score,
                        candidates_evaluated=kf.keyframes_considered,
                        scan_points_used=0,
                        in_map_points=0,
                        hit_rate=kf.primary_hit_rate,
                        ray_score=0.0,
                        ray_mae_m=yaw_probe.ray_mae_m,
                    )
                    match_mode = "keyframe"
                    good = kf.score >= cfg.mapping_revisit_keyframe_min_score and (
                        not math.isfinite(yaw_probe.ray_mae_m)
                        or yaw_probe.ray_mae_m <= cfg.mapping_revisit_max_ray_mae_m
                    )
                    score = float(kf.score)
                    ray_mae = (
                        float(yaw_probe.ray_mae_m)
                        if math.isfinite(yaw_probe.ray_mae_m)
                        else None
                    )

        # Corridor 180° ambiguity: same XY often scores similarly both ways.
        # Re-score pose vs pose+π and prefer IMU-nearer heading on a near-tie.
        yaw_choice = await self.host._localizer.run(
            "choose_yaw_or_flip",
            occ_map,
            scan,
            result.pose,
            reference_theta=current.theta,
        )
        if yaw_flip:
            chosen_pose = yaw_choice.alt_pose
            chosen_score = yaw_choice.alt_score
            chosen_mae = yaw_choice.alt_ray_mae_m
            did_flip = not yaw_choice.flipped
        else:
            chosen_pose = yaw_choice.pose
            chosen_score = yaw_choice.score
            chosen_mae = yaw_choice.ray_mae_m
            did_flip = yaw_choice.flipped
        if match_mode == "keyframe":
            # Keep keyframe hit-rate as the score (occupancy score scale differs).
            result = replace(
                result,
                pose=chosen_pose,
                ray_mae_m=chosen_mae,
            )
            ray_mae = float(chosen_mae) if math.isfinite(chosen_mae) else None
            good = score >= cfg.mapping_revisit_keyframe_min_score and (
                ray_mae is None or ray_mae <= cfg.mapping_revisit_max_ray_mae_m
            )
        else:
            result = replace(
                result,
                pose=chosen_pose,
                score=chosen_score,
                ray_mae_m=chosen_mae,
            )
            good, score, ray_mae = self.host._revisit_match_quality(result, cfg)

        shift_m = math.hypot(result.pose.x - current.x, result.pose.y - current.y)
        shift_deg = abs(
            math.degrees(conv.normalize_angle(result.pose.theta - current.theta))
        )
        drifted = (
            shift_m >= cfg.mapping_revisit_min_shift_m
            or shift_deg >= cfg.mapping_revisit_min_shift_deg
        )
        sane = shift_m <= cfg.mapping_revisit_max_shift_m

        out: dict = {
            "status": "ok",
            "match_mode": match_mode,
            "tiers": tiers_tried,
            "score": round(score, 3),
            "ray_mae_m": None if ray_mae is None else round(ray_mae, 3),
            "shift_m": round(shift_m, 3),
            "shift_deg": round(shift_deg, 2),
            "good_match": good,
            "drifted": drifted,
            "pose": {
                "x": result.pose.x,
                "y": result.pose.y,
                "theta": result.pose.theta,
            },
            "yaw_flipped": did_flip,
            "yaw_alt": {
                "theta": yaw_choice.alt_pose.theta,
                "score": round(yaw_choice.alt_score, 3),
                "ray_mae_m": round(yaw_choice.alt_ray_mae_m, 3),
            },
            "corrected": False,
        }
        if kf_info is not None:
            out["keyframe_match"] = kf_info
        if self.host._pause_keyframes is not None:
            out["keyframes"] = self.host._pause_keyframes.status()

        should_apply = apply_override is True or (
            apply_override is None and good and drifted and sane
        )
        if not sane and apply_override is not True:
            out["status"] = "rejected_shift"
            LOGGER.warning(
                "mapping revisit: match %.1f m away exceeds "
                "mapping_revisit_max_shift_m=%.1f; not correcting (score=%.2f)",
                shift_m,
                cfg.mapping_revisit_max_shift_m,
                score,
            )
            return self.host._publish_revisit_check(out)
        if not good and apply_override is not True:
            out["status"] = "low_quality"
            return self.host._publish_revisit_check(out)

        # Multi-height-slice verification: the occupancy map only encodes the
        # primary z-band, which is self-similar across desk clutter. Before
        # shifting odom, require the proposed pose to also agree with any
        # height bands we have reference geometry for (recorded from earlier
        # trusted pause scans). An explicit apply_override skips the veto.
        if should_apply and drifted and library is not None:
            verdict = await asyncio.to_thread(library.verify, band_points, result.pose)
            out["slice_verify"] = verdict
            if verdict.get("pass") is False and apply_override is not True:
                out["status"] = "slice_verify_failed"
                LOGGER.warning(
                    "mapping revisit: 2D match at (%.2f, %.2f) rejected — "
                    "%d/%d height slices disagree with stored geometry",
                    result.pose.x,
                    result.pose.y,
                    verdict.get("bands_failed", 0),
                    verdict.get("bands_checked", 0),
                )
                return self.host._publish_revisit_check(out)

        if should_apply:
            decision = self.host._confirm_pose_jump(
                current, result.pose, force=apply_override is True
            )
            out.update(self.host._jump_decision_dict(decision))
            if not decision.should_apply:
                out["status"] = "awaiting_confirm"
                out["corrected"] = False
                LOGGER.info(
                    "mapping revisit: large jump %.2f m / %.1f deg awaiting "
                    "confirm %d/%d (score=%.2f)",
                    decision.shift_m,
                    decision.shift_deg,
                    decision.confirm_count,
                    decision.confirm_needed,
                    score,
                )
                return self.host._publish_revisit_check(out)
            applied = await asyncio.to_thread(
                mgr.apply_map_pose_correction, result.pose
            )
            out["correction"] = applied
            if applied.get("applied"):
                out["status"] = "corrected"
                out["corrected"] = True
                LOGGER.info(
                    "mapping revisit: odom shifted to rejoin map via %s "
                    "(shift=%.2f m, %.1f deg, score=%.2f, ray_mae=%s, jump=%s)",
                    match_mode,
                    shift_m,
                    shift_deg,
                    score,
                    ray_mae,
                    decision.status,
                )
            else:
                out["status"] = "correction_failed"
        else:
            self.host._pose_jump_gate.clear()

        # Grow the slice library from confirmed in-place / corrected poses as a
        # backup to the still-publish keyframe path (no-ops when already dense).
        if library is not None and good:
            record_pose: Optional[conv.Pose2D] = None
            if out["status"] == "corrected":
                record_pose = result.pose
            elif out["status"] == "ok" and not drifted:
                record_pose = current
            if record_pose is not None and any(
                np.asarray(pts).size for pts in band_points
            ):
                await asyncio.to_thread(library.record, band_points, record_pose)
                out["slice_scans_recorded"] = library.scans_recorded
        return self.host._publish_revisit_check(out)

    async def _stuck_pose_refine(self) -> Mapping[str, ValueTypes]:
        """One local scan match around the current pose, for a stuck navigator.

        Search stays inside about a meter and ±30° of heading. A better small
        shift is applied in mapping and localizing. An ambiguous or hallway-scale
        peak is reported and left alone. This does not run a full-map search.
        """
        cfg = self.host._cfg
        mgr = self.host._manager
        if cfg is None or mgr is None:
            return {"status": "unconfigured", "corrected": False}
        if cfg.mode not in (MODE_MAPPING, MODE_LOCALIZING):
            return {
                "status": "skipped",
                "reason": "bad_mode",
                "corrected": False,
            }
        current = mgr.get_pose_in_map()
        if current is None:
            return {
                "status": "skipped",
                "reason": "no_pose",
                "corrected": False,
            }
        try:
            occ_map, map_source = self.host._load_active_occupancy_map("live")
        except Exception as exc:  # noqa: BLE001 - live map not ready
            return {
                "status": "skipped",
                "reason": f"no_live_map: {exc}",
                "corrected": False,
            }
        try:
            scan = await self.host._read_merged_scan()
            result = await self.host._localizer.run(
                "global_localize_scan",
                occ_map,
                scan,
                hint=current,
                full_map=False,
                search_radius_m=1.0,
                local_yaw_window_deg=30.0,
                coarse_position_step_m=0.2,
                coarse_yaw_step_deg=10.0,
                fine_position_step_m=0.05,
                fine_yaw_step_deg=2.0,
                fine_window_m=0.35,
                fine_yaw_window_deg=12.0,
                ray_refine_candidates=16,
            )
            prior_score, prior_ray_mae = await self.host._localizer.run(
                "score_pose_with_rays",
                occ_map,
                scan,
                current,
            )
        except Exception as exc:  # noqa: BLE001 - match must not kill navigation
            LOGGER.warning("stuck pose refine failed: %s", exc)
            return {
                "status": "error",
                "reason": str(exc).strip() or type(exc).__name__,
                "corrected": False,
            }

        shift_m = math.hypot(result.pose.x - current.x, result.pose.y - current.y)
        shift_deg = abs(
            math.degrees(conv.normalize_angle(result.pose.theta - current.theta))
        )
        score = float(result.score)
        ray_mae = float(result.ray_mae_m) if math.isfinite(result.ray_mae_m) else None
        prior_f = float(prior_score) if prior_score is not None else None
        prior_mae = (
            float(prior_ray_mae)
            if prior_ray_mae is not None and math.isfinite(float(prior_ray_mae))
            else None
        )
        decision = stuck_refine_decision(
            shift_m=shift_m,
            shift_deg=shift_deg,
            score=score,
            prior_score=prior_f,
            ambiguous=bool(getattr(result, "ambiguous", False)),
            prior_ray_mae_m=prior_mae,
            ray_mae_m=ray_mae,
        )
        out: dict = {
            "status": decision,
            "match_mode": "local",
            "slam_mode": cfg.mode,
            "map_source": map_source,
            "score": round(score, 3) if math.isfinite(score) else None,
            "prior_score": (
                round(prior_f, 3)
                if prior_f is not None and math.isfinite(prior_f)
                else None
            ),
            "ray_mae_m": None if ray_mae is None else round(ray_mae, 3),
            "shift_m": round(shift_m, 3),
            "shift_deg": round(shift_deg, 2),
            "ambiguous": bool(getattr(result, "ambiguous", False)),
            "corrected": False,
            "pose": {
                "x": result.pose.x,
                "y": result.pose.y,
                "theta": result.pose.theta,
            },
        }
        if decision != "apply":
            LOGGER.info(
                "stuck pose refine: not applying (%s) shift=%.2f m %.1f deg "
                "score=%s prior=%s ambiguous=%s mode=%s",
                decision,
                shift_m,
                shift_deg,
                out["score"],
                out["prior_score"],
                out["ambiguous"],
                cfg.mode,
            )
            return out

        apply = getattr(mgr, "apply_map_pose_correction", None)
        if not callable(apply):
            out["status"] = "correction_failed"
            out["reason"] = "no_apply"
            return out
        applied = await asyncio.to_thread(apply, result.pose)
        self.host._pose_jump_gate.clear()
        did = bool(isinstance(applied, Mapping) and applied.get("applied"))
        out["corrected"] = did
        out["status"] = "corrected" if did else "correction_failed"
        if isinstance(applied, Mapping):
            out["correction"] = dict(applied)
        if did:
            LOGGER.info(
                "stuck pose refine: applied shift=%.2f m %.1f deg score=%s "
                "prior=%s mode=%s",
                shift_m,
                shift_deg,
                out["score"],
                out["prior_score"],
                cfg.mode,
            )
        return out

    async def _relocalize_cycle_guarded(self) -> None:
        try:
            await self.host._periodic_relocalize_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - watchdog must survive hiccups
            LOGGER.warning("periodic relocalize cycle failed: %s", exc)
            self.host._publish_relocalize_check({"status": "error", "error": str(exc)})

    def _publish_relocalize_check(self, result: Mapping[str, ValueTypes]) -> dict:
        self.host._loc_state.observe(result)
        self.host._last_relocalize_check.clear()
        self.host._last_relocalize_check.update(result)
        # The shared dict is what builtin nav reads; the state rides along so
        # a goal refusal can say how long and what to do.
        self.host._last_relocalize_check["state"] = self.host._loc_state.snapshot()
        return dict(self.host._last_relocalize_check)

    def _periodic_relocalize_preflight(
        self, cfg: SlamConfig
    ) -> Optional[Mapping[str, ValueTypes]]:
        """Skip cycles taken on smeared or stale scans (returns a status dict)."""
        yaw_rate = 0.0
        scan_age: Optional[float] = None
        mgr = self.host._manager
        try:
            node = getattr(mgr, "node", None) if mgr is not None else None
            status_fn = getattr(node, "slam_bridge_status", None) if node else None
            if callable(status_fn):
                vel = (status_fn() or {}).get("odom_velocity") or {}
                yaw_rate = abs(float(vel.get("vtheta", 0.0)))
        except Exception:  # noqa: BLE001
            pass
        try:
            engine = self.host._engine
            if engine is not None:
                diag = engine.diagnostics()
                age_raw = diag.get("last_scan_age_s")
                if age_raw is not None:
                    scan_age = float(age_raw)
                if yaw_rate <= 1e-9:
                    yaw_rate = abs(math.radians(float(diag.get("yaw_rate_deg_s", 0.0))))
        except Exception:  # noqa: BLE001
            pass

        max_yaw = float(cfg.periodic_relocalize_max_yaw_rate_rad_s)
        if max_yaw > 0.0 and yaw_rate >= max_yaw:
            return {
                "status": "skipped",
                "reason": "spinning",
                "yaw_rate_rad_s": round(yaw_rate, 3),
            }
        max_age = float(cfg.periodic_relocalize_max_scan_age_s)
        if (
            max_age > 0.0
            and scan_age is not None
            and math.isfinite(scan_age)
            and scan_age > max_age
        ):
            return {
                "status": "skipped",
                "reason": "stale_scan",
                "scan_age_s": round(scan_age, 3),
            }
        return None

    def _maybe_bypass_startup_localize(
        self, cfg: SlamConfig
    ) -> Optional[Mapping[str, ValueTypes]]:
        """Cancel a stuck/long startup localize so the watchdog can recover.

        Returns None when startup is not blocking. Otherwise either cancels and
        returns None (caller proceeds) or returns a skip status dict.
        """
        startup = self.host._startup_global_localize_task
        if startup is None or startup.done():
            return None
        bad, tick_score = self.host._published_pose_is_bad(cfg)
        bypass_after = float(cfg.periodic_relocalize_bypass_startup_after_s)
        started = self.host._startup_localize_started_at
        age_s = (time.monotonic() - float(started)) if started is not None else None
        timed_out = bypass_after > 0.0 and age_s is not None and age_s >= bypass_after
        # A restored-pose startup searches locally. Once that has had time for
        # one match, a score under the trusted bar should not keep the parked
        # check waiting through the rest of the refine loop. 0.007 was just
        # above the "still bad" line and therefore waited out the whole cap.
        soft = tick_score is not None and tick_score < float(
            cfg.periodic_relocalize_min_score
        )
        local_grace_s = 15.0
        early = (
            self.host._startup_localize_local
            and (bad or soft)
            and age_s is not None
            and age_s >= local_grace_s
        )
        if not bad and not timed_out and not early:
            return {
                "status": "skipped",
                "reason": "startup_localize_running",
                "startup_age_s": None if age_s is None else round(age_s, 1),
                "tick_match_score": tick_score,
            }
        LOGGER.warning(
            "bypassing startup global_localize for drift watchdog "
            "(bad_tick=%s score=%s age_s=%s)",
            bad,
            tick_score,
            None if age_s is None else round(age_s, 1),
        )
        self.host._cancel_startup_global_localize_task()
        return None

    async def _periodic_relocalize_cycle(
        self,
        *,
        apply_override: Optional[bool] = None,
        full_map_escalation: str = "low_quality",
        allow_during_navigation: bool = False,
        stopped_recovery: bool = False,
    ) -> Mapping[str, ValueTypes]:
        """Run one drift check; correct pose when a trusted match has moved.

        Stays on a local search whenever a map pose exists. A weak score, a
        bad prior, or several nav recoveries do not escalate: that whole-map
        search is the 15–20 s pause before the first command, and in a hallway
        the peak is usually an ambiguous twin that gets refused. Full-map runs
        only when there is no published pose to search around.

        ``full_map_escalation`` is accepted for callers and no longer starts a
        full-map search while a pose exists.

        ``allow_during_navigation``: run a local cycle while a goal is active
        (used by builtin nav's lidar↔map refine). Does not force apply, and
        never escalates to full-map — a hallway twin snap while driving.

        ``stopped_recovery``: base already halted for bad loc. Widen the local
        search, allow full-map when the published pose is still wrong, and
        permit a larger *trusted* apply so nav can fix once and continue.
        """
        escalation = str(full_map_escalation or "low_quality").strip().lower()
        if escalation not in ("low_quality", "still_bad"):
            escalation = "low_quality"
        cfg = self.host._cfg
        mgr = self.host._manager
        if cfg is None or mgr is None:
            return self.host._publish_relocalize_check({"status": "unconfigured"})
        if cfg.mode != MODE_LOCALIZING:
            return self.host._publish_relocalize_check(
                {"status": "skipped", "reason": "not_localizing"}
            )
        if apply_override is not True:
            blocked = self.host._maybe_bypass_startup_localize(cfg)
            if blocked is not None:
                return self.host._publish_relocalize_check(blocked)

        nav_active = self.host._is_navigation_active()
        # A goal that is active but not moving is the parked case: the stall
        # that just failed eliot never got a check because this skip fired
        # the whole time. Driving stays skipped.
        nav_stopped = bool(nav_active and self.host._base_is_stopped())
        driving = bool(nav_active and not nav_stopped)
        if (
            apply_override is not True
            and driving
            and not cfg.periodic_relocalize_during_navigation
            and not allow_during_navigation
            and not stopped_recovery
        ):
            return self.host._publish_relocalize_check(
                {"status": "skipped", "reason": "navigation_active"}
            )
        if apply_override is not True:
            preflight = self.host._periodic_relocalize_preflight(cfg)
            if preflight is not None:
                return self.host._publish_relocalize_check(preflight)

        nav_recoveries = 0
        if nav_active:
            try:
                nav_recoveries = int(mgr.nav_status().get("number_of_recoveries", 0))
            except Exception:  # noqa: BLE001
                nav_recoveries = 0

        still_bad, tick_score = self.host._published_pose_is_bad(cfg)
        current = mgr.get_pose_in_map()
        # Full-map only with no published pose. A weak score around a known
        # pose stays local.
        have_pose = current is not None
        force_full_map = not have_pose
        # Tick scores just above zero (the paused hallway was 0.12) are not
        # "still bad", but they are also not a pose the button would leave
        # alone. Try the 1 m window whenever the parked score is under the
        # trusted-match bar. That window needs a pose to center on.
        parked_soft = (
            have_pose
            and tick_score is not None
            and tick_score < float(cfg.periodic_relocalize_min_score)
        )
        parked_button = (
            have_pose
            and not driving
            and escalation == "low_quality"
            and not stopped_recovery
            and apply_override is not True
            and nav_recoveries < cfg.periodic_relocalize_nav_recoveries_threshold
            and (bool(still_bad) or parked_soft)
        )

        base_command: dict = {"command": "global_localize"}
        base_command.update(dict(cfg.periodic_relocalize_options))
        base_command["apply"] = False
        # A local request must not widen itself into a full-map search.
        # Full-map is chosen only when ``force_full_map`` (no published pose).
        base_command["auto_full_map_fallback"] = False
        if stopped_recovery:
            # Wider local window while halted — still prefer local before full-map.
            try:
                base_command["search_radius_m"] = max(
                    6.0, float(base_command.get("search_radius_m") or 0.0)
                )
            except (TypeError, ValueError):
                base_command["search_radius_m"] = 6.0
            try:
                base_command["local_yaw_window_deg"] = max(
                    120.0, float(base_command.get("local_yaw_window_deg") or 0.0)
                )
            except (TypeError, ValueError):
                base_command["local_yaw_window_deg"] = 120.0

        match_mode = "local" if parked_button or not force_full_map else "full_map"
        match_command = dict(base_command)
        if force_full_map:
            match_command["full_map"] = True
        if parked_button:
            match_command["full_map"] = False
            match_command["auto_full_map_fallback"] = False
            match_command["search_radius_m"] = float(
                cfg.periodic_relocalize_parked_radius_m
            )
            match_command["local_yaw_window_deg"] = float(
                cfg.periodic_relocalize_parked_yaw_window_deg
            )
        match = await self.host._global_localize(match_command)

        # If the local command auto-fell-back to a full-map search inside
        # ``_global_localize``, treat it as full-map for ambiguity / recovery
        # gates. Leaving match_mode as ``local`` previously skipped the
        # ambiguous-twin refuse and let the jump gate confirm a 30 m yank.
        if match_mode == "local" and (
            bool(match.get("fallback_used")) or bool(match.get("full_map"))
        ):
            match_mode = "full_map_via_local_fallback"

        good_match, score, ray_mae = self.host._localization_match_quality(match, cfg)
        matched_pose = match.get("pose")
        shift_m, shift_deg = self.host._pose_shift_from_current(current, matched_pose)

        parked_hit = False
        if parked_button and match_mode == "local":
            parked_hit = self.host._parked_refine_worth_applying(
                cfg, match, shift_m, shift_deg
            )
            if parked_hit:
                match_mode = "local_parked"
            else:
                # Soft parked score: the 1 m peak was not the button's match.
                # Continue with the ordinary local search.
                match_command = dict(base_command)
                match_command["apply"] = False
                match_command["full_map"] = False
                match = await self.host._global_localize(match_command)
                match_mode = "local"
                good_match, score, ray_mae = self.host._localization_match_quality(
                    match, cfg
                )
                matched_pose = match.get("pose")
                shift_m, shift_deg = self.host._pose_shift_from_current(
                    current, matched_pose
                )

        # An ambiguous peak is usually a corridor twin. Do not confirm it onto
        # the pose, whether the search was local or full-map. Manual apply:true
        # still bypasses via apply_override.
        if apply_override is not True and bool(match.get("ambiguous")):
            result = {
                "status": "ambiguous",
                "match_mode": match_mode,
                "score": score,
                "ray_mae_m": ray_mae,
                "shift_m": None if math.isinf(shift_m) else round(shift_m, 3),
                "shift_deg": None if math.isinf(shift_deg) else round(shift_deg, 2),
                "good_match": good_match,
                "drifted": True,
                "recovery_apply": False,
                "corrected": False,
                "ambiguous": True,
                "second_best_score": match.get("second_best_score"),
                "tick_match_score": tick_score,
                "navigation_active": nav_active,
                "nav_recoveries": nav_recoveries,
            }
            prior_score = match.get("prior_score")
            if prior_score is not None:
                result["previous_score"] = prior_score
                result["previous_ray_mae_m"] = match.get("prior_ray_mae_m")
            # Drop any in-progress large-jump confirm so a twin cannot finish
            # confirming across cycles that alternate ambiguous / not.
            self.host._pose_jump_gate.clear()
            LOGGER.warning(
                "periodic relocalize: refusing ambiguous peak "
                "(score=%.2f second_best=%s mode=%s)",
                score,
                match.get("second_best_score"),
                match_mode,
            )
            return self.host._publish_relocalize_check(result)

        drifted = (
            shift_m >= cfg.periodic_relocalize_min_shift_m
            or shift_deg >= cfg.periodic_relocalize_min_shift_deg
        )
        full_map_recovery = match_mode.startswith("full_map")
        # In a recovery situation the robot is (or looks) lost, so mirror a manual
        # global_localize: trust the best full-map match once its score clears the
        # recovery floor, ignoring the stricter ray_mae gate that good_match uses.
        # This is the case that previously left the robot stuck for minutes -- the
        # full-map match was correct (and manual apply fixed it) but ray_mae was
        # above periodic_relocalize_max_ray_mae_m so good_match was False.
        recovery_apply = (
            full_map_recovery and score >= cfg.periodic_relocalize_recovery_min_score
        )
        stopped_local = self.host._stopped_local_beats_published(
            cfg,
            match,
            shift_m,
            shift_deg,
            match_command,
            driving=driving,
            match_mode=match_mode,
        )
        should_apply = apply_override is True or (
            apply_override is None
            and (
                recovery_apply
                or (good_match and drifted)
                or parked_hit
                or stopped_local
            )
        )

        result: dict = {
            "status": "ok",
            "match_mode": match_mode,
            "score": score,
            "ray_mae_m": ray_mae,
            "shift_m": None if math.isinf(shift_m) else round(shift_m, 3),
            "shift_deg": None if math.isinf(shift_deg) else round(shift_deg, 2),
            "good_match": good_match,
            "drifted": drifted,
            "recovery_apply": recovery_apply,
            "navigation_active": nav_active,
            "nav_recoveries": nav_recoveries,
            "corrected": False,
            "tick_match_score": tick_score,
        }
        prior_score = match.get("prior_score")
        prior_ray_mae = match.get("prior_ray_mae_m")
        if prior_score is not None:
            result["previous_score"] = prior_score
            result["previous_ray_mae_m"] = prior_ray_mae

        large_jump = (
            not math.isinf(shift_m) and shift_m >= cfg.localize_jump_large_m
        ) or (not math.isinf(shift_deg) and shift_deg >= cfg.localize_jump_large_deg)
        prefer_previous = False
        if large_jump and prior_score is not None and math.isfinite(float(prior_score)):
            # Only treat "still at previous" when the prior itself is plausible —
            # two equally bad scores must not resume driving (nav_hold instead).
            previous_ok = (
                float(prior_score) >= cfg.periodic_relocalize_recovery_min_score
            )
            if (
                not previous_ok
                and prior_ray_mae is not None
                and math.isfinite(float(prior_ray_mae))
            ):
                previous_ok = (
                    float(prior_ray_mae) <= cfg.periodic_relocalize_max_ray_mae_m
                )
            prefer_previous = previous_ok and not candidate_beats_previous(
                previous_score=float(prior_score),
                candidate_score=float(score),
                shift_m=0.0 if math.isinf(shift_m) else float(shift_m),
                shift_deg=0.0 if math.isinf(shift_deg) else float(shift_deg),
                previous_ray_mae_m=(
                    float(prior_ray_mae) if prior_ray_mae is not None else None
                ),
                candidate_ray_mae_m=ray_mae,
            )
            result["prefer_previous"] = prefer_previous
            result["previous_ok"] = previous_ok
            result["large_jump"] = True

        if prefer_previous:
            # Scan still fits the published pose — treat the far peak as a false
            # match and keep driving (do not apply, do not hold).
            self.host._pose_jump_gate.clear()
            result["status"] = "previous_better"
            result["corrected"] = False
            LOGGER.info(
                "periodic relocalize: keeping previous pose "
                "(shift=%.2f m score=%.2f prior=%.2f); candidate rejected",
                0.0 if math.isinf(shift_m) else shift_m,
                score,
                float(prior_score),
            )
            return self.host._publish_relocalize_check(result)

        if not should_apply and apply_override is not True and not good_match:
            # During nav, do not keep driving on an untrusted pose.
            # - Mid-nav refine + large jump → refused_large (keep published pose)
            # - Periodic large jump + weak match → nav_hold (until confirm)
            # - Soft published pose, no jump → brief nav_hold, then resume on the
            #   published pose (odom continuity) so we do not sit forever at
            #   borderline scores like 0.49.
            # - Weak candidate while published pose is still a good_match → keep going
            if driving:
                previous_ok = False
                if prior_score is not None and math.isfinite(float(prior_score)):
                    # Use the *good_match* bars, not the lower recovery floor —
                    # recovery is for applying a full-map fix, not for deciding
                    # that it is safe to keep driving.
                    previous_ok = (
                        float(prior_score) >= cfg.periodic_relocalize_min_score
                    )
                    if previous_ok and prior_ray_mae is not None:
                        previous_ok = (
                            math.isfinite(float(prior_ray_mae))
                            and float(prior_ray_mae)
                            <= cfg.periodic_relocalize_max_ray_mae_m
                        )
                soft_previous = prior_score is None or not previous_ok
                if large_jump or soft_previous:
                    result["large_jump"] = bool(large_jump)
                    result["soft_loc"] = bool(soft_previous and not large_jump)
                    if prior_score is not None:
                        result["previous_ok"] = previous_ok
                    # Mid-nav refine: a hallway-sized yank we will not apply
                    # must not sticky-hold the base (that is the lobby pause).
                    if self.host._should_refuse_mid_nav_jump(
                        nav_active=nav_active,
                        allow_during_navigation=allow_during_navigation,
                        apply_override=apply_override,
                        shift_m=shift_m,
                        shift_deg=shift_deg,
                        large_jump=large_jump,
                        cfg=cfg,
                        stopped_recovery=stopped_recovery,
                        good_match=good_match,
                        score=float(score),
                    ):
                        self.host._pose_jump_gate.clear()
                        self.host._soft_nav_hold_since = None
                        self.host._soft_nav_hold_released = True
                        result["status"] = "refused_large"
                        result["corrected"] = False
                        result["large_jump"] = True
                        result["stopped_recovery"] = bool(stopped_recovery)
                        LOGGER.info(
                            "periodic relocalize: refusing large mid-nav jump "
                            "(%.2f m / %.1f deg score=%.2f); "
                            "continuing on published pose",
                            0.0 if math.isinf(shift_m) else shift_m,
                            0.0 if math.isinf(shift_deg) else shift_deg,
                            score,
                        )
                        return self.host._publish_relocalize_check(result)
                    soft_hold_max = float(cfg.periodic_relocalize_soft_hold_max_s)
                    if soft_previous and not large_jump and soft_hold_max > 0.0:
                        now = time.monotonic()
                        if self.host._soft_nav_hold_released:
                            result["status"] = "soft_loc_resume"
                            result["hold_max_s"] = soft_hold_max
                            LOGGER.info(
                                "periodic relocalize: soft loc resume "
                                "(no jump; continuing on published pose "
                                "score=%.2f prior=%s)",
                                score,
                                prior_score,
                            )
                            return self.host._publish_relocalize_check(result)
                        if self.host._soft_nav_hold_since is None:
                            self.host._soft_nav_hold_since = now
                        held_s = now - float(self.host._soft_nav_hold_since)
                        result["hold_elapsed_s"] = round(held_s, 1)
                        result["hold_max_s"] = soft_hold_max
                        if held_s >= soft_hold_max:
                            self.host._soft_nav_hold_released = True
                            result["status"] = "soft_loc_resume"
                            LOGGER.info(
                                "periodic relocalize: soft hold timed out after "
                                "%.1fs — resuming on published pose "
                                "(shift=%.2f m score=%.2f)",
                                held_s,
                                0.0 if math.isinf(shift_m) else shift_m,
                                score,
                            )
                            return self.host._publish_relocalize_check(result)
                    elif large_jump:
                        # Disputed jump: do not auto-resume on the soft timer.
                        self.host._soft_nav_hold_since = None
                        self.host._soft_nav_hold_released = False
                    result["status"] = "nav_hold"
                    LOGGER.warning(
                        "periodic relocalize: soft loc during nav — holding "
                        "(shift=%.2f m / %.1f deg score=%.2f ray_mae=%s "
                        "prior=%s prior_mae=%s large_jump=%s soft_previous=%s)",
                        0.0 if math.isinf(shift_m) else shift_m,
                        0.0 if math.isinf(shift_deg) else shift_deg,
                        score,
                        ray_mae,
                        prior_score,
                        prior_ray_mae,
                        large_jump,
                        soft_previous,
                    )
                    return self.host._publish_relocalize_check(result)
            self.host._soft_nav_hold_since = None
            self.host._soft_nav_hold_released = False
            result["status"] = "low_quality"
            LOGGER.warning(
                "periodic relocalize: no trusted match after %s "
                "(score=%.2f ray_mae=%s recovery_floor=%.2f nav_recoveries=%d); "
                "not correcting",
                match_mode,
                score,
                ray_mae,
                cfg.periodic_relocalize_recovery_min_score,
                nav_recoveries,
            )
            return self.host._publish_relocalize_check(result)

        # Trusted path (or apply override): clear soft-hold bookkeeping.
        self.host._soft_nav_hold_since = None
        self.host._soft_nav_hold_released = False

        if should_apply and isinstance(matched_pose, Mapping):
            if self.host._should_refuse_mid_nav_jump(
                nav_active=nav_active,
                allow_during_navigation=allow_during_navigation,
                apply_override=apply_override,
                shift_m=shift_m,
                shift_deg=shift_deg,
                large_jump=large_jump,
                cfg=cfg,
                stopped_recovery=stopped_recovery,
                good_match=good_match,
                score=float(score),
            ):
                self.host._pose_jump_gate.clear()
                result["status"] = "refused_large"
                result["corrected"] = False
                result["large_jump"] = True
                result["stopped_recovery"] = bool(stopped_recovery)
                LOGGER.info(
                    "periodic relocalize: refusing large mid-nav jump "
                    "(%.2f m / %.1f deg score=%.2f apply=%s recovery=%s); "
                    "continuing on published pose",
                    0.0 if math.isinf(shift_m) else shift_m,
                    0.0 if math.isinf(shift_deg) else shift_deg,
                    score,
                    apply_override,
                    stopped_recovery,
                )
                return self.host._publish_relocalize_check(result)
            candidate = self.host._pose_from_mapping(matched_pose)
            # Stopped recovery: a trusted large match may apply once without
            # waiting for a second agreeing frame (base is already halted).
            force_jump = apply_override is True or (
                stopped_recovery and good_match and large_jump
            )
            decision = self.host._confirm_pose_jump(
                current, candidate, force=force_jump
            )
            result.update(self.host._jump_decision_dict(decision))
            if not decision.should_apply:
                result["status"] = "awaiting_confirm"
                result["corrected"] = False
                LOGGER.info(
                    "periodic relocalize: large jump %.2f m / %.1f deg awaiting "
                    "confirm %d/%d (score=%.2f prior=%s)",
                    decision.shift_m,
                    decision.shift_deg,
                    decision.confirm_count,
                    decision.confirm_needed,
                    score,
                    prior_score,
                )
                return self.host._publish_relocalize_check(result)
            await self.host.do_command(
                {
                    "command": "relocalize",
                    "pose": {
                        "x": float(matched_pose.get("x", 0.0)),
                        "y": float(matched_pose.get("y", 0.0)),
                        "theta": float(matched_pose.get("theta", 0.0)),
                    },
                    "position_variance_m2": 0.25,
                    "yaw_variance_rad2": 0.06853891945200942,
                }
            )
            result["status"] = "corrected"
            result["corrected"] = True
            LOGGER.info(
                "periodic relocalize: corrected via %s (shift=%.2f m, %.1f deg, "
                "score=%.2f, ray_mae=%s, recovery=%s, nav_recoveries=%d, jump=%s)",
                match_mode,
                0.0 if math.isinf(shift_m) else shift_m,
                0.0 if math.isinf(shift_deg) else shift_deg,
                score,
                ray_mae,
                recovery_apply and not good_match,
                nav_recoveries,
                decision.status,
            )
        else:
            if not should_apply:
                self.host._pose_jump_gate.clear()
            LOGGER.debug(
                "periodic relocalize: pose ok (%s shift=%.2f m score=%.2f)",
                match_mode,
                0.0 if math.isinf(shift_m) else shift_m,
                score,
            )

        return self.host._publish_relocalize_check(result)

    @staticmethod
    def _startup_global_localize_quality(
        result: Mapping[str, ValueTypes],
    ) -> tuple[float, float, float]:
        score = float(result.get("score", float("-inf")))
        ray_mae_raw = result.get("ray_mae_m")
        ray_mae = float(ray_mae_raw) if ray_mae_raw is not None else float("inf")
        hit_rate = float(result.get("hit_rate", 0.0))
        return (score, -ray_mae, hit_rate)

    @staticmethod
    def _startup_global_localize_meets_target(
        result: Mapping[str, ValueTypes],
        *,
        target_score: float,
        target_ray_mae_m: float,
    ) -> bool:
        score = float(result.get("score", float("-inf")))
        ray_mae_raw = result.get("ray_mae_m")
        if ray_mae_raw is None:
            return False
        ray_mae = float(ray_mae_raw)
        return score >= target_score and ray_mae <= target_ray_mae_m

    async def _wait_for_startup_localize_ready(
        self, *, timeout_s: float = 90.0, poll_interval_s: float = 2.0
    ) -> bool:
        """Hold the startup auto-localize until scan matching can actually work.

        A fixed post-reconfigure delay fires while MiR rosbridge sessions are
        still connecting and the map is still loading; the
        retry attempts then burn out and the robot stays mislocalized until a
        manual global_localize. Ready = slam process up, a merged scan with
        returns readable, and an occupancy map loadable.
        """
        mgr = self.host._manager
        if mgr is None:
            return True
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while time.monotonic() < deadline:
            try:
                if mgr.slam_running():
                    await self.host._read_merged_scan()  # raises when no returns yet
                    self.host._load_active_occupancy_map("auto")
                    LOGGER.info("startup global_localize readiness: slam+scan+map ok")
                    return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - not ready yet; keep polling
                now = time.monotonic()
                if now - last_log >= 10.0:
                    LOGGER.info(
                        "startup global_localize waiting for readiness: %s",
                        exc,
                    )
                    last_log = now
            await asyncio.sleep(poll_interval_s)
        if timeout_s > 0.0:
            LOGGER.warning(
                "startup global_localize readiness wait timed out after %.0fs; "
                "attempting anyway",
                timeout_s,
            )
        return False

    async def _run_startup_global_localize(
        self,
        options: Mapping[str, ValueTypes],
        *,
        delay_s: float = 4.0,
        max_attempts: int = 5,
        retry_delay_s: float = 5.0,
        readiness_timeout_s: float = 90.0,
        run_refine_pass: bool = False,
        refine_delay_s: float = 8.0,
        refine_max_passes: int = 3,
        target_score: float = 0.7,
        target_ray_mae_m: float = 0.4,
        refine_options: Optional[Mapping[str, ValueTypes]] = None,
        run_post_apply_refine: bool = True,
        post_apply_refine_delay_s: float = 3.0,
        post_apply_refine_options: Optional[Mapping[str, ValueTypes]] = None,
    ) -> None:
        if self.host._is_navigation_active():
            LOGGER.info("startup global_localize skipped: navigation already active")
            return
        await self.host._wait_for_startup_localize_ready(timeout_s=readiness_timeout_s)
        # Extra settle time after readiness so SLAM has processed a few
        # scans against the loaded posegraph before we sample one for matching.
        if delay_s > 0.0:
            await asyncio.sleep(delay_s)
        if self.host._is_navigation_active():
            LOGGER.info("startup global_localize skipped: navigation already active")
            return
        command: dict = {
            "command": "global_localize",
            # Evaluate candidates first; apply only the best pose at the end.
            "apply": False,
            "auto_full_map_fallback": False,
        }
        command.update(dict(options))
        command["apply"] = False
        # Restored last pose: try the same 1 m window as the parked auto-refine
        # before the wider startup search. The wider search is what marked this
        # hallway slip ambiguous and then spent minutes refusing it.
        if not bool(command.get("full_map", True)):
            if await self.host._startup_apply_parked_refine():
                return
        best_result: Optional[Mapping[str, ValueTypes]] = None
        applied = False

        async def _observe_startup_pose(pose_map: object, match: Mapping) -> bool:
            """Feed one match into the jump gate; apply when confirmed + trusted."""
            if not isinstance(pose_map, Mapping):
                return False
            # Do not lock continuous matching onto a weak / ambiguous full-map peak.
            if bool(match.get("ambiguous")):
                LOGGER.info(
                    "startup global_localize: skipping apply (ambiguous peak "
                    "score=%s second_best=%s)",
                    match.get("score"),
                    match.get("second_best_score"),
                )
                return False
            score = match.get("score")
            ray_mae = match.get("ray_mae_m")
            # Corridor twins often clear 0.55 while still wrong; require a
            # clearer peak before boot auto-apply (manual seed remains the
            # recovery path when the map is repetitive).
            if score is not None and float(score) < 0.65:
                LOGGER.info(
                    "startup global_localize: skipping apply (low score=%.3f)",
                    float(score),
                )
                return False
            if (
                ray_mae is not None
                and math.isfinite(float(ray_mae))
                and float(ray_mae) > 0.45
            ):
                LOGGER.info(
                    "startup global_localize: skipping apply (ray_mae=%.3f)",
                    float(ray_mae),
                )
                return False
            # Prefer a clear margin over second-best when reported.
            second = match.get("second_best_score")
            if (
                score is not None
                and second is not None
                and math.isfinite(float(second))
                and float(score) - float(second) < 0.08
            ):
                LOGGER.info(
                    "startup global_localize: skipping apply (near-tie "
                    "score=%.3f second_best=%.3f)",
                    float(score),
                    float(second),
                )
                return False
            mgr = self.host._manager
            current = mgr.get_pose_in_map() if mgr is not None else None
            candidate = self.host._pose_from_mapping(pose_map)
            decision = self.host._confirm_pose_jump(current, candidate, force=False)
            if not decision.should_apply:
                LOGGER.info(
                    "startup global_localize: large jump %.2f m awaiting confirm %d/%d",
                    decision.shift_m,
                    decision.confirm_count,
                    decision.confirm_needed,
                )
                return False
            await self.host.do_command(
                {
                    "command": "relocalize",
                    "pose": {
                        "x": float(pose_map.get("x", 0.0)),
                        "y": float(pose_map.get("y", 0.0)),
                        "theta": float(pose_map.get("theta", 0.0)),
                    },
                    "position_variance_m2": 0.25,
                    "yaw_variance_rad2": 0.06853891945200942,
                }
            )
            LOGGER.info(
                "startup global_localize applied (%.2f, %.2f, %.2f) jump=%s "
                "score=%s ray_mae=%s",
                candidate.x,
                candidate.y,
                candidate.theta,
                decision.status,
                match.get("score"),
                match.get("ray_mae_m"),
            )
            return True

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.host.do_command(command)
                best_result = result
                LOGGER.info(
                    "startup global_localize matched on attempt %d: score=%s ray_mae=%s",
                    attempt,
                    result.get("score"),
                    result.get("ray_mae_m"),
                )
                if await _observe_startup_pose(result.get("pose"), result):
                    applied = True
                if run_refine_pass and not applied:
                    passes = max(0, int(refine_max_passes))
                    for refine_pass in range(1, passes + 1):
                        if self.host._is_navigation_active():
                            LOGGER.info(
                                "startup global_localize refinement aborted: navigation active"
                            )
                            break
                        if self.host._startup_global_localize_meets_target(
                            best_result,
                            target_score=target_score,
                            target_ray_mae_m=target_ray_mae_m,
                        ):
                            break
                        if refine_delay_s > 0.0:
                            await asyncio.sleep(refine_delay_s)
                        refine_command = dict(command)
                        # Refine around the best candidate pose found so far.
                        refine_command["full_map"] = False
                        if refine_options:
                            refine_command.update(dict(refine_options))
                        best_pose = (
                            best_result.get("pose") if best_result is not None else None
                        )
                        if isinstance(best_pose, Mapping):
                            refine_command["pose"] = {
                                "x": float(best_pose.get("x", 0.0)),
                                "y": float(best_pose.get("y", 0.0)),
                                "theta": float(best_pose.get("theta", 0.0)),
                            }
                        refine_result = await self.host.do_command(refine_command)
                        if self.host._startup_global_localize_quality(
                            refine_result
                        ) > self.host._startup_global_localize_quality(best_result):
                            best_result = refine_result
                        LOGGER.info(
                            "startup global_localize refinement pass %d/%d: "
                            "score=%s ray_mae=%s",
                            refine_pass,
                            passes,
                            refine_result.get("score"),
                            refine_result.get("ray_mae_m"),
                        )
                        if await _observe_startup_pose(
                            best_result.get("pose"), best_result
                        ):
                            applied = True
                            break

                if not applied:
                    if attempt < max_attempts:
                        await asyncio.sleep(max(retry_delay_s, 0.0))
                        continue
                    LOGGER.warning(
                        "startup global_localize: large jump never confirmed; "
                        "not applying"
                    )
                    return

                if run_post_apply_refine and best_result is not None:
                    if post_apply_refine_delay_s > 0.0:
                        await asyncio.sleep(post_apply_refine_delay_s)
                    if self.host._is_navigation_active():
                        LOGGER.info(
                            "startup global_localize post-apply skipped: navigation active"
                        )
                        return
                    post_command: dict = {
                        "command": "global_localize",
                        # Match known-good manual behavior exactly.
                        "apply": True,
                    }
                    if post_apply_refine_options:
                        post_command.update(dict(post_apply_refine_options))
                    post_command["apply"] = True
                    post_result = await self.host.do_command(post_command)
                    best_result = post_result
                    LOGGER.info(
                        "startup global_localize post-apply pass: score=%s ray_mae=%s",
                        post_result.get("score"),
                        post_result.get("ray_mae_m"),
                    )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - startup best-effort
                LOGGER.warning(
                    "startup global_localize attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(max(retry_delay_s, 0.0))

    # -- ROS IO --------------------------------------------------------------
