"""Deterministic rule engine producing flags and the pre-verdict from an assembled analysis"""
from typing import Any

from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG

LIKELY_FLAGS = {
    "HARD_BREAK", "CONFIDENCE_SHIFT", "CLASS_SHIFT", "BOX_COUNT_SHIFT", "BOX_GEOMETRY_SHIFT", "FEEDBACK_DEGRADATION"
}
SUSPICIOUS_FLAGS = {"CONFIDENCE_SHIFT_MODERATE", "CLASS_SHIFT_MODERATE", "THRESHOLD_PRESSURE", "TRANSIENT_OUTLIER"}


def compute_flags(
        split: dict[str, Any] | None,
        trend: dict[str, dict[str, Any]],
        outliers: list[dict[str, Any]],
        hard_breaks: list[dict[str, Any]],
        insufficient_data: bool,
        insufficient_buckets: bool,
        config: DriftConfig = DEFAULT_CONFIG
) -> list[str]:
    flags: list[str] = []

    if insufficient_data:
        flags.append("INSUFFICIENT_DATA")
    if insufficient_buckets:
        flags.append("INSUFFICIENT_BUCKETS")
    if any(item["kind"] != "elapsed_time" for item in hard_breaks):
        flags.append("HARD_BREAK")

    if split is not None and split.get("sides_sufficient"):
        flags.extend(_split_flags(split, config))

    trend_flags = [f"TREND_{name.upper()}" for name, item in trend.items() if item.get("meaningful")]
    flags.extend(trend_flags)

    shift_or_trend = trend_flags or any(
        flag in flags for flag in ("CONFIDENCE_SHIFT", "CLASS_SHIFT", "BOX_COUNT_SHIFT", "BOX_GEOMETRY_SHIFT")
    )
    if outliers and not shift_or_trend:
        flags.append("TRANSIENT_OUTLIER")

    return flags


def _split_flags(split: dict[str, Any], config: DriftConfig) -> list[str]:
    flags: list[str] = []

    psi_conf = split.get("psi_conf", 0.0)
    if psi_conf >= config.psi_significant:
        flags.append("CONFIDENCE_SHIFT")
    elif psi_conf >= config.psi_moderate:
        flags.append("CONFIDENCE_SHIFT_MODERATE")

    psi_class = split.get("psi_class", 0.0)
    chi2_p = split.get("chi2_class", {}).get("p", 1.0)
    prop_change = split.get("max_class_prop_change", 0.0)
    if psi_class >= config.psi_significant or (chi2_p < config.chi2_p and prop_change >= config.class_prop_change):
        flags.append("CLASS_SHIFT")
    elif psi_class >= config.psi_moderate:
        flags.append("CLASS_SHIFT_MODERATE")

    psi_boxes = split.get("psi_boxes_per_image")
    if psi_boxes is not None and psi_boxes >= config.psi_significant:
        flags.append("BOX_COUNT_SHIFT")

    for name in ("ks_area_norm", "ks_cx_norm", "ks_cy_norm"):
        ks = split.get(name)
        if ks and ks["d"] >= config.ks_d_geometry and ks["p"] < config.ks_p_geometry:
            flags.append("BOX_GEOMETRY_SHIFT")
            break

    before_rate = split["before"].get("below_threshold_rate")
    after_rate = split["after"].get("below_threshold_rate")
    if before_rate is not None and after_rate is not None:
        if after_rate >= config.threshold_pressure_ratio * before_rate \
                and after_rate - before_rate >= config.threshold_pressure_abs:
            flags.append("THRESHOLD_PRESSURE")

    before_fb, after_fb = split["before"], split["after"]
    if before_fb.get("n_feedback", 0) >= config.feedback_min and after_fb.get("n_feedback", 0) >= config.feedback_min:
        before_mismatch = before_fb.get("feedback_mismatch_rate") or 0.0
        after_mismatch = after_fb.get("feedback_mismatch_rate") or 0.0
        if after_mismatch >= config.feedback_ratio * before_mismatch and after_mismatch > before_mismatch:
            flags.append("FEEDBACK_DEGRADATION")

    return flags


def pre_verdict(flags: list[str]) -> str:
    if any(flag.startswith("INSUFFICIENT_") for flag in flags):
        return "undetermined"

    trend_count = sum(1 for flag in flags if flag.startswith("TREND_"))
    if any(flag in LIKELY_FLAGS for flag in flags) or trend_count >= 2:
        return "drift_likely"
    if any(flag in SUSPICIOUS_FLAGS for flag in flags) or trend_count == 1:
        return "suspicious"
    return "stable"
