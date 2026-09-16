from typing import Any

from common.config import SETTINGS
from common.constants import (
    CLASS_SHARE_PREFIX, DriftFlag, PreVerdict, HardBreakKind, BoxGeometry, FLAG_VERDICT, TREND_FLAG_FAMILY
)


class DriftRules:
    SHIFT_FLAGS = [
        DriftFlag.CONFIDENCE_SHIFT, DriftFlag.CLASS_SHIFT, DriftFlag.BOX_COUNT_SHIFT, DriftFlag.BOX_GEOMETRY_SHIFT
    ]

    def __init__(self):
        self.config = SETTINGS.data_drift

    def flags(
            self,
            split: dict[str, Any] | None,
            trend: dict[str, dict[str, Any]],
            outliers: list[dict[str, Any]],
            hard_breaks: list[dict[str, Any]],
            insufficient_data: bool,
            insufficient_buckets: bool
    ) -> list[DriftFlag]:
        flags: list[DriftFlag] = []

        if insufficient_data:
            flags.append(DriftFlag.INSUFFICIENT_DATA)
        if insufficient_buckets:
            flags.append(DriftFlag.INSUFFICIENT_BUCKETS)
        if any(item["kind"] != HardBreakKind.ELAPSED_TIME for item in hard_breaks):
            flags.append(DriftFlag.HARD_BREAK)

        if split is not None and split.get("sides_sufficient"):
            flags.extend(self._split_flags(split))

        trend_flags: list[DriftFlag] = []
        for name, item in trend.items():
            if not item.get("meaningful"):
                continue
            flag = DriftFlag.TREND_CLASS_SHARE if name.startswith(CLASS_SHARE_PREFIX) \
                else DriftFlag(f"TREND_{name.upper()}")
            if flag not in trend_flags:
                trend_flags.append(flag)
        flags.extend(trend_flags)

        shift_or_trend = trend_flags or any(flag in flags for flag in self.SHIFT_FLAGS)
        if outliers and not shift_or_trend:
            flags.append(DriftFlag.TRANSIENT_OUTLIER)

        return flags

    def _split_flags(self, split: dict[str, Any]) -> list[DriftFlag]:
        config = self.config
        flags = []

        psi_confidence = split.get("psi_confidence", 0.0)
        if psi_confidence >= config.psi_significant:
            flags.append(DriftFlag.CONFIDENCE_SHIFT)
        elif psi_confidence >= config.psi_moderate:
            flags.append(DriftFlag.CONFIDENCE_SHIFT_MODERATE)

        psi_class = split.get("psi_class", 0.0)
        chi2_p = split.get("chi2_class", {}).get("p", 1.0)
        prop_change = split.get("max_class_proportion_change", 0.0)
        if psi_class >= config.psi_significant or (chi2_p < config.chi2_p and prop_change >= config.class_prop_change):
            flags.append(DriftFlag.CLASS_SHIFT)
        elif psi_class >= config.psi_moderate:
            flags.append(DriftFlag.CLASS_SHIFT_MODERATE)

        psi_boxes = split.get("psi_boxes_per_image")
        if psi_boxes is not None and psi_boxes >= config.psi_significant:
            flags.append(DriftFlag.BOX_COUNT_SHIFT)

        for measure in BoxGeometry:
            ks = split.get(f"ks_{measure}")
            if ks and ks["d"] >= config.ks_d_geometry and ks["p"] < config.ks_p_geometry:
                flags.append(DriftFlag.BOX_GEOMETRY_SHIFT)
                break

        before_rate = split["before"].get("below_threshold_rate")
        after_rate = split["after"].get("below_threshold_rate")
        if before_rate is not None and after_rate is not None:
            if after_rate >= config.threshold_pressure_ratio * before_rate \
                    and after_rate - before_rate >= config.threshold_pressure_abs:
                flags.append(DriftFlag.THRESHOLD_PRESSURE)

        return flags

    @staticmethod
    def pre_verdict(flags: list[DriftFlag]) -> PreVerdict:
        if any(flag.startswith("INSUFFICIENT_") for flag in flags):
            return PreVerdict.UNDETERMINED

        trend_count = len({TREND_FLAG_FAMILY[flag] for flag in flags if flag in TREND_FLAG_FAMILY})
        verdicts = {FLAG_VERDICT[flag] for flag in flags if flag in FLAG_VERDICT}
        if PreVerdict.DRIFT_LIKELY in verdicts or trend_count >= 2:
            return PreVerdict.DRIFT_LIKELY
        if PreVerdict.SUSPICIOUS in verdicts or trend_count == 1:
            return PreVerdict.SUSPICIOUS
        return PreVerdict.STABLE
