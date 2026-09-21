from typing import Any

from common.config import SETTINGS
from common.constants import DriftFlag, PreVerdict, FLAG_VERDICT, INSUFFICIENCY_FLAGS


class DriftRules:
    def __init__(self):
        self.config = SETTINGS.data_drift

    def flags(
            self,
            split: dict[str, Any] | None,
            hard_breaks: list[dict[str, Any]],
            insufficient_data: bool,
            insufficient_periods: bool,
            incompatible_bins: bool
    ) -> list[DriftFlag]:
        flags: list[DriftFlag] = []

        if insufficient_data:
            flags.append(DriftFlag.INSUFFICIENT_DATA)
        if insufficient_periods:
            flags.append(DriftFlag.INSUFFICIENT_PERIODS)
        if incompatible_bins:
            flags.append(DriftFlag.INCOMPATIBLE_BINS)
        if hard_breaks:
            flags.append(DriftFlag.HARD_BREAK)

        if split is not None and split.get("sides_sufficient"):
            flags.extend(self._split_flags(split))

        return flags

    def _split_flags(self, split: dict[str, Any]) -> list[DriftFlag]:
        config = self.config
        flags = []

        psi_confidence = split.get("psi_confidence")
        if psi_confidence is not None:
            if psi_confidence >= config.psi_significant:
                flags.append(DriftFlag.CONFIDENCE_SHIFT)
            elif psi_confidence >= config.psi_moderate:
                flags.append(DriftFlag.CONFIDENCE_SHIFT_MODERATE)

        psi_class = split.get("psi_class")
        chi2_p = (split.get("chi2_class") or {}).get("p", 1.0)
        prop_change = split.get("max_class_proportion_change") or 0.0
        if psi_class is not None:
            if psi_class >= config.psi_significant \
                    or (chi2_p < config.chi2_p and prop_change >= config.class_prop_change):
                flags.append(DriftFlag.CLASS_SHIFT)
            elif psi_class >= config.psi_moderate:
                flags.append(DriftFlag.CLASS_SHIFT_MODERATE)

        psi_boxes = split.get("psi_boxes_per_image")
        if psi_boxes is not None and psi_boxes >= config.psi_significant:
            flags.append(DriftFlag.BOX_COUNT_SHIFT)

        before_rate = split["before"].below_threshold_rate
        after_rate = split["after"].below_threshold_rate
        if before_rate is not None and after_rate is not None:
            if after_rate >= config.threshold_pressure_ratio * before_rate \
                    and after_rate - before_rate >= config.threshold_pressure_abs:
                flags.append(DriftFlag.THRESHOLD_PRESSURE)

        return flags

    @staticmethod
    def pre_verdict(flags: list[DriftFlag]) -> PreVerdict:
        if any(flag in INSUFFICIENCY_FLAGS for flag in flags):
            return PreVerdict.UNDETERMINED

        verdicts = {FLAG_VERDICT[flag] for flag in flags if flag in FLAG_VERDICT}
        if PreVerdict.DRIFT_LIKELY in verdicts:
            return PreVerdict.DRIFT_LIKELY
        if PreVerdict.SUSPICIOUS in verdicts:
            return PreVerdict.SUSPICIOUS
        return PreVerdict.STABLE
