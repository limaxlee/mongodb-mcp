from datetime import timedelta

from common.config import SETTINGS
from common.constants import DriftFlag, PreVerdict, HardBreakKind
from mongodb_mcp.schemas import SideSummary
from mongodb_mcp.drift import DriftRules
from tests.drift.synthetic import START

CONFIG = SETTINGS.data_drift


def side(below_rate=0.01):
    return SideSummary(
        start_date=START, end_date=START + timedelta(days=1), period_count=1, prediction_count=1000,
        below_threshold_rate=below_rate
    )


def split(**overrides):
    report = {
        "sides_sufficient": True,
        "psi_confidence": 0.0,
        "psi_class": 0.0,
        "chi2_class": {"stat": 0.0, "p": 1.0, "dof": 1},
        "max_class_proportion_change": 0.0,
        "psi_boxes_per_image": None,
        "before": side(),
        "after": side()
    }
    report.update(overrides)
    return report


def hard_break(kind=HardBreakKind.BACKEND):
    return {"kind": kind, "class_name": None, "date": START, "from": "ts", "to": "trt"}


class TestFlags:
    def test_quiet_split_gives_no_flags(self):
        assert DriftRules().flags(split(), [], False, False, False) == []

    def test_insufficiency_flags(self):
        flags = DriftRules().flags(None, [], True, True, True)
        assert flags == [DriftFlag.INSUFFICIENT_DATA, DriftFlag.INSUFFICIENT_PERIODS, DriftFlag.INCOMPATIBLE_BINS]

    def test_hard_break(self):
        assert DriftRules().flags(split(), [hard_break()], False, False, False) == [DriftFlag.HARD_BREAK]

    def test_confidence_shift_levels(self):
        rules = DriftRules()
        assert DriftFlag.CONFIDENCE_SHIFT in rules.flags(split(psi_confidence=CONFIG.psi_significant), [], False, False, False)
        moderate = rules.flags(split(psi_confidence=CONFIG.psi_moderate), [], False, False, False)
        assert moderate == [DriftFlag.CONFIDENCE_SHIFT_MODERATE]
        assert rules.flags(split(psi_confidence=None), [], False, False, False) == []

    def test_class_shift_routes(self):
        rules = DriftRules()
        assert rules.flags(split(psi_class=0.3), [], False, False, False) == [DriftFlag.CLASS_SHIFT]
        by_test = split(psi_class=0.05, chi2_class={"stat": 30.0, "p": 1e-6, "dof": 1}, max_class_proportion_change=0.06)
        assert rules.flags(by_test, [], False, False, False) == [DriftFlag.CLASS_SHIFT]
        small_move = split(psi_class=0.05, chi2_class={"stat": 30.0, "p": 1e-6, "dof": 1}, max_class_proportion_change=0.01)
        assert rules.flags(small_move, [], False, False, False) == []
        assert rules.flags(split(psi_class=0.15), [], False, False, False) == [DriftFlag.CLASS_SHIFT_MODERATE]
        assert rules.flags(split(psi_class=None), [], False, False, False) == []

    def test_box_count_shift(self):
        assert DriftRules().flags(split(psi_boxes_per_image=0.3), [], False, False, False) == [DriftFlag.BOX_COUNT_SHIFT]
        assert DriftRules().flags(split(psi_boxes_per_image=0.1), [], False, False, False) == []

    def test_threshold_pressure(self):
        rules = DriftRules()
        pressure = split(before=side(0.01), after=side(0.05))
        assert rules.flags(pressure, [], False, False, False) == [DriftFlag.THRESHOLD_PRESSURE]
        ratio_only = split(before=side(0.001), after=side(0.005))
        assert rules.flags(ratio_only, [], False, False, False) == []
        no_threshold = split(before=side(None), after=side(0.05))
        assert rules.flags(no_threshold, [], False, False, False) == []

    def test_insufficient_sides_block_shift_flags(self):
        loud = split(psi_confidence=0.9, psi_class=0.9, sides_sufficient=False)
        assert DriftRules().flags(loud, [], True, False, False) == [DriftFlag.INSUFFICIENT_DATA]


class TestPreVerdict:
    def test_insufficiency_overrides_everything(self):
        verdict = DriftRules.pre_verdict
        assert verdict([DriftFlag.INSUFFICIENT_DATA, DriftFlag.CONFIDENCE_SHIFT]) == PreVerdict.UNDETERMINED
        assert verdict([DriftFlag.INSUFFICIENT_PERIODS]) == PreVerdict.UNDETERMINED
        assert verdict([DriftFlag.INCOMPATIBLE_BINS]) == PreVerdict.UNDETERMINED

    def test_likely_flags(self):
        verdict = DriftRules.pre_verdict
        for flag in (DriftFlag.CONFIDENCE_SHIFT, DriftFlag.CLASS_SHIFT, DriftFlag.BOX_COUNT_SHIFT):
            assert verdict([flag]) == PreVerdict.DRIFT_LIKELY
        assert verdict([DriftFlag.HARD_BREAK, DriftFlag.CONFIDENCE_SHIFT]) == PreVerdict.DRIFT_LIKELY

    def test_suspicious_flags(self):
        verdict = DriftRules.pre_verdict
        for flag in (DriftFlag.CONFIDENCE_SHIFT_MODERATE, DriftFlag.CLASS_SHIFT_MODERATE,
                     DriftFlag.THRESHOLD_PRESSURE, DriftFlag.HARD_BREAK):
            assert verdict([flag]) == PreVerdict.SUSPICIOUS

    def test_stable(self):
        assert DriftRules.pre_verdict([]) == PreVerdict.STABLE
