from common.config import SETTINGS
from common.constants import DriftFlag, PreVerdict, HardBreakKind, CLASS_SHARE_PREFIX
from mongodb_mcp.drift.rules import DriftRules


def split_report(**overrides):
    """A sufficient, quiet split report; the tests move single values off it"""
    report = {
        "sides_sufficient": True,
        "psi_confidence": 0.01,
        "psi_class": 0.01,
        "chi2_class": {"stat": 0.1, "p": 0.9, "dof": 1},
        "max_class_proportion_change": 0.005,
        "before": {"below_threshold_rate": 0.03},
        "after": {"below_threshold_rate": 0.03}
    }
    report.update(overrides)
    return report


def trend(meaningful=True):
    return {"tau": -1.0, "p": 0.001, "slope_per_bucket": -0.01, "first": 0.9, "last": 0.8, "point_count": 8,
            "meaningful": meaningful}


class TestFlags:
    def test_quiet_split_gives_no_flags(self):
        assert DriftRules().flags(split_report(), {}, [], [], False, False) == []
        assert DriftRules().flags(None, {}, [], [], False, False) == []

    def test_insufficiency_flags(self):
        flags = DriftRules().flags(None, {}, [], [], True, True)
        assert flags == [DriftFlag.INSUFFICIENT_DATA, DriftFlag.INSUFFICIENT_BUCKETS]

    def test_hard_break_ignores_elapsed_time(self):
        rules = DriftRules()
        elapsed = [{"kind": HardBreakKind.ELAPSED_TIME}]
        assert rules.flags(None, {}, [], elapsed, False, False) == []
        assert rules.flags(None, {}, [], elapsed + [{"kind": HardBreakKind.THRESHOLD}], False, False) == \
            [DriftFlag.HARD_BREAK]

    def test_confidence_shift_levels(self):
        config = SETTINGS.data_drift
        rules = DriftRules()
        assert rules.flags(split_report(psi_confidence=config.psi_significant), {}, [], [], False, False) == \
            [DriftFlag.CONFIDENCE_SHIFT]
        assert rules.flags(split_report(psi_confidence=config.psi_moderate), {}, [], [], False, False) == \
            [DriftFlag.CONFIDENCE_SHIFT_MODERATE]
        assert rules.flags(split_report(psi_confidence=config.psi_moderate, sides_sufficient=False),
                           {}, [], [], False, False) == []

    def test_class_shift_routes(self):
        config = SETTINGS.data_drift
        rules = DriftRules()
        assert rules.flags(split_report(psi_class=0.3), {}, [], [], False, False) == [DriftFlag.CLASS_SHIFT]
        assert rules.flags(split_report(psi_class=0.15), {}, [], [], False, False) == [DriftFlag.CLASS_SHIFT_MODERATE]

        by_chi2 = split_report(chi2_class={"stat": 30.0, "p": 1e-6, "dof": 1},
                               max_class_proportion_change=config.class_prop_change)
        assert rules.flags(by_chi2, {}, [], [], False, False) == [DriftFlag.CLASS_SHIFT]

        tiny_but_significant = split_report(chi2_class={"stat": 30.0, "p": 1e-6, "dof": 1},
                                            max_class_proportion_change=0.01)
        assert rules.flags(tiny_but_significant, {}, [], [], False, False) == []

    def test_detection_flags(self):
        rules = DriftRules()
        assert rules.flags(split_report(psi_boxes_per_image=0.3), {}, [], [], False, False) == \
            [DriftFlag.BOX_COUNT_SHIFT]

        moved = split_report(ks_normalized_area={"d": 0.05, "p": 0.5}, ks_normalized_cx={"d": 0.2, "p": 0.001})
        assert rules.flags(moved, {}, [], [], False, False) == [DriftFlag.BOX_GEOMETRY_SHIFT]

        not_significant = split_report(ks_normalized_cx={"d": 0.2, "p": 0.05}, ks_normalized_cy=None)
        assert rules.flags(not_significant, {}, [], [], False, False) == []

    def test_threshold_pressure(self):
        rules = DriftRules()
        pressed = split_report(before={"below_threshold_rate": 0.03}, after={"below_threshold_rate": 0.07})
        assert rules.flags(pressed, {}, [], [], False, False) == [DriftFlag.THRESHOLD_PRESSURE]

        small = split_report(before={"below_threshold_rate": 0.005}, after={"below_threshold_rate": 0.012})
        assert rules.flags(small, {}, [], [], False, False) == []

        unknown = split_report(before={"below_threshold_rate": None}, after={"below_threshold_rate": 0.5})
        assert rules.flags(unknown, {}, [], [], False, False) == []

    def test_trend_flags_and_class_share_collapse(self):
        trends = {
            "median_confidence": trend(),
            "mean_confidence": trend(meaningful=False),
            "no_box_rate": trend(),
            f"{CLASS_SHARE_PREFIX}Good": trend(),
            f"{CLASS_SHARE_PREFIX}NG": trend()
        }
        flags = DriftRules().flags(None, trends, [], [], False, False)
        assert flags == [DriftFlag.TREND_MEDIAN_CONFIDENCE, DriftFlag.TREND_NO_BOX_RATE, DriftFlag.TREND_CLASS_SHARE]

    def test_transient_outlier_only_without_shift_or_trend(self):
        rules = DriftRules()
        outlier = [{"bucket_index": 3, "series": "median_confidence", "z": 5.0, "value": 0.6}]
        assert rules.flags(split_report(), {}, outlier, [], False, False) == [DriftFlag.TRANSIENT_OUTLIER]
        assert rules.flags(split_report(psi_confidence=0.3), {}, outlier, [], False, False) == \
            [DriftFlag.CONFIDENCE_SHIFT]
        assert rules.flags(None, {"median_confidence": trend()}, outlier, [], False, False) == \
            [DriftFlag.TREND_MEDIAN_CONFIDENCE]
        assert rules.flags(split_report(psi_confidence=0.15), {}, outlier, [], False, False) == \
            [DriftFlag.CONFIDENCE_SHIFT_MODERATE, DriftFlag.TRANSIENT_OUTLIER]


class TestPreVerdict:
    def test_insufficiency_overrides_everything(self):
        assert DriftRules.pre_verdict([DriftFlag.INSUFFICIENT_DATA, DriftFlag.CONFIDENCE_SHIFT]) == \
            PreVerdict.UNDETERMINED
        assert DriftRules.pre_verdict([DriftFlag.INSUFFICIENT_BUCKETS]) == PreVerdict.UNDETERMINED

    def test_likely_flags(self):
        for flag in (DriftFlag.HARD_BREAK, DriftFlag.CONFIDENCE_SHIFT, DriftFlag.CLASS_SHIFT,
                     DriftFlag.BOX_COUNT_SHIFT, DriftFlag.BOX_GEOMETRY_SHIFT):
            assert DriftRules.pre_verdict([flag]) == PreVerdict.DRIFT_LIKELY

    def test_suspicious_flags(self):
        for flag in (DriftFlag.CONFIDENCE_SHIFT_MODERATE, DriftFlag.CLASS_SHIFT_MODERATE,
                     DriftFlag.THRESHOLD_PRESSURE, DriftFlag.TRANSIENT_OUTLIER):
            assert DriftRules.pre_verdict([flag]) == PreVerdict.SUSPICIOUS
        assert DriftRules.pre_verdict([DriftFlag.THRESHOLD_PRESSURE, DriftFlag.CONFIDENCE_SHIFT]) == \
            PreVerdict.DRIFT_LIKELY

    def test_trends_count_per_family(self):
        confidence_family = [DriftFlag.TREND_MEDIAN_CONFIDENCE, DriftFlag.TREND_MEAN_CONFIDENCE,
                             DriftFlag.TREND_BELOW_THRESHOLD_RATE]
        assert DriftRules.pre_verdict(confidence_family) == PreVerdict.SUSPICIOUS
        assert DriftRules.pre_verdict([DriftFlag.TREND_NO_BOX_RATE]) == PreVerdict.SUSPICIOUS
        assert DriftRules.pre_verdict([DriftFlag.TREND_NO_BOX_RATE, DriftFlag.TREND_MEAN_BOXES_PER_IMAGE]) == \
            PreVerdict.SUSPICIOUS
        assert DriftRules.pre_verdict([DriftFlag.TREND_MEDIAN_CONFIDENCE, DriftFlag.TREND_CLASS_SHARE]) == \
            PreVerdict.DRIFT_LIKELY
        assert DriftRules.pre_verdict([DriftFlag.TREND_NO_BOX_RATE, DriftFlag.TREND_MEDIAN_NORMALIZED_CX]) == \
            PreVerdict.DRIFT_LIKELY

    def test_stable(self):
        assert DriftRules.pre_verdict([]) == PreVerdict.STABLE
