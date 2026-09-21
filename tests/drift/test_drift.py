import math
from datetime import datetime, timedelta, timezone

from common.config import SETTINGS
from common.constants import DriftFlag, PreVerdict, DriftWindowMode, HardBreakKind, Granularity, MedianSource
from mongodb_mcp.schemas import DriftFilters, DriftAnalysisResult
from mongodb_mcp.drift import DriftWindow, DriftAnalyzer, build_periods
from tests.drift.synthetic import START, EDGES, generate_documents, aggregate

CONFIG = SETTINGS.data_drift


def analyse(
        task="cls",
        confidences=(0.93, 0.93, 0.93, 0.93, 0.84, 0.84, 0.84),
        reference_confidences=None,
        n=600,
        granularity="daily",
        equipment_id=None,
        detail="full",
        mode="production",
        **kwargs
):
    hours = {"hourly": 1, "shift": 12, "daily": 24, "weekly": 168}[granularity]
    documents = generate_documents(task, len(confidences), lambda i: confidences[i], n=n, granularity=granularity, **kwargs)
    periods = build_periods(aggregate(documents, equipment_id), DriftWindowMode.CURRENT)
    reference_start = reference_end = None
    if reference_confidences is not None:
        reference_start = START - timedelta(hours=hours * len(reference_confidences))
        reference_documents = generate_documents(
            task, len(reference_confidences), lambda i: reference_confidences[i], n=n, granularity=granularity,
            start=reference_start, seed=11, **kwargs
        )
        periods = build_periods(aggregate(reference_documents, equipment_id), DriftWindowMode.REFERENCE) + periods
        reference_end = START

    windows = DriftWindow(START, START + timedelta(hours=hours * len(confidences)), reference_start, reference_end)
    windows.validate()
    analyzer = DriftAnalyzer(
        "MetalCls", "1.0", task, mode, periods, windows,
        DriftFilters(gbm="SEV", equipment_id=equipment_id), Granularity(granularity), detail
    )
    return analyzer.run()


class TestRangeMode:
    def test_step_is_found_and_flagged(self):
        result = analyse()
        assert isinstance(result, DriftAnalysisResult)
        assert result.analysis_mode == "range" and result.task == "cls" and result.mode == "production"
        assert result.granularity == Granularity.DAILY and result.detail == "full"
        assert result.status.analysis_possible and result.status.split_ran and not result.status.comparison_ran
        assert result.status.period_count == 7 and result.status.reference_period_count == 0
        assert result.change_point.period_index == 4 and result.change_point.date == START + timedelta(days=4)
        assert result.change_point.psi_confidence > CONFIG.psi_significant
        assert result.change_point.sides_sufficient
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags and DriftFlag.THRESHOLD_PRESSURE in result.flags
        assert DriftFlag.CLASS_SHIFT not in result.flags
        assert result.pre_verdict == PreVerdict.DRIFT_LIKELY
        assert result.comparison is None and result.reference_range is None

    def test_series_and_consecutive_divergence(self):
        result = analyse()
        assert len(result.periods) == 7 and all(item.window == "current" for item in result.periods)
        assert [item.start_date for item in result.periods] == [START + timedelta(days=i) for i in range(7)]
        assert all(item.median_source == MedianSource.EXACT for item in result.periods)
        assert result.periods[0].mean_confidence > result.periods[-1].mean_confidence + 0.05

        assert [item.period_index for item in result.consecutive] == list(range(1, 7))
        spike = max(result.consecutive, key=lambda item: item.psi_confidence)
        assert spike.date == START + timedelta(days=4) and spike.psi_confidence > CONFIG.psi_significant
        assert all(item.psi_confidence < CONFIG.psi_moderate for item in result.consecutive if item is not spike)
        assert all(item.sufficient for item in result.consecutive)

    def test_quiet_series_is_stable(self):
        result = analyse(confidences=(0.93,) * 6)
        assert result.flags == [] and result.pre_verdict == PreVerdict.STABLE
        assert result.change_point is not None and result.change_point.psi_confidence < CONFIG.psi_moderate

    def test_identity_and_sites(self):
        result = analyse()
        assert result.model_name == "MetalCls" and result.model_version == "1.0"
        assert result.filters.gbm == "SEV" and result.filters.equipment_id is None
        assert result.sites_seen.gbms == ["SEV"] and result.sites_seen.processes == ["SMD"]
        assert result.sites_seen.modes == ["production"] and result.sites_seen.equipment_ids == ["EQ-01"]
        assert result.classes == ["Good", "NG"]
        assert result.bins.confidence_edges == EDGES and result.bins.near_threshold_margin == 0.05
        assert result.range.days == 7.0
        assert result.config["psi_significant"] == CONFIG.psi_significant

    def test_data_quality(self):
        result = analyse(products=3)
        quality = result.data_quality
        assert quality.document_count == 21 and quality.product_count == 21
        assert quality.prediction_count == 7 * 600 and quality.box_count is None
        assert quality.missing_confidence_count == 0 and quality.parse_error_count == 0
        assert quality.partial_periods == [] and quality.periods_missing_equipment == []
        assert not quality.incompatible_bins
        assert all(item.median_source == MedianSource.HISTOGRAM for item in result.periods)
        assert all(item.document_count == 3 and item.product_count == 3 for item in result.periods)

    def test_partial_period_is_reported(self):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        documents = generate_documents("cls", 3, lambda i: 0.9, n=600, granularity="hourly", start=now - timedelta(hours=2))
        periods = build_periods(aggregate(documents), DriftWindowMode.CURRENT)
        windows = DriftWindow(now - timedelta(hours=2), now + timedelta(hours=1))
        result = DriftAnalyzer("MetalCls", "1.0", "cls", None, periods, windows, DriftFilters(), Granularity.HOURLY).run()
        assert result.data_quality.partial_periods == [now]
        assert result.periods[-1].partial and not result.periods[0].partial

    def test_compact_detail(self):
        result = analyse(detail="compact")
        assert result.detail == "compact"
        assert all(item.confidence_histogram is None and item.per_class == {} for item in result.periods)
        assert result.change_point.before.confidence_histogram is not None
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags

    def test_too_few_periods(self):
        result = analyse(confidences=(0.93, 0.93, 0.84))
        assert result.change_point is None and not result.status.analysis_possible
        assert DriftFlag.INSUFFICIENT_PERIODS in result.flags
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert len(result.consecutive) == 2

    def test_too_little_data(self):
        result = analyse(n=40)
        assert DriftFlag.INSUFFICIENT_DATA in result.flags
        assert result.change_point is not None and not result.change_point.sides_sufficient
        assert DriftFlag.CONFIDENCE_SHIFT not in result.flags
        assert result.pre_verdict == PreVerdict.UNDETERMINED

    def test_class_shift(self):
        result = analyse(
            confidences=(0.93,) * 6, stats_kwargs_for_period=lambda i: {"ng_rate": 0.02 if i < 3 else 0.3}
        )
        assert DriftFlag.CLASS_SHIFT in result.flags and result.change_point.period_index == 3
        assert result.change_point.max_class_proportion_change > 0.2
        assert DriftFlag.CONFIDENCE_SHIFT not in result.flags


class TestComparisonMode:
    def test_reference_versus_current(self):
        result = analyse(confidences=(0.84, 0.84, 0.84), reference_confidences=(0.93, 0.93, 0.93))
        assert result.analysis_mode == "comparison" and result.status.comparison_ran and not result.status.split_ran
        assert result.status.period_count == 3 and result.status.reference_period_count == 3
        assert result.reference_range.days == 3.0
        assert [item.window for item in result.periods] == ["reference"] * 3 + ["current"] * 3
        assert result.comparison.period_index is None and result.comparison.date == START
        assert result.comparison.candidate_count == 1
        assert result.comparison.before.period_count == 3 and result.comparison.after.period_count == 3
        assert result.comparison.psi_confidence > CONFIG.psi_significant
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags and result.change_point is None
        assert result.pre_verdict == PreVerdict.DRIFT_LIKELY

    def test_consecutive_crosses_the_boundary(self):
        result = analyse(confidences=(0.84, 0.84), reference_confidences=(0.93, 0.93))
        boundary = next(item for item in result.consecutive if item.date == START)
        assert boundary.period_index == 2 and boundary.psi_confidence > CONFIG.psi_significant

    def test_same_distribution_is_stable(self):
        result = analyse(confidences=(0.93, 0.93, 0.93), reference_confidences=(0.93, 0.93, 0.93))
        assert result.flags == [] and result.pre_verdict == PreVerdict.STABLE

    def test_missing_side_is_insufficient(self):
        documents = generate_documents("cls", 3, lambda i: 0.9, n=600)
        periods = build_periods(aggregate(documents), DriftWindowMode.CURRENT)
        windows = DriftWindow(START, START + timedelta(days=3), START - timedelta(days=3), START)
        result = DriftAnalyzer("MetalCls", "1.0", "cls", None, periods, windows, DriftFilters(), Granularity.DAILY).run()
        assert result.comparison is None and DriftFlag.INSUFFICIENT_DATA in result.flags
        assert result.pre_verdict == PreVerdict.UNDETERMINED


class TestDetection:
    def test_detection_series_and_box_shift(self):
        result = analyse(
            task="det", confidences=(0.9,) * 6, n=400,
            stats_kwargs_for_period=lambda i: {"boxes_mean": 1.0 if i < 3 else 3.0}
        )
        assert result.task == "det"
        assert result.data_quality.box_count > 0
        assert all(item.box_count > 0 and item.mean_boxes_per_image is not None for item in result.periods)
        assert result.periods[0].boxes_per_image_histogram is not None
        assert result.periods[0].thresholds_by_class == {"Good": [0.8], "NG": [0.8]}
        assert all(item.psi_boxes_per_image is not None for item in result.consecutive)
        assert result.change_point.period_index == 3
        assert DriftFlag.BOX_COUNT_SHIFT in result.flags and result.pre_verdict == PreVerdict.DRIFT_LIKELY

    def test_detection_confidence_shift(self):
        result = analyse(task="det", confidences=(0.93, 0.93, 0.93, 0.80, 0.80, 0.80), n=400)
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert result.change_point.psi_confidence_by_class["Good"] > CONFIG.psi_significant


class TestHardBreaks:
    def test_backend_and_threshold_break(self):
        result = analyse(
            confidences=(0.93,) * 6,
            stats_kwargs_for_period=lambda i: {"backend": "ts" if i < 2 else "trt", "threshold": 0.8 if i < 4 else 0.9}
        )
        kinds = [(item.kind, item.class_name, item.date, item.from_value, item.to_value) for item in result.hard_breaks]
        assert (HardBreakKind.BACKEND, None, START + timedelta(days=2), "ts", "trt") in kinds
        assert (HardBreakKind.THRESHOLD, "Good", START + timedelta(days=4), 0.8, 0.9) in kinds
        assert (HardBreakKind.THRESHOLD, "NG", START + timedelta(days=4), 0.8, 0.9) in kinds
        assert len(result.hard_breaks) == 3
        assert DriftFlag.HARD_BREAK in result.flags and result.pre_verdict == PreVerdict.SUSPICIOUS

    def test_threshold_break_carries_the_class(self):
        result = analyse(
            task="det", confidences=(0.9,) * 4, n=400,
            stats_kwargs_for_period=lambda i: {"threshold": 0.8 if i < 2 else 0.7}
        )
        thresholds = [item for item in result.hard_breaks if item.kind == HardBreakKind.THRESHOLD]
        assert {item.class_name for item in thresholds} == {"Good", "NG"}
        assert all(item.date == START + timedelta(days=2) and item.from_value == 0.8 and item.to_value == 0.7
                   for item in thresholds)

    def test_class_list_break(self):
        result = analyse(
            confidences=(0.93,) * 4,
            doc_kwargs_for_period=lambda i: {"classes": ["Good", "NG"] if i < 2 else ["Good", "NG", "Scratch"]}
        )
        classes = [item for item in result.hard_breaks if item.kind == HardBreakKind.CLASSES]
        assert len(classes) == 1 and classes[0].to_value == ["Good", "NG", "Scratch"]
        assert result.classes == ["Good", "NG", "Scratch"]

    def test_several_values_inside_one_period(self):
        # Two products in the same period with different backends
        documents = generate_documents("cls", 2, lambda i: 0.9, n=600)
        documents += generate_documents(
            "cls", 2, lambda i: 0.9, n=600, seed=5, stats_kwargs_for_period=lambda i: {"backend": "trt"},
            doc_kwargs_for_period=lambda i: {"product_id": "PR-X"}
        )
        for doc in documents[2:]:
            doc["productId"] = "PR-X-" + doc["productId"]
        periods = build_periods(aggregate(documents), DriftWindowMode.CURRENT)
        windows = DriftWindow(START, START + timedelta(days=2))
        result = DriftAnalyzer("MetalCls", "1.0", "cls", None, periods, windows, DriftFilters(), Granularity.DAILY).run()
        backend_breaks = [item for item in result.hard_breaks if item.kind == HardBreakKind.BACKEND]
        assert len(backend_breaks) == 1
        assert backend_breaks[0].date == START and backend_breaks[0].to_value == ["trt", "ts"]
        assert result.periods[0].backends == ["trt", "ts"]


class TestBinsAndEquipment:
    def test_incompatible_bins(self):
        other = {"confidenceEdges": [0.0, 0.5, 1.0], "nearThresholdMargin": 0.05, "boxesPerImageMax": 5}
        result = analyse(
            confidences=(0.93,) * 6,
            doc_kwargs_for_period=lambda i: {"bins": other} if i >= 3 else {}
        )
        assert result.data_quality.incompatible_bins
        assert DriftFlag.INCOMPATIBLE_BINS in result.flags and result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.change_point is None
        crossing = next(item for item in result.consecutive if item.date == START + timedelta(days=3))
        assert crossing.psi_confidence is None and crossing.psi_class is not None
        assert len(result.periods) == 6

    def test_equipment_entry_and_missing_periods(self):
        documents = generate_documents(
            "cls", 4, lambda i: 0.9, n=600, equipments=[("EQ-01", "Line_01"), ("EQ-02", "Line_02")]
        )
        for doc in documents[2:]:
            doc["equipments"] = [item for item in doc["equipments"] if item["equipmentId"] != "EQ-02"]
        periods = build_periods(aggregate(documents, "EQ-02"), DriftWindowMode.CURRENT)
        windows = DriftWindow(START, START + timedelta(days=4))
        result = DriftAnalyzer(
            "MetalCls", "1.0", "cls", None, periods, windows, DriftFilters(equipment_id="EQ-02"), Granularity.DAILY
        ).run()
        assert len(result.periods) == 2
        assert result.data_quality.periods_missing_equipment == [START + timedelta(days=2), START + timedelta(days=3)]
        assert result.sites_seen.equipment_ids == ["EQ-02"]
        assert result.periods[0].prediction_count == 600


class TestRounding:
    def test_floats_are_rounded(self):
        result = analyse()
        for item in result.periods:
            assert item.mean_confidence == round(item.mean_confidence, CONFIG.decimals)
            for value in item.confidence_histogram:
                assert value == round(value, CONFIG.decimals)
        assert result.change_point.psi_confidence == round(result.change_point.psi_confidence, CONFIG.decimals)
        assert math.isclose(sum(result.change_point.before.class_distribution.values()), 1.0, abs_tol=1e-3)
