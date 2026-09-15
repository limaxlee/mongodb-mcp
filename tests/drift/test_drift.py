import json
import random
import pytest
from datetime import timedelta, timezone

from common.config import SETTINGS, DataDriftConfig
from common.constants import DriftWindow, DriftFlag, PreVerdict, HardBreakKind
from mongodb_mcp.drift import DriftAnalyzer, RecordExtractor
from mongodb_mcp.schemas import DriftWindows, DriftAnalysisResult
from tests.drift.synthetic import generate_rows, make_row, cls_prediction, det_prediction, START

DAYS = 14
END = START + timedelta(days=DAYS)
FILTERS = {"gbm": "SEV", "process": None, "location": None, "equipment_id": None, "mode": "production"}


def extract(rows, task=None, reference_rows=None):
    extractor = RecordExtractor(task=task)
    for row in reference_rows or []:
        extractor.add(row, DriftWindow.REFERENCE)
    for row in rows:
        extractor.add(row)
    extractor.quality.scanned_document_count = len(rows) + len(reference_rows or [])
    return extractor


def analyze(rows, task=None, days=DAYS, windows=None, **kwargs):
    extractor = extract(rows, task)
    windows = windows or DriftWindows(start_date=START, end_date=START + timedelta(days=days))
    return DriftAnalyzer("Metal", "1.0", extractor.resolve_task(task), extractor, windows, FILTERS, **kwargs).run()


class TestScenarios:
    def test_no_drift_is_stable(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93, 0.2), seed=3)
        result = analyze(rows)

        assert result.pre_verdict == PreVerdict.STABLE
        assert result.flags == []
        assert result.change_point is not None and result.change_point.psi_confidence < 0.1
        assert result.change_point.candidate_count == DAYS - 2 * SETTINGS.data_drift.min_segment_buckets + 1
        assert result.status.analysis_possible and result.status.trend_ran and result.status.outlier_ran
        assert result.bucket == "1d" and result.task == "cls" and result.mode == "range"
        assert result.status.bucket_count == DAYS

    def test_step_change_is_found_on_day_eight(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93 if day < 7 else 0.8, 0.2))
        result = analyze(rows)

        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert result.pre_verdict == PreVerdict.DRIFT_LIKELY
        assert result.change_point.bucket_index == 7
        assert result.change_point.date == START + timedelta(days=7)
        assert result.change_point.sides_sufficient
        assert result.change_point.before.confidence_quantiles.p50 > result.change_point.after.confidence_quantiles.p50
        assert result.change_point.psi_confidence >= SETTINGS.data_drift.psi_significant

    def test_gradual_trend_is_flagged(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93 - 0.006 * day, 0.2))
        result = analyze(rows)

        assert DriftFlag.TREND_MEDIAN_CONFIDENCE in result.flags
        assert result.trend["median_confidence"].tau < -0.5
        assert result.trend["median_confidence"].slope_per_bucket < 0
        assert result.trend["median_confidence"].meaningful

    def test_one_bad_day_is_a_transient(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.6 if day == 6 else 0.9, 0.2))
        result = analyze(rows)

        assert result.flags == [DriftFlag.TRANSIENT_OUTLIER]
        assert result.pre_verdict == PreVerdict.SUSPICIOUS
        assert 6 in {item.bucket_index for item in result.outlier_buckets}
        # The bad bucket is left out of the split search, so no persistent shift is reported
        assert result.change_point.psi_confidence < 0.1
        assert result.change_point.bucket_index != 6

    def test_two_bad_days_are_a_shift(self):
        rows = generate_rows(
            "cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.6 if day in (6, 7) else 0.9, 0.2)
        )
        result = analyze(rows)
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        # Two flagged neighbours are a level change and stay in the search; the split lands on either edge of them
        assert result.change_point.bucket_index in (6, 8)

    def test_transient_next_to_a_step_does_not_hide_it(self):
        rows = generate_rows(
            "cls", DAYS, 300,
            lambda rng, day: cls_prediction(rng, 0.6 if day == 3 else (0.9 if day < 8 else 0.75), 0.2)
        )
        result = analyze(rows)
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert result.change_point.bucket_index == 8
        assert result.change_point.date == START + timedelta(days=8)

    def test_detection_transient_on_box_count(self):
        rows = generate_rows(
            "det", DAYS, 250,
            lambda rng, day: det_prediction(rng, 0.93, 0.1, no_box_rate=0.3 if day == 6 else 0.02)
        )
        result = analyze(rows)

        assert result.flags == [DriftFlag.TRANSIENT_OUTLIER]
        assert result.pre_verdict == PreVerdict.SUSPICIOUS
        assert {item.series for item in result.outlier_buckets} <= {"no_box_rate", "mean_boxes_per_image"}
        assert {item.bucket_index for item in result.outlier_buckets} == {6}

    def test_resolution_change_is_a_hard_break(self):
        rng = random.Random(3)
        rows = []
        for day in range(DAYS):
            scale = 1 if day < 7 else 2
            for slot in range(250):
                prediction = det_prediction(rng, 0.93, 0.1)
                for detection in prediction["detections"]:
                    detection["bbox"] = [value * scale for value in detection["bbox"]]
                rows.append(make_row(len(rows), START + timedelta(days=day, seconds=slot * 345), "det", prediction,
                                     image=(400 * scale, 400 * scale, 3)))
        result = analyze(rows)

        assert result.flags == [DriftFlag.HARD_BREAK]
        assert result.pre_verdict == PreVerdict.DRIFT_LIKELY
        assert [item.kind for item in result.hard_breaks] == [HardBreakKind.IMAGE_SPEC]
        assert result.hard_breaks[0].from_value == [400, 400, 3] and result.hard_breaks[0].to_value == [800, 800, 3]
        assert result.hard_breaks[0].date == START + timedelta(days=7)
        assert result.hard_breaks[0].class_name is None

    def test_detection_thresholds_are_tracked_per_class(self):
        def rows_for(ng_threshold_for_day):
            rng = random.Random(2)
            rows = []
            for day in range(8):
                for slot in range(200):
                    prediction = det_prediction(rng, 0.85, 0.3)
                    for detection in prediction["detections"]:
                        detection["threshold"] = ng_threshold_for_day(day) if detection["prediction"] == "NG" else 0.5
                    rows.append(make_row(len(rows), START + timedelta(days=day, seconds=slot * 432), "det", prediction))
            return rows

        stable = analyze(rows_for(lambda day: 0.8), days=8)
        assert stable.hard_breaks == [] and DriftFlag.HARD_BREAK not in stable.flags

        changed = analyze(rows_for(lambda day: 0.8 if day < 5 else 0.6), days=8)
        assert len(changed.hard_breaks) == 1
        item = changed.hard_breaks[0]
        assert item.kind == HardBreakKind.THRESHOLD and item.class_name == "NG"
        assert item.from_value == 0.8 and item.to_value == 0.6
        # The break is dated at the first record of day 5 that carries an NG box
        assert START + timedelta(days=5) <= item.date < START + timedelta(days=6)

    def test_classification_threshold_change_is_a_hard_break(self):
        rows = generate_rows(
            "cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2, threshold=0.8 if day < 5 else 0.7)
        )
        result = analyze(rows)
        assert [item.kind for item in result.hard_breaks] == [HardBreakKind.THRESHOLD]
        assert result.hard_breaks[0].from_value == 0.8 and result.hard_breaks[0].to_value == 0.7
        assert result.hard_breaks[0].inspection_id == rows[5 * 300]["_id"]

    def test_backend_and_class_list_changes(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        for row in rows[8 * 300:]:
            row["backend"] = "onnx"
            row["classes"] = ["Good", "NG", "Scratch"]
        result = analyze(rows)

        assert {item.kind for item in result.hard_breaks} == {HardBreakKind.BACKEND, HardBreakKind.CLASSES}
        assert result.classes == ["Good", "NG", "Scratch"]
        assert result.pre_verdict == PreVerdict.DRIFT_LIKELY

    def test_list_confidence_in_detections(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.9, 0.1, list_confidence=True))
        result = analyze(rows)

        assert result.data_quality.missing_confidence_count == 0
        assert result.buckets[3].median_confidence == pytest.approx(0.9, abs=0.02)

    def test_model_repeated_in_document(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        rows += [
            dict(row, entryIndex=1, prediction=cls_prediction(random.Random(index), 0.93, 0.2))
            for index, row in enumerate(rows)
        ]
        result = analyze(rows)

        assert result.data_quality.record_count == 2 * DAYS * 300
        assert result.data_quality.matched_document_count == DAYS * 300
        assert result.data_quality.matched_entry_count == 2 * DAYS * 300


class TestInsufficiency:
    def test_too_little_data(self):
        rows = generate_rows("cls", 2, 30, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        result = analyze(rows, days=2)

        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert DriftFlag.INSUFFICIENT_DATA in result.flags and DriftFlag.INSUFFICIENT_BUCKETS in result.flags
        assert result.status.analysis_possible is False
        assert result.change_point is None

    def test_no_side_can_be_sufficient(self):
        rows = generate_rows("cls", 4, 220, lambda rng, day: cls_prediction(rng, 0.93 if day < 3 else 0.6, 0.2))
        result = analyze(rows, days=4)

        assert result.flags == [DriftFlag.INSUFFICIENT_DATA]
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.change_point is not None and not result.change_point.sides_sufficient
        assert result.change_point.psi_confidence > 1.0

    def test_three_buckets_cannot_be_split(self):
        rows = generate_rows("cls", 3, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        result = analyze(rows, days=3)

        assert result.flags == [DriftFlag.INSUFFICIENT_BUCKETS]
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.status.analysis_possible is False and result.status.bucket_count == 3

    def test_detection_needs_boxes_on_each_side(self):
        rows = generate_rows(
            "det", 8, 150, lambda rng, day: det_prediction(rng, 0.85, 0.3, boxes_mean=0.3, no_box_rate=0.6)
        )
        result = analyze(rows, days=8)
        assert DriftFlag.INSUFFICIENT_DATA in result.flags
        assert result.data_quality.record_count == 1200 and result.data_quality.box_count < 600

    def test_no_records_at_all(self):
        result = analyze([], task="det")
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.status.bucket_count == 0
        assert result.data_quality.record_count == 0 and result.data_quality.box_count == 0
        assert result.buckets == [] and result.trend == {} and result.max_pairwise == {}


class TestModesAndOutput:
    def test_comparison_mode(self):
        reference_rows = generate_rows(
            "cls", 7, 400, lambda rng, day: cls_prediction(rng, 0.93, 0.2), start=START - timedelta(days=7)
        )
        current_rows = generate_rows("cls", 7, 400, lambda rng, day: cls_prediction(rng, 0.84, 0.2))
        extractor = extract(current_rows, reference_rows=reference_rows)
        windows = DriftWindows(
            start_date=START, end_date=START + timedelta(days=7),
            reference_start_date=START - timedelta(days=7), reference_end_date=START
        )
        result = DriftAnalyzer("Metal", "1.0", "cls", extractor, windows, FILTERS).run()

        assert result.mode == "comparison"
        assert result.change_point is None and result.comparison is not None
        assert result.status.comparison_ran and result.status.change_point_ran is False
        assert result.comparison.date == START and result.comparison.bucket_index is None
        assert result.comparison.candidate_count == 1
        assert result.comparison.before_count == len(reference_rows)
        assert result.comparison.after_count == len(current_rows)
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert [bucket.window for bucket in result.buckets] == [DriftWindow.REFERENCE] * 7 + [DriftWindow.CURRENT] * 7
        assert result.reference_range.days == 7.0 and result.reference_range.end_date == START
        assert result.data_quality.record_count == len(reference_rows) + len(current_rows)
        assert result.data_quality.scanned_document_count == len(reference_rows) + len(current_rows)

    def test_comparison_with_a_small_reference_is_undetermined(self):
        reference_rows = generate_rows(
            "cls", 2, 100, lambda rng, day: cls_prediction(rng, 0.93, 0.2), start=START - timedelta(days=2)
        )
        current_rows = generate_rows("cls", 5, 300, lambda rng, day: cls_prediction(rng, 0.7, 0.5))
        extractor = extract(current_rows, reference_rows=reference_rows)
        windows = DriftWindows(
            start_date=START, end_date=START + timedelta(days=5),
            reference_start_date=START - timedelta(days=2), reference_end_date=START
        )
        result = DriftAnalyzer("Metal", "1.0", "cls", extractor, windows, FILTERS).run()

        assert DriftFlag.INSUFFICIENT_DATA in result.flags
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.comparison is not None and not result.comparison.sides_sufficient

    def test_explicit_bucket(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        result = analyze(rows, bucket="1w")
        assert result.bucket == "1w" and result.status.bucket_count == 3

    def test_compact_detail_drops_histograms(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        full = analyze(rows, detail="full")
        compact = analyze(rows, detail="compact")

        assert full.detail == "full" and compact.detail == "compact"
        assert full.buckets[0].confidence_histogram is not None and full.buckets[0].box is not None
        assert compact.buckets[0].confidence_histogram is None and compact.buckets[0].box is None
        assert compact.buckets[0].median_confidence == full.buckets[0].median_confidence
        assert compact.buckets[0].mean_boxes_per_image == full.buckets[0].mean_boxes_per_image
        assert compact.flags == full.flags and compact.change_point == full.change_point
        assert len(compact.model_dump_json()) < len(full.model_dump_json())

    def test_result_is_json_serialisable(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        result = analyze(rows)
        payload = json.loads(result.model_dump_json(by_alias=True))

        assert payload["modelName"] == "Metal" and payload["modelVersion"] == "1.0"
        assert payload["filters"] == FILTERS
        assert payload["range"] == {"startDate": "2026-09-01T00:00:00Z", "endDate": "2026-09-15T00:00:00Z", "days": 14.0}
        assert payload["referenceRange"] is None
        assert payload["buckets"][0]["startDate"] == "2026-09-01T00:00:00Z"
        assert set(payload["status"]) == {
            "analysisPossible", "changePointRan", "comparisonRan", "trendRan", "outlierRan", "bucketCount"
        }
        assert "psiConfidence" in payload["changePoint"] and "ksNormalizedArea" in payload["changePoint"]
        assert set(payload["maxPairwise"]) == {"psi_confidence", "psi_class"}
        assert payload["config"]["confidence_bin_edges"] == [round(index / 10, 1) for index in range(11)]
        assert DriftAnalysisResult.model_validate(payload) == result

    def test_values_are_rounded(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        result = analyze(rows)
        decimals = SETTINGS.data_drift.decimals
        assert result.buckets[0].mean_confidence == round(result.buckets[0].mean_confidence, decimals)
        assert result.change_point.psi_confidence == round(result.change_point.psi_confidence, decimals)

    def test_full_detection_result_stays_small(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        payload = analyze(rows).model_dump_json(by_alias=True)
        assert len(payload) < 40 * 1024

    def test_config_is_echoed(self):
        rows = generate_rows("cls", 4, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.2))
        result = analyze(rows, days=4)
        assert set(result.config) == set(DataDriftConfig.model_fields)
        assert DataDriftConfig.model_validate(result.config) == SETTINGS.data_drift

    def test_max_pairwise_names_the_buckets(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93 if day < 7 else 0.8, 0.2))
        result = analyze(rows)
        pairwise = result.max_pairwise["psi_confidence"]
        assert pairwise.value >= result.change_point.psi_confidence
        assert pairwise.pair[0] < START + timedelta(days=7) <= pairwise.pair[1]

    def test_elapsed_time_break(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93 if day < 7 else 0.8, 0.2))
        for row in rows[7 * 300:]:
            row["prediction"]["elapsedTime"] = 0.02
        result = analyze(rows)

        elapsed = [item for item in result.hard_breaks if item.kind == HardBreakKind.ELAPSED_TIME]
        assert len(elapsed) == 1
        assert elapsed[0].from_value == 0.008 and elapsed[0].to_value == 0.02 and elapsed[0].inspection_id is None
        assert DriftFlag.HARD_BREAK not in result.flags


class TestWindows:
    def test_validation(self):
        with pytest.raises(ValueError, match="before end_date"):
            DriftWindows(start_date=END, end_date=START)
        with pytest.raises(ValueError, match="maximum is 30"):
            DriftWindows(start_date=START, end_date=START + timedelta(days=31))
        with pytest.raises(ValueError, match="given together"):
            DriftWindows(start_date=START, end_date=END, reference_start_date=START - timedelta(days=7))
        with pytest.raises(ValueError, match="before reference_end_date"):
            DriftWindows(start_date=START, end_date=END, reference_start_date=START, reference_end_date=START)
        with pytest.raises(ValueError, match="end before the current"):
            DriftWindows(start_date=START, end_date=END, reference_start_date=START - timedelta(days=1),
                         reference_end_date=START + timedelta(days=1))
        with pytest.raises(ValueError, match="maximum is 30"):
            DriftWindows(start_date=START, end_date=END, reference_start_date=START - timedelta(days=20),
                         reference_end_date=START)

    def test_ranges(self):
        windows = DriftWindows(start_date=START, end_date=END, reference_start_date=START - timedelta(days=14),
                               reference_end_date=START)
        assert windows.comparison and windows.total_days == 28.0
        assert windows.current.start_date == START and windows.current.days == 14.0
        assert windows.reference.end_date == START and windows.reference.days == 14.0

        windows = DriftWindows(start_date=START, end_date=END)
        assert not windows.comparison and windows.reference is None and windows.total_days == 14.0

    def test_naive_dates_are_treated_as_utc(self):
        windows = DriftWindows(start_date=START.replace(tzinfo=None), end_date=END.replace(tzinfo=None))
        assert windows.start_date == START and windows.start_date.tzinfo == timezone.utc
        assert windows.end_date == END
