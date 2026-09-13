import json
import pytest
from datetime import timedelta, datetime, timezone

from mongodb_mcp.drift import RecordExtractor, analyze_records, validate_windows, resolve_task
from mongodb_mcp.drift.config import DriftConfig
from mongodb_mcp.schemas import DriftAnalysisResult
from tests.drift.synthetic import generate_rows, cls_prediction, det_prediction, START

DAYS = 14
END = START + timedelta(days=DAYS)


def extract(rows, task=None):
    extractor = RecordExtractor(task=task)
    for row in rows:
        extractor.add(row)
    return extractor


def analyze(rows, task=None, **kwargs):
    extractor = extract(rows, task)
    resolved = resolve_task([extractor], task)
    return analyze_records("Metal", "1.0", resolved, extractor, START, END, {"gbm": "SEV"}, **kwargs)


class TestScenarios:
    def test_no_drift_is_stable(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93, 0.02))
        result = analyze(rows)

        assert result.pre_verdict == "stable"
        assert result.flags == []
        assert result.change_point is not None and result.change_point.psi_conf < 0.1
        assert result.status.analysis_possible and result.status.trend_ran
        assert result.bucket == "1d" and result.task == "cls" and result.mode == "range"

    def test_step_change_is_found_on_day_eight(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93 if day < 7 else 0.85, 0.02))
        result = analyze(rows)

        assert "CONFIDENCE_SHIFT" in result.flags
        assert result.pre_verdict == "drift_likely"
        assert result.change_point.date.date() == (START + timedelta(days=7)).date()
        assert result.change_point.sides_sufficient
        assert result.change_point.before.conf_quantiles.p50 > result.change_point.after.conf_quantiles.p50
        assert result.change_point.psi_conf >= 0.25

    def test_gradual_trend_is_flagged(self):
        rows = generate_rows("cls", DAYS, 400, lambda rng, day: cls_prediction(rng, 0.93 - 0.004 * day, 0.02))
        result = analyze(rows)

        assert "TREND_CONF_P50" in result.flags
        assert result.trend["conf_p50"].tau < -0.5
        assert result.trend["conf_p50"].slope_per_bucket < 0
        assert result.pre_verdict == "drift_likely"

    def test_transient_outlier_only(self):
        rows = generate_rows(
            "det", DAYS, 250,
            lambda rng, day: det_prediction(rng, 0.93, 0.1, no_box_rate=0.3 if day == 6 else 0.02)
        )
        result = analyze(rows)

        assert result.flags == ["TRANSIENT_OUTLIER"]
        assert result.pre_verdict == "suspicious"
        assert {item.series for item in result.outlier_buckets} == {"no_box_rate"}
        # UTC day 6 spans two local buckets in Seoul
        assert {item.bucket_index for item in result.outlier_buckets} <= {6, 7}

    def test_resolution_change_is_a_hard_break(self):
        def image_for(day):
            return {"image": (400, 400, 3) if day < 7 else (800, 800, 3)}

        def prediction(rng, day):
            scale = 1.0 if day < 7 else 2.0
            return det_prediction(rng, 0.93, 0.1, box_size=60 * scale, center=(200 * scale, 200 * scale))

        result = analyze(generate_rows("det", DAYS, 250, prediction, image_for))

        assert "HARD_BREAK" in result.flags
        assert result.pre_verdict == "drift_likely"
        assert [item.kind for item in result.hard_breaks] == ["image_spec"]
        assert result.hard_breaks[0].from_value == [400, 400, 3] and result.hard_breaks[0].to_value == [800, 800, 3]
        # Geometry is normalised, so the scaled boxes do not look like a geometry shift
        assert "BOX_GEOMETRY_SHIFT" not in result.flags

    def test_list_confidence_in_detections(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.9, 0.1, list_confidence=True))
        result = analyze(rows)

        assert result.data_quality.n_missing_confidence == 0
        assert result.buckets[3].conf_p50 == pytest.approx(0.9, abs=0.02)

    def test_model_repeated_in_document(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.02))
        rows += [
            dict(row, entryIndex=1, prediction=cls_prediction(__import__("random").Random(i), 0.93, 0.02))
            for i, row in enumerate(rows)
        ]
        result = analyze(rows)

        assert result.data_quality.n_records == 2 * DAYS * 300
        assert result.data_quality.n_docs_matched == DAYS * 300
        assert result.data_quality.n_entries_matched == 2 * DAYS * 300

    def test_too_little_data(self):
        rows = generate_rows("cls", 2, 30, lambda rng, day: cls_prediction(rng, 0.93, 0.02))
        result = analyze(rows)

        assert result.pre_verdict == "undetermined"
        assert "INSUFFICIENT_DATA" in result.flags and "INSUFFICIENT_BUCKETS" in result.flags
        assert result.status.analysis_possible is False
        assert result.change_point is None

    def test_no_records_at_all(self):
        result = analyze([], task="det")
        assert result.pre_verdict == "undetermined"
        assert result.status.n_buckets == 0
        assert result.data_quality.n_records == 0


class TestModesAndOutput:
    def test_comparison_mode(self):
        reference_rows = generate_rows("cls", 7, 400, lambda rng, day: cls_prediction(rng, 0.93, 0.02))
        current_rows = generate_rows(
            "cls", 7, 400, lambda rng, day: cls_prediction(rng, 0.84, 0.02), start=START + timedelta(days=30)
        )
        current, reference = extract(current_rows), extract(reference_rows)
        result = analyze_records(
            "Metal", "1.0", "cls", current, START + timedelta(days=30), START + timedelta(days=37), {},
            reference_extractor=reference, reference_start=START, reference_end=START + timedelta(days=7)
        )

        assert result.mode == "comparison"
        assert result.change_point is None and result.comparison is not None
        assert result.status.comparison_ran and result.status.change_point_ran is False
        assert result.comparison.date == START + timedelta(days=30)
        assert result.comparison.n_before == len(reference_rows) and result.comparison.n_after == len(current_rows)
        assert "CONFIDENCE_SHIFT" in result.flags
        assert {bucket.window for bucket in result.buckets} == {"reference", "current"}
        assert result.reference_range.days == 7.0
        assert result.data_quality.n_records == len(reference_rows) + len(current_rows)

    def test_compact_detail_drops_histograms(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        full = analyze(rows, detail="full")
        compact = analyze(rows, detail="compact")

        assert full.buckets[0].conf_hist is not None and full.buckets[0].box is not None
        assert compact.buckets[0].conf_hist is None and compact.buckets[0].box is None
        assert compact.buckets[0].conf_p50 == full.buckets[0].conf_p50
        assert compact.flags == full.flags
        assert len(compact.model_dump_json()) < len(full.model_dump_json())

    def test_result_is_json_serialisable(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        result = analyze(rows)
        payload = json.loads(result.model_dump_json(by_alias=True))

        assert payload["modelName"] == "Metal" and payload["modelVersion"] == "1.0"
        assert payload["buckets"][0]["bucketStart"].endswith("+09:00")
        assert payload["config"]["conf_bin_edges"] == [round(i / 10, 1) for i in range(11)]

    def test_full_detection_result_stays_small(self):
        rows = generate_rows("det", DAYS, 250, lambda rng, day: det_prediction(rng, 0.93, 0.1))
        payload = analyze(rows).model_dump_json(by_alias=True)
        assert len(payload) < 40 * 1024

    def test_defect_classes_default_and_override(self):
        rows = generate_rows("cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.1))
        assert analyze(rows).defect_classes == ["NG"]
        result = analyze(rows, defect_classes=["Good"])
        assert result.defect_classes == ["Good"]
        assert result.buckets[2].defect_rate == pytest.approx(result.buckets[2].class_dist["Good"])

    def test_threshold_change_is_a_hard_break(self):
        rows = generate_rows(
            "cls", DAYS, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.02, threshold=0.8 if day < 5 else 0.7)
        )
        result = analyze(rows)
        assert [item.kind for item in result.hard_breaks] == ["threshold"]
        assert result.hard_breaks[0].from_value == 0.8 and result.hard_breaks[0].to_value == 0.7

    def test_config_is_echoed(self):
        rows = generate_rows("cls", 3, 300, lambda rng, day: cls_prediction(rng, 0.93, 0.02))
        result = analyze(rows, config=DriftConfig(max_total_days=45))
        assert result.config["max_total_days"] == 45


class TestValidation:
    def test_windows(self):
        with pytest.raises(ValueError, match="before end_date"):
            validate_windows(END, START, None, None)
        with pytest.raises(ValueError, match="maximum is 30"):
            validate_windows(START, START + timedelta(days=31), None, None)
        with pytest.raises(ValueError, match="given together"):
            validate_windows(START, END, START - timedelta(days=7), None)
        with pytest.raises(ValueError, match="end before the current"):
            validate_windows(START, END, START - timedelta(days=1), START + timedelta(days=1))
        with pytest.raises(ValueError, match="maximum is 30"):
            validate_windows(START, END, START - timedelta(days=20), START)
        validate_windows(START, END, START - timedelta(days=14), START)

    def test_resolve_task(self):
        cls_rows = generate_rows("cls", 1, 5, lambda rng, day: cls_prediction(rng, 0.9, 0.0))
        det_rows = generate_rows("det", 1, 5, lambda rng, day: det_prediction(rng, 0.9, 0.0))

        assert resolve_task([extract(cls_rows)], None) == "cls"
        assert resolve_task([extract(det_rows)], None) == "det"
        assert resolve_task([extract(cls_rows), extract(det_rows)], "det") == "det"
        with pytest.raises(ValueError, match="multiple tasks"):
            resolve_task([extract(cls_rows), extract(det_rows)], None)
        with pytest.raises(ValueError, match="not supported"):
            resolve_task([], "seg")
