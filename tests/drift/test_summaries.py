import random
import pytest
from datetime import timedelta

from common.constants import BoxGeometry, BucketSize, CLASS_SHARE_PREFIX
from mongodb_mcp.schemas import SideCounts, CompactBucketSummary
from mongodb_mcp.drift import RecordExtractor
from mongodb_mcp.drift.buckets import BucketBuilder
from mongodb_mcp.drift.summaries import Summarizer
from tests.drift.synthetic import generate_rows, make_row, cls_prediction, det_prediction, START


def records_for(task: str, days: int = 2, per_day: int = 100, **kwargs):
    if task == "det":
        rows = generate_rows(task, days, per_day, lambda rng, day: det_prediction(rng, 0.9, 0.3, **kwargs))
    else:
        rows = generate_rows(task, days, per_day, lambda rng, day: cls_prediction(rng, 0.9, 0.3, **kwargs))
    extractor = RecordExtractor()
    for row in rows:
        extractor.add_record(row)
    return extractor.records


class TestRecordAccess:
    def test_quantile_points_come_from_the_schema(self):
        summarizer = Summarizer("cls", ["Good", "NG"])
        assert summarizer.confidence_points == [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        assert summarizer.three_points == [0.10, 0.50, 0.90]
        assert summarizer.detection is False and Summarizer("det", []).detection is True

    def test_classification_values_come_from_records(self):
        records = records_for("cls")
        summarizer = Summarizer("cls", ["Good", "NG"])
        assert summarizer.confidences(records) == [record.max_confidence for record in records]
        assert summarizer.labels(records) == [record.prediction for record in records]
        assert summarizer.boxes(records) == []
        assert summarizer.thresholds(records) == [0.8]

    def test_detection_values_come_from_boxes(self):
        records = records_for("det")
        summarizer = Summarizer("det", ["Good", "NG"])
        boxes = summarizer.boxes(records)
        assert len(boxes) == sum(record.box_count for record in records)
        assert summarizer.confidences(records) == [box.max_confidence for box in boxes]
        assert summarizer.labels(records) == [box.prediction for box in boxes]
        assert summarizer.geometry(records, BoxGeometry.NORMALIZED_AREA) == [box.normalized_area for box in boxes]

    def test_class_counts_include_every_class(self):
        summarizer = Summarizer("cls", ["Good", "NG", "Scratch"])
        counts = summarizer.class_counts(records_for("cls"))
        assert set(counts) == {"Good", "NG", "Scratch"} and counts["Scratch"] == 0
        distribution = summarizer.class_distribution(counts)
        assert sum(distribution.values()) == pytest.approx(1.0)
        assert summarizer.class_distribution({"Good": 0}) == {"Good": 0.0}

    def test_below_threshold_rate(self):
        summarizer = Summarizer("cls", ["Good", "NG"])
        records = records_for("cls")
        below = [record.below_threshold for record in records]
        assert summarizer.below_threshold_rate(records) == pytest.approx(sum(below) / len(below))
        assert summarizer.below_threshold_rate(records_for("cls", threshold=None)) is None


class TestSideCounts:
    def test_counts_are_additive(self):
        summarizer = Summarizer("det", ["Good", "NG"])
        records = records_for("det", days=2, per_day=80)
        first, second = records[:70], records[70:]

        total = summarizer.side_counts(records)
        combined = summarizer.side_counts(first) + summarizer.side_counts(second)
        assert combined == total
        assert total - summarizer.side_counts(first) == summarizer.side_counts(second)
        assert total.record_count == len(records)
        assert sum(total.confidence_counts) == len(summarizer.confidences(records))
        assert len(total.boxes_per_image_counts) == 6

    def test_empty_side(self):
        counts = SideCounts() + SideCounts(record_count=2, confidence_counts=[1, 1])
        assert counts.record_count == 2 and counts.confidence_counts == [1, 1]
        assert Summarizer("cls", []).side_counts([]).box_count == 0


class TestSummaries:
    def test_bucket_summary_classification(self):
        records = records_for("cls", days=1, per_day=150)
        bucket = BucketBuilder("cls").build(records, BucketSize.DAY)[0]
        summary = Summarizer("cls", ["Good", "NG"]).bucket(bucket)

        assert summary["start_date"] == START and summary["end_date"] == START + timedelta(days=1)
        assert summary["record_count"] == 150 and summary["merged_from"] == 1
        assert set(summary["class_distribution"]) == {"Good", "NG"}
        assert 0.8 < summary["median_confidence"] < 1.0
        assert sum(summary["confidence_histogram"]) == pytest.approx(1.0)
        assert set(summary["confidence_quantiles"]) == {"p05", "p10", "p25", "p50", "p75", "p90", "p95"}
        assert summary["threshold_values"] == [0.8]
        assert summary["image_specs"] == [[400, 400, 3]]
        assert "box" not in summary and "mean_boxes_per_image" not in summary

    def test_bucket_summary_detection(self):
        records = records_for("det", days=1, per_day=150)
        bucket = BucketBuilder("det").build(records, BucketSize.DAY)[0]
        summary = Summarizer("det", ["Good", "NG"]).bucket(bucket)

        assert summary["box_count"] == sum(record.box_count for record in records)
        assert summary["mean_boxes_per_image"] == pytest.approx(summary["box_count"] / 150)
        assert len(summary["boxes_per_image_histogram"]) == 6
        assert set(summary["boxes_by_class_per_image"]) == {"Good", "NG"}
        assert set(summary["box"]["normalized_area_quantiles"]) == {"p10", "p50", "p90"}
        assert 0 < summary["box"]["normalized_cx_quantiles"]["p50"] < 1
        assert sum(summary["box"]["normalized_area_histogram"]) == pytest.approx(1.0)
        assert "near_threshold_rate" not in summary

    def test_side_summary(self):
        records = records_for("cls")
        summary = Summarizer("cls", ["Good", "NG"]).side(records)
        assert set(summary) == {
            "record_count", "box_count", "class_distribution", "confidence_histogram", "confidence_quantiles",
            "below_threshold_rate", "boxes_per_image_histogram", "median_elapsed_time"
        }
        assert summary["record_count"] == len(records)
        assert summary["box_count"] is None and summary["boxes_per_image_histogram"] is None
        assert summary["median_elapsed_time"] == 0.008

    def test_empty_bucket_summary(self):
        prediction = det_prediction(random.Random(1), 0.9, 0.0, no_box_rate=1.0)
        extractor = RecordExtractor()
        extractor.add_record(make_row(1, START, "det", prediction))
        records = extractor.records
        bucket = BucketBuilder("det").build(records, BucketSize.DAY)[0]
        summary = Summarizer("det", ["Good", "NG"]).bucket(bucket)
        assert summary["median_confidence"] is None and summary["confidence_quantiles"] == {}
        assert summary["no_box_rate"] == 1.0
        assert summary["box"]["normalized_area_quantiles"] == {"p10": None, "p50": None, "p90": None}
        assert summary["box"]["normalized_area_histogram"] is None

    def test_compact_keeps_the_schema_fields(self):
        records = records_for("det", days=1, per_day=150)
        bucket = BucketBuilder("det").build(records, BucketSize.DAY)[0]
        summarizer = Summarizer("det", ["Good", "NG"])
        full = summarizer.bucket(bucket)
        compact = summarizer.compact(full)

        assert set(compact) == set(CompactBucketSummary.model_fields)
        assert compact["median_confidence"] == full["median_confidence"]
        assert "confidence_histogram" not in compact and "box" not in compact


class TestSeriesAccess:
    def test_class_share_series(self):
        assert Summarizer("cls", ["Good", "NG"]).class_share_series() == [
            f"{CLASS_SHARE_PREFIX}Good", f"{CLASS_SHARE_PREFIX}NG"
        ]

    def test_series_value_reads_top_level_nested_and_class_share(self):
        summary = {
            "median_confidence": 0.9,
            "class_distribution": {"Good": 0.7, "NG": 0.3},
            "box": {"normalized_area_quantiles": {"p10": 0.01, "p50": 0.02, "p90": 0.03}}
        }
        assert Summarizer.series_value(summary, "median_confidence") == 0.9
        assert Summarizer.series_value(summary, f"{CLASS_SHARE_PREFIX}NG") == 0.3
        assert Summarizer.series_value(summary, f"{CLASS_SHARE_PREFIX}Scratch") is None
        assert Summarizer.series_value(summary, "median_normalized_area") == 0.02
        assert Summarizer.series_value(summary, "median_normalized_cx") is None
        assert Summarizer.series_value(summary, "no_box_rate") is None
        assert Summarizer.series_value({"median_confidence": True}, "median_confidence") is None
