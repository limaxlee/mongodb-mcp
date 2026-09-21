import math
import random
from datetime import timedelta

from common.constants import DriftWindowMode, MedianSource
from mongodb_mcp.schemas import PeriodCounts, ConfidenceCounts
from mongodb_mcp.drift import build_periods, sum_periods, PeriodSummarizer
from tests.drift.synthetic import (
    START, EDGES, generate_documents, aggregate, make_document, cls_equipment, det_equipment, total_block
)


def documents_and_periods(task="cls", periods=3, products=1, equipment_id=None, **kwargs):
    documents = generate_documents(task, periods, lambda i: 0.93, products=products, **kwargs)
    return documents, build_periods(aggregate(documents, equipment_id), DriftWindowMode.CURRENT)


class TestBuildPeriods:
    def test_one_period_per_document_date(self):
        documents, periods = documents_and_periods(periods=3)
        assert [item.start_date for item in periods] == [START + timedelta(days=i) for i in range(3)]
        assert all(item.window == DriftWindowMode.CURRENT for item in periods)
        assert all(item.end_date == item.start_date + timedelta(days=1) for item in periods)

    def test_total_row_fills_period_level_fields(self):
        documents, periods = documents_and_periods(periods=1)
        period, doc = periods[0], documents[0]
        assert period.document_count == 1 and period.missing_block_count == 0
        assert period.tasks == ["cls"] and period.product_ids == [doc["productId"]]
        assert period.gbms == ["SEV"] and period.processes == ["SMD"] and period.modes == ["production"]
        assert period.equipment_ids == ["EQ-01"] and period.classes == ["Good", "NG"]
        assert period.bins == [doc["bins"]]
        assert period.prediction_count == 300 and period.inspection_count == 30 and period.box_count is None
        assert period.no_box_count is None and period.boxes_per_image_histogram == []
        assert period.backends == ["ts"] and period.thresholds == [0.8]
        assert period.elapsed_count == 300 and math.isclose(period.elapsed_sum, 300 * 0.008)

    def test_confidence_counts_of_a_single_document_keep_the_quantiles(self):
        documents, periods = documents_and_periods(periods=1)
        confidence = periods[0].confidence
        block = documents[0]["total"]["confidence"]
        assert confidence.count == block["count"] and confidence.histogram == block["histogram"]
        assert confidence.below_threshold_count == block["belowThresholdCount"]
        assert confidence.near_threshold_count == block["nearThresholdCount"]
        assert confidence.min == block["min"] and confidence.max == block["max"]
        assert confidence.quantiles == block["quantiles"]

    def test_class_rows(self):
        documents, periods = documents_and_periods(periods=1)
        period = periods[0]
        assert set(period.per_class) == {"Good", "NG"}
        assert period.class_counts == documents[0]["total"]["classCounts"]
        assert period.per_class["Good"].count == documents[0]["total"]["perClass"]["Good"]["count"]

    def test_several_products_are_summed_and_lose_the_quantiles(self):
        documents, periods = documents_and_periods(periods=1, products=3)
        period = periods[0]
        assert period.document_count == 3 and len(period.product_ids) == 3
        assert period.prediction_count == sum(doc["total"]["predictionCount"] for doc in documents)
        assert period.confidence.count == sum(doc["total"]["confidence"]["count"] for doc in documents)
        assert period.confidence.histogram == [
            sum(values) for values in zip(*(doc["total"]["confidence"]["histogram"] for doc in documents))
        ]
        assert period.confidence.quantiles is None

    def test_equipment_entry_is_used_and_missing_equipment_is_counted(self):
        documents, periods = documents_and_periods(
            periods=2, equipments=[("EQ-01", "Line_01"), ("EQ-02", "Line_02")], equipment_id="EQ-02"
        )
        entry = next(item for item in documents[0]["equipments"] if item["equipmentId"] == "EQ-02")
        assert periods[0].prediction_count == entry["predictionCount"]
        assert periods[0].equipment_ids == ["EQ-01", "EQ-02"]

        rows = aggregate(documents, "EQ-03")
        missing = build_periods(rows, DriftWindowMode.REFERENCE)
        assert len(missing) == 2
        assert all(item.document_count == 0 and item.missing_block_count == 1 for item in missing)
        assert all(item.window == DriftWindowMode.REFERENCE for item in missing)

    def test_detection_fields(self):
        documents, periods = documents_and_periods(task="det", periods=1)
        period, total = periods[0], documents[0]["total"]
        assert period.box_count == total["boxCount"] and period.no_box_count == total["images"]["noBoxCount"]
        assert period.boxes_per_image_sum == total["images"]["boxesPerImage"]["sum"]
        assert period.boxes_per_image_histogram == total["images"]["boxesPerImage"]["histogram"]
        assert period.thresholds == [] and period.per_class["Good"].thresholds == [0.8]

    def test_null_threshold_gives_none_counts(self):
        documents, periods = documents_and_periods(periods=1, stats_kwargs_for_period=lambda i: {"threshold": None})
        assert periods[0].confidence.below_threshold_count is None
        assert periods[0].confidence.near_threshold_count is None
        assert periods[0].thresholds == []

    def test_rows_without_date_are_skipped(self):
        assert build_periods([{"_id": {"kind": "total", "k": None}}], DriftWindowMode.CURRENT) == []


class TestSumPeriods:
    def test_sum_is_additive(self):
        _, periods = documents_and_periods(periods=3)
        total = sum_periods(periods)
        assert total.start_date == periods[0].start_date and total.end_date == periods[-1].end_date
        assert total.prediction_count == sum(item.prediction_count for item in periods)
        assert total.document_count == 3 and len(total.product_ids) == 3
        assert total.confidence.count == sum(item.confidence.count for item in periods)
        assert total.confidence.histogram == [
            sum(values) for values in zip(*(item.confidence.histogram for item in periods))
        ]
        assert total.confidence.quantiles is None
        assert total.class_counts["Good"] == sum(item.class_counts["Good"] for item in periods)
        assert total.per_class["NG"].count == sum(item.per_class["NG"].count for item in periods)

    def test_optional_counts(self):
        a = ConfidenceCounts(count=1, below_threshold_count=None, min=0.5, max=0.9)
        b = ConfidenceCounts(count=2, below_threshold_count=3, min=0.4, max=0.8)
        merged = a + b
        assert merged.below_threshold_count == 3 and merged.near_threshold_count is None
        assert merged.min == 0.4 and merged.max == 0.9


class TestPeriodSummarizer:
    def test_period_summary_from_exact_document(self):
        documents, periods = documents_and_periods(periods=1)
        summarizer = PeriodSummarizer("cls", ["Good", "NG"], EDGES)
        summary = summarizer.period(periods[0], START + timedelta(days=30))
        total = documents[0]["total"]

        assert summary.partial is False and summary.document_count == 1 and summary.product_count == 1
        assert math.isclose(summary.mean_confidence, total["confidence"]["mean"])
        assert math.isclose(summary.std_confidence, total["confidence"]["std"], abs_tol=1e-9)
        assert summary.median_confidence == total["confidence"]["quantiles"]["p50"]
        assert summary.median_source == MedianSource.EXACT
        assert math.isclose(sum(summary.class_distribution.values()), 1.0)
        assert math.isclose(summary.below_threshold_rate, total["confidence"]["belowThresholdCount"] / 300)
        assert math.isclose(sum(summary.confidence_histogram), 1.0)
        assert summary.confidence_quantiles.p50 == total["confidence"]["quantiles"]["p50"]
        assert summary.mean_boxes_per_image is None and summary.no_box_rate is None
        assert math.isclose(summary.mean_elapsed_time, 0.008)
        assert summary.backends == ["ts"] and summary.thresholds == [0.8] and summary.thresholds_by_class == {}
        assert set(summary.per_class) == {"Good", "NG"}
        assert summary.per_class["Good"].count == total["classCounts"]["Good"]
        assert math.isclose(summary.per_class["Good"].share, summary.class_distribution["Good"])
        assert summary.per_class["Good"].confidence_histogram is not None

    def test_period_summary_from_summed_documents_uses_histogram_median(self):
        _, periods = documents_and_periods(periods=1, products=2)
        summary = PeriodSummarizer("cls", ["Good", "NG"], EDGES).period(periods[0], START + timedelta(days=30))
        assert summary.median_source == MedianSource.HISTOGRAM
        assert 0.85 <= summary.median_confidence <= 1.0
        assert summary.confidence_quantiles is None

    def test_partial_period(self):
        _, periods = documents_and_periods(periods=1)
        summary = PeriodSummarizer("cls", ["Good", "NG"], EDGES).period(periods[0], START + timedelta(hours=1))
        assert summary.partial is True

    def test_compact_detail_drops_histograms_and_classes(self):
        _, periods = documents_and_periods(periods=1)
        summary = PeriodSummarizer("cls", ["Good", "NG"], EDGES, detail="compact").period(periods[0], START)
        assert summary.confidence_histogram is None and summary.confidence_quantiles is None
        assert summary.per_class == {} and summary.mean_confidence is not None

    def test_detection_summary(self):
        documents, periods = documents_and_periods(task="det", periods=1)
        summary = PeriodSummarizer("det", ["Good", "NG"], EDGES).period(periods[0], START + timedelta(days=30))
        total = documents[0]["total"]
        assert summary.box_count == total["boxCount"]
        assert math.isclose(summary.mean_boxes_per_image, total["boxCount"] / total["predictionCount"])
        assert math.isclose(summary.no_box_rate, total["images"]["noBoxCount"] / total["predictionCount"])
        assert math.isclose(sum(summary.boxes_per_image_histogram), 1.0)
        assert summary.thresholds == [] and summary.thresholds_by_class == {"Good": [0.8], "NG": [0.8]}
        assert summary.per_class["Good"].threshold == 0.8

    def test_missing_class_gets_zero(self):
        _, periods = documents_and_periods(periods=1)
        summary = PeriodSummarizer("cls", ["Good", "NG", "Scratch"], EDGES).period(periods[0], START)
        assert summary.class_distribution["Scratch"] == 0.0
        assert summary.per_class["Scratch"].count == 0 and summary.per_class["Scratch"].mean_confidence is None

    def test_side_summary(self):
        _, periods = documents_and_periods(periods=3)
        summarizer = PeriodSummarizer("cls", ["Good", "NG"], EDGES)
        side = summarizer.side(sum_periods(periods), 3)
        assert side.period_count == 3 and side.document_count == 3
        assert side.prediction_count == 900 and side.median_confidence is not None
        assert math.isclose(sum(side.confidence_histogram), 1.0)
        assert set(side.per_class) == {"Good", "NG"}

    def test_empty_counts(self):
        period = PeriodCounts(start_date=START, end_date=START + timedelta(days=1))
        summary = PeriodSummarizer("cls", ["Good"], EDGES).period(period, START)
        assert summary.mean_confidence is None and summary.median_confidence is None
        assert summary.confidence_histogram is None and summary.class_distribution == {"Good": 0.0}
        assert summary.mean_elapsed_time is None


class TestSyntheticTotals:
    def test_total_block_sums_equipments(self):
        rng = random.Random(1)
        a = dict(cls_equipment(rng, 100, 0.9, 0.1), equipmentId="A", location="L1")
        b = dict(cls_equipment(rng, 50, 0.8, 0.2), equipmentId="B", location="L2")
        total = total_block([a, b])
        assert total["predictionCount"] == 150 and total["confidence"]["count"] == 150
        assert total["classCounts"]["NG"] == a["classCounts"]["NG"] + b["classCounts"]["NG"]
        assert total["confidence"]["quantiles"] is None
        doc = make_document(START, "cls", [("A", "L1", a), ("B", "L2", b)])
        assert doc["equipmentCount"] == 2 and doc["total"]["predictionCount"] == 150

    def test_det_total_block(self):
        rng = random.Random(1)
        a = det_equipment(rng, 40, 0.9, 0.1)
        total = total_block([dict(a, equipmentId="A", location="L1")])
        assert total["boxCount"] == a["boxCount"]
        assert total["images"]["boxesPerImage"]["histogram"] == a["images"]["boxesPerImage"]["histogram"]
