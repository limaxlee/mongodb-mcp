import pytest
from datetime import timedelta

from common.config import SETTINGS
from common.constants import BucketSize
from mongodb_mcp.drift import RecordExtractor
from mongodb_mcp.drift.buckets import BucketBuilder
from mongodb_mcp.drift.summaries import Summarizer
from mongodb_mcp.drift.splits import SplitFinder
from tests.drift.synthetic import generate_rows, cls_prediction, det_prediction, START


def records_for(task, days, per_day, confidence_for_day, ng_rate=0.2, **kwargs):
    if task == "det":
        rows = generate_rows(task, days, per_day, lambda rng, day: det_prediction(rng, confidence_for_day(day), ng_rate, **kwargs))
    else:
        rows = generate_rows(task, days, per_day, lambda rng, day: cls_prediction(rng, confidence_for_day(day), ng_rate, **kwargs))
    extractor = RecordExtractor()
    for row in rows:
        extractor.add(row)
    return extractor.records


def buckets_and_counts(task, records):
    summarizer = Summarizer(task, ["Good", "NG"])
    buckets = BucketBuilder(task).build(records, BucketSize.DAY)
    return summarizer, buckets, [summarizer.side_counts(bucket.records) for bucket in buckets]


class TestSufficiencyAndScore:
    def test_minimums_follow_the_task(self):
        assert SplitFinder(Summarizer("cls", [])).min_side_records == SETTINGS.data_drift.min_side_records_cls
        assert SplitFinder(Summarizer("det", [])).min_side_records == SETTINGS.data_drift.min_side_records_det

    def test_sides_sufficient(self):
        records = records_for("cls", 4, 300, lambda day: 0.9)
        summarizer = Summarizer("cls", ["Good", "NG"])
        finder = SplitFinder(summarizer)
        assert finder.sides_sufficient(summarizer.side_counts(records[:600]), summarizer.side_counts(records[600:]))
        assert not finder.sides_sufficient(summarizer.side_counts(records[:400]), summarizer.side_counts(records[400:]))

        records = records_for("det", 4, 300, lambda day: 0.9, boxes_mean=0.4, no_box_rate=0.6)
        summarizer = Summarizer("det", ["Good", "NG"])
        before, after = summarizer.side_counts(records[:600]), summarizer.side_counts(records[600:])
        assert before.record_count >= 300 and before.box_count < 300
        assert not SplitFinder(summarizer).sides_sufficient(before, after)

    def test_score_grows_with_divergence(self):
        summarizer = Summarizer("cls", ["Good", "NG"])
        finder = SplitFinder(summarizer)
        same = records_for("cls", 2, 300, lambda day: 0.9)
        moved = records_for("cls", 2, 300, lambda day: 0.6, ng_rate=0.6)
        assert finder.score(summarizer.side_counts(same[:300]), summarizer.side_counts(same[300:])) < 0.1
        assert finder.score(summarizer.side_counts(same), summarizer.side_counts(moved)) > 1.0

    def test_candidate_count(self):
        finder = SplitFinder(Summarizer("cls", []))
        min_segment = SETTINGS.data_drift.min_segment_buckets
        assert finder.candidate_count(2 * min_segment) == 1
        assert finder.candidate_count(12) == 12 - 2 * min_segment + 1
        assert finder.candidate_count(1) == 1


class TestCompare:
    def test_classification_report(self):
        before = records_for("cls", 3, 300, lambda day: 0.9)
        after = records_for("cls", 3, 300, lambda day: 0.7, ng_rate=0.5)
        summarizer = Summarizer("cls", ["Good", "NG"])
        report = SplitFinder(summarizer).compare(before, after, START, bucket_index=3)

        assert report["bucket_index"] == 3 and report["date"] == START and report["candidate_count"] == 1
        assert report["before_count"] == 900 and report["after_count"] == 900 and report["sides_sufficient"]
        assert report["psi_confidence"] > 0.25 and report["js_confidence"] > 0.1
        assert report["ks_confidence"]["d"] > 0.5 and report["ks_confidence"]["p"] < 0.001
        assert report["psi_class"] > 0.25 and report["chi2_class"]["p"] < 0.001 and report["chi2_class"]["dof"] == 1
        assert report["max_class_proportion_change"] == pytest.approx(0.3, abs=0.05)
        assert report["before"]["record_count"] == 900 and report["after"]["below_threshold_rate"] > 0.5
        assert "psi_boxes_per_image" not in report and "ks_normalized_area" not in report

    def test_detection_report_has_geometry(self):
        before = records_for("det", 3, 200, lambda day: 0.9)
        after = records_for("det", 3, 200, lambda day: 0.9, center=(300.0, 200.0))
        report = SplitFinder(Summarizer("det", ["Good", "NG"])).compare(before, after, START)

        assert report["psi_confidence"] < 0.1
        assert report["psi_boxes_per_image"] < 0.1
        assert report["ks_normalized_area"]["d"] < 0.1
        assert report["ks_normalized_cx"]["d"] > 0.5 and report["ks_normalized_cy"]["d"] < 0.1

    def test_p_values_are_corrected_for_candidates(self):
        before = records_for("cls", 2, 300, lambda day: 0.9, ng_rate=0.2)
        after = records_for("cls", 2, 300, lambda day: 0.9, ng_rate=0.2)
        finder = SplitFinder(Summarizer("cls", ["Good", "NG"]))
        single = finder.compare(before, after, START)
        searched = finder.compare(before, after, START, candidate_count=8)

        assert searched["candidate_count"] == 8
        assert searched["ks_confidence"]["p"] == min(single["ks_confidence"]["p"] * 8, 1.0)
        assert searched["chi2_class"]["p"] == min(single["chi2_class"]["p"] * 8, 1.0)
        assert searched["psi_confidence"] == single["psi_confidence"]

    def test_missing_confidences_give_a_neutral_ks(self):
        records = records_for("cls", 1, 50, lambda day: 0.9, threshold=None)
        for record in records:
            record.max_confidence = None
        report = SplitFinder(Summarizer("cls", ["Good", "NG"])).compare(records[:25], records[25:], START)
        assert report["ks_confidence"] == {"d": 0.0, "p": 1.0, "before_count": 0, "after_count": 0}


class TestSearch:
    def test_best_split_finds_the_step(self):
        records = records_for("cls", 10, 300, lambda day: 0.9 if day < 6 else 0.7)
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        assert SplitFinder(summarizer).best_split(counts) == 6

    def test_best_split_prefers_sufficient_sides(self):
        # The step sits after the first bucket, where the before side is too small, so a sufficient split wins
        records = records_for("cls", 6, 300, lambda day: 0.9 if day < 1 else 0.7)
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        assert SplitFinder(summarizer).best_split(counts) == 2

    def test_best_split_too_short(self):
        records = records_for("cls", 3, 300, lambda day: 0.9)
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        assert SplitFinder(summarizer).best_split(counts) is None

    def test_primary_and_secondary(self):
        records = records_for("cls", 12, 300, lambda day: 0.9 if day < 4 else (0.8 if day < 8 else 0.6))
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        finder = SplitFinder(summarizer)

        primary = finder.primary(buckets, counts)
        assert primary["bucket_index"] == 8 and primary["date"] == START + timedelta(days=8)
        assert primary["candidate_count"] == finder.candidate_count(12)
        assert primary["sides_sufficient"] and primary["psi_confidence"] > 0.25

        secondary = finder.secondary(buckets, counts, primary)
        assert all(item["sides_sufficient"] for item in secondary)
        by_index = {item["bucket_index"]: item for item in secondary}
        assert 4 in by_index and all(index != 8 for index in by_index)
        assert by_index[4]["date"] == START + timedelta(days=4)
        assert by_index[4]["candidate_count"] == finder.candidate_count(8)
        assert by_index[4]["psi_confidence"] > 0.25

    def test_secondary_needs_room_on_each_side(self):
        records = records_for("cls", 6, 300, lambda day: 0.9 if day < 3 else 0.7)
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        finder = SplitFinder(summarizer)
        primary = finder.primary(buckets, counts)
        assert primary["bucket_index"] == 3
        assert finder.secondary(buckets, counts, primary) == []

    def test_primary_none_when_too_short(self):
        records = records_for("cls", 3, 300, lambda day: 0.9)
        summarizer, buckets, counts = buckets_and_counts("cls", records)
        assert SplitFinder(summarizer).primary(buckets, counts) is None
