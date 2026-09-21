import math
from datetime import timedelta

from common.config import SETTINGS
from common.constants import DriftWindowMode
from mongodb_mcp.schemas import ConfidenceCounts
from mongodb_mcp.drift import build_periods, sum_periods, PeriodSummarizer, SplitFinder
from tests.drift.synthetic import START, EDGES, generate_documents, aggregate


def periods_for(task, confidences, n=600, **kwargs):
    documents = generate_documents(task, len(confidences), lambda i: confidences[i], n=n, **kwargs)
    return build_periods(aggregate(documents), DriftWindowMode.CURRENT)


def finder(task="cls"):
    return SplitFinder(PeriodSummarizer(task, ["Good", "NG"], EDGES))


class TestSufficiency:
    def test_classification_needs_predictions(self):
        periods = periods_for("cls", [0.9, 0.9], n=SETTINGS.data_drift.min_side_records_cls)
        assert finder().sides_sufficient(periods[0], periods[1])
        small = periods_for("cls", [0.9, 0.9], n=SETTINGS.data_drift.min_side_records_cls - 1)
        assert not finder().sides_sufficient(small[0], small[1])

    def test_detection_needs_boxes_too(self):
        periods = periods_for("det", [0.9, 0.9], n=SETTINGS.data_drift.min_side_records_det)
        assert finder("det").sides_sufficient(periods[0], periods[1])
        few_boxes = periods_for(
            "det", [0.9, 0.9], n=SETTINGS.data_drift.min_side_records_det,
            stats_kwargs_for_period=lambda i: {"boxes_mean": 0.5, "no_box_rate": 0.8}
        )
        assert not finder("det").sides_sufficient(few_boxes[0], few_boxes[1])


class TestDivergence:
    def test_identical_sides_have_no_divergence(self):
        periods = periods_for("cls", [0.9, 0.9], seed=3)
        split = finder()
        assert split.psi_confidence(periods[0].confidence, periods[0].confidence) == 0.0
        assert split.psi_class(periods[0], periods[0]) == 0.0
        assert split.psi_confidence(periods[0].confidence, periods[1].confidence) < 0.1

    def test_confidence_drop_is_a_large_psi(self):
        periods = periods_for("cls", [0.93, 0.80])
        assert finder().psi_confidence(periods[0].confidence, periods[1].confidence) > 0.25
        assert finder().js_confidence(periods[0].confidence, periods[1].confidence) > 0.1

    def test_incomparable_histograms_give_none(self):
        split = finder()
        assert split.psi_confidence(ConfidenceCounts(), ConfidenceCounts(histogram=[1, 2])) is None
        assert split.psi_confidence(ConfidenceCounts(histogram=[1, 2]), ConfidenceCounts(histogram=[1, 2, 3])) is None
        assert split.js_confidence(ConfidenceCounts(histogram=[0, 0]), ConfidenceCounts(histogram=[1, 2])) is None

    def test_boxes_per_image_only_for_detection(self):
        cls_periods = periods_for("cls", [0.9, 0.9])
        assert finder().psi_boxes_per_image(cls_periods[0], cls_periods[1]) is None
        det_periods = periods_for("det", [0.9, 0.9])
        assert finder("det").psi_boxes_per_image(det_periods[0], det_periods[1]) is not None


class TestCompare:
    def test_report_fields(self):
        periods = periods_for("cls", [0.93, 0.93, 0.80, 0.80])
        report = finder().compare(periods[:2], periods[2:], periods[2].start_date, period_index=2, candidate_count=3)

        assert report["period_index"] == 2 and report["date"] == START + timedelta(days=2)
        assert report["candidate_count"] == 3 and report["sides_sufficient"]
        assert report["before_count"] == 1200 and report["after_count"] == 1200
        assert report["psi_confidence"] > 0.25 and report["js_confidence"] > 0
        assert set(report["psi_confidence_by_class"]) == {"Good", "NG"}
        assert report["psi_confidence_by_class"]["Good"] > 0.25
        assert report["psi_class"] < 0.1 and report["chi2_class"]["dof"] == 1
        assert report["max_class_proportion_change"] < 0.05
        assert report["psi_boxes_per_image"] is None
        assert report["mean_confidence_delta"] < -0.1
        assert report["below_threshold_rate_delta"] > 0
        assert math.isclose(report["elapsed_time_ratio"], 1.0)
        assert report["before"].period_count == 2 and report["after"].period_count == 2
        assert math.isclose(report["score"], report["psi_confidence"] + report["psi_class"])

    def test_chi2_p_is_corrected_by_candidate_count(self):
        periods = periods_for("cls", [0.9, 0.9, 0.9, 0.9], stats_kwargs_for_period=lambda i: {"ng_rate": 0.02 + 0.2 * (i >= 2)})
        single = finder().compare(periods[:2], periods[2:], periods[2].start_date, candidate_count=1)
        corrected = finder().compare(periods[:2], periods[2:], periods[2].start_date, candidate_count=5)
        assert corrected["chi2_class"]["p"] == min(single["chi2_class"]["p"] * 5, 1.0)
        assert corrected["psi_class"] > 0.25

    def test_detection_report(self):
        periods = periods_for("det", [0.9, 0.9, 0.9, 0.9], stats_kwargs_for_period=lambda i: {"boxes_mean": 1.0 + 2.0 * (i >= 2)})
        report = finder("det").compare(periods[:2], periods[2:], periods[2].start_date)
        assert report["psi_boxes_per_image"] > 0.25
        assert report["before"].mean_boxes_per_image < report["after"].mean_boxes_per_image


class TestBestSplit:
    def test_finds_the_step(self):
        periods = periods_for("cls", [0.93, 0.93, 0.93, 0.80, 0.80, 0.80])
        assert finder().best_split(periods) == 3
        report = finder().primary(periods)
        assert report["period_index"] == 3 and report["date"] == START + timedelta(days=3)
        assert report["candidate_count"] == finder().candidate_count(6)

    def test_needs_two_segments(self):
        periods = periods_for("cls", [0.93, 0.80, 0.80])
        assert finder().best_split(periods) is None and finder().primary(periods) is None

    def test_prefers_sufficient_split(self):
        # Tiny periods: no split is sufficient, the best of the insufficient ones is still returned
        periods = periods_for("cls", [0.93, 0.93, 0.80, 0.80], n=50)
        k = finder().best_split(periods)
        assert k == 2
        assert not finder().primary(periods)["sides_sufficient"]

    def test_candidate_count(self):
        assert finder().candidate_count(6) == 3
        assert finder().candidate_count(4) == 1
        assert finder().candidate_count(2) == 1

    def test_sum_of_split_sides_is_everything(self):
        periods = periods_for("cls", [0.9, 0.9, 0.9, 0.9])
        everything = sum_periods(periods)
        before, after = sum_periods(periods[:2]), sum_periods(periods[2:])
        assert before.prediction_count + after.prediction_count == everything.prediction_count
        assert [a + b for a, b in zip(before.confidence.histogram, after.confidence.histogram)] == everything.confidence.histogram
