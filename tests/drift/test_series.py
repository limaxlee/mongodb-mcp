import pytest
from datetime import timedelta

from common.config import SETTINGS
from common.constants import SeriesKind, TrendSeries, OutlierSeries, CLASS_SHARE_PREFIX
from mongodb_mcp.drift.summaries import Summarizer
from mongodb_mcp.drift.series import SeriesAnalyzer
from tests.drift.synthetic import START


def summaries_for(values, series="median_confidence", good_share=None):
    """Minimal bucket summaries carrying one scalar series and optionally a class share series"""
    summaries = []
    for index, value in enumerate(values):
        summary = {"start_date": START + timedelta(days=index), series: value}
        if good_share is not None:
            summary["class_distribution"] = {"Good": good_share[index], "NG": 1 - good_share[index]}
        summaries.append(summary)
    return summaries


class TestKindAndRelevance:
    def test_kind_is_decided_by_the_name(self):
        assert SeriesAnalyzer.kind("median_confidence") == SeriesKind.VALUE
        assert SeriesAnalyzer.kind("median_normalized_area") == SeriesKind.VALUE
        assert SeriesAnalyzer.kind("below_threshold_rate") == SeriesKind.RATE
        assert SeriesAnalyzer.kind("no_box_rate") == SeriesKind.RATE
        assert SeriesAnalyzer.kind("mean_boxes_per_image") == SeriesKind.COUNT
        assert SeriesAnalyzer.kind(f"{CLASS_SHARE_PREFIX}NG") == SeriesKind.CLASS_SHARE

    def test_relevance_thresholds(self):
        config = SETTINGS.data_drift
        analyzer = SeriesAnalyzer(Summarizer("det", ["Good", "NG"]))

        assert analyzer.relevant("median_confidence", 0.90, 0.90 - config.trend_value_change, require_both=False)
        assert not analyzer.relevant("median_confidence", 0.90, 0.89, require_both=False)

        # A rate needs the absolute or the relative move for a trend, both for an outlier
        assert analyzer.relevant("no_box_rate", 0.10, 0.13, require_both=False)
        assert not analyzer.relevant("no_box_rate", 0.10, 0.13, require_both=True)
        assert analyzer.relevant("no_box_rate", 0.02, 0.05, require_both=True)
        assert not analyzer.relevant("no_box_rate", 0.0, 0.01, require_both=False)

        assert analyzer.relevant("mean_boxes_per_image", 2.0, 2.5, require_both=True)
        assert not analyzer.relevant("mean_boxes_per_image", 2.0, 2.1, require_both=False)
        assert not analyzer.relevant("mean_boxes_per_image", 0.0, 1.0, require_both=False)


class TestPoints:
    def test_points_skip_missing_values(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        summaries = summaries_for([0.9, None, 0.8], good_share=[0.7, 0.6, 0.5])
        assert analyzer.points(summaries, "median_confidence") == {0: 0.9, 2: 0.8}
        assert analyzer.points(summaries, f"{CLASS_SHARE_PREFIX}NG") == pytest.approx({0: 0.3, 1: 0.4, 2: 0.5})
        assert analyzer.points(summaries, "no_box_rate") == {}


class TestTrends:
    def test_monotonic_decline_is_meaningful(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        trend = analyzer.trends(summaries_for([0.95, 0.94, 0.92, 0.91, 0.89, 0.88, 0.86, 0.85]))

        assert set(trend) == {"median_confidence"}
        item = trend["median_confidence"]
        assert item["tau"] == -1.0 and item["p"] < 0.01
        assert item["slope_per_bucket"] < 0
        assert item["first"] == 0.95 and item["last"] == 0.85 and item["point_count"] == 8
        assert item["meaningful"]

    def test_significant_but_tiny_move_is_not_meaningful(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        trend = analyzer.trends(summaries_for([0.950, 0.949, 0.948, 0.947, 0.946, 0.945, 0.944, 0.943]))
        assert trend["median_confidence"]["tau"] == -1.0
        assert not trend["median_confidence"]["meaningful"]

    def test_noise_is_not_a_trend(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        trend = analyzer.trends(summaries_for([0.90, 0.93, 0.89, 0.94, 0.91, 0.92, 0.90, 0.93]))
        assert not trend["median_confidence"]["meaningful"]

    def test_short_series_are_skipped(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        assert analyzer.trends(summaries_for([0.9, 0.8, 0.7, 0.6, 0.5])) == {}

    def test_every_series_and_class_share_is_tested(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        summaries = summaries_for([0.9] * 6, good_share=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        trend = analyzer.trends(summaries)
        assert set(trend) == {"median_confidence", f"{CLASS_SHARE_PREFIX}Good", f"{CLASS_SHARE_PREFIX}NG"}
        assert trend[f"{CLASS_SHARE_PREFIX}Good"]["meaningful"] and trend[f"{CLASS_SHARE_PREFIX}NG"]["meaningful"]
        assert trend[f"{CLASS_SHARE_PREFIX}Good"]["tau"] == -1.0 and trend[f"{CLASS_SHARE_PREFIX}NG"]["tau"] == 1.0
        assert all(name in [item.value for item in TrendSeries] or name.startswith(CLASS_SHARE_PREFIX) for name in trend)


class TestOutliers:
    def test_single_spike_is_an_outlier(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        outliers = analyzer.outliers(summaries_for([0.90, 0.91, 0.90, 0.60, 0.90, 0.91, 0.90]))

        assert len(outliers) == 1
        item = outliers[0]
        assert item["bucket_index"] == 3 and item["start_date"] == START + timedelta(days=3)
        assert item["series"] == "median_confidence" and item["value"] == 0.60 and abs(item["z"]) > 3.5

    def test_extreme_but_negligible_move_is_ignored(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        assert analyzer.outliers(summaries_for([0.900, 0.900, 0.900, 0.905, 0.900, 0.900, 0.900])) == []

    def test_rate_outlier_needs_both_thresholds(self):
        analyzer = SeriesAnalyzer(Summarizer("det", ["Good", "NG"]))
        assert analyzer.outliers(summaries_for([0.10] * 6 + [0.125] + [0.10] * 5, series="no_box_rate")) == []
        outliers = analyzer.outliers(summaries_for([0.10] * 6 + [0.30] + [0.10] * 5, series="no_box_rate"))
        assert [item["bucket_index"] for item in outliers] == [6]

    def test_only_outlier_series_are_tested(self):
        analyzer = SeriesAnalyzer(Summarizer("det", ["Good", "NG"]))
        assert "mean_confidence" not in [item.value for item in OutlierSeries]
        assert analyzer.outliers(summaries_for([0.9, 0.9, 0.9, 0.5, 0.9, 0.9, 0.9], series="mean_confidence")) == []

    def test_short_series_are_skipped(self):
        analyzer = SeriesAnalyzer(Summarizer("cls", ["Good", "NG"]))
        assert analyzer.outliers(summaries_for([0.9, 0.9, 0.5, 0.9, 0.9])) == []
