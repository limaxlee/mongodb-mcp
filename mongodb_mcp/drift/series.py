"""Trend and outlier detection over the scalar series of the bucket summaries"""
from typing import Any, Sequence

from common.config import SETTINGS
from common.constants import CLASS_SHARE_PREFIX, TrendSeries, OutlierSeries, SeriesKind
from mongodb_mcp.utils import stats
from mongodb_mcp.drift.summaries import Summarizer


class SeriesAnalyzer:
    """Reads the scalar series out of chronological bucket summaries and tests them for trends and outliers"""

    def __init__(self, summarizer: Summarizer):
        self.config = SETTINGS.data_drift
        self.summarizer = summarizer

    @staticmethod
    def kind(name: str) -> SeriesKind:
        """Which practical relevance threshold applies to a series, decided by its name"""
        if name.startswith(CLASS_SHARE_PREFIX):
            return SeriesKind.CLASS_SHARE
        if name.endswith("_rate"):
            return SeriesKind.RATE
        if name == TrendSeries.MEAN_BOXES_PER_IMAGE:
            return SeriesKind.COUNT
        return SeriesKind.VALUE

    def relevant(self, name: str, reference: float, value: float, require_both: bool) -> bool:
        """Whether a move from reference to value is large enough to matter, statistical significance aside"""
        config = self.config
        change = abs(value - reference)
        kind = self.kind(name)
        if kind in (SeriesKind.RATE, SeriesKind.CLASS_SHARE):
            absolute = change >= config.trend_rate_abs
            relative = reference > 0 and change / reference >= config.trend_rate_rel
            return (absolute and relative) if require_both else (absolute or relative)
        if kind == SeriesKind.COUNT:
            return reference > 0 and change / reference >= config.trend_count_rel
        return change >= config.trend_value_change

    def points(self, summaries: Sequence[dict[str, Any]], name: str) -> dict[int, float]:
        """Bucket index to series value, skipping the buckets where the series is missing"""
        points: dict[int, float] = {}
        for index, summary in enumerate(summaries):
            value = self.summarizer.series_value(summary, name)
            if value is not None:
                points[index] = value
        return points

    def trends(self, summaries: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        config = self.config
        trend: dict[str, dict[str, Any]] = {}
        for name in [item.value for item in TrendSeries] + self.summarizer.class_share_series():
            points = self.points(summaries, name)
            if len(points) < config.min_buckets_trend:
                continue

            indices = [float(index) for index in points]
            values = list(points.values())
            tau, p = stats.kendall_tau(indices, values)
            slope = stats.theil_sen_slope(values)
            first, last = values[0], values[-1]

            trend[name] = {
                "tau": tau,
                "p": p,
                "slope_per_bucket": slope,
                "first": first,
                "last": last,
                "point_count": len(values),
                "meaningful": p < config.trend_p and abs(tau) >= config.trend_tau
                and self.relevant(name, first, last, require_both=False)
            }

        return trend

    def outliers(self, summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        config = self.config
        outliers: list[dict[str, Any]] = []
        for name in [item.value for item in OutlierSeries] + self.summarizer.class_share_series():
            points = self.points(summaries, name)
            if len(points) < config.min_buckets_trend:
                continue

            values = list(points.values())
            scores = stats.robust_z_scores(values)
            for position, (index, z) in enumerate(zip(points, scores)):
                value = values[position]
                others = values[:position] + values[position + 1:]
                center = stats.median(others)
                # A statistically extreme point that moved by a negligible amount is noise, not a transient
                if z is None or abs(z) <= config.robust_z or center is None:
                    continue
                if self.relevant(name, center, value, require_both=True):
                    outliers.append({
                        "bucket_index": index,
                        "start_date": summaries[index]["start_date"],
                        "series": name,
                        "z": z,
                        "value": value
                    })

        return outliers
