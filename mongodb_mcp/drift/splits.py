from typing import Any, Sequence
from datetime import datetime

from common.config import SETTINGS
from mongodb_mcp.utils import stats
from mongodb_mcp.schemas import ConfidenceCounts, PeriodCounts
from mongodb_mcp.drift.period import PeriodSummarizer, sum_periods


class SplitFinder:
    """Divergence between two groups of periods, and the search for the split that maximises it"""

    def __init__(self, summarizer: PeriodSummarizer):
        self.config = SETTINGS.data_drift
        self.summarizer = summarizer
        self.detection = summarizer.detection
        self.min_side_records = self.config.min_side_records_det if self.detection \
            else self.config.min_side_records_cls

    def sides_sufficient(self, before: PeriodCounts, after: PeriodCounts) -> bool:
        if before.prediction_count < self.min_side_records or after.prediction_count < self.min_side_records:
            return False
        if self.detection and ((before.box_count or 0) < self.config.min_side_boxes
                               or (after.box_count or 0) < self.config.min_side_boxes):
            return False
        return True

    @staticmethod
    def comparable(before: ConfidenceCounts, after: ConfidenceCounts) -> bool:
        return bool(before.histogram) and bool(after.histogram) \
            and len(before.histogram) == len(after.histogram) \
            and sum(before.histogram) > 0 and sum(after.histogram) > 0

    def psi_confidence(self, before: ConfidenceCounts, after: ConfidenceCounts) -> float | None:
        if not self.comparable(before, after):
            return None
        return stats.psi(stats.proportions(before.histogram), stats.proportions(after.histogram), self.config.eps)

    def js_confidence(self, before: ConfidenceCounts, after: ConfidenceCounts) -> float | None:
        if not self.comparable(before, after):
            return None
        return stats.js_divergence(
            stats.proportions(before.histogram), stats.proportions(after.histogram), self.config.eps
        )

    @staticmethod
    def class_table(before: PeriodCounts, after: PeriodCounts) -> tuple[list[str], list[int], list[int]]:
        keys = list(before.class_counts)
        keys.extend(key for key in after.class_counts if key not in keys)
        return keys, [before.class_counts.get(key, 0) for key in keys], [after.class_counts.get(key, 0) for key in keys]

    def psi_class(self, before: PeriodCounts, after: PeriodCounts) -> float | None:
        _, before_class, after_class = self.class_table(before, after)
        if sum(before_class) <= 0 or sum(after_class) <= 0:
            return None
        return stats.psi(stats.proportions(before_class), stats.proportions(after_class), self.config.eps)

    def psi_boxes_per_image(self, before: PeriodCounts, after: PeriodCounts) -> float | None:
        if not self.detection:
            return None
        before_hist, after_hist = before.boxes_per_image_histogram, after.boxes_per_image_histogram
        if not before_hist or not after_hist or len(before_hist) != len(after_hist) \
                or sum(before_hist) <= 0 or sum(after_hist) <= 0:
            return None
        return stats.psi(stats.proportions(before_hist), stats.proportions(after_hist), self.config.eps)

    def score(self, before: PeriodCounts, after: PeriodCounts) -> float:
        values = [
            self.psi_confidence(before.confidence, after.confidence),
            self.psi_class(before, after),
            self.psi_boxes_per_image(before, after)
        ]
        return sum(value for value in values if value is not None)

    def compare(
            self,
            before_periods: Sequence[PeriodCounts],
            after_periods: Sequence[PeriodCounts],
            date: datetime,
            period_index: int | None = None,
            candidate_count: int = 1
    ) -> dict[str, Any]:
        summarizer = self.summarizer
        before = sum_periods(before_periods)
        after = sum_periods(after_periods)

        keys, before_class, after_class = self.class_table(before, after)
        chi2_class = None
        max_prop_change = None
        if sum(before_class) > 0 and sum(after_class) > 0:
            chi2_stat, chi2_p, dof = stats.chi2_contingency([before_class, after_class])
            chi2_class = {"stat": chi2_stat, "p": min(chi2_p * candidate_count, 1.0), "dof": dof}
            before_dist = summarizer.class_distribution(dict(zip(keys, before_class)))
            after_dist = summarizer.class_distribution(dict(zip(keys, after_class)))
            max_prop_change = max((abs(before_dist[key] - after_dist[key]) for key in keys), default=0.0)

        class_names = list(before.per_class)
        class_names.extend(name for name in after.per_class if name not in class_names)

        before_side = summarizer.side(before, len(before_periods))
        after_side = summarizer.side(after, len(after_periods))

        return {
            "period_index": period_index,
            "date": date,
            "candidate_count": candidate_count,
            "score": self.score(before, after),
            "before_count": before.prediction_count,
            "after_count": after.prediction_count,
            "sides_sufficient": self.sides_sufficient(before, after),
            "psi_confidence": self.psi_confidence(before.confidence, after.confidence),
            "js_confidence": self.js_confidence(before.confidence, after.confidence),
            "psi_confidence_by_class": {
                name: self.psi_confidence(
                    before.per_class.get(name, ConfidenceCounts()), after.per_class.get(name, ConfidenceCounts())
                )
                for name in class_names
            },
            "psi_class": self.psi_class(before, after),
            "chi2_class": chi2_class,
            "max_class_proportion_change": max_prop_change,
            "psi_boxes_per_image": self.psi_boxes_per_image(before, after),
            "mean_confidence_delta": self._delta(before_side.mean_confidence, after_side.mean_confidence),
            "below_threshold_rate_delta": self._delta(
                before_side.below_threshold_rate, after_side.below_threshold_rate
            ),
            "elapsed_time_ratio": self._ratio(before_side.mean_elapsed_time, after_side.mean_elapsed_time),
            "before": before_side,
            "after": after_side
        }

    @staticmethod
    def _delta(before: float | None, after: float | None) -> float | None:
        if before is None or after is None:
            return None
        return after - before

    @staticmethod
    def _ratio(before: float | None, after: float | None) -> float | None:
        if not before or after is None:
            return None
        return after / before

    def candidate_count(self, period_count: int) -> int:
        return max(period_count - 2 * self.config.min_segment_periods + 1, 1)

    def best_split(self, periods: Sequence[PeriodCounts]) -> int | None:
        """Index of the first "after" period of the split with the largest PSI sum, sufficient sides preferred"""
        total = len(periods)
        min_seg = self.config.min_segment_periods
        if total < 2 * min_seg:
            return None

        prefix: list[PeriodCounts] = []
        running: PeriodCounts | None = None
        for item in periods:
            running = item if running is None else running + item
            prefix.append(running)

        best_sufficient: int | None = None
        best_sufficient_score = 0.0
        best_any: int | None = None
        best_any_score = 0.0
        for k in range(min_seg, total - min_seg + 1):
            before = prefix[k - 1]
            after = sum_periods(periods[k:])
            score = self.score(before, after)
            if best_any is None or score > best_any_score:
                best_any, best_any_score = k, score
            if self.sides_sufficient(before, after) and (best_sufficient is None or score > best_sufficient_score):
                best_sufficient, best_sufficient_score = k, score

        return best_sufficient if best_sufficient is not None else best_any

    def primary(self, periods: Sequence[PeriodCounts]) -> dict[str, Any] | None:
        k = self.best_split(periods)
        if k is None:
            return None

        return self.compare(
            periods[:k], periods[k:], periods[k].start_date,
            period_index=k, candidate_count=self.candidate_count(len(periods))
        )
