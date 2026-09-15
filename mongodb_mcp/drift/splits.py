"""Before/after comparison of two groups of records, and the search for the split that separates them most"""
from typing import Any, Sequence
from datetime import datetime

from common.config import SETTINGS
from common.constants import BoxGeometry
from mongodb_mcp.utils import stats
from mongodb_mcp.schemas import Record, Bucket, SideCounts
from mongodb_mcp.drift.summaries import Summarizer


class SplitFinder:
    """Divergence between two sides and the best split of a bucket sequence, for one summarizer"""

    def __init__(self, summarizer: Summarizer):
        self.config = SETTINGS.data_drift
        self.summarizer = summarizer
        self.detection = summarizer.detection
        self.min_side_records = self.config.min_side_records_det if self.detection \
            else self.config.min_side_records_cls

    def sides_sufficient(self, before: SideCounts, after: SideCounts) -> bool:
        if before.record_count < self.min_side_records or after.record_count < self.min_side_records:
            return False
        if self.detection and (before.box_count < self.config.min_side_boxes or after.box_count < self.config.min_side_boxes):
            return False
        return True

    def score(self, before: SideCounts, after: SideCounts) -> float:
        """Sum of the PSI values of every distribution the split is searched on"""
        eps = self.config.eps
        score = stats.psi(stats.proportions(before.confidence_counts), stats.proportions(after.confidence_counts), eps)
        keys = sorted(set(before.class_counts) | set(after.class_counts))
        score += stats.psi(
            stats.proportions([before.class_counts.get(key, 0) for key in keys]),
            stats.proportions([after.class_counts.get(key, 0) for key in keys]),
            eps
        )
        if self.detection:
            score += stats.psi(
                stats.proportions(before.boxes_per_image_counts), stats.proportions(after.boxes_per_image_counts), eps
            )
        return score

    @staticmethod
    def _ks(before: Sequence[float], after: Sequence[float]) -> dict[str, Any] | None:
        if not before or not after:
            return None
        d, p = stats.ks_2samp(before, after)
        return {"d": d, "p": p, "before_count": len(before), "after_count": len(after)}

    def compare(
            self,
            before: Sequence[Record],
            after: Sequence[Record],
            date: datetime,
            bucket_index: int | None = None,
            candidate_count: int = 1
    ) -> dict[str, Any]:
        """Full divergence report between two groups of records, computed from the records themselves

        A split chosen among candidate_count positions carries p-values corrected for that choice (Bonferroni), since
        the maximum of many noisy tests always looks more significant than a single test would.
        """
        eps = self.config.eps
        summarizer = self.summarizer
        before_counts = summarizer.side_counts(before)
        after_counts = summarizer.side_counts(after)

        before_hist = stats.proportions(before_counts.confidence_counts)
        after_hist = stats.proportions(after_counts.confidence_counts)

        keys = sorted(set(before_counts.class_counts) | set(after_counts.class_counts))
        before_class = [before_counts.class_counts.get(key, 0) for key in keys]
        after_class = [after_counts.class_counts.get(key, 0) for key in keys]
        chi2_stat, chi2_p, dof = stats.chi2_contingency([before_class, after_class])
        before_dist = summarizer.class_distribution(dict(zip(keys, before_class)))
        after_dist = summarizer.class_distribution(dict(zip(keys, after_class)))
        max_prop_change = max((abs(before_dist[key] - after_dist[key]) for key in keys), default=0.0)

        report: dict[str, Any] = {
            "bucket_index": bucket_index,
            "date": date,
            "candidate_count": candidate_count,
            "score": self.score(before_counts, after_counts),
            "before_count": before_counts.record_count,
            "after_count": after_counts.record_count,
            "sides_sufficient": self.sides_sufficient(before_counts, after_counts),
            "psi_confidence": stats.psi(before_hist, after_hist, eps),
            "js_confidence": stats.js_divergence(before_hist, after_hist, eps),
            "ks_confidence": self._ks(summarizer.confidences(before), summarizer.confidences(after))
            or {"d": 0.0, "p": 1.0, "before_count": 0, "after_count": 0},
            "psi_class": stats.psi(stats.proportions(before_class), stats.proportions(after_class), eps),
            "chi2_class": {"stat": chi2_stat, "p": chi2_p, "dof": dof},
            "max_class_proportion_change": max_prop_change,
            "before": summarizer.side(before),
            "after": summarizer.side(after)
        }

        if self.detection:
            report["psi_boxes_per_image"] = stats.psi(
                stats.proportions(before_counts.boxes_per_image_counts),
                stats.proportions(after_counts.boxes_per_image_counts),
                eps
            )
            for measure in BoxGeometry:
                report[f"ks_{measure}"] = self._ks(
                    summarizer.geometry(before, measure), summarizer.geometry(after, measure)
                )

        if candidate_count > 1:
            for key in ["ks_confidence", "chi2_class"] + [f"ks_{measure}" for measure in BoxGeometry]:
                test = report.get(key)
                if test:
                    test["p"] = min(test["p"] * candidate_count, 1.0)

        return report

    def candidate_count(self, bucket_count: int) -> int:
        """How many split positions the search over bucket_count tries"""
        return max(bucket_count - 2 * self.config.min_segment_buckets + 1, 1)

    def best_split(self, counts: Sequence[SideCounts]) -> int | None:
        """Index k maximising the divergence between buckets[:k] and buckets[k:], preferring sufficient sides

        When no split has enough records on both sides the best insufficient split is returned so the reader still
        sees where the largest movement is, and the flags stay silent. None when the sequence is too short.
        """
        total = len(counts)
        min_seg = self.config.min_segment_buckets
        if total < 2 * min_seg:
            return None

        prefix: list[SideCounts] = []
        running = SideCounts()
        for item in counts:
            running = running + item
            prefix.append(running)
        grand_total = prefix[-1]

        best_sufficient: int | None = None
        best_sufficient_score = 0.0
        best_any: int | None = None
        best_any_score = 0.0
        for k in range(min_seg, total - min_seg + 1):
            before = prefix[k - 1]
            after = grand_total - before
            score = self.score(before, after)
            if best_any is None or score > best_any_score:
                best_any, best_any_score = k, score
            if self.sides_sufficient(before, after) and (best_sufficient is None or score > best_sufficient_score):
                best_sufficient, best_sufficient_score = k, score

        return best_sufficient if best_sufficient is not None else best_any

    def primary(self, buckets: Sequence[Bucket], counts: Sequence[SideCounts]) -> dict[str, Any] | None:
        """The split with the largest divergence over the whole bucket sequence"""
        return self._report(buckets, counts, offset=0)

    def secondary(
            self,
            buckets: Sequence[Bucket],
            counts: Sequence[SideCounts],
            primary: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """At most one sufficient split on each side of the primary one (one level of binary segmentation)"""
        k = primary["bucket_index"]
        reports: list[dict[str, Any]] = []
        for lo, hi in ((0, k), (k, len(buckets))):
            if hi - lo >= 2 * self.config.min_segment_buckets:
                report = self._report(buckets[lo:hi], counts[lo:hi], offset=lo)
                if report is not None and report["sides_sufficient"]:
                    reports.append(report)

        return reports

    def _report(
            self,
            buckets: Sequence[Bucket],
            counts: Sequence[SideCounts],
            offset: int
    ) -> dict[str, Any] | None:
        k = self.best_split(counts)
        if k is None:
            return None

        before = [record for bucket in buckets[:k] for record in bucket.records]
        after = [record for bucket in buckets[k:] for record in bucket.records]
        return self.compare(
            before, after, buckets[k].start_date, bucket_index=offset + k, candidate_count=self.candidate_count(len(buckets))
        )
