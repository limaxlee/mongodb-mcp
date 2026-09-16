from typing import Any, Sequence

from common.config import SETTINGS
from common.constants import AUTO_BUCKET, DriftDetail, DriftMode, DriftWindowMode, HardBreakKind
from mongodb_mcp.utils import stats
from mongodb_mcp.schemas import Record, Bucket, DriftAnalysisResult
from mongodb_mcp.drift.window import DriftWindow
from mongodb_mcp.drift.extractor import RecordExtractor
from mongodb_mcp.drift.buckets import BucketBuilder
from mongodb_mcp.drift.summaries import Summarizer
from mongodb_mcp.drift.splits import SplitFinder
from mongodb_mcp.drift.series import SeriesAnalyzer
from mongodb_mcp.drift.rules import DriftRules


class DriftAnalyzer:
    def __init__(
            self,
            model_name: str,
            model_version: str,
            task: str,
            extractor: RecordExtractor,
            windows: DriftWindow,
            filters: dict[str, str | None],
            bucket: str = AUTO_BUCKET,
            detail: str = DriftDetail.FULL
    ):
        self.config = SETTINGS.data_drift
        self.model_name = model_name
        self.model_version = model_version
        self.task = task
        self.extractor = extractor
        self.windows = windows
        self.filters = filters
        self.requested_bucket = bucket
        self.detail = DriftDetail(detail)

        self.records = sorted(extractor.records, key=lambda item: item.created_at)
        self.current = [record for record in self.records if record.window == DriftWindowMode.CURRENT]
        self.reference = [record for record in self.records if record.window == DriftWindowMode.REFERENCE]

        self.summarizer = Summarizer(task, extractor.classes)
        self.builder = BucketBuilder(task)
        self.splits = SplitFinder(self.summarizer)
        self.series = SeriesAnalyzer(self.summarizer)
        self.rules = DriftRules()

    def run(self) -> DriftAnalysisResult:
        config = self.config
        comparison_mode = self.windows.comparison

        bucket, buckets = self._build_buckets()
        summaries = [self.summarizer.bucket(item) for item in buckets]

        trend: dict[str, dict[str, Any]] = {}
        outliers: list[dict[str, Any]] = []
        series_ran = len(buckets) >= config.min_buckets_trend
        if series_ran:
            trend = self.series.trends(summaries)
            outliers = self.series.outliers(summaries)

        min_side = self.splits.min_side_records
        change_point = None
        secondary: list[dict[str, Any]] = []
        comparison = None
        if comparison_mode:
            if self.current and self.reference:
                comparison = self.splits.compare(self.reference, self.current, self.windows.start_date)
            insufficient_data = comparison is None or not comparison["sides_sufficient"]
            insufficient_buckets = False
            split = comparison
        else:
            insufficient_buckets = len(buckets) < config.min_buckets_changepoint
            if not insufficient_buckets:
                change_point, secondary = self._change_points(buckets, outliers)
                insufficient_buckets = change_point is None
            insufficient_data = len(self.current) < min_side or (
                change_point is not None and not change_point["sides_sufficient"]
            )
            split = change_point

        hard_breaks = self._hard_breaks(self.records)
        if split is not None:
            hard_breaks.extend(self._elapsed_break(split))

        flags = self.rules.flags(split, trend, outliers, hard_breaks, insufficient_data, insufficient_buckets)

        box_count = sum(record.box_count for record in self.records) if self.summarizer.detection else None
        quality = self.extractor.quality.model_dump()
        quality.update({
            "record_count": len(self.records),
            "box_count": box_count,
            "merged_buckets": self.builder.merged_start_dates
        })
        reference_range = self.windows.reference

        result = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "mode": DriftMode.COMPARISON if comparison_mode else DriftMode.RANGE,
            "filters": self.filters,
            "range": self.windows.current.model_dump(),
            "reference_range": reference_range.model_dump() if reference_range else None,
            "bucket": bucket,
            "detail": self.detail,
            "status": {
                "analysis_possible": (change_point is not None) or (comparison is not None),
                "change_point_ran": change_point is not None,
                "comparison_ran": comparison is not None,
                "trend_ran": series_ran,
                "outlier_ran": series_ran,
                "bucket_count": len(buckets)
            },
            "data_quality": quality,
            "classes": self.summarizer.classes,
            "buckets": summaries if self.detail == DriftDetail.FULL
            else [self.summarizer.compact(item) for item in summaries],
            "change_point": change_point,
            "secondary_change_points": secondary,
            "comparison": comparison,
            "max_pairwise": self._max_pairwise(buckets),
            "outlier_buckets": outliers,
            "trend": trend,
            "hard_breaks": hard_breaks,
            "flags": flags,
            "pre_verdict": self.rules.pre_verdict(flags),
            "config": config.model_dump()
        }

        return DriftAnalysisResult.model_validate(self._round(result, config.decimals))

    def _change_points(
            self,
            buckets: Sequence[Bucket],
            outliers: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        flagged = {item["bucket_index"] for item in outliers}
        excluded = {index for index in flagged if index - 1 not in flagged and index + 1 not in flagged}
        kept = [index for index in range(len(buckets)) if index not in excluded]
        searched = [buckets[index] for index in kept]
        counts = [self.summarizer.side_counts(item.records) for item in searched]

        primary = self.splits.primary(searched, counts)
        if primary is None:
            return None, []
        secondary = self.splits.secondary(searched, counts, primary)
        for report in [primary] + secondary:
            report["bucket_index"] = kept[report["bucket_index"]]

        return primary, secondary

    def _build_buckets(self) -> tuple[str, list[Bucket]]:
        bucket = self.builder.choose(self.records, self.windows.total_days, self.requested_bucket)
        buckets: list[Bucket] = []
        if self.windows.comparison:
            buckets.extend(self.builder.merge_small(self.builder.build(self.reference, bucket, DriftWindowMode.REFERENCE)))
        buckets.extend(self.builder.merge_small(self.builder.build(self.current, bucket, DriftWindowMode.CURRENT)))
        return bucket, buckets

    def _hard_breaks(self, records: Sequence[Record]) -> list[dict[str, Any]]:
        breaks: list[dict[str, Any]] = []
        last: dict[str, Any] = {}

        for record in records:
            observed: list[dict[str, Any]] = [
                {"kind": HardBreakKind.IMAGE_SPEC, "class_name": None, "value": record.image_spec},
                {"kind": HardBreakKind.BACKEND, "class_name": None, "value": record.backend},
                {"kind": HardBreakKind.CLASSES, "class_name": None,
                 "value": list(record.classes) if record.classes else None}
            ]
            if self.summarizer.detection:
                thresholds: dict[str, float] = {}
                for box in record.boxes:
                    if box.threshold is not None and box.prediction not in thresholds:
                        thresholds[box.prediction] = box.threshold
                observed.extend(
                    {"kind": HardBreakKind.THRESHOLD, "class_name": name, "value": value}
                    for name, value in thresholds.items()
                )
            else:
                observed.append({"kind": HardBreakKind.THRESHOLD, "class_name": None, "value": record.threshold})

            for item in observed:
                value = item["value"]
                if value is None:
                    continue
                key = f"{item['kind']}:{item['class_name']}"
                if key in last and last[key] != value:
                    breaks.append({
                        "kind": item["kind"],
                        "class_name": item["class_name"],
                        "date": record.created_at,
                        "inspection_id": record.inspection_id,
                        "from": last[key],
                        "to": value
                    })
                last[key] = value

        return breaks

    def _elapsed_break(self, split: dict[str, Any]) -> list[dict[str, Any]]:
        before = split["before"].get("median_elapsed_time")
        after = split["after"].get("median_elapsed_time")
        if not before or after is None:
            return []

        ratio = after / before
        if ratio > self.config.elapsed_ratio_high or ratio < self.config.elapsed_ratio_low:
            return [{
                "kind": HardBreakKind.ELAPSED_TIME,
                "date": split["date"],
                "inspection_id": None,
                "from": before,
                "to": after
            }]
        return []

    def _max_pairwise(self, buckets: Sequence[Bucket]) -> dict[str, dict[str, Any]]:
        if len(buckets) < 2:
            return {}

        counts = [self.summarizer.side_counts(item.records) for item in buckets]
        class_keys = sorted({key for item in counts for key in item.class_counts})
        distributions = {
            "psi_confidence": [stats.proportions(item.confidence_counts) for item in counts],
            "psi_class": [stats.proportions([item.class_counts.get(key, 0) for key in class_keys]) for item in counts]
        }

        best: dict[str, dict[str, Any]] = {}
        for name, dists in distributions.items():
            best[name] = {"value": 0.0, "pair": [buckets[0].start_date, buckets[1].start_date]}
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    value = stats.psi(dists[i], dists[j], self.config.eps)
                    if value > best[name]["value"]:
                        best[name] = {"value": value, "pair": [buckets[i].start_date, buckets[j].start_date]}

        return best

    @classmethod
    def _round(cls, value: Any, decimals: int) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, decimals)
        if isinstance(value, dict):
            return {key: cls._round(item, decimals) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._round(item, decimals) for item in value]
        return value
