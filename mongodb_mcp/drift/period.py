"""Periods: the pipeline rows parsed into additive counts, and the summaries built from those counts"""
import math
from typing import Any, Sequence
from datetime import datetime
from functools import reduce
from operator import add

from common.constants import DriftDetail, DriftTask, DriftWindowMode, MedianSource
from mongodb_mcp.utils import stats, to_utc
from mongodb_mcp.schemas import (
    ConfidenceCounts, PeriodCounts, ClassSummary, PeriodSummary, SideSummary, ConfidenceQuantiles
)
from mongodb_mcp.drift.query import ENTRY_TOTAL, ENTRY_CLASS, ENTRY_COUNT


def _clean(values: Any) -> list[Any]:
    """Distinct non-null values in first-seen order"""
    result: list[Any] = []
    for value in values or []:
        if value is not None and value not in result:
            result.append(value)
    return result


def _flatten(sets: Any) -> list[Any]:
    """Union of a list of lists, in first-seen order"""
    result: list[Any] = []
    for values in sets or []:
        if isinstance(values, list):
            for value in values:
                if value is not None and value not in result:
                    result.append(value)
    return result


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _confidence_counts(row: dict[str, Any]) -> ConfidenceCounts:
    quantiles = row.get("quantiles") if _int(row.get("documentCount")) == 1 else None
    return ConfidenceCounts(
        count=_int(row.get("count")),
        sum=_float(row.get("sum")),
        sum_sq=_float(row.get("sumSq")),
        min=_optional_float(row.get("min")),
        max=_optional_float(row.get("max")),
        histogram=[_int(value) for value in row.get("histogram") or []],
        below_threshold_count=_int(row.get("belowThresholdCount")) if row.get("belowPresent") else None,
        near_threshold_count=_int(row.get("nearThresholdCount")) if row.get("nearPresent") else None,
        thresholds=sorted(value for value in _clean(row.get("entryThresholds")) if isinstance(value, (int, float))),
        quantiles=quantiles if isinstance(quantiles, dict) else None
    )


def build_periods(rows: Sequence[dict[str, Any]], window: DriftWindowMode) -> list[PeriodCounts]:
    """Groups the pipeline rows of one window by period start, in chronological order"""
    periods: dict[datetime, PeriodCounts] = {}

    for row in rows:
        key = row.get("_id") or {}
        date = to_utc(key.get("date"))
        if date is None:
            continue

        period = periods.get(date)
        if period is None:
            period = PeriodCounts(start_date=date, end_date=to_utc(row.get("endDate")) or date, window=window)
            periods[date] = period

        kind, name = key.get("kind"), key.get("k")
        if kind == ENTRY_TOTAL:
            _fill_period(period, row)
        elif kind == ENTRY_CLASS and name is not None:
            period.per_class[str(name)] = _confidence_counts(row)
        elif kind == ENTRY_COUNT and name is not None:
            period.class_counts[str(name)] = _int(row.get("classCount"))

    return [periods[date] for date in sorted(periods)]


def _fill_period(period: PeriodCounts, row: dict[str, Any]) -> None:
    period.end_date = to_utc(row.get("endDate")) or period.end_date
    period.tasks = [str(task) for task in _clean(row.get("tasks"))]
    period.document_count = _int(row.get("documentCount"))
    period.missing_block_count = _int(row.get("missingBlockCount"))
    period.product_ids = [str(item) for item in _clean(row.get("productIds"))]
    period.gbms = [str(item) for item in _clean(row.get("gbms"))]
    period.processes = [str(item) for item in _clean(row.get("processes"))]
    period.modes = [str(item) for item in _clean(row.get("modes"))]
    period.equipment_ids = [str(item) for item in _flatten(row.get("equipmentIdSets"))]
    period.classes = [str(item) for item in _flatten(row.get("classSets"))]
    period.bins = [item for item in row.get("bins") or [] if isinstance(item, dict)]
    period.inspection_count = _int(row.get("inspectionCount"))
    period.prediction_count = _int(row.get("predictionCount"))
    period.box_count = _int(row.get("boxCount")) if row.get("boxCountPresent") else None
    period.missing_confidence_count = _int(row.get("missingConfidenceCount"))
    period.parse_error_count = _int(row.get("parseErrorCount"))
    period.elapsed_count = _int(row.get("elapsedCount"))
    period.elapsed_sum = _float(row.get("elapsedSum"))
    period.no_box_count = _int(row.get("noBoxCount")) if row.get("imagesPresent") else None
    period.boxes_per_image_sum = _int(row.get("boxesPerImageSum"))
    period.boxes_per_image_sum_sq = _int(row.get("boxesPerImageSumSq"))
    period.boxes_per_image_histogram = [_int(value) for value in row.get("boxesPerImageHistogram") or []]
    period.backends = sorted(str(item) for item in _flatten(row.get("backendSets")))
    period.thresholds = sorted(
        value for value in _flatten(row.get("thresholdSets")) if isinstance(value, (int, float))
    )
    period.confidence = _confidence_counts(row)


def sum_periods(periods: Sequence[PeriodCounts]) -> PeriodCounts:
    return reduce(add, periods)


class PeriodSummarizer:
    """Turns additive counts into the proportions, means, rates and medians of the result"""

    def __init__(
            self,
            task: str,
            classes: Sequence[str],
            edges: Sequence[float] | None,
            detail: str = DriftDetail.FULL
    ):
        self.task = task
        self.detection = task == DriftTask.DETECTION
        self.classes = list(classes)
        self.edges = list(edges) if edges else None
        self.detail = DriftDetail(detail)

    @property
    def full(self) -> bool:
        return self.detail == DriftDetail.FULL

    def class_keys(self, class_counts: dict[str, int]) -> list[str]:
        keys = list(self.classes)
        keys.extend(key for key in class_counts if key not in keys)
        return keys

    def class_distribution(self, class_counts: dict[str, int]) -> dict[str, float]:
        total = sum(class_counts.values())
        return {
            key: class_counts.get(key, 0) / total if total > 0 else 0.0
            for key in self.class_keys(class_counts)
        }

    @staticmethod
    def mean(counts: ConfidenceCounts) -> float | None:
        return counts.sum / counts.count if counts.count > 0 else None

    @staticmethod
    def std(counts: ConfidenceCounts) -> float | None:
        if counts.count <= 0:
            return None
        mean = counts.sum / counts.count
        return math.sqrt(max(counts.sum_sq / counts.count - mean * mean, 0.0))

    def median(self, counts: ConfidenceCounts) -> tuple[float | None, MedianSource | None]:
        if counts.quantiles and counts.quantiles.get("p50") is not None:
            return float(counts.quantiles["p50"]), MedianSource.EXACT
        if self.edges and counts.histogram and sum(counts.histogram) > 0 \
                and len(counts.histogram) == len(self.edges) - 1:
            return stats.histogram_quantile(counts.histogram, self.edges, 0.5), MedianSource.HISTOGRAM
        return None, None

    @staticmethod
    def rate(numerator: int | None, count: int) -> float | None:
        if numerator is None or count <= 0:
            return None
        return numerator / count

    @staticmethod
    def histogram(counts: Sequence[int]) -> list[float] | None:
        return stats.proportions(counts) if counts and sum(counts) > 0 else None

    @staticmethod
    def quantiles(counts: ConfidenceCounts) -> ConfidenceQuantiles | None:
        if not counts.quantiles:
            return None
        return ConfidenceQuantiles(**{
            key: counts.quantiles.get(key) for key in ConfidenceQuantiles.model_fields
        })

    def class_summary(self, name: str, period: PeriodCounts, distribution: dict[str, float]) -> ClassSummary:
        counts = period.per_class.get(name, ConfidenceCounts())
        return ClassSummary(
            count=period.class_counts.get(name, counts.count),
            share=distribution.get(name, 0.0),
            mean_confidence=self.mean(counts),
            std_confidence=self.std(counts),
            below_threshold_rate=self.rate(counts.below_threshold_count, counts.count),
            near_threshold_rate=self.rate(counts.near_threshold_count, counts.count),
            threshold=counts.thresholds[0] if len(counts.thresholds) == 1 else None,
            confidence_histogram=self.histogram(counts.histogram) if self.full else None
        )

    def per_class(self, period: PeriodCounts, distribution: dict[str, float]) -> dict[str, ClassSummary]:
        keys = self.class_keys(period.class_counts)
        keys.extend(key for key in period.per_class if key not in keys)
        return {name: self.class_summary(name, period, distribution) for name in keys}

    def boxes_per_image(self, period: PeriodCounts) -> tuple[float | None, float | None]:
        if not self.detection or period.prediction_count <= 0 or period.no_box_count is None:
            return None, None
        return period.boxes_per_image_sum / period.prediction_count, period.no_box_count / period.prediction_count

    def period(self, period: PeriodCounts, now: datetime) -> PeriodSummary:
        confidence = period.confidence
        distribution = self.class_distribution(period.class_counts)
        median, source = self.median(confidence)
        mean_boxes, no_box_rate = self.boxes_per_image(period)

        return PeriodSummary(
            start_date=period.start_date,
            end_date=period.end_date,
            window=period.window,
            partial=period.end_date > now,
            document_count=period.document_count,
            product_count=len(period.product_ids),
            inspection_count=period.inspection_count,
            prediction_count=period.prediction_count,
            box_count=period.box_count if self.detection else None,
            class_distribution=distribution,
            mean_confidence=self.mean(confidence),
            std_confidence=self.std(confidence),
            median_confidence=median,
            median_source=source,
            below_threshold_rate=self.rate(confidence.below_threshold_count, confidence.count),
            near_threshold_rate=self.rate(confidence.near_threshold_count, confidence.count),
            mean_boxes_per_image=mean_boxes,
            no_box_rate=no_box_rate,
            mean_elapsed_time=period.elapsed_sum / period.elapsed_count if period.elapsed_count > 0 else None,
            backends=list(period.backends),
            thresholds=list(period.thresholds),
            thresholds_by_class={
                name: list(counts.thresholds) for name, counts in period.per_class.items() if counts.thresholds
            },
            confidence_histogram=self.histogram(confidence.histogram) if self.full else None,
            confidence_quantiles=self.quantiles(confidence) if self.full else None,
            boxes_per_image_histogram=self.histogram(period.boxes_per_image_histogram)
            if self.full and self.detection else None,
            per_class=self.per_class(period, distribution) if self.full else {}
        )

    def side(self, counts: PeriodCounts, period_count: int) -> SideSummary:
        confidence = counts.confidence
        distribution = self.class_distribution(counts.class_counts)
        median, _ = self.median(confidence)
        mean_boxes, no_box_rate = self.boxes_per_image(counts)

        return SideSummary(
            start_date=counts.start_date,
            end_date=counts.end_date,
            period_count=period_count,
            document_count=counts.document_count,
            prediction_count=counts.prediction_count,
            box_count=counts.box_count if self.detection else None,
            class_distribution=distribution,
            mean_confidence=self.mean(confidence),
            median_confidence=median,
            below_threshold_rate=self.rate(confidence.below_threshold_count, confidence.count),
            near_threshold_rate=self.rate(confidence.near_threshold_count, confidence.count),
            mean_boxes_per_image=mean_boxes,
            no_box_rate=no_box_rate,
            mean_elapsed_time=counts.elapsed_sum / counts.elapsed_count if counts.elapsed_count > 0 else None,
            confidence_histogram=self.histogram(confidence.histogram),
            boxes_per_image_histogram=self.histogram(counts.boxes_per_image_histogram) if self.detection else None,
            per_class=self.per_class(counts, distribution)
        )
