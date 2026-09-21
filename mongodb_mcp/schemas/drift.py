from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from common.constants import (
    DriftTask, DriftMode, DriftWindowMode, DriftFlag, PreVerdict, HardBreakKind, Granularity, MedianSource
)


class DriftModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


# ---------------------------------------------------------------------------------------------------------------------
# Internal additive counts, read from the statistics documents and summed over periods and sides
# ---------------------------------------------------------------------------------------------------------------------

def _add_histograms(a: list[int], b: list[int]) -> list[int]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    return [x + y for x, y in zip(a, b)]


def _add_optional(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _min_optional(a: float | None, b: float | None) -> float | None:
    values = [value for value in (a, b) if value is not None]
    return min(values) if values else None


def _max_optional(a: float | None, b: float | None) -> float | None:
    values = [value for value in (a, b) if value is not None]
    return max(values) if values else None


def _union(a: list[Any], b: list[Any]) -> list[Any]:
    merged = list(a)
    for item in b:
        if item not in merged:
            merged.append(item)
    return merged


class ConfidenceCounts(BaseModel):
    """Additive confidence statistics of one group of predictions (a period, a class, a side)"""
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float | None = None
    max: float | None = None
    histogram: list[int] = []
    below_threshold_count: int | None = None
    near_threshold_count: int | None = None
    thresholds: list[float] = []
    quantiles: dict[str, float | None] | None = None   # kept only while the group is a single document

    def __add__(self, other: "ConfidenceCounts") -> "ConfidenceCounts":
        return ConfidenceCounts(
            count=self.count + other.count,
            sum=self.sum + other.sum,
            sum_sq=self.sum_sq + other.sum_sq,
            min=_min_optional(self.min, other.min),
            max=_max_optional(self.max, other.max),
            histogram=_add_histograms(self.histogram, other.histogram),
            below_threshold_count=_add_optional(self.below_threshold_count, other.below_threshold_count),
            near_threshold_count=_add_optional(self.near_threshold_count, other.near_threshold_count),
            thresholds=_union(self.thresholds, other.thresholds),
            quantiles=None
        )


class PeriodCounts(BaseModel):
    """Every additive number of one period, as the aggregation pipeline returns it; sums give sides"""
    start_date: datetime
    end_date: datetime
    window: DriftWindowMode = DriftWindowMode.CURRENT
    tasks: list[str] = []
    document_count: int = 0
    missing_block_count: int = 0
    product_ids: list[str] = []
    gbms: list[str] = []
    processes: list[str] = []
    modes: list[str] = []
    equipment_ids: list[str] = []
    classes: list[str] = []
    bins: list[dict[str, Any]] = []
    inspection_count: int = 0
    prediction_count: int = 0
    box_count: int | None = None
    missing_confidence_count: int = 0
    parse_error_count: int = 0
    elapsed_count: int = 0
    elapsed_sum: float = 0.0
    no_box_count: int | None = None
    boxes_per_image_sum: int = 0
    boxes_per_image_sum_sq: int = 0
    boxes_per_image_histogram: list[int] = []
    backends: list[str] = []
    class_counts: dict[str, int] = {}
    confidence: ConfidenceCounts = ConfidenceCounts()
    per_class: dict[str, ConfidenceCounts] = {}

    def __add__(self, other: "PeriodCounts") -> "PeriodCounts":
        per_class = dict(self.per_class)
        for name, counts in other.per_class.items():
            per_class[name] = per_class[name] + counts if name in per_class else counts
        return PeriodCounts(
            start_date=min(self.start_date, other.start_date),
            end_date=max(self.end_date, other.end_date),
            window=self.window,
            tasks=_union(self.tasks, other.tasks),
            document_count=self.document_count + other.document_count,
            missing_block_count=self.missing_block_count + other.missing_block_count,
            product_ids=_union(self.product_ids, other.product_ids),
            gbms=_union(self.gbms, other.gbms),
            processes=_union(self.processes, other.processes),
            modes=_union(self.modes, other.modes),
            equipment_ids=_union(self.equipment_ids, other.equipment_ids),
            classes=_union(self.classes, other.classes),
            bins=_union(self.bins, other.bins),
            inspection_count=self.inspection_count + other.inspection_count,
            prediction_count=self.prediction_count + other.prediction_count,
            box_count=_add_optional(self.box_count, other.box_count),
            missing_confidence_count=self.missing_confidence_count + other.missing_confidence_count,
            parse_error_count=self.parse_error_count + other.parse_error_count,
            elapsed_count=self.elapsed_count + other.elapsed_count,
            elapsed_sum=self.elapsed_sum + other.elapsed_sum,
            no_box_count=_add_optional(self.no_box_count, other.no_box_count),
            boxes_per_image_sum=self.boxes_per_image_sum + other.boxes_per_image_sum,
            boxes_per_image_sum_sq=self.boxes_per_image_sum_sq + other.boxes_per_image_sum_sq,
            boxes_per_image_histogram=_add_histograms(self.boxes_per_image_histogram, other.boxes_per_image_histogram),
            backends=_union(self.backends, other.backends),
            class_counts={
                key: self.class_counts.get(key, 0) + other.class_counts.get(key, 0)
                for key in _union(list(self.class_counts), list(other.class_counts))
            },
            confidence=self.confidence + other.confidence,
            per_class=per_class
        )

    @property
    def confidence_edges(self) -> list[float] | None:
        for item in self.bins:
            edges = item.get("confidenceEdges")
            if edges:
                return list(edges)
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------------------------------------------------

class DateRange(DriftModel):
    start_date: datetime
    end_date: datetime
    days: float


class DriftFilters(DriftModel):
    gbm: str | None = None
    process: str | None = None
    equipment_id: str | None = None
    product_id: str | None = None


class SitesSeen(DriftModel):
    """Distinct values summed into the analysis when the matching filter was null"""
    gbms: list[str] = []
    processes: list[str] = []
    modes: list[str] = []
    equipment_ids: list[str] = []


class DriftBins(DriftModel):
    """Bin definitions as read from the statistics documents"""
    confidence_edges: list[float] = []
    near_threshold_margin: float | None = None
    boxes_per_image_max: int | None = None


class AnalysisStatus(DriftModel):
    analysis_possible: bool
    split_ran: bool = False
    comparison_ran: bool = False
    period_count: int = 0
    reference_period_count: int = 0


class DataQuality(DriftModel):
    document_count: int = 0
    product_count: int = 0
    inspection_count: int = 0
    prediction_count: int = 0
    box_count: int | None = None
    missing_confidence_count: int = 0
    parse_error_count: int = 0
    partial_periods: list[datetime] = []
    periods_missing_equipment: list[datetime] = []
    incompatible_bins: bool = False


class ConfidenceQuantiles(DriftModel):
    p05: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None


class ClassSummary(DriftModel):
    count: int = 0
    share: float = 0.0
    mean_confidence: float | None = None
    std_confidence: float | None = None
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    threshold: float | None = None
    confidence_histogram: list[float] | None = None


class CompactPeriodSummary(DriftModel):
    start_date: datetime
    end_date: datetime
    window: DriftWindowMode = DriftWindowMode.CURRENT
    partial: bool = False
    document_count: int = 0
    product_count: int = 0
    inspection_count: int = 0
    prediction_count: int = 0
    box_count: int | None = None
    class_distribution: dict[str, float] = {}
    mean_confidence: float | None = None
    std_confidence: float | None = None
    median_confidence: float | None = None
    median_source: MedianSource | None = None
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    mean_boxes_per_image: float | None = None
    no_box_rate: float | None = None
    mean_elapsed_time: float | None = None
    backends: list[str] = []
    thresholds_by_class: dict[str, list[float]] = {}


class PeriodSummary(CompactPeriodSummary):
    confidence_histogram: list[float] | None = None
    confidence_quantiles: ConfidenceQuantiles | None = None
    boxes_per_image_histogram: list[float] | None = None
    per_class: dict[str, ClassSummary] = {}


class ConsecutiveDivergence(DriftModel):
    period_index: int
    date: datetime
    psi_confidence: float | None = None
    psi_class: float | None = None
    psi_boxes_per_image: float | None = None
    sufficient: bool = False


class Chi2Result(DriftModel):
    stat: float
    p: float
    dof: int


class SideSummary(DriftModel):
    start_date: datetime
    end_date: datetime
    period_count: int
    document_count: int = 0
    prediction_count: int = 0
    box_count: int | None = None
    class_distribution: dict[str, float] = {}
    mean_confidence: float | None = None
    median_confidence: float | None = None
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    mean_boxes_per_image: float | None = None
    no_box_rate: float | None = None
    mean_elapsed_time: float | None = None
    confidence_histogram: list[float] | None = None
    boxes_per_image_histogram: list[float] | None = None
    per_class: dict[str, ClassSummary] = {}


class SplitResult(DriftModel):
    period_index: int | None = None
    date: datetime
    candidate_count: int = 1
    score: float = 0.0
    before_count: int
    after_count: int
    sides_sufficient: bool
    psi_confidence: float | None = None
    js_confidence: float | None = None
    psi_confidence_by_class: dict[str, float | None] = {}
    psi_class: float | None = None
    chi2_class: Chi2Result | None = None
    max_class_proportion_change: float | None = None
    psi_boxes_per_image: float | None = None
    mean_confidence_delta: float | None = None
    below_threshold_rate_delta: float | None = None
    elapsed_time_ratio: float | None = None
    before: SideSummary
    after: SideSummary


class HardBreak(DriftModel):
    kind: HardBreakKind
    class_name: str | None = None
    date: datetime
    from_value: Any = Field(None, alias="from")
    to_value: Any = Field(None, alias="to")


class DriftAnalysisResult(DriftModel):
    model_name: str
    model_version: str
    task: DriftTask
    mode: str | None = None
    analysis_mode: DriftMode
    filters: DriftFilters = DriftFilters()
    sites_seen: SitesSeen = SitesSeen()
    granularity: Granularity
    detail: str
    range: DateRange
    reference_range: DateRange | None = None
    status: AnalysisStatus
    data_quality: DataQuality
    classes: list[str] = []
    bins: DriftBins = DriftBins()
    periods: list[PeriodSummary] = []
    consecutive: list[ConsecutiveDivergence] = []
    change_point: SplitResult | None = None
    comparison: SplitResult | None = None
    hard_breaks: list[HardBreak] = []
    flags: list[DriftFlag] = []
    pre_verdict: PreVerdict
    config: dict[str, Any] = {}
