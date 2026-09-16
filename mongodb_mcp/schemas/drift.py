from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from common.constants import (
    DriftTask, DriftMode, DriftWindowMode, DriftDetail,
    DriftFlag, PreVerdict, HardBreakKind, BucketSize
)


class DriftModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class BoxRecord(BaseModel):
    bbox_id: int | None = None
    prediction: str
    max_confidence: float | None = None
    threshold: float | None = None
    below_threshold: bool | None = None
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float
    aspect: float | None = None
    cx: float
    cy: float
    normalized_width: float | None = None
    normalized_height: float | None = None
    normalized_area: float | None = None
    normalized_cx: float | None = None
    normalized_cy: float | None = None


class Record(BaseModel):
    window: DriftWindowMode = DriftWindowMode.CURRENT
    inspection_id: str
    prediction_id: int | None = None
    created_at: datetime
    gbm: str | None = None
    process: str | None = None
    location: str | None = None
    equipment_id: str | None = None
    product_id: str | None = None
    mode: str | None = None
    backend: str | None = None
    classes: list[str] = []
    threshold: float | None = None
    elapsed_time: float | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_channels: int | None = None
    # Classification
    prediction: str | None = None
    max_confidence: float | None = None
    below_threshold: bool | None = None
    near_threshold: bool | None = None
    # Detection
    boxes: list[BoxRecord] = []
    box_count: int = 0
    box_count_by_class: dict[str, int] = {}
    below_threshold_count: int = 0
    mean_confidence: float | None = None
    min_confidence: float | None = None

    @property
    def has_no_boxes(self) -> bool:
        return self.box_count == 0

    @property
    def image_spec(self) -> list[int] | None:
        if self.image_width is None or self.image_height is None:
            return None
        return [self.image_width, self.image_height, self.image_channels if self.image_channels is not None else 0]


class Bucket(BaseModel):
    start_date: datetime
    end_date: datetime
    window: DriftWindowMode = DriftWindowMode.CURRENT
    records: list[Record] = []
    merged_from: int = 1

    @property
    def record_count(self) -> int:
        return len(self.records)


class SideCounts(BaseModel):
    record_count: int = 0
    box_count: int = 0
    confidence_counts: list[int] = []
    class_counts: dict[str, int] = {}
    boxes_per_image_counts: list[int] = []

    def __add__(self, other: "SideCounts") -> "SideCounts":
        return SideCounts(
            record_count=self.record_count + other.record_count,
            box_count=self.box_count + other.box_count,
            confidence_counts=self._combine(self.confidence_counts, other.confidence_counts, 1),
            class_counts={
                key: self.class_counts.get(key, 0) + other.class_counts.get(key, 0)
                for key in set(self.class_counts) | set(other.class_counts)
            },
            boxes_per_image_counts=self._combine(self.boxes_per_image_counts, other.boxes_per_image_counts, 1)
        )

    def __sub__(self, other: "SideCounts") -> "SideCounts":
        return SideCounts(
            record_count=self.record_count - other.record_count,
            box_count=self.box_count - other.box_count,
            confidence_counts=self._combine(self.confidence_counts, other.confidence_counts, -1),
            class_counts={
                key: self.class_counts.get(key, 0) - other.class_counts.get(key, 0)
                for key in set(self.class_counts) | set(other.class_counts)
            },
            boxes_per_image_counts=self._combine(self.boxes_per_image_counts, other.boxes_per_image_counts, -1)
        )

    @staticmethod
    def _combine(a: list[int], b: list[int], sign: int) -> list[int]:
        if not a:
            return [sign * value for value in b]
        if not b:
            return list(a)
        return [x + sign * y for x, y in zip(a, b)]


class DateRange(DriftModel):
    start_date: datetime
    end_date: datetime
    days: float


class ConfidenceQuantiles(DriftModel):
    p05: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None


class Quantiles(DriftModel):
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


class KsResult(DriftModel):
    d: float
    p: float
    before_count: int
    after_count: int


class Chi2Result(DriftModel):
    stat: float
    p: float
    dof: int


class BoxSummary(DriftModel):
    box_count: int
    normalized_area_quantiles: Quantiles = Quantiles()
    normalized_width_quantiles: Quantiles = Quantiles()
    normalized_height_quantiles: Quantiles = Quantiles()
    aspect_quantiles: Quantiles = Quantiles()
    normalized_cx_quantiles: Quantiles = Quantiles()
    normalized_cy_quantiles: Quantiles = Quantiles()
    normalized_area_histogram: list[float] | None = None


class CompactBucketSummary(DriftModel):
    start_date: datetime
    end_date: datetime
    window: DriftWindowMode = DriftWindowMode.CURRENT
    record_count: int
    merged_from: int = 1
    class_distribution: dict[str, float] = {}
    median_confidence: float | None = None
    mean_confidence: float | None = None
    below_threshold_rate: float | None = None
    # Classification only
    near_threshold_rate: float | None = None
    # Detection only
    mean_boxes_per_image: float | None = None
    no_box_rate: float | None = None


class BucketSummary(CompactBucketSummary):
    confidence_histogram: list[float] | None = None
    confidence_quantiles: ConfidenceQuantiles | None = None
    std_confidence: float | None = None
    threshold_values: list[float] | None = None
    image_specs: list[list[int]] | None = None
    median_elapsed_time: float | None = None
    # Detection only
    box_count: int | None = None
    std_boxes_per_image: float | None = None
    boxes_per_image_histogram: list[float] | None = None
    boxes_by_class_per_image: dict[str, float] | None = None
    box: BoxSummary | None = None


class SideSummary(DriftModel):
    record_count: int
    box_count: int | None = None
    class_distribution: dict[str, float] = {}
    confidence_histogram: list[float] = []
    confidence_quantiles: ConfidenceQuantiles = ConfidenceQuantiles()
    below_threshold_rate: float | None = None
    boxes_per_image_histogram: list[float] | None = None
    median_elapsed_time: float | None = None


class SplitComparison(DriftModel):
    bucket_index: int | None = None
    date: datetime
    candidate_count: int = 1
    score: float
    before_count: int
    after_count: int
    sides_sufficient: bool
    psi_confidence: float
    js_confidence: float
    ks_confidence: KsResult
    psi_class: float
    chi2_class: Chi2Result
    max_class_proportion_change: float
    psi_boxes_per_image: float | None = None
    ks_normalized_area: KsResult | None = None
    ks_normalized_cx: KsResult | None = None
    ks_normalized_cy: KsResult | None = None
    before: SideSummary
    after: SideSummary


class PairwiseMax(DriftModel):
    value: float
    pair: list[datetime]


class OutlierBucket(DriftModel):
    bucket_index: int
    start_date: datetime
    series: str
    z: float
    value: float


class TrendStat(DriftModel):
    tau: float
    p: float
    slope_per_bucket: float
    first: float
    last: float
    point_count: int
    meaningful: bool


class HardBreak(DriftModel):
    kind: HardBreakKind
    class_name: str | None = None
    date: datetime
    inspection_id: str | None = None
    from_value: Any = Field(None, alias="from")
    to_value: Any = Field(None, alias="to")


class DataQuality(DriftModel):
    scanned_document_count: int = 0
    matched_document_count: int = 0
    matched_entry_count: int = 0
    record_count: int = 0
    box_count: int | None = None
    missing_confidence_count: int = 0
    missing_image_spec_count: int = 0
    parse_error_count: int = 0
    parse_error_examples: list[str] = []
    merged_buckets: list[datetime] = []


class AnalysisStatus(DriftModel):
    analysis_possible: bool
    change_point_ran: bool = False
    comparison_ran: bool = False
    trend_ran: bool = False
    outlier_ran: bool = False
    bucket_count: int = 0


class DriftAnalysisResult(DriftModel):
    model_name: str
    model_version: str
    task: DriftTask
    mode: DriftMode
    filters: dict[str, str | None] = {}
    range: DateRange
    reference_range: DateRange | None = None
    bucket: BucketSize
    detail: DriftDetail = DriftDetail.FULL
    status: AnalysisStatus
    data_quality: DataQuality
    classes: list[str] = []
    buckets: list[BucketSummary] = []
    change_point: SplitComparison | None = None
    secondary_change_points: list[SplitComparison] = []
    comparison: SplitComparison | None = None
    max_pairwise: dict[str, PairwiseMax] = {}
    outlier_buckets: list[OutlierBucket] = []
    trend: dict[str, TrendStat] = {}
    hard_breaks: list[HardBreak] = []
    flags: list[DriftFlag] = []
    pre_verdict: PreVerdict
    config: dict[str, Any] = {}
