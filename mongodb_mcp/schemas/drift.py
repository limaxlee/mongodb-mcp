from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class DriftModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ConfQuantiles(DriftModel):
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
    n_before: int
    n_after: int


class Chi2Result(DriftModel):
    stat: float
    p: float
    dof: int


class BoxSummary(DriftModel):
    n_boxes: int
    area_norm_quantiles: Quantiles = Quantiles()
    w_norm_quantiles: Quantiles = Quantiles()
    h_norm_quantiles: Quantiles = Quantiles()
    aspect_quantiles: Quantiles = Quantiles()
    cx_norm_quantiles: Quantiles = Quantiles()
    cy_norm_quantiles: Quantiles = Quantiles()
    area_norm_hist: list[float] | None = None


class BucketSummary(DriftModel):
    bucket_start: datetime
    bucket_end: datetime
    window: Literal["current", "reference"] = "current"
    n: int
    merged_from: int = 1
    class_dist: dict[str, float] = {}
    defect_rate: float | None = None
    conf_p50: float | None = None
    conf_mean: float | None = None
    below_threshold_rate: float | None = None
    n_feedback: int = 0
    feedback_mismatch_rate: float | None = None
    n_feedback_other: int = 0
    # Full detail only
    conf_hist: list[float] | None = None
    conf_quantiles: ConfQuantiles | None = None
    conf_std: float | None = None
    threshold_values: list[float] | None = None
    image_specs: list[list[int]] | None = None
    elapsed_time_p50: float | None = None
    # Classification only
    margin_quantiles: Quantiles | None = None
    entropy_quantiles: Quantiles | None = None
    decision_differs_rate: float | None = None
    patch_w_p50: float | None = None
    patch_h_p50: float | None = None
    near_threshold_rate: float | None = None
    # Detection only
    n_boxes: int | None = None
    boxes_per_image_mean: float | None = None
    boxes_per_image_std: float | None = None
    boxes_per_image_hist: list[float] | None = None
    no_box_rate: float | None = None
    boxes_by_class_per_image: dict[str, float] | None = None
    box: BoxSummary | None = None


class SideSummary(DriftModel):
    n: int
    n_boxes: int | None = None
    class_dist: dict[str, float] = {}
    conf_hist: list[float] = []
    conf_quantiles: ConfQuantiles = ConfQuantiles()
    below_threshold_rate: float | None = None
    boxes_per_image_hist: list[float] | None = None
    n_feedback: int = 0
    feedback_mismatch_rate: float | None = None
    elapsed_time_p50: float | None = None


class SplitComparison(DriftModel):
    bucket_index: int | None = None
    date: datetime
    score: float
    n_before: int
    n_after: int
    sides_sufficient: bool
    psi_conf: float
    js_conf: float
    ks_conf: KsResult
    psi_class: float
    chi2_class: Chi2Result
    max_class_prop_change: float
    psi_boxes_per_image: float | None = None
    ks_area_norm: KsResult | None = None
    ks_cx_norm: KsResult | None = None
    ks_cy_norm: KsResult | None = None
    ks_margin: KsResult | None = None
    ks_entropy: KsResult | None = None
    before: SideSummary
    after: SideSummary


class PairwiseMax(DriftModel):
    value: float
    pair: list[datetime]


class OutlierBucket(DriftModel):
    bucket_index: int
    bucket_start: datetime
    series: str
    z: float
    value: float


class TrendStat(DriftModel):
    tau: float
    p: float
    slope_per_bucket: float
    first: float
    last: float
    n_points: int
    meaningful: bool


class HardBreak(DriftModel):
    kind: Literal["image_spec", "threshold", "backend", "classes", "elapsed_time"]
    date: datetime
    inspection_id: str | None = None
    from_value: Any = Field(None, alias="from")
    to_value: Any = Field(None, alias="to")


class DataQuality(DriftModel):
    n_docs_scanned: int = 0
    n_docs_matched: int = 0
    n_entries_matched: int = 0
    n_records: int = 0
    n_boxes: int | None = None
    n_missing_confidence: int = 0
    n_missing_image_spec: int = 0
    n_parse_errors: int = 0
    parse_error_examples: list[str] = []
    merged_buckets: list[datetime] = []


class AnalysisStatus(DriftModel):
    analysis_possible: bool
    change_point_ran: bool = False
    comparison_ran: bool = False
    trend_ran: bool = False
    outlier_ran: bool = False
    n_buckets: int = 0


class DateRange(DriftModel):
    start: datetime
    end: datetime
    days: float


class DriftAnalysisResult(DriftModel):
    model_name: str
    model_version: str
    task: Literal["cls", "det"]
    mode: Literal["range", "comparison"]
    filters: dict[str, str | None] = {}
    range: DateRange
    reference_range: DateRange | None = None
    bucket: Literal["1h", "shift", "1d", "1w"]
    detail: Literal["full", "compact"] = "full"
    status: AnalysisStatus
    data_quality: DataQuality
    classes: list[str] = []
    defect_classes: list[str] = []
    warnings: list[str] = []
    buckets: list[BucketSummary] = []
    change_point: SplitComparison | None = None
    secondary_change_points: list[SplitComparison] = []
    comparison: SplitComparison | None = None
    max_pairwise: dict[str, PairwiseMax] = {}
    outlier_buckets: list[OutlierBucket] = []
    trend: dict[str, TrendStat] = {}
    hard_breaks: list[HardBreak] = []
    flags: list[str] = []
    pre_verdict: Literal["stable", "suspicious", "drift_likely", "undetermined"]
    config: dict[str, Any] = {}
