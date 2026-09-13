from typing import Any
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DriftConfig:
    """Every tunable of the drift analysis, echoed back in the result for reproducibility"""

    # Range limits
    max_total_days: int = 30
    max_buckets: int = 60

    # Minimum records per bucket, used by the automatic bucket choice and for merging small buckets
    min_bucket_records_cls: int = 200
    min_bucket_records_det: int = 100
    small_bucket_share: float = 0.3

    # Minimum records on each side of a split before divergence measures are trusted
    min_side_records_cls: int = 500
    min_side_records_det: int = 300
    min_side_boxes: int = 300

    # Minimum number of buckets for each kind of analysis
    min_buckets_changepoint: int = 3
    min_buckets_trend: int = 6
    min_segment_buckets: int = 2

    # Bucket boundaries
    shift_hours: tuple[int, ...] = (6, 14, 22)

    # Fixed histogram edges, mandatory so that histograms are comparable across buckets
    conf_bin_edges: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    area_log_bin_edges: tuple[float, ...] = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0)
    boxes_per_image_max_bin: int = 5

    # Divergence thresholds
    eps: float = 1e-4
    psi_moderate: float = 0.10
    psi_significant: float = 0.25
    ks_d_geometry: float = 0.15
    ks_p_geometry: float = 0.01
    chi2_p: float = 0.001
    class_prop_change: float = 0.05
    threshold_pressure_ratio: float = 2.0
    threshold_pressure_abs: float = 0.02

    # Trend thresholds
    trend_p: float = 0.05
    trend_tau: float = 0.5
    trend_value_change: float = 0.03
    trend_rate_abs: float = 0.02
    trend_rate_rel: float = 0.5
    trend_count_rel: float = 0.2

    # Outlier and feedback thresholds
    robust_z: float = 3.5
    feedback_min: int = 30
    feedback_ratio: float = 1.5

    # Misc
    near_threshold_margin: float = 0.05
    elapsed_ratio_high: float = 1.5
    elapsed_ratio_low: float = 0.67
    decimals: int = 4
    max_parse_error_examples: int = 10
    good_class: str = "Good"

    def as_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


DEFAULT_CONFIG = DriftConfig()
