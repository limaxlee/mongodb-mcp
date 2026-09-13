"""Orchestrates the drift analysis over extracted records and assembles the result"""
from typing import Any, Sequence
from datetime import datetime, timezone

from common.constants import ModelTasks
from mongodb_mcp.drift import stats
from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG
from mongodb_mcp.drift.extract import Record, RecordExtractor
from mongodb_mcp.drift.bucketing import Bucket, build_buckets, choose_bucket, merge_small_buckets, min_bucket_records
from mongodb_mcp.drift.summarize import summarize_bucket, side_counts, series_value, compact_summary, is_detection
from mongodb_mcp.drift.changepoint import find_change_points, compare_sides, min_side_records
from mongodb_mcp.drift.flags import compute_flags, pre_verdict
from mongodb_mcp.schemas import DriftAnalysisResult

OUTLIER_SERIES = ("conf_p50", "below_threshold_rate", "defect_rate", "boxes_per_image_mean", "no_box_rate")
TREND_SERIES = OUTLIER_SERIES + (
    "conf_mean", "margin_p50", "entropy_p50", "area_norm_p50", "cx_norm_p50", "cy_norm_p50"
)
RATE_SERIES = {"below_threshold_rate", "defect_rate", "no_box_rate"}
COUNT_SERIES = {"boxes_per_image_mean"}


def analyze_records(
        model_name: str,
        model_version: str,
        task: str,
        extractor: RecordExtractor,
        start_date: datetime,
        end_date: datetime,
        filters: dict[str, str | None],
        reference_extractor: RecordExtractor | None = None,
        reference_start: datetime | None = None,
        reference_end: datetime | None = None,
        bucket: str = "auto",
        detail: str = "full",
        defect_classes: Sequence[str] | None = None,
        config: DriftConfig = DEFAULT_CONFIG
) -> DriftAnalysisResult:
    comparison_mode = reference_extractor is not None
    records = sorted(extractor.records, key=lambda item: item.created_at)
    reference_records = sorted(reference_extractor.records, key=lambda item: item.created_at) \
        if reference_extractor else []

    warnings = list(extractor.warnings)
    if reference_extractor:
        warnings.extend(message for message in reference_extractor.warnings if message not in warnings)

    classes = list(extractor.classes_seen)
    if reference_extractor:
        classes.extend(name for name in reference_extractor.classes_seen if name not in classes)
    if defect_classes is None:
        defect_classes = [name for name in classes if name != config.good_class]

    timezones = extractor.timezones_seen | (reference_extractor.timezones_seen if reference_extractor else set())
    if len(timezones) > 1:
        warnings.append(f"Records span several local timezones {sorted(timezones)}, buckets use each record's own")

    days = _days(start_date, end_date)
    reference_days = _days(reference_start, reference_end) if comparison_mode else 0.0

    # Bucketing
    all_records = reference_records + records
    resolved_bucket, bucket_warnings = choose_bucket(all_records, days + reference_days, task, bucket, config)
    warnings.extend(bucket_warnings)
    min_n = min_bucket_records(task, config)

    merged_starts: list[datetime] = []
    buckets: list[Bucket] = []
    if comparison_mode:
        reference_buckets, merged = merge_small_buckets(
            build_buckets(reference_records, resolved_bucket, "reference", config), min_n
        )
        merged_starts.extend(merged)
        buckets.extend(reference_buckets)
    current_buckets, merged = merge_small_buckets(build_buckets(records, resolved_bucket, "current", config), min_n)
    merged_starts.extend(merged)
    buckets.extend(current_buckets)

    summaries = [summarize_bucket(item, task, classes, defect_classes, config) for item in buckets]

    # Split analysis
    min_side = min_side_records(task, config)
    change_point = None
    secondary: list[dict[str, Any]] = []
    comparison = None
    if comparison_mode:
        insufficient_data = len(records) < min_side or len(reference_records) < min_side
        insufficient_buckets = False
        if records and reference_records:
            comparison = compare_sides(reference_records, records, task, classes, start_date, config)
        split = comparison
    else:
        insufficient_data = len(records) < min_side
        insufficient_buckets = len(buckets) < config.min_buckets_changepoint
        if not insufficient_buckets:
            change_point, secondary = find_change_points(buckets, task, classes, config)
        split = change_point

    # Series based analysis
    trend: dict[str, dict[str, Any]] = {}
    outliers: list[dict[str, Any]] = []
    series_ran = len(buckets) >= config.min_buckets_trend
    if series_ran:
        trend = _trends(summaries, task, config)
        outliers = _outliers(summaries, task, config)

    hard_breaks = _hard_breaks(all_records)
    if split is not None:
        hard_breaks.extend(_elapsed_break(split, config))

    flags = compute_flags(split, trend, outliers, hard_breaks, insufficient_data, insufficient_buckets, config)
    n_boxes = sum(record.n_boxes for record in all_records) if is_detection(task) else None

    quality = extractor.quality.as_dict(len(all_records), n_boxes)
    if reference_extractor:
        reference_quality = reference_extractor.quality.as_dict(0, None)
        for key in ("n_docs_scanned", "n_docs_matched", "n_entries_matched", "n_missing_confidence",
                    "n_missing_image_spec", "n_parse_errors"):
            quality[key] += reference_quality[key]
        quality["parse_error_examples"] = (
            quality["parse_error_examples"] + reference_quality["parse_error_examples"]
        )[:config.max_parse_error_examples]
    quality["merged_buckets"] = merged_starts

    result = {
        "model_name": model_name,
        "model_version": model_version,
        "task": task,
        "mode": "comparison" if comparison_mode else "range",
        "filters": filters,
        "range": {"start": start_date, "end": end_date, "days": days},
        "reference_range": {"start": reference_start, "end": reference_end, "days": reference_days}
        if comparison_mode else None,
        "bucket": resolved_bucket,
        "detail": detail,
        "status": {
            "analysis_possible": (change_point is not None) or (comparison is not None),
            "change_point_ran": change_point is not None,
            "comparison_ran": comparison is not None,
            "trend_ran": series_ran,
            "outlier_ran": series_ran,
            "n_buckets": len(buckets)
        },
        "data_quality": quality,
        "classes": classes,
        "defect_classes": list(defect_classes),
        "warnings": warnings,
        "buckets": summaries if detail == "full" else [compact_summary(item) for item in summaries],
        "change_point": change_point,
        "secondary_change_points": secondary,
        "comparison": comparison,
        "max_pairwise": _max_pairwise(buckets, task, classes, config),
        "outlier_buckets": outliers,
        "trend": trend,
        "hard_breaks": hard_breaks,
        "flags": flags,
        "pre_verdict": pre_verdict(flags),
        "config": config.as_dict()
    }

    return DriftAnalysisResult.model_validate(_round(result, config.decimals))


def _days(start: datetime | None, end: datetime | None) -> float:
    if start is None or end is None:
        return 0.0
    return (end - start).total_seconds() / 86400


def _trends(summaries: list[dict[str, Any]], task: str, config: DriftConfig) -> dict[str, dict[str, Any]]:
    trend: dict[str, dict[str, Any]] = {}
    for name in TREND_SERIES:
        points = [(index, series_value(item, name)) for index, item in enumerate(summaries)]
        points = [(index, value) for index, value in points if value is not None]
        if len(points) < config.min_buckets_trend:
            continue

        indices = [float(index) for index, _ in points]
        values = [value for _, value in points]
        tau, p = stats.kendall_tau(indices, values)
        slope = stats.theil_sen_slope(values)
        first, last = values[0], values[-1]

        trend[name] = {
            "tau": tau,
            "p": p,
            "slope_per_bucket": slope,
            "first": first,
            "last": last,
            "n_points": len(values),
            "meaningful": p < config.trend_p and abs(tau) >= config.trend_tau
            and _practically_relevant(name, first, last, config, require_both=False)
        }

    return trend


def _practically_relevant(name: str, reference: float, value: float, config: DriftConfig, require_both: bool) -> bool:
    """Whether a move from reference to value is large enough to matter, statistical significance aside"""
    change = abs(value - reference)
    if name in RATE_SERIES:
        absolute = change >= config.trend_rate_abs
        relative = reference > 0 and change / reference >= config.trend_rate_rel
        return (absolute and relative) if require_both else (absolute or relative)
    if name in COUNT_SERIES:
        return reference > 0 and change / reference >= config.trend_count_rel
    return change >= config.trend_value_change


def _outliers(summaries: list[dict[str, Any]], task: str, config: DriftConfig) -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    for name in OUTLIER_SERIES:
        points = [(index, series_value(item, name)) for index, item in enumerate(summaries)]
        points = [(index, value) for index, value in points if value is not None]
        if len(points) < config.min_buckets_trend:
            continue

        values = [value for _, value in points]
        scores = stats.robust_z_scores(values)
        for position, ((index, value), z) in enumerate(zip(points, scores)):
            others = values[:position] + values[position + 1:]
            center = stats.median(others)
            # A statistically extreme point that moved by a negligible amount is noise, not a transient
            if z is None or abs(z) <= config.robust_z or center is None:
                continue
            if _practically_relevant(name, center, value, config, require_both=True):
                outliers.append({
                    "bucket_index": index,
                    "bucket_start": summaries[index]["bucket_start"],
                    "series": name,
                    "z": z,
                    "value": value
                })

    return outliers


def _hard_breaks(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Every position in the ordered records where a value that should never change silently changes"""
    breaks: list[dict[str, Any]] = []
    last: dict[str, Any] = {}

    for record in records:
        observed = {
            "image_spec": list(record.image_spec) if record.image_spec is not None else None,
            "threshold": record.threshold,
            "backend": record.backend,
            "classes": list(record.classes) if record.classes else None
        }
        for kind, value in observed.items():
            if value is None:
                continue
            if kind in last and last[kind] != value:
                breaks.append({
                    "kind": kind,
                    "date": record.created_at,
                    "inspection_id": record.inspection_id,
                    "from": last[kind],
                    "to": value
                })
            last[kind] = value

    return breaks


def _elapsed_break(split: dict[str, Any], config: DriftConfig) -> list[dict[str, Any]]:
    before = split["before"].get("elapsed_time_p50")
    after = split["after"].get("elapsed_time_p50")
    if not before or after is None:
        return []

    ratio = after / before
    if ratio > config.elapsed_ratio_high or ratio < config.elapsed_ratio_low:
        return [{"kind": "elapsed_time", "date": split["date"], "inspection_id": None, "from": before, "to": after}]
    return []


def _max_pairwise(
        buckets: Sequence[Bucket],
        task: str,
        classes: Sequence[str],
        config: DriftConfig
) -> dict[str, dict[str, Any]]:
    if len(buckets) < 2:
        return {}

    counts = [side_counts(item.records, task, classes, config) for item in buckets]
    conf = [stats.proportions(item.conf_counts) for item in counts]
    class_keys = sorted({key for item in counts for key in item.class_counts})
    class_dist = [stats.proportions([item.class_counts.get(key, 0) for key in class_keys]) for item in counts]

    best = {"psi_conf": (0.0, 0, 1), "psi_class": (0.0, 0, 1)}
    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            for name, dists in (("psi_conf", conf), ("psi_class", class_dist)):
                value = stats.psi(dists[i], dists[j], config.eps)
                if value > best[name][0]:
                    best[name] = (value, i, j)

    return {
        name: {"value": value, "pair": [buckets[i].start, buckets[j].start]}
        for name, (value, i, j) in best.items()
    }


def _round(value: Any, decimals: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {key: _round(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item, decimals) for item in value]
    return value


def utc(value: datetime) -> datetime:
    """Treats naive datetimes as UTC, which is how MongoDB stores them"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def validate_windows(
        start_date: datetime,
        end_date: datetime,
        reference_start: datetime | None,
        reference_end: datetime | None,
        config: DriftConfig = DEFAULT_CONFIG
) -> None:
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    total_days = _days(start_date, end_date)
    if (reference_start is None) != (reference_end is None):
        raise ValueError("reference_start and reference_end must be given together")
    if reference_start is not None:
        if reference_start >= reference_end:
            raise ValueError("reference_start must be before reference_end")
        if reference_end > start_date:
            raise ValueError("The reference window must end before the current window starts")
        total_days += _days(reference_start, reference_end)

    if total_days > config.max_total_days:
        raise ValueError(f"The analysed windows cover {total_days:.1f} days, the maximum is {config.max_total_days}")


def resolve_task(extractors: Sequence[RecordExtractor], requested: str | None) -> str:
    if requested is not None:
        if requested not in (ModelTasks.CLASSIFICATION.value, ModelTasks.DETECTION.value):
            raise ValueError(f"Task {requested!r} is not supported for drift analysis, only cls and det are")
        return requested

    tasks = set()
    for extractor in extractors:
        tasks |= extractor.tasks_seen
    if len(tasks) > 1:
        raise ValueError(f"The model has multiple tasks {sorted(tasks)}, specify task")
    return next(iter(tasks)) if tasks else ModelTasks.CLASSIFICATION.value
