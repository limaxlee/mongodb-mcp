"""Per bucket and per side summary statistics

For detection the confidence and class statistics of a bucket are computed over boxes, since an image row has no
confidence of its own. The box block of a bucket then only carries geometry.
"""
import math
from typing import Any, Sequence
from dataclasses import dataclass, field

from common.constants import ModelTasks
from mongodb_mcp.drift import stats
from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG
from mongodb_mcp.drift.extract import Record, BoxRecord
from mongodb_mcp.drift.bucketing import Bucket

CONF_POINTS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
THREE_POINTS = (0.10, 0.50, 0.90)


@dataclass
class SideCounts:
    """Raw counts of a group of records, additive across groups so before/after histograms are exact sums"""
    n: int = 0
    n_boxes: int = 0
    conf_counts: list[int] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    boxes_hist_counts: list[int] = field(default_factory=list)

    def __add__(self, other: "SideCounts") -> "SideCounts":
        return SideCounts(
            n=self.n + other.n,
            n_boxes=self.n_boxes + other.n_boxes,
            conf_counts=_add_lists(self.conf_counts, other.conf_counts),
            class_counts={
                key: self.class_counts.get(key, 0) + other.class_counts.get(key, 0)
                for key in set(self.class_counts) | set(other.class_counts)
            },
            boxes_hist_counts=_add_lists(self.boxes_hist_counts, other.boxes_hist_counts)
        )


def _add_lists(a: list[int], b: list[int]) -> list[int]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    return [x + y for x, y in zip(a, b)]


def is_detection(task: str) -> bool:
    return task == ModelTasks.DETECTION.value


def all_boxes(records: Sequence[Record]) -> list[BoxRecord]:
    return [box for record in records for box in record.boxes]


def confidence_values(records: Sequence[Record], task: str) -> list[float]:
    if is_detection(task):
        return [box.conf_max for box in all_boxes(records) if box.conf_max is not None]
    return [record.conf_max for record in records if record.conf_max is not None]


def class_labels(records: Sequence[Record], task: str) -> list[str]:
    if is_detection(task):
        return [box.prediction for box in all_boxes(records)]
    return [record.prediction for record in records if record.prediction is not None]


def class_counts(records: Sequence[Record], task: str, classes: Sequence[str]) -> dict[str, int]:
    counts = {name: 0 for name in classes}
    for label in class_labels(records, task):
        counts[label] = counts.get(label, 0) + 1
    return counts


def class_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {name: (count / total if total else 0.0) for name, count in counts.items()}


def boxes_per_image_counts(records: Sequence[Record], config: DriftConfig) -> list[int]:
    counts = [0] * (config.boxes_per_image_max_bin + 1)
    for record in records:
        counts[min(record.n_boxes, config.boxes_per_image_max_bin)] += 1
    return counts


def side_counts(
        records: Sequence[Record],
        task: str,
        classes: Sequence[str],
        config: DriftConfig = DEFAULT_CONFIG
) -> SideCounts:
    counts = SideCounts(
        n=len(records),
        conf_counts=stats.histogram(confidence_values(records, task), config.conf_bin_edges),
        class_counts=class_counts(records, task, classes)
    )
    if is_detection(task):
        counts.n_boxes = sum(record.n_boxes for record in records)
        counts.boxes_hist_counts = boxes_per_image_counts(records, config)
    return counts


def _threshold_values(records: Sequence[Record], task: str) -> list[float]:
    if is_detection(task):
        values = {box.threshold for box in all_boxes(records) if box.threshold is not None}
    else:
        values = {record.threshold for record in records if record.threshold is not None}
    return sorted(values)


def _below_threshold_rate(records: Sequence[Record], task: str) -> float | None:
    if is_detection(task):
        flags = [box.below_threshold for box in all_boxes(records) if box.below_threshold is not None]
    else:
        flags = [record.below_threshold for record in records if record.below_threshold is not None]
    return sum(flags) / len(flags) if flags else None


def feedback_stats(records: Sequence[Record], task: str) -> tuple[int, float | None, int]:
    """Returns (n labelled feedback, mismatch rate over labelled feedback, n other feedback)"""
    items: Sequence[Record | BoxRecord] = all_boxes(records) if is_detection(task) else records
    labelled = [item.feedback_mismatch for item in items if item.feedback_label is not None]
    other = sum(1 for item in items if item.feedback_other)
    rate = sum(labelled) / len(labelled) if labelled else None
    return len(labelled), rate, other


def elapsed_time_p50(records: Sequence[Record]) -> float | None:
    return stats.median([record.elapsed_time for record in records if record.elapsed_time is not None])


def _quantiles3(values: Sequence[float]) -> dict[str, float | None]:
    return stats.quantiles(values, THREE_POINTS) if values else {"p10": None, "p50": None, "p90": None}


def summarize_side(
        records: Sequence[Record],
        task: str,
        classes: Sequence[str],
        config: DriftConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    confidences = confidence_values(records, task)
    counts = side_counts(records, task, classes, config)
    n_feedback, mismatch_rate, _ = feedback_stats(records, task)

    return {
        "n": counts.n,
        "n_boxes": counts.n_boxes if is_detection(task) else None,
        "class_dist": class_distribution(counts.class_counts),
        "conf_hist": stats.proportions(counts.conf_counts),
        "conf_quantiles": stats.quantiles(confidences, CONF_POINTS) if confidences else {},
        "below_threshold_rate": _below_threshold_rate(records, task),
        "boxes_per_image_hist": stats.proportions(counts.boxes_hist_counts) if is_detection(task) else None,
        "n_feedback": n_feedback,
        "feedback_mismatch_rate": mismatch_rate,
        "elapsed_time_p50": elapsed_time_p50(records)
    }


def summarize_bucket(
        bucket: Bucket,
        task: str,
        classes: Sequence[str],
        defect_classes: Sequence[str],
        config: DriftConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Full summary of one bucket, the compact form is derived from it when the result is assembled"""
    records = bucket.records
    confidences = confidence_values(records, task)
    counts = side_counts(records, task, classes, config)
    class_dist = class_distribution(counts.class_counts)
    n_feedback, mismatch_rate, n_other = feedback_stats(records, task)
    image_specs = sorted({record.image_spec for record in records if record.image_spec is not None})

    summary: dict[str, Any] = {
        "bucket_start": bucket.start,
        "bucket_end": bucket.end,
        "window": bucket.window,
        "n": counts.n,
        "merged_from": bucket.merged_from,
        "class_dist": class_dist,
        "defect_rate": sum(class_dist.get(name, 0.0) for name in defect_classes) if class_dist else None,
        "conf_p50": stats.median(confidences),
        "conf_mean": stats.mean(confidences),
        "conf_std": stats.std(confidences),
        "conf_hist": stats.proportions(counts.conf_counts),
        "conf_quantiles": stats.quantiles(confidences, CONF_POINTS) if confidences else {},
        "below_threshold_rate": _below_threshold_rate(records, task),
        "threshold_values": _threshold_values(records, task),
        "image_specs": [list(spec) for spec in image_specs],
        "elapsed_time_p50": elapsed_time_p50(records),
        "n_feedback": n_feedback,
        "feedback_mismatch_rate": mismatch_rate,
        "n_feedback_other": n_other
    }

    if is_detection(task):
        summary.update(_detection_summary(records, counts, classes, config))
    else:
        summary.update(_classification_summary(records))

    return summary


def _classification_summary(records: Sequence[Record]) -> dict[str, Any]:
    margins = [record.margin for record in records if record.margin is not None]
    entropies = [record.entropy for record in records if record.entropy is not None]
    near = [record.near_threshold for record in records if record.near_threshold is not None]
    patch_w = [record.patch_w for record in records if record.patch_w is not None]
    patch_h = [record.patch_h for record in records if record.patch_h is not None]

    return {
        "margin_quantiles": _quantiles3(margins) if margins else None,
        "entropy_quantiles": _quantiles3(entropies) if entropies else None,
        "decision_differs_rate": sum(r.decision_differs for r in records) / len(records) if records else None,
        "patch_w_p50": stats.median(patch_w),
        "patch_h_p50": stats.median(patch_h),
        "near_threshold_rate": sum(near) / len(near) if near else None
    }


def _detection_summary(
        records: Sequence[Record],
        counts: SideCounts,
        classes: Sequence[str],
        config: DriftConfig
) -> dict[str, Any]:
    boxes = all_boxes(records)
    n_images = len(records)
    per_image = [float(record.n_boxes) for record in records]

    by_class: dict[str, float] = {}
    for name in list(classes) + [box.prediction for box in boxes if box.prediction not in classes]:
        by_class[name] = sum(record.n_boxes_by_class.get(name, 0) for record in records) / n_images if n_images else 0.0

    area_norm = [box.area_norm for box in boxes if box.area_norm is not None]
    log_area = [math.log10(value) for value in area_norm if value > 0]

    return {
        "n_boxes": counts.n_boxes,
        "boxes_per_image_mean": stats.mean(per_image),
        "boxes_per_image_std": stats.std(per_image),
        "boxes_per_image_hist": stats.proportions(counts.boxes_hist_counts),
        "no_box_rate": sum(record.has_no_boxes for record in records) / n_images if n_images else None,
        "boxes_by_class_per_image": by_class,
        "box": {
            "n_boxes": counts.n_boxes,
            "area_norm_quantiles": _quantiles3(area_norm),
            "w_norm_quantiles": _quantiles3([box.w_norm for box in boxes if box.w_norm is not None]),
            "h_norm_quantiles": _quantiles3([box.h_norm for box in boxes if box.h_norm is not None]),
            "aspect_quantiles": _quantiles3([box.aspect for box in boxes if box.aspect is not None]),
            "cx_norm_quantiles": _quantiles3([box.cx_norm for box in boxes if box.cx_norm is not None]),
            "cy_norm_quantiles": _quantiles3([box.cy_norm for box in boxes if box.cy_norm is not None]),
            "area_norm_hist": stats.proportions(stats.histogram(log_area, config.area_log_bin_edges))
            if log_area else None
        }
    }


def series_value(summary: dict[str, Any], name: str) -> float | None:
    """Reads one scalar series value out of a full bucket summary"""
    direct = ("conf_p50", "conf_mean", "below_threshold_rate", "defect_rate", "boxes_per_image_mean", "no_box_rate")
    if name in direct:
        return summary.get(name)

    nested = {
        "margin_p50": ("margin_quantiles", "p50"),
        "entropy_p50": ("entropy_quantiles", "p50"),
        "area_norm_p50": ("box", "area_norm_quantiles", "p50"),
        "cx_norm_p50": ("box", "cx_norm_quantiles", "p50"),
        "cy_norm_p50": ("box", "cy_norm_quantiles", "p50")
    }
    value: Any = summary
    for key in nested.get(name, ()):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, (int, float)) else None


COMPACT_KEYS = (
    "bucket_start", "bucket_end", "window", "n", "merged_from", "class_dist", "defect_rate", "conf_p50", "conf_mean",
    "below_threshold_rate", "n_feedback", "feedback_mismatch_rate", "n_feedback_other", "near_threshold_rate",
    "boxes_per_image_mean", "no_box_rate"
)


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary[key] for key in COMPACT_KEYS if key in summary}
