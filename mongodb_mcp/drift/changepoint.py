"""Before/after comparison of two groups of records, and the search for the split that separates them most"""
from typing import Any, Sequence
from datetime import datetime

from mongodb_mcp.drift import stats
from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG
from mongodb_mcp.drift.extract import Record
from mongodb_mcp.drift.bucketing import Bucket
from mongodb_mcp.drift.summarize import (
    SideCounts, side_counts, summarize_side, confidence_values, all_boxes, is_detection, class_distribution
)


def min_side_records(task: str, config: DriftConfig) -> int:
    return config.min_side_records_det if is_detection(task) else config.min_side_records_cls


def sides_sufficient(before: SideCounts, after: SideCounts, task: str, config: DriftConfig) -> bool:
    min_records = min_side_records(task, config)
    if before.n < min_records or after.n < min_records:
        return False
    if is_detection(task) and (before.n_boxes < config.min_side_boxes or after.n_boxes < config.min_side_boxes):
        return False
    return True


def split_score(before: SideCounts, after: SideCounts, task: str, config: DriftConfig) -> float:
    score = stats.psi(stats.proportions(before.conf_counts), stats.proportions(after.conf_counts), config.eps)
    keys = sorted(set(before.class_counts) | set(after.class_counts))
    score += stats.psi(
        stats.proportions([before.class_counts.get(key, 0) for key in keys]),
        stats.proportions([after.class_counts.get(key, 0) for key in keys]),
        config.eps
    )
    if is_detection(task):
        score += stats.psi(
            stats.proportions(before.boxes_hist_counts), stats.proportions(after.boxes_hist_counts), config.eps
        )
    return score


def _ks(before: Sequence[float], after: Sequence[float]) -> dict[str, Any] | None:
    if not before or not after:
        return None
    d, p = stats.ks_2samp(before, after)
    return {"d": d, "p": p, "n_before": len(before), "n_after": len(after)}


def compare_sides(
        before: Sequence[Record],
        after: Sequence[Record],
        task: str,
        classes: Sequence[str],
        date: datetime,
        config: DriftConfig = DEFAULT_CONFIG,
        bucket_index: int | None = None
) -> dict[str, Any]:
    """Full divergence report between two groups of records, computed from the records themselves"""
    before_counts = side_counts(before, task, classes, config)
    after_counts = side_counts(after, task, classes, config)

    before_conf = confidence_values(before, task)
    after_conf = confidence_values(after, task)
    before_hist = stats.proportions(before_counts.conf_counts)
    after_hist = stats.proportions(after_counts.conf_counts)

    keys = sorted(set(before_counts.class_counts) | set(after_counts.class_counts))
    before_class = [before_counts.class_counts.get(key, 0) for key in keys]
    after_class = [after_counts.class_counts.get(key, 0) for key in keys]
    chi2_stat, chi2_p, dof = stats.chi2_contingency([before_class, after_class])
    before_dist = class_distribution(dict(zip(keys, before_class)))
    after_dist = class_distribution(dict(zip(keys, after_class)))
    max_prop_change = max((abs(before_dist[key] - after_dist[key]) for key in keys), default=0.0)

    report: dict[str, Any] = {
        "bucket_index": bucket_index,
        "date": date,
        "score": split_score(before_counts, after_counts, task, config),
        "n_before": before_counts.n,
        "n_after": after_counts.n,
        "sides_sufficient": sides_sufficient(before_counts, after_counts, task, config),
        "psi_conf": stats.psi(before_hist, after_hist, config.eps),
        "js_conf": stats.js_divergence(before_hist, after_hist, config.eps),
        "ks_conf": _ks(before_conf, after_conf) or {"d": 0.0, "p": 1.0, "n_before": 0, "n_after": 0},
        "psi_class": stats.psi(stats.proportions(before_class), stats.proportions(after_class), config.eps),
        "chi2_class": {"stat": chi2_stat, "p": chi2_p, "dof": dof},
        "max_class_prop_change": max_prop_change,
        "before": summarize_side(before, task, classes, config),
        "after": summarize_side(after, task, classes, config)
    }

    if is_detection(task):
        report["psi_boxes_per_image"] = stats.psi(
            stats.proportions(before_counts.boxes_hist_counts),
            stats.proportions(after_counts.boxes_hist_counts),
            config.eps
        )
        before_boxes, after_boxes = all_boxes(before), all_boxes(after)
        for name in ("area_norm", "cx_norm", "cy_norm"):
            report[f"ks_{name}"] = _ks(
                [getattr(box, name) for box in before_boxes if getattr(box, name) is not None],
                [getattr(box, name) for box in after_boxes if getattr(box, name) is not None]
            )
    else:
        for name in ("margin", "entropy"):
            report[f"ks_{name}"] = _ks(
                [getattr(record, name) for record in before if getattr(record, name) is not None],
                [getattr(record, name) for record in after if getattr(record, name) is not None]
            )

    return report


def best_split(
        counts: Sequence[SideCounts],
        task: str,
        config: DriftConfig = DEFAULT_CONFIG
) -> tuple[int, float, bool] | None:
    """Index k maximising the divergence between buckets[:k] and buckets[k:], preferring sufficient sides

    Returns (k, score, sufficient). When no split has enough records on both sides the best insufficient split is
    returned so the reader still sees where the largest movement is, and the flags stay silent.
    """
    total = len(counts)
    min_seg = config.min_segment_buckets
    if total < 2 * min_seg:
        return None

    prefix: list[SideCounts] = []
    running = SideCounts()
    for item in counts:
        running = running + item
        prefix.append(running)
    grand_total = prefix[-1]

    best_sufficient: tuple[int, float] | None = None
    best_any: tuple[int, float] | None = None
    for k in range(min_seg, total - min_seg + 1):
        before = prefix[k - 1]
        after = SideCounts(
            n=grand_total.n - before.n,
            n_boxes=grand_total.n_boxes - before.n_boxes,
            conf_counts=[a - b for a, b in zip(grand_total.conf_counts, before.conf_counts)],
            class_counts={key: grand_total.class_counts.get(key, 0) - before.class_counts.get(key, 0)
                          for key in grand_total.class_counts},
            boxes_hist_counts=[a - b for a, b in zip(grand_total.boxes_hist_counts, before.boxes_hist_counts)]
        )
        score = split_score(before, after, task, config)
        if best_any is None or score > best_any[1]:
            best_any = (k, score)
        if sides_sufficient(before, after, task, config) and (best_sufficient is None or score > best_sufficient[1]):
            best_sufficient = (k, score)

    if best_sufficient is not None:
        return best_sufficient[0], best_sufficient[1], True
    if best_any is not None:
        return best_any[0], best_any[1], False
    return None


def find_change_points(
        buckets: Sequence[Bucket],
        task: str,
        classes: Sequence[str],
        config: DriftConfig = DEFAULT_CONFIG
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Primary split plus at most one secondary split on each side (one level of binary segmentation)"""
    counts = [side_counts(bucket.records, task, classes, config) for bucket in buckets]
    primary = _split_report(buckets, counts, task, classes, config, offset=0)
    if primary is None:
        return None, []

    k = primary["bucket_index"]
    secondary: list[dict[str, Any]] = []
    for lo, hi in ((0, k), (k, len(buckets))):
        if hi - lo >= 2 * config.min_segment_buckets:
            report = _split_report(buckets[lo:hi], counts[lo:hi], task, classes, config, offset=lo)
            if report is not None and report["sides_sufficient"]:
                secondary.append(report)

    return primary, secondary


def _split_report(
        buckets: Sequence[Bucket],
        counts: Sequence[SideCounts],
        task: str,
        classes: Sequence[str],
        config: DriftConfig,
        offset: int
) -> dict[str, Any] | None:
    found = best_split(counts, task, config)
    if found is None:
        return None

    k, _, _ = found
    before = [record for bucket in buckets[:k] for record in bucket.records]
    after = [record for bucket in buckets[k:] for record in bucket.records]
    return compare_sides(before, after, task, classes, buckets[k].start, config, bucket_index=offset + k)
