"""Synthetic inspection statistics documents, and a Python emulation of the tool's aggregation pipeline

`make_document` / `generate_documents` build documents shaped like the inspectionStatistics collection, the way
the Data Service would write them. `aggregate` turns documents into the rows the pipeline of
`StatisticsQuery.build_pipeline` returns, with the same accumulator semantics ($sum ignores non-numbers, $addToSet
keeps nulls, $min/$max ignore nulls, $first takes the first document), so that the parsing and the analysis can be
tested without MongoDB.
"""
import math
import random
from typing import Any, Callable, Sequence
from datetime import datetime, timedelta, timezone

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
CLASSES = ["Good", "NG"]
EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
BINS = {"confidenceEdges": EDGES, "nearThresholdMargin": 0.05, "boxesPerImageMax": 5}
QUANTILE_POINTS = {"p05": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95}
PERIOD_HOURS = {"hourly": 1, "shift": 12, "daily": 24, "weekly": 168}


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    last = len(edges) - 2
    if value >= edges[-1]:
        return last
    for index in range(last + 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return 0


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    position = q * (n - 1)
    lower = math.floor(position)
    upper = min(lower + 1, n - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def confidence_stats(
        values: Sequence[float],
        threshold: float | None,
        edges: Sequence[float] = EDGES,
        margin: float = 0.05
) -> dict[str, Any]:
    """The ConfidenceStats block of the schema, computed from raw confidences"""
    count = len(values)
    histogram = [0] * (len(edges) - 1)
    for value in values:
        histogram[_bin_index(value, edges)] += 1
    if count == 0:
        return {"count": 0, "sum": 0.0, "sumSq": 0.0, "mean": None, "std": None, "min": None, "max": None,
                "quantiles": None, "histogram": histogram, "belowThresholdCount": None, "nearThresholdCount": None}

    ordered = sorted(values)
    total = sum(values)
    sum_sq = sum(value * value for value in values)
    mean = total / count
    return {
        "count": count,
        "sum": total,
        "sumSq": sum_sq,
        "mean": mean,
        "std": math.sqrt(max(sum_sq / count - mean * mean, 0.0)),
        "min": ordered[0],
        "max": ordered[-1],
        "quantiles": {key: _quantile(ordered, point) for key, point in QUANTILE_POINTS.items()},
        "histogram": histogram,
        "belowThresholdCount": sum(1 for value in values if value < threshold) if threshold is not None else None,
        "nearThresholdCount": sum(1 for value in values if abs(value - threshold) < margin)
        if threshold is not None else None
    }


def _elapsed(count: int, elapsed: float) -> dict[str, Any]:
    return {"count": count, "sum": count * elapsed, "mean": elapsed, "min": elapsed, "max": elapsed,
            "p50": elapsed, "p90": elapsed}


def cls_equipment(
        rng: random.Random,
        n: int,
        conf_mean: float,
        ng_rate: float,
        threshold: float | None = 0.8,
        classes: Sequence[str] = CLASSES,
        backend: str = "ts",
        elapsed: float = 0.008,
        conf_sd: float = 0.05,
        edges: Sequence[float] = EDGES
) -> dict[str, Any]:
    """EquipmentStats of a classification model: `n` predictions with the given mean confidence and NG share"""
    labels = [classes[1] if rng.random() < ng_rate else classes[0] for _ in range(n)]
    values = [_clip(rng.gauss(conf_mean, conf_sd)) for _ in range(n)]
    by_class: dict[str, list[float]] = {name: [] for name in classes}
    for label, value in zip(labels, values):
        by_class.setdefault(label, []).append(value)

    return {
        "inspectionCount": max(n // 10, 1),
        "predictionCount": n,
        "backend": backend,
        "threshold": threshold,
        "quality": {"missingConfidenceCount": 0, "parseErrorCount": 0},
        "elapsedTime": _elapsed(n, elapsed),
        "classCounts": {name: len(items) for name, items in by_class.items()},
        "confidence": confidence_stats(values, threshold, edges),
        "perClass": {name: confidence_stats(items, threshold, edges) for name, items in by_class.items()}
    }


def det_equipment(
        rng: random.Random,
        n_images: int,
        conf_mean: float,
        ng_rate: float,
        boxes_mean: float = 2.0,
        threshold: float | None = 0.8,
        no_box_rate: float = 0.02,
        classes: Sequence[str] = CLASSES,
        backend: str = "trt",
        elapsed: float = 0.02,
        conf_sd: float = 0.05,
        edges: Sequence[float] = EDGES,
        max_bin: int = 5
) -> dict[str, Any]:
    """EquipmentStats of a detection model: confidence and class blocks over boxes, images block over images"""
    by_class: dict[str, list[float]] = {name: [] for name in classes}
    box_counts: list[int] = []
    for _ in range(n_images):
        n_boxes = 0 if rng.random() < no_box_rate else max(1, round(rng.gauss(boxes_mean, 0.7)))
        box_counts.append(n_boxes)
        for _ in range(n_boxes):
            label = classes[1] if rng.random() < ng_rate else classes[0]
            by_class.setdefault(label, []).append(_clip(rng.gauss(conf_mean, conf_sd)))

    values = [value for items in by_class.values() for value in items]
    histogram = [0] * (max_bin + 1)
    for count in box_counts:
        histogram[min(count, max_bin)] += 1
    total_boxes = sum(box_counts)

    per_class = {}
    for name, items in by_class.items():
        block = confidence_stats(items, threshold, edges)
        block["threshold"] = threshold
        per_class[name] = block

    return {
        "inspectionCount": max(n_images // 10, 1),
        "predictionCount": n_images,
        "boxCount": total_boxes,
        "backend": backend,
        "threshold": None,
        "quality": {"missingConfidenceCount": 0, "parseErrorCount": 0},
        "elapsedTime": _elapsed(n_images, elapsed),
        "images": {
            "noBoxCount": sum(1 for count in box_counts if count == 0),
            "boxesPerImage": {
                "sum": total_boxes,
                "sumSq": sum(count * count for count in box_counts),
                "mean": total_boxes / n_images if n_images else None,
                "std": None,
                "max": max(box_counts, default=None),
                "histogram": histogram
            },
            "boxesPerImageByClass": {
                name: len(items) / n_images if n_images else 0.0 for name, items in by_class.items()
            }
        },
        "classCounts": {name: len(items) for name, items in by_class.items()},
        "confidence": confidence_stats(values, threshold, edges),
        "perClass": per_class
    }


def _merge_confidence(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = sum(block["count"] for block in blocks)
    total = sum(block["sum"] for block in blocks)
    sum_sq = sum(block["sumSq"] for block in blocks)
    histogram = [sum(values) for values in zip(*(block["histogram"] for block in blocks))]
    mins = [block["min"] for block in blocks if block["min"] is not None]
    maxs = [block["max"] for block in blocks if block["max"] is not None]
    below = [block["belowThresholdCount"] for block in blocks if block["belowThresholdCount"] is not None]
    near = [block["nearThresholdCount"] for block in blocks if block["nearThresholdCount"] is not None]
    mean = total / count if count else None
    merged = {
        "count": count,
        "sum": total,
        "sumSq": sum_sq,
        "mean": mean,
        "std": math.sqrt(max(sum_sq / count - mean * mean, 0.0)) if count else None,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "quantiles": blocks[0]["quantiles"] if len(blocks) == 1 else None,
        "histogram": histogram,
        "belowThresholdCount": sum(below) if below else None,
        "nearThresholdCount": sum(near) if near else None
    }
    thresholds = {block.get("threshold") for block in blocks if block.get("threshold") is not None}
    if len(thresholds) == 1:
        merged["threshold"] = thresholds.pop()
    return merged


def total_block(equipments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The total block: the additive sum of the equipment entries"""
    stats = [{key: value for key, value in item.items() if key not in ("equipmentId", "location")}
             for item in equipments]
    class_names: list[str] = []
    for item in stats:
        class_names.extend(name for name in item["perClass"] if name not in class_names)

    total: dict[str, Any] = {
        "inspectionCount": sum(item["inspectionCount"] for item in stats),
        "predictionCount": sum(item["predictionCount"] for item in stats),
        "quality": {
            "missingConfidenceCount": sum(item["quality"]["missingConfidenceCount"] for item in stats),
            "parseErrorCount": sum(item["quality"]["parseErrorCount"] for item in stats)
        },
        "elapsedTime": {
            "count": sum(item["elapsedTime"]["count"] for item in stats),
            "sum": sum(item["elapsedTime"]["sum"] for item in stats)
        },
        "classCounts": {
            name: sum(item["classCounts"].get(name, 0) for item in stats) for name in class_names
        },
        "confidence": _merge_confidence([item["confidence"] for item in stats]),
        "perClass": {
            name: _merge_confidence([item["perClass"][name] for item in stats if name in item["perClass"]])
            for name in class_names
        }
    }
    if any("boxCount" in item for item in stats):
        total["boxCount"] = sum(item.get("boxCount", 0) for item in stats)
        images = [item["images"] for item in stats if item.get("images")]
        total["images"] = {
            "noBoxCount": sum(item["noBoxCount"] for item in images),
            "boxesPerImage": {
                "sum": sum(item["boxesPerImage"]["sum"] for item in images),
                "sumSq": sum(item["boxesPerImage"]["sumSq"] for item in images),
                "histogram": [sum(values) for values in zip(*(item["boxesPerImage"]["histogram"] for item in images))]
            }
        }
    return total


def period_bounds(start: datetime, granularity: str) -> tuple[datetime, datetime]:
    return start, start + timedelta(hours=PERIOD_HOURS[granularity])


def make_document(
        start: datetime,
        task: str,
        equipments: Sequence[tuple[str, str, dict[str, Any]]],
        granularity: str = "daily",
        model_name: str = "MetalCls",
        model_version: str = "1.0",
        mode: str = "production",
        gbm: str = "SEV",
        process: str = "SMD",
        product_id: str = "PR-01",
        classes: Sequence[str] = CLASSES,
        bins: dict[str, Any] = BINS,
        local_timezone: str = "Asia/Seoul"
) -> dict[str, Any]:
    """One inspection statistics document"""
    start_date, end_date = period_bounds(start, granularity)
    entries = [dict(stats, equipmentId=equipment_id, location=location)
               for equipment_id, location, stats in equipments]
    return {
        "_id": f"{abs(hash((start, model_name, product_id))):024x}"[:24],
        "modelName": model_name,
        "modelVersion": model_version,
        "task": task,
        "mode": mode,
        "gbm": gbm,
        "process": process,
        "productId": product_id,
        "granularity": granularity,
        "startDate": start_date,
        "endDate": end_date,
        "classes": list(classes),
        "localTimezone": local_timezone,
        "bins": dict(bins),
        "computedAt": end_date + timedelta(minutes=5),
        "equipmentCount": len(entries),
        "total": total_block(entries),
        "equipments": entries
    }


def generate_documents(
        task: str,
        periods: int,
        conf_for_period: Callable[[int], float],
        n: int = 300,
        ng_rate: float = 0.05,
        granularity: str = "daily",
        products: int = 1,
        equipments: Sequence[tuple[str, str]] = (("EQ-01", "Line_01"),),
        stats_kwargs_for_period: Callable[[int], dict[str, Any]] | None = None,
        doc_kwargs_for_period: Callable[[int], dict[str, Any]] | None = None,
        seed: int = 7,
        start: datetime = START
) -> list[dict[str, Any]]:
    """`periods` consecutive periods, `products` documents per period, every document with every equipment

    `conf_for_period(index)` gives the mean confidence of a period, `stats_kwargs_for_period(index)` extra
    arguments of the equipment generator (threshold, backend, ...), `doc_kwargs_for_period(index)` extra document
    fields (bins, classes, mode, ...).
    """
    rng = random.Random(seed)
    generator = det_equipment if task == "det" else cls_equipment
    documents = []
    hours = PERIOD_HOURS[granularity]
    for index in range(periods):
        period_start = start + timedelta(hours=index * hours)
        stats_kwargs = {"ng_rate": ng_rate, **(stats_kwargs_for_period(index) if stats_kwargs_for_period else {})}
        for product in range(products):
            doc_kwargs = {
                "granularity": granularity,
                "product_id": f"PR-{index:03d}-{product:03d}",
                **(doc_kwargs_for_period(index) if doc_kwargs_for_period else {})
            }
            entries = [
                (equipment_id, location, generator(rng, max(n // products, 1), conf_for_period(index), **stats_kwargs))
                for equipment_id, location in equipments
            ]
            documents.append(make_document(period_start, task, entries, **doc_kwargs))
    return documents


# ---------------------------------------------------------------------------------------------------------------------
# Pipeline emulation
# ---------------------------------------------------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get(item: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(item, dict):
            return None
        item = item.get(key)
    return item


def _add_to_set(target: list[Any], value: Any) -> None:
    if value not in target:
        target.append(value)


def _new_group(kind: str, name: str | None, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": {"date": doc["startDate"], "kind": kind, "k": name},
        "endDate": doc.get("endDate"),
        "documentCount": 0, "missingBlockCount": 0,
        "tasks": [], "productIds": [], "gbms": [], "processes": [], "modes": [],
        "equipmentIdSets": [], "classSets": [], "bins": [],
        "inspectionCount": 0, "predictionCount": 0, "boxCount": 0, "boxCountPresent": 0,
        "missingConfidenceCount": 0, "parseErrorCount": 0, "elapsedCount": 0, "elapsedSum": 0.0,
        "imagesPresent": 0, "noBoxCount": 0, "boxesPerImageSum": 0, "boxesPerImageSumSq": 0,
        "boxesPerImageHistograms": [], "backendSets": [], "thresholdSets": [],
        "count": 0, "sum": 0.0, "sumSq": 0.0, "min": None, "max": None,
        "belowThresholdCount": 0, "belowPresent": 0, "nearThresholdCount": 0, "nearPresent": 0,
        "histograms": [], "entryThresholds": [], "quantiles": None, "quantilesSet": False, "classCount": 0
    }


def _sum_arrays(arrays: Sequence[Any]) -> list[int]:
    result: list[int] = []
    for item in arrays:
        if not isinstance(item, list):
            continue
        result = list(item) if not result else [a + b for a, b in zip(result, item)]
    return result


def aggregate(documents: Sequence[dict[str, Any]], equipment_id: str | None = None) -> list[dict[str, Any]]:
    """Rows of StatisticsQuery.build_pipeline over the given documents (the $match stage is not applied)"""
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}

    for doc in documents:
        if equipment_id is None:
            block = doc.get("total")
        else:
            block = next((item for item in doc.get("equipments", []) if item.get("equipmentId") == equipment_id), None)
        present = block is not None
        block = block or {}
        equipment_ids = [item.get("equipmentId") for item in doc.get("equipments", [])]
        if equipment_id is None:
            backends = [item.get("backend") for item in doc.get("equipments", []) if "backend" in item]
            thresholds = [item.get("threshold") for item in doc.get("equipments", []) if "threshold" in item]
        else:
            backends, thresholds = [block.get("backend")], [block.get("threshold")]

        entries: list[tuple[str, str | None, dict[str, Any]]] = [("total", None, block.get("confidence") or {})]
        entries.extend(("class", name, value) for name, value in (block.get("perClass") or {}).items())
        entries.extend(("count", name, {"classCount": value}) for name, value in (block.get("classCounts") or {}).items())

        for kind, name, value in entries:
            key = (doc["startDate"], kind, name)
            group = groups.get(key)
            if group is None:
                group = _new_group(kind, name, doc)
                groups[key] = group

            group["documentCount"] += 1 if present else 0
            group["missingBlockCount"] += 0 if present else 1
            _add_to_set(group["tasks"], doc.get("task"))
            _add_to_set(group["productIds"], doc.get("productId") if present else None)
            _add_to_set(group["gbms"], doc.get("gbm") if present else None)
            _add_to_set(group["processes"], doc.get("process") if present else None)
            _add_to_set(group["modes"], doc.get("mode") if present else None)
            _add_to_set(group["equipmentIdSets"], equipment_ids if present else None)
            _add_to_set(group["classSets"], doc.get("classes") if present else None)
            _add_to_set(group["bins"], doc.get("bins") if present else None)

            for field, path in (
                    ("inspectionCount", "inspectionCount"), ("predictionCount", "predictionCount"),
                    ("boxCount", "boxCount"), ("missingConfidenceCount", "quality.missingConfidenceCount"),
                    ("parseErrorCount", "quality.parseErrorCount"), ("elapsedCount", "elapsedTime.count"),
                    ("elapsedSum", "elapsedTime.sum"), ("noBoxCount", "images.noBoxCount"),
                    ("boxesPerImageSum", "images.boxesPerImage.sum"),
                    ("boxesPerImageSumSq", "images.boxesPerImage.sumSq")
            ):
                item = _get(block, path)
                if _is_number(item):
                    group[field] += item
            group["boxCountPresent"] = max(group["boxCountPresent"], 1 if block.get("boxCount") is not None else 0)
            group["imagesPresent"] = max(group["imagesPresent"], 1 if block.get("images") is not None else 0)
            group["boxesPerImageHistograms"].append(_get(block, "images.boxesPerImage.histogram"))
            _add_to_set(group["backendSets"], backends if present else None)
            _add_to_set(group["thresholdSets"], thresholds if present else None)

            for field in ("count", "sum", "sumSq", "belowThresholdCount", "nearThresholdCount", "classCount"):
                item = value.get(field)
                if _is_number(item):
                    group[field] += item
            for field, chooser in (("min", min), ("max", max)):
                item = value.get(field)
                if _is_number(item):
                    group[field] = item if group[field] is None else chooser(group[field], item)
            group["belowPresent"] = max(group["belowPresent"], 1 if value.get("belowThresholdCount") is not None else 0)
            group["nearPresent"] = max(group["nearPresent"], 1 if value.get("nearThresholdCount") is not None else 0)
            group["histograms"].append(value.get("histogram"))
            _add_to_set(group["entryThresholds"], value.get("threshold"))
            if not group["quantilesSet"]:
                group["quantiles"] = value.get("quantiles")
                group["quantilesSet"] = True

    rows = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2] or "")):
        group = groups[key]
        group["histogram"] = _sum_arrays(group.pop("histograms"))
        group["boxesPerImageHistogram"] = _sum_arrays(group.pop("boxesPerImageHistograms"))
        group.pop("quantilesSet")
        rows.append(group)
    return rows
