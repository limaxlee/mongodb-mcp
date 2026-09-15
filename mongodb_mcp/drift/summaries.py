"""Per bucket and per side summary statistics

For detection the confidence and class statistics of a bucket are computed over boxes, since an image row has no
confidence of its own. The box block of a bucket then only carries geometry.
"""
import math
from typing import Any, Sequence

from common.config import SETTINGS
from common.constants import DriftTask, CLASS_SHARE_PREFIX, BoxGeometry
from mongodb_mcp.utils import stats
from mongodb_mcp.schemas import Record, BoxRecord, Bucket, SideCounts, ConfidenceQuantiles, Quantiles, CompactBucketSummary


class Summarizer:
    """Computes the statistics of a group of records for one task and one fixed set of classes"""

    def __init__(self, task: str, classes: Sequence[str]):
        self.config = SETTINGS.data_drift
        self.task = task
        self.classes = list(classes)
        self.detection = task == DriftTask.DETECTION
        self.confidence_points = self._points(ConfidenceQuantiles)
        self.three_points = self._points(Quantiles)

    @staticmethod
    def _points(model: type[ConfidenceQuantiles] | type[Quantiles]) -> list[float]:
        """The quantile points a schema asks for, read from its field names (p05 -> 0.05)"""
        return [int(name[1:]) / 100 for name in model.model_fields]

    # Record access

    @staticmethod
    def boxes(records: Sequence[Record]) -> list[BoxRecord]:
        return [box for record in records for box in record.boxes]

    def confidences(self, records: Sequence[Record]) -> list[float]:
        if self.detection:
            return [box.max_confidence for box in self.boxes(records) if box.max_confidence is not None]
        return [record.max_confidence for record in records if record.max_confidence is not None]

    def labels(self, records: Sequence[Record]) -> list[str]:
        if self.detection:
            return [box.prediction for box in self.boxes(records)]
        return [record.prediction for record in records if record.prediction is not None]

    def class_counts(self, records: Sequence[Record]) -> dict[str, int]:
        counts = {name: 0 for name in self.classes}
        for label in self.labels(records):
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def class_distribution(counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        return {name: (count / total if total else 0.0) for name, count in counts.items()}

    def boxes_per_image_counts(self, records: Sequence[Record]) -> list[int]:
        counts = [0] * (self.config.boxes_per_image_max_bin + 1)
        for record in records:
            counts[min(record.box_count, self.config.boxes_per_image_max_bin)] += 1
        return counts

    def side_counts(self, records: Sequence[Record]) -> SideCounts:
        counts = SideCounts(
            record_count=len(records),
            confidence_counts=stats.histogram(self.confidences(records), self.config.confidence_bin_edges),
            class_counts=self.class_counts(records)
        )
        if self.detection:
            counts.box_count = sum(record.box_count for record in records)
            counts.boxes_per_image_counts = self.boxes_per_image_counts(records)
        return counts

    def geometry(self, records: Sequence[Record], measure: BoxGeometry) -> list[float]:
        """One normalised geometry measure of every box that has it"""
        values = [getattr(box, measure.value) for box in self.boxes(records)]
        return [value for value in values if value is not None]

    def thresholds(self, records: Sequence[Record]) -> list[float]:
        if self.detection:
            values = {box.threshold for box in self.boxes(records) if box.threshold is not None}
        else:
            values = {record.threshold for record in records if record.threshold is not None}
        return sorted(values)

    def below_threshold_rate(self, records: Sequence[Record]) -> float | None:
        if self.detection:
            flags = [box.below_threshold for box in self.boxes(records) if box.below_threshold is not None]
        else:
            flags = [record.below_threshold for record in records if record.below_threshold is not None]
        return sum(flags) / len(flags) if flags else None

    @staticmethod
    def median_elapsed_time(records: Sequence[Record]) -> float | None:
        return stats.median([record.elapsed_time for record in records if record.elapsed_time is not None])

    # Summaries

    def side(self, records: Sequence[Record]) -> dict[str, Any]:
        """Summary of one side of a split"""
        confidences = self.confidences(records)
        counts = self.side_counts(records)

        return {
            "record_count": counts.record_count,
            "box_count": counts.box_count if self.detection else None,
            "class_distribution": self.class_distribution(counts.class_counts),
            "confidence_histogram": stats.proportions(counts.confidence_counts),
            "confidence_quantiles": stats.quantiles(confidences, self.confidence_points) if confidences else {},
            "below_threshold_rate": self.below_threshold_rate(records),
            "boxes_per_image_histogram": stats.proportions(counts.boxes_per_image_counts) if self.detection else None,
            "median_elapsed_time": self.median_elapsed_time(records)
        }

    def bucket(self, bucket: Bucket) -> dict[str, Any]:
        """Full summary of one bucket, the compact form is derived from it when the result is assembled"""
        records = bucket.records
        confidences = self.confidences(records)
        counts = self.side_counts(records)

        image_specs: list[list[int]] = []
        for record in records:
            spec = record.image_spec
            if spec is not None and spec not in image_specs:
                image_specs.append(spec)

        summary: dict[str, Any] = {
            "start_date": bucket.start_date,
            "end_date": bucket.end_date,
            "window": bucket.window,
            "record_count": counts.record_count,
            "merged_from": bucket.merged_from,
            "class_distribution": self.class_distribution(counts.class_counts),
            "median_confidence": stats.median(confidences),
            "mean_confidence": stats.mean(confidences),
            "std_confidence": stats.std(confidences),
            "confidence_histogram": stats.proportions(counts.confidence_counts),
            "confidence_quantiles": stats.quantiles(confidences, self.confidence_points) if confidences else {},
            "below_threshold_rate": self.below_threshold_rate(records),
            "threshold_values": self.thresholds(records),
            "image_specs": sorted(image_specs),
            "median_elapsed_time": self.median_elapsed_time(records)
        }

        if self.detection:
            summary.update(self._detection_summary(records, counts))
        else:
            summary.update(self._classification_summary(records))

        return summary

    @staticmethod
    def _classification_summary(records: Sequence[Record]) -> dict[str, Any]:
        near = [record.near_threshold for record in records if record.near_threshold is not None]
        patch_width = [record.patch_width for record in records if record.patch_width is not None]
        patch_height = [record.patch_height for record in records if record.patch_height is not None]

        return {
            "decision_differs_rate": sum(r.decision_differs for r in records) / len(records) if records else None,
            "median_patch_width": stats.median(patch_width),
            "median_patch_height": stats.median(patch_height),
            "near_threshold_rate": sum(near) / len(near) if near else None
        }

    def _detection_summary(self, records: Sequence[Record], counts: SideCounts) -> dict[str, Any]:
        boxes = self.boxes(records)
        image_count = len(records)
        per_image = [float(record.box_count) for record in records]

        by_class: dict[str, float] = {}
        for name in self.classes + [box.prediction for box in boxes if box.prediction not in self.classes]:
            by_class[name] = sum(record.box_count_by_class.get(name, 0) for record in records) / image_count \
                if image_count else 0.0

        normalized_area = self.geometry(records, BoxGeometry.NORMALIZED_AREA)
        log_area = [math.log10(value) for value in normalized_area if value > 0]

        return {
            "box_count": counts.box_count,
            "mean_boxes_per_image": stats.mean(per_image),
            "std_boxes_per_image": stats.std(per_image),
            "boxes_per_image_histogram": stats.proportions(counts.boxes_per_image_counts),
            "no_box_rate": sum(record.has_no_boxes for record in records) / image_count if image_count else None,
            "boxes_by_class_per_image": by_class,
            "box": {
                "box_count": counts.box_count,
                "normalized_area_quantiles": self._quantiles3(normalized_area),
                "normalized_width_quantiles": self._quantiles3([box.normalized_width for box in boxes if box.normalized_width is not None]),
                "normalized_height_quantiles": self._quantiles3([box.normalized_height for box in boxes if box.normalized_height is not None]),
                "aspect_quantiles": self._quantiles3([box.aspect for box in boxes if box.aspect is not None]),
                "normalized_cx_quantiles": self._quantiles3(self.geometry(records, BoxGeometry.NORMALIZED_CX)),
                "normalized_cy_quantiles": self._quantiles3(self.geometry(records, BoxGeometry.NORMALIZED_CY)),
                "normalized_area_histogram": stats.proportions(stats.histogram(log_area, self.config.area_log_bin_edges))
                if log_area else None
            }
        }

    def _quantiles3(self, values: Sequence[float]) -> dict[str, float | None]:
        if values:
            return stats.quantiles(values, self.three_points)
        return {name: None for name in Quantiles.model_fields}

    # Series access

    def class_share_series(self) -> list[str]:
        return [f"{CLASS_SHARE_PREFIX}{name}" for name in self.classes]

    @staticmethod
    def series_value(summary: dict[str, Any], name: str) -> float | None:
        """Reads one scalar series value out of a full bucket summary"""
        if name.startswith(CLASS_SHARE_PREFIX):
            class_distribution = summary.get("class_distribution") or {}
            return class_distribution.get(name[len(CLASS_SHARE_PREFIX):]) if class_distribution else None

        value: Any = summary
        # Geometry series live in the box block as the median of their quantiles, everything else is a top level key
        measure = name.removeprefix("median_")
        if name != measure and measure in list(BoxGeometry):
            for key in ("box", f"{measure}_quantiles", "p50"):
                if not isinstance(value, dict):
                    return None
                value = value.get(key)
        else:
            value = summary.get(name)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @staticmethod
    def compact(summary: dict[str, Any]) -> dict[str, Any]:
        """The scalar series of a full bucket summary, exactly the fields of the compact schema"""
        return CompactBucketSummary.model_validate(summary).model_dump()
