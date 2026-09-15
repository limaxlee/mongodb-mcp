"""Turns flattened inspection rows into the flat records every later statistic is computed from

A row is one prediction of one matched aiResults entry, as produced by the aggregation pipeline of DriftQuery.
Extraction never raises on a malformed row, it counts the problem and moves on.
"""
import logging
from typing import Any

from common.config import SETTINGS
from common.constants import DriftTask, DriftWindow
from mongodb_mcp.utils import to_utc
from mongodb_mcp.schemas import Record, BoxRecord, DataQuality

logger = logging.getLogger(__name__)


class UnsupportedTaskError(ValueError):
    def __init__(self, task: Any):
        supported = ", ".join(item.value for item in DriftTask)
        super().__init__(f"Task {task!r} is not supported for drift analysis, supported tasks are {supported}")


class RecordExtractor:
    """Accumulates the records of both windows from flattened rows and tracks everything needed for data quality"""

    def __init__(self, task: str | None = None):
        self.requested_task = task
        self.records: list[Record] = []
        self.quality = DataQuality()
        self.warnings: list[str] = []
        self.tasks_seen: set[str] = set()
        self.classes_seen: list[str] = []
        self._warned: set[str] = set()
        self._matched_docs: set[str] = set()
        self._matched_entries: set[str] = set()

    def warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            self.warnings.append(message)

    def resolve_task(self, requested: str | None = None) -> str:
        """The task to analyse, either the requested one or the single task seen in the rows"""
        if requested is not None:
            if requested not in list(DriftTask):
                raise UnsupportedTaskError(requested)
            return requested
        if len(self.tasks_seen) > 1:
            raise ValueError(f"The model has multiple tasks {sorted(self.tasks_seen)}, specify task")
        return next(iter(self.tasks_seen)) if self.tasks_seen else DriftTask.CLASSIFICATION.value

    def add(self, row: dict[str, Any], window: DriftWindow = DriftWindow.CURRENT) -> Record | None:
        inspection_id = str(row.get("_id", ""))
        try:
            record = self._extract(row, inspection_id, window)
        except UnsupportedTaskError:
            raise
        except Exception as e:  # never let one bad row abort the analysis
            self.quality.parse_error_count += 1
            if len(self.quality.parse_error_examples) < SETTINGS.data_drift.max_parse_error_examples:
                self.quality.parse_error_examples.append(inspection_id)
            logger.debug(f"Skipped malformed inspection row {inspection_id}: {e}")
            return None

        if record is not None:
            self.records.append(record)
        return record

    def _extract(self, row: dict[str, Any], inspection_id: str, window: DriftWindow) -> Record | None:
        task = row.get("task")
        if task not in list(DriftTask):
            raise UnsupportedTaskError(task)
        if self.requested_task and task != self.requested_task:
            return None

        self.tasks_seen.add(task)
        self._matched_docs.add(inspection_id)
        self._matched_entries.add(f"{inspection_id}:{int(row.get('entryIndex', 0) or 0)}")
        self.quality.matched_document_count = len(self._matched_docs)
        self.quality.matched_entry_count = len(self._matched_entries)

        schema_version = row.get("schemaVersion")
        if schema_version not in (None, "1.0"):
            self.warn_once(f"Unknown inspection schema version {schema_version}, extracted on a best effort basis")

        classes = [str(name) for name in (row.get("classes") or [])]
        for name in classes:
            if name not in self.classes_seen:
                self.classes_seen.append(name)

        prediction = row.get("prediction") or {}
        created_at = to_utc(row.get("createdAt"))
        if created_at is None:
            raise ValueError("Missing createdAt")

        image_width, image_height, image_channels = self._image_spec(row.get("dataSpec"), prediction.get("fileIndex"))

        patch_spec = prediction.get("patchSpec") or {}
        is_patch = bool(prediction.get("isPatch", False))
        patch_width = patch_height = None
        if is_patch and all(key in patch_spec for key in ("x1", "x2", "y1", "y2")):
            patch_width = float(patch_spec["x2"]) - float(patch_spec["x1"])
            patch_height = float(patch_spec["y2"]) - float(patch_spec["y1"])

        fields: dict[str, Any] = {
            "window": window,
            "inspection_id": inspection_id,
            "prediction_id": self._as_int(prediction.get("predictionId")),
            "created_at": created_at,
            "gbm": row.get("gbm"),
            "process": row.get("process"),
            "location": row.get("location"),
            "equipment_id": row.get("equipmentId"),
            "product_id": row.get("productId"),
            "mode": row.get("mode"),
            "backend": row.get("backend"),
            "classes": classes,
            "threshold": self._as_float(prediction.get("threshold")),
            "elapsed_time": self._as_float(prediction.get("elapsedTime")),
            "is_patch": is_patch,
            "patch_width": patch_width,
            "patch_height": patch_height,
            "image_width": image_width,
            "image_height": image_height,
            "image_channels": image_channels
        }

        if task == DriftTask.CLASSIFICATION:
            fields.update(self._classification_fields(prediction, classes, fields["threshold"]))
        else:
            fields.update(self._detection_fields(prediction, classes, fields["threshold"], image_width, image_height))

        return Record(**fields)

    def _classification_fields(
            self,
            prediction: dict[str, Any],
            classes: list[str],
            threshold: float | None
    ) -> dict[str, Any]:
        label = str(prediction.get("prediction"))
        decision = prediction.get("decision")
        if label not in classes:
            self.warn_once(f"Prediction class {label!r} is not in the model classes")

        max_confidence = self.normalize_confidence(prediction.get("confidence"))
        if max_confidence is None:
            self.quality.missing_confidence_count += 1

        below_threshold = near_threshold = None
        if threshold is not None and max_confidence is not None:
            below_threshold = max_confidence < threshold
            near_threshold = abs(max_confidence - threshold) < SETTINGS.data_drift.near_threshold_margin

        return {
            "prediction": label,
            "decision": decision,
            "max_confidence": max_confidence,
            "below_threshold": below_threshold,
            "near_threshold": near_threshold,
            "decision_differs": decision is not None and decision != label
        }

    def _detection_fields(
            self,
            prediction: dict[str, Any],
            classes: list[str],
            threshold: float | None,
            image_width: int | None,
            image_height: int | None
    ) -> dict[str, Any]:
        boxes: list[BoxRecord] = []
        for detection in prediction.get("detections") or []:
            box = self._extract_box(detection, classes, image_width, image_height)
            if box is not None:
                boxes.append(box)

        box_count_by_class = {name: 0 for name in classes}
        for box in boxes:
            box_count_by_class[box.prediction] = box_count_by_class.get(box.prediction, 0) + 1

        confidences = [box.max_confidence for box in boxes if box.max_confidence is not None]
        # A detection threshold usually lives on the boxes, the record keeps the first one seen for hard break scanning
        if threshold is None:
            thresholds = [box.threshold for box in boxes if box.threshold is not None]
            threshold = thresholds[0] if thresholds else None

        return {
            "threshold": threshold,
            "boxes": boxes,
            "box_count": len(boxes),
            "box_count_by_class": box_count_by_class,
            "below_threshold_count": sum(1 for box in boxes if box.below_threshold),
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "min_confidence": min(confidences) if confidences else None
        }

    def _extract_box(
            self,
            detection: dict[str, Any],
            classes: list[str],
            image_width: int | None,
            image_height: int | None
    ) -> BoxRecord | None:
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            self.quality.parse_error_count += 1
            return None

        x1, y1, x2, y2 = (float(value) for value in bbox)
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect = width / height if height != 0 else None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        normalized_width = normalized_height = normalized_area = normalized_cx = normalized_cy = None
        if image_width and image_height:
            normalized_width, normalized_height = width / image_width, height / image_height
            normalized_area = area / (image_width * image_height)
            normalized_cx, normalized_cy = cx / image_width, cy / image_height

        label = str(detection.get("prediction"))
        if label not in classes:
            self.warn_once(f"Prediction class {label!r} is not in the model classes")

        max_confidence = self.normalize_confidence(detection.get("confidence"))
        if max_confidence is None:
            self.quality.missing_confidence_count += 1
        threshold = self._as_float(detection.get("threshold"))
        below_threshold = max_confidence < threshold if threshold is not None and max_confidence is not None else None

        return BoxRecord(
            bbox_id=self._as_int(detection.get("bboxId")),
            prediction=label,
            max_confidence=max_confidence,
            threshold=threshold,
            below_threshold=below_threshold,
            x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height, area=area, aspect=aspect, cx=cx, cy=cy,
            normalized_width=normalized_width, normalized_height=normalized_height, normalized_area=normalized_area, normalized_cx=normalized_cx, normalized_cy=normalized_cy
        )

    def _image_spec(self, data_spec: Any, file_index: Any) -> tuple[int | None, int | None, int | None]:
        index = self._as_int(file_index)
        if not isinstance(data_spec, list) or index is None or index < 0 or index >= len(data_spec):
            self.quality.missing_image_spec_count += 1
            return None, None, None

        spec = data_spec[index] or {}
        width, height = self._as_int(spec.get("width")), self._as_int(spec.get("height"))
        if width is None or height is None:
            self.quality.missing_image_spec_count += 1
            return None, None, None

        return width, height, self._as_int(spec.get("channels"))

    @staticmethod
    def normalize_confidence(confidence: Any) -> float | None:
        """A list of confidences collapses to its maximum, the other values carry no information and are ignored"""
        if isinstance(confidence, (list, tuple)):
            values = [float(value) for value in confidence if isinstance(value, (int, float)) and not isinstance(value, bool)]
            return max(values) if values else None

        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            return float(confidence)

        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)
