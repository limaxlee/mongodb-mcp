import logging
from typing import Any

from common.config import SETTINGS
from common.constants import DriftWindowMode, ModelTasks
from mongodb_mcp.utils import to_utc
from mongodb_mcp.schemas import Record, BoxRecord, DataQuality

logger = logging.getLogger(__name__)


class RecordExtractor:
    def __init__(self, task: str | None = None):
        self.target_task = task
        self.records = []
        self.quality = DataQuality()
        self.classes = []
        self._matched_docs = set()
        self._matched_entries = set()
        self._seen_predictions = set()

    def get_task(self) -> str:
        return self.target_task

    def add_record(self, item: dict[str, Any], window: DriftWindowMode = DriftWindowMode.CURRENT):
        inspection_id = str(item.get("_id", ""))
        try:
            record = self._extract(item, inspection_id, window)
            if record is not None:
                self.records.append(record)
        except Exception as e:
            self.quality.parse_error_count += 1
            if len(self.quality.parse_error_examples) < SETTINGS.data_drift.max_parse_error_examples:
                self.quality.parse_error_examples.append(inspection_id)
            logger.warning(f"Skipped malformed inspection row {inspection_id}: {e}")
            return

    def _extract(self, item: dict[str, Any], inspection_id: str, window: DriftWindowMode) -> Record | None:
        task = item.get("task")
        if not self.target_task:
            self.target_task = task
        if self.target_task and task != self.target_task:
            return None

        prediction = item.get("prediction") or {}
        entry_index = int(item.get("entryIndex", 0) or 0)
        prediction_key = f"{inspection_id}:{entry_index}:{prediction.get('predictionId')}"
        if prediction_key in self._seen_predictions:
            return None
        self._seen_predictions.add(prediction_key)

        self._matched_docs.add(inspection_id)
        self._matched_entries.add(f"{inspection_id}:{entry_index}")
        self.quality.matched_document_count = len(self._matched_docs)
        self.quality.matched_entry_count = len(self._matched_entries)

        row_classes = [str(name) for name in (item.get("classes") or [])]
        for name in row_classes:
            if name not in self.classes:
                self.classes.append(name)

        created_at = to_utc(item.get("createdAt"))

        image_width, image_height, image_channels = self._image_spec(item.get("dataSpec"), prediction.get("fileIndex"))

        fields = {
            "window": window,
            "inspection_id": inspection_id,
            "prediction_id": self._as_int(prediction.get("predictionId")),
            "created_at": created_at,
            "gbm": item.get("gbm"),
            "process": item.get("process"),
            "location": item.get("location"),
            "equipment_id": item.get("equipmentId"),
            "product_id": item.get("productId"),
            "mode": item.get("mode"),
            "backend": item.get("backend"),
            "classes": row_classes,
            "threshold": self._as_float(prediction.get("threshold")),
            "elapsed_time": self._as_float(prediction.get("elapsedTime")),
            "image_width": image_width,
            "image_height": image_height,
            "image_channels": image_channels
        }

        if task == ModelTasks.CLASSIFICATION:
            fields.update(self._get_cls_field(prediction, fields["threshold"]))
        else:
            fields.update(self._get_det_field(prediction, row_classes, fields["threshold"], image_width, image_height))

        return Record(**fields)

    def _get_cls_field(
            self,
            prediction: dict[str, Any],
            threshold: float | None
    ) -> dict[str, Any]:
        label = prediction.get("prediction")

        max_confidence = self.get_confidence(prediction.get("confidence"))
        if max_confidence is None:
            self.quality.missing_confidence_count += 1

        below_threshold = near_threshold = None
        if threshold is not None and max_confidence is not None:
            below_threshold = max_confidence < threshold
            near_threshold = abs(max_confidence - threshold) < SETTINGS.data_drift.near_threshold_margin

        return {
            "prediction": label,
            "max_confidence": max_confidence,
            "below_threshold": below_threshold,
            "near_threshold": near_threshold
        }

    def _get_det_field(
            self,
            prediction: dict[str, Any],
            classes: list[str],
            threshold: float | None,
            image_width: int | None,
            image_height: int | None
    ) -> dict[str, Any]:
        boxes = []
        for detection in prediction.get("detections", []):
            box = self._extract_box(detection, image_width, image_height)
            if box is not None:
                boxes.append(box)

        box_count_by_class = {name: 0 for name in classes}
        for box in boxes:
            box_count_by_class[box.prediction] = box_count_by_class.get(box.prediction, 0) + 1

        confidences = [box.max_confidence for box in boxes if box.max_confidence is not None]
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

        max_confidence = self.get_confidence(detection.get("confidence"))
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

    def get_confidence(self, confidence: Any) -> float | None:
        """A list of confidences collapses to its maximum, anything that is not a number becomes None"""
        if isinstance(confidence, list):
            values = [self._as_float(value) for value in confidence]
            values = [value for value in values if value is not None]
            return max(values) if values else None

        return self._as_float(confidence)

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
