"""Turns flattened inspection rows into the flat records every later statistic is computed from

A row is one prediction of one matched aiResults entry, as produced by the aggregation pipeline in the connector.
Extraction never raises on a malformed row, it counts the problem and moves on.
"""
import math
import logging
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from common.constants import ModelTasks
from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = (ModelTasks.CLASSIFICATION.value, ModelTasks.DETECTION.value)


def normalize_confidence(conf: Any) -> tuple[float | None, float | None, float | None]:
    """Returns (conf_max, margin, entropy); a list of confidences collapses to its maximum"""
    if isinstance(conf, (list, tuple)):
        values = [float(value) for value in conf if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not values:
            return None, None, None

        ordered = sorted(values, reverse=True)
        conf_max = ordered[0]
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else None

        probabilities = [max(value, 1e-12) for value in values]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        entropy = -sum(value * math.log(value) for value in probabilities)
        return conf_max, margin, entropy

    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return float(conf), None, None

    return None, None, None


@dataclass(slots=True)
class BoxRecord:
    bbox_id: int | None
    prediction: str
    conf_max: float | None
    threshold: float | None
    below_threshold: bool | None
    x1: float
    y1: float
    x2: float
    y2: float
    w: float
    h: float
    area: float
    aspect: float | None
    cx: float
    cy: float
    w_norm: float | None
    h_norm: float | None
    area_norm: float | None
    cx_norm: float | None
    cy_norm: float | None
    feedback_label: str | None
    feedback_mismatch: bool | None
    feedback_other: bool


@dataclass(slots=True)
class Record:
    """One prediction of the analysed model; detection records also carry their boxes"""
    inspection_id: str
    prediction_id: int | None
    created_at: datetime
    local_created_at: datetime
    gbm: str | None
    process: str | None
    location: str | None
    equipment_id: str | None
    product_id: str | None
    mode: str | None
    backend: str | None
    classes: tuple[str, ...]
    threshold: float | None
    elapsed_time: float | None
    is_patch: bool
    patch_w: float | None
    patch_h: float | None
    image_w: int | None
    image_h: int | None
    image_c: int | None
    conclusion: str | None
    feedback_label: str | None
    feedback_mismatch: bool | None
    feedback_other: bool
    # Classification
    prediction: str | None = None
    decision: str | None = None
    conf_max: float | None = None
    margin: float | None = None
    entropy: float | None = None
    below_threshold: bool | None = None
    decision_differs: bool = False
    near_threshold: bool | None = None
    # Detection
    boxes: list[BoxRecord] = field(default_factory=list)
    n_boxes: int = 0
    n_boxes_by_class: dict[str, int] = field(default_factory=dict)
    n_below_threshold: int = 0
    mean_conf: float | None = None
    min_conf: float | None = None

    @property
    def has_no_boxes(self) -> bool:
        return self.n_boxes == 0

    @property
    def image_spec(self) -> tuple[int, int, int] | None:
        if self.image_w is None or self.image_h is None:
            return None
        return self.image_w, self.image_h, self.image_c if self.image_c is not None else 0


@dataclass
class DataQualityCounters:
    n_docs_scanned: int = 0
    n_missing_confidence: int = 0
    n_missing_image_spec: int = 0
    n_parse_errors: int = 0
    parse_error_examples: list[str] = field(default_factory=list)
    matched_docs: set[str] = field(default_factory=set)
    matched_entries: set[tuple[str, int]] = field(default_factory=set)

    def as_dict(self, n_records: int, n_boxes: int | None) -> dict[str, Any]:
        return {
            "n_docs_scanned": self.n_docs_scanned,
            "n_docs_matched": len(self.matched_docs),
            "n_entries_matched": len(self.matched_entries),
            "n_records": n_records,
            "n_boxes": n_boxes,
            "n_missing_confidence": self.n_missing_confidence,
            "n_missing_image_spec": self.n_missing_image_spec,
            "n_parse_errors": self.n_parse_errors,
            "parse_error_examples": list(self.parse_error_examples)
        }


class RecordExtractor:
    """Accumulates records from flattened rows and tracks everything needed for data quality reporting"""

    def __init__(self, task: str | None = None, config: DriftConfig = DEFAULT_CONFIG):
        self.config = config
        self.requested_task = task
        self.records: list[Record] = []
        self.quality = DataQualityCounters()
        self.warnings: list[str] = []
        self.tasks_seen: set[str] = set()
        self.classes_seen: list[str] = []
        self.timezones_seen: set[str] = set()
        self._warned: set[str] = set()
        self._tz_cache: dict[str, ZoneInfo | None] = {}

    def warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            self.warnings.append(message)

    def add(self, row: dict[str, Any]) -> Record | None:
        inspection_id = str(row.get("_id", ""))
        try:
            record = self._extract(row, inspection_id)
        except UnsupportedTaskError:
            raise
        except Exception as e:  # never let one bad row abort the analysis
            self.quality.n_parse_errors += 1
            if len(self.quality.parse_error_examples) < self.config.max_parse_error_examples:
                self.quality.parse_error_examples.append(inspection_id)
            logger.debug(f"Skipped malformed inspection row {inspection_id}: {e}")
            return None

        if record is not None:
            self.records.append(record)
        return record

    def _extract(self, row: dict[str, Any], inspection_id: str) -> Record | None:
        task = row.get("task")
        if task not in SUPPORTED_TASKS:
            raise UnsupportedTaskError(f"Task {task!r} is not supported for drift analysis, only cls and det are")
        if self.requested_task and task != self.requested_task:
            return None

        self.tasks_seen.add(task)
        self.quality.matched_docs.add(inspection_id)
        self.quality.matched_entries.add((inspection_id, int(row.get("entryIndex", 0) or 0)))

        schema_version = row.get("schemaVersion")
        if schema_version not in (None, "1.0"):
            self.warn_once(f"Unknown inspection schema version {schema_version}, extracted on a best effort basis")

        classes = tuple(str(name) for name in (row.get("classes") or []))
        for name in classes:
            if name not in self.classes_seen:
                self.classes_seen.append(name)

        prediction = row.get("prediction") or {}
        created_at = _as_utc(row.get("createdAt"))
        if created_at is None:
            raise ValueError("Missing createdAt")
        local_created_at = self._to_local(created_at, row.get("localTimezone"))

        file_index = prediction.get("fileIndex")
        image_w, image_h, image_c = self._image_spec(row.get("dataSpec"), file_index)

        patch_spec = prediction.get("patchSpec") or {}
        is_patch = bool(prediction.get("isPatch", False))
        patch_w = patch_h = None
        if is_patch and all(key in patch_spec for key in ("x1", "x2", "y1", "y2")):
            patch_w = float(patch_spec["x2"]) - float(patch_spec["x1"])
            patch_h = float(patch_spec["y2"]) - float(patch_spec["y1"])

        feedback_label, feedback_mismatch, feedback_other = _resolve_feedback(
            prediction.get("feedbacks"), classes, prediction.get("prediction")
        )

        record = Record(
            inspection_id=inspection_id,
            prediction_id=_as_int(prediction.get("predictionId")),
            created_at=created_at,
            local_created_at=local_created_at,
            gbm=row.get("gbm"),
            process=row.get("process"),
            location=row.get("location"),
            equipment_id=row.get("equipmentId"),
            product_id=row.get("productId"),
            mode=row.get("mode"),
            backend=row.get("backend"),
            classes=classes,
            threshold=_as_float(prediction.get("threshold")),
            elapsed_time=_as_float(prediction.get("elapsedTime")),
            is_patch=is_patch,
            patch_w=patch_w,
            patch_h=patch_h,
            image_w=image_w,
            image_h=image_h,
            image_c=image_c,
            conclusion=row.get("conclusion"),
            feedback_label=feedback_label,
            feedback_mismatch=feedback_mismatch,
            feedback_other=feedback_other
        )

        if task == ModelTasks.CLASSIFICATION.value:
            self._fill_classification(record, prediction, classes)
        else:
            self._fill_detection(record, prediction, classes)

        return record

    def _fill_classification(self, record: Record, prediction: dict[str, Any], classes: tuple[str, ...]) -> None:
        record.prediction = str(prediction.get("prediction"))
        record.decision = prediction.get("decision")
        if record.prediction not in classes:
            self.warn_once(f"Prediction class {record.prediction!r} is not in the model classes")

        conf_max, margin, entropy = normalize_confidence(prediction.get("confidence"))
        if conf_max is None:
            self.quality.n_missing_confidence += 1
        record.conf_max, record.margin, record.entropy = conf_max, margin, entropy

        if record.threshold is not None and conf_max is not None:
            record.below_threshold = conf_max < record.threshold
            record.near_threshold = abs(conf_max - record.threshold) < self.config.near_threshold_margin
        record.decision_differs = record.decision is not None and record.decision != record.prediction

    def _fill_detection(self, record: Record, prediction: dict[str, Any], classes: tuple[str, ...]) -> None:
        # A detection threshold lives on the boxes, the record keeps the first one seen for hard break scanning
        boxes: list[BoxRecord] = []
        for detection in prediction.get("detections") or []:
            box = self._extract_box(detection, record, classes)
            if box is not None:
                boxes.append(box)
                if record.threshold is None:
                    record.threshold = box.threshold

        record.boxes = boxes
        record.n_boxes = len(boxes)
        record.n_boxes_by_class = {name: 0 for name in classes}
        for box in boxes:
            record.n_boxes_by_class[box.prediction] = record.n_boxes_by_class.get(box.prediction, 0) + 1
        record.n_below_threshold = sum(1 for box in boxes if box.below_threshold)

        confidences = [box.conf_max for box in boxes if box.conf_max is not None]
        if confidences:
            record.mean_conf = sum(confidences) / len(confidences)
            record.min_conf = min(confidences)

    def _extract_box(self, detection: dict[str, Any], record: Record, classes: tuple[str, ...]) -> BoxRecord | None:
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            self.quality.n_parse_errors += 1
            return None

        x1, y1, x2, y2 = (float(value) for value in bbox)
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect = w / h if h != 0 else None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w_norm = h_norm = area_norm = cx_norm = cy_norm = None
        if record.image_w and record.image_h:
            w_norm, h_norm = w / record.image_w, h / record.image_h
            area_norm = area / (record.image_w * record.image_h)
            cx_norm, cy_norm = cx / record.image_w, cy / record.image_h

        prediction = str(detection.get("prediction"))
        if prediction not in classes:
            self.warn_once(f"Prediction class {prediction!r} is not in the model classes")

        conf_max, _, _ = normalize_confidence(detection.get("confidence"))
        if conf_max is None:
            self.quality.n_missing_confidence += 1
        threshold = _as_float(detection.get("threshold"))
        below_threshold = conf_max < threshold if threshold is not None and conf_max is not None else None

        feedback_label, feedback_mismatch, feedback_other = _resolve_feedback(
            detection.get("feedbacks"), classes, prediction
        )

        return BoxRecord(
            bbox_id=_as_int(detection.get("bboxId")),
            prediction=prediction,
            conf_max=conf_max,
            threshold=threshold,
            below_threshold=below_threshold,
            x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, area=area, aspect=aspect, cx=cx, cy=cy,
            w_norm=w_norm, h_norm=h_norm, area_norm=area_norm, cx_norm=cx_norm, cy_norm=cy_norm,
            feedback_label=feedback_label,
            feedback_mismatch=feedback_mismatch,
            feedback_other=feedback_other
        )

    def _image_spec(self, data_spec: Any, file_index: Any) -> tuple[int | None, int | None, int | None]:
        index = _as_int(file_index)
        if not isinstance(data_spec, list) or index is None or index < 0 or index >= len(data_spec):
            self.quality.n_missing_image_spec += 1
            return None, None, None

        spec = data_spec[index] or {}
        width, height = _as_int(spec.get("width")), _as_int(spec.get("height"))
        if width is None or height is None:
            self.quality.n_missing_image_spec += 1
            return None, None, None

        return width, height, _as_int(spec.get("channels"))

    def _to_local(self, created_at: datetime, timezone_name: Any) -> datetime:
        if not timezone_name:
            self.warn_once("Some documents have no local timezone, UTC was used for their bucket boundaries")
            return created_at

        if timezone_name not in self._tz_cache:
            try:
                self._tz_cache[timezone_name] = ZoneInfo(str(timezone_name))
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                self._tz_cache[timezone_name] = None
                self.warn_once(f"Unknown timezone {timezone_name!r}, UTC was used for its bucket boundaries")

        tz = self._tz_cache[timezone_name]
        if tz is None:
            return created_at

        self.timezones_seen.add(str(timezone_name))
        return created_at.astimezone(tz)


class UnsupportedTaskError(ValueError):
    pass


def _resolve_feedback(
        feedbacks: Any,
        classes: tuple[str, ...],
        prediction: Any
) -> tuple[str | None, bool | None, bool]:
    """Most recent feedback wins; a feedback text equal to a class name is a corrected label, anything else a comment"""
    if not isinstance(feedbacks, list) or not feedbacks:
        return None, None, False

    def registered_at(item: dict[str, Any]) -> datetime:
        return _as_utc(item.get("registeredAt")) or datetime.min.replace(tzinfo=timezone.utc)

    entries = [item for item in feedbacks if isinstance(item, dict)]
    if not entries:
        return None, None, False

    latest = max(entries, key=registered_at)
    text = str(latest.get("feedback", "")).strip()
    by_folded = {name.casefold(): name for name in classes}
    label = by_folded.get(text.casefold())
    if label is None:
        return None, None, True

    return label, label != str(prediction), False


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
