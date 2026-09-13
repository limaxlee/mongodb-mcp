"""Synthetic flattened inspection rows, shaped like the output of the connector's aggregation pipeline"""
import random
from typing import Any, Callable
from datetime import datetime, timedelta, timezone

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
CLASSES = ["Good", "NG"]


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def make_row(
        index: int,
        created_at: datetime,
        task: str,
        prediction: dict[str, Any],
        classes: list[str] = CLASSES,
        image: tuple[int, int, int] = (400, 400, 3),
        backend: str = "ts",
        entry_index: int = 0,
        timezone_name: str | None = "Asia/Seoul"
) -> dict[str, Any]:
    return {
        "_id": f"{index:024x}",
        "createdAt": created_at,
        "localTimezone": timezone_name,
        "gbm": "SEV",
        "process": "SMD",
        "location": "Line_01",
        "equipmentId": "Metal_Inspector_01",
        "productId": None,
        "mode": "production",
        "conclusion": "NG" if prediction.get("prediction") == "NG" else "Good",
        "schemaVersion": "1.0",
        "dataSpec": [{"type": "image", "width": image[0], "height": image[1], "channels": image[2]}],
        "entryIndex": entry_index,
        "backend": backend,
        "aiModel": f"Metal{'Det' if task == 'det' else 'Cls'}/1.0",
        "task": task,
        "classes": classes,
        "prediction": prediction
    }


def cls_prediction(
        rng: random.Random,
        conf_mean: float,
        ng_rate: float,
        threshold: float | None = 0.8,
        as_list: bool = True,
        feedback: str | None = None
) -> dict[str, Any]:
    label = "NG" if rng.random() < ng_rate else "Good"
    conf = _clip(rng.gauss(conf_mean, 0.05))
    return {
        "predictionId": 3,
        "fileIndex": 0,
        "elapsedTime": 0.008,
        "isPatch": True,
        "patchSpec": {"x1": 100.0, "y1": 50.0, "x2": 166.0, "y2": 90.0, "channels": 3},
        "threshold": threshold,
        "prediction": label,
        "decision": label,
        "confidence": [conf, 1 - conf] if as_list else conf,
        "feedbacks": [{"feedback": feedback, "registeredAt": START}] if feedback else []
    }


def det_prediction(
        rng: random.Random,
        conf_mean: float,
        ng_rate: float,
        boxes_mean: float = 2.0,
        threshold: float | None = 0.8,
        box_size: float = 60.0,
        center: tuple[float, float] = (200.0, 200.0),
        no_box_rate: float = 0.02,
        list_confidence: bool = False
) -> dict[str, Any]:
    n_boxes = 0 if rng.random() < no_box_rate else max(1, round(rng.gauss(boxes_mean, 0.7)))
    detections = []
    for bbox_id in range(n_boxes):
        conf = _clip(rng.gauss(conf_mean, 0.05))
        w, h = abs(rng.gauss(box_size, 8.0)), abs(rng.gauss(box_size * 0.6, 5.0))
        cx, cy = rng.gauss(center[0], 20.0), rng.gauss(center[1], 20.0)
        detections.append({
            "bboxId": bbox_id,
            "prediction": "NG" if rng.random() < ng_rate else "Good",
            "confidence": [conf, _clip(conf - 0.3)] if list_confidence else conf,
            "threshold": threshold,
            "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            "childIds": [],
            "feedbacks": []
        })

    return {
        "predictionId": 2,
        "fileIndex": 0,
        "elapsedTime": 0.008,
        "isPatch": False,
        "patchSpec": {},
        "detections": detections
    }


def generate_rows(
        task: str,
        days: int,
        per_day: int,
        prediction_for_day: Callable[[random.Random, int], dict[str, Any]],
        row_kwargs_for_day: Callable[[int], dict[str, Any]] | None = None,
        seed: int = 7,
        start: datetime = START
) -> list[dict[str, Any]]:
    """One row per prediction spread evenly over each day, `prediction_for_day(rng, day)` shapes the prediction"""
    rng = random.Random(seed)
    rows = []
    index = 0
    for day in range(days):
        extra = row_kwargs_for_day(day) if row_kwargs_for_day else {}
        for slot in range(per_day):
            created_at = start + timedelta(days=day, seconds=slot * 86400 / per_day)
            rows.append(make_row(index, created_at, task, prediction_for_day(rng, day), **extra))
            index += 1
    return rows
