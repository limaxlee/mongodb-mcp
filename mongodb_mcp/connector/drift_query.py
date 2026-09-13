"""Query building for the data drift analysis over the inspections collection"""
from typing import Any
from datetime import datetime


def build_drift_match(
        model: str,
        start_date: datetime,
        end_date: datetime,
        task: str | None = None,
        gbm: str | None = None,
        process: str | None = None,
        location: str | None = None,
        equipment_id: str | None = None,
        mode: str | None = None
) -> dict[str, Any]:
    """Document level filter; the model string is matched exactly so that versions never blend"""
    element: dict[str, Any] = {"aiModel": model}
    if task:
        element["task"] = task

    match: dict[str, Any] = {
        "isDeleted": False,
        "metadata.createdAt": {"$gte": start_date, "$lt": end_date},
        "inspectionResult.aiResults": {"$elemMatch": element}
    }
    for field, value in (
            ("gbm", gbm), ("process", process), ("location", location), ("equipmentId", equipment_id), ("mode", mode)
    ):
        if value is not None:
            match[f"metadata.{field}"] = value

    return match


def build_drift_pipeline(match: dict[str, Any], model: str, task: str | None = None) -> list[dict[str, Any]]:
    """Flattens matching documents to one row per prediction of the analysed model, in chronological order

    The sort happens before any unwind so the createdAt index carries it, and the unwinds preserve the order.
    Detections stay nested in their prediction so an image row arrives together with its boxes.
    """
    entry_match: dict[str, Any] = {"inspectionResult.aiResults.aiModel": model}
    if task:
        entry_match["inspectionResult.aiResults.task"] = task

    return [
        {"$match": match},
        {"$sort": {"metadata.createdAt": 1}},
        {"$project": {
            "metadata": 1,
            "dataSpec": 1,
            "schemaVersion": 1,
            "inspectionResult.conclusion": 1,
            "inspectionResult.aiResults": 1
        }},
        {"$unwind": {"path": "$inspectionResult.aiResults", "includeArrayIndex": "entryIndex"}},
        {"$match": entry_match},
        {"$unwind": "$inspectionResult.aiResults.predictions"},
        {"$project": {
            "_id": 1,
            "createdAt": "$metadata.createdAt",
            "localTimezone": "$metadata.localTimezone",
            "gbm": "$metadata.gbm",
            "process": "$metadata.process",
            "location": "$metadata.location",
            "equipmentId": "$metadata.equipmentId",
            "productId": "$metadata.productId",
            "mode": "$metadata.mode",
            "conclusion": "$inspectionResult.conclusion",
            "schemaVersion": 1,
            "dataSpec": 1,
            "entryIndex": 1,
            "backend": "$inspectionResult.aiResults.backend",
            "aiModel": "$inspectionResult.aiResults.aiModel",
            "task": "$inspectionResult.aiResults.task",
            "classes": "$inspectionResult.aiResults.classes",
            "prediction": "$inspectionResult.aiResults.predictions"
        }}
    ]
