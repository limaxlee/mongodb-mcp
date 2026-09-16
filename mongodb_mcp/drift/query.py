from datetime import datetime
from typing import Any

from pydantic.alias_generators import to_camel

from common.constants import ModelTasks


class DriftQueryBuilder:
    def __init__(
            self,
            model: str,
            start_date: datetime,
            end_date: datetime,
            task: str | None = None,
            filters: dict[str, str | None] | None = None
    ):
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.task = task
        self.filters = {key: item for key, item in (filters or {}).items() if item is not None}

    def build(self) -> dict[str, Any]:
        element = {
            "aiModel": self.model,
            "task": {"$in": [ModelTasks.CLASSIFICATION.value, ModelTasks.DETECTION.value]}
        }
        if self.task:
            element["task"] = self.task

        query = {
            "isDeleted": False,
            "metadata.createdAt": {"$gte": self.start_date, "$lte": self.end_date},
            "inspectionResult.aiResults": {"$elemMatch": element}
        }
        for key, item in self.filters.items():
            query[f"metadata.{to_camel(key)}"] = item

        return query

    def build_pipeline(self) -> list[dict[str, Any]]:
        pipeline = {"inspectionResult.aiResults.aiModel": self.model}
        if self.task:
            pipeline["inspectionResult.aiResults.task"] = self.task

        return [
            {"$match": self.build()},
            {"$sort": {"metadata.createdAt": 1}},
            {"$project": {
                "metadata": 1,
                "dataSpec": 1,
                "schemaVersion": 1,
                "inspectionResult.aiResults": 1
            }},
            {"$unwind": {"path": "$inspectionResult.aiResults", "includeArrayIndex": "entryIndex"}},
            {"$match": pipeline},
            {"$unwind": "$inspectionResult.aiResults.predictions"},
            {"$project": {
                "_id": 1,
                "createdAt": "$metadata.createdAt",
                "gbm": "$metadata.gbm",
                "process": "$metadata.process",
                "location": "$metadata.location",
                "equipmentId": "$metadata.equipmentId",
                "productId": "$metadata.productId",
                "mode": "$metadata.mode",
                "schemaVersion": 1,
                "dataSpec": 1,
                "entryIndex": 1,
                "backend": "$inspectionResult.aiResults.backend",
                "aiModel": "$inspectionResult.aiResults.aiModel",
                "task": "$inspectionResult.aiResults.task",
                "classes": "$inspectionResult.aiResults.classes",
                "prediction": "$inspectionResult.aiResults.predictions"
            }},
            {"$unset": ["prediction.feedbacks", "prediction.detections.feedbacks"]}
        ]
