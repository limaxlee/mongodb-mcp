"""Query building for the data drift analysis over the inspections collection"""
from typing import Any
from datetime import datetime
from pydantic.alias_generators import to_camel


class DriftQuery:
    """Document filter and aggregation pipeline that flatten the inspections of one model to one row per prediction

    The model string is matched exactly so that versions never blend. The filters are exact matches on metadata
    fields, given with their snake case names and translated to the camel case names of the documents.
    """

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
        self.filters = {key: value for key, value in (filters or {}).items() if value is not None}

    def match(self) -> dict[str, Any]:
        """Document level filter"""
        element: dict[str, Any] = {"aiModel": self.model}
        if self.task:
            element["task"] = self.task

        match: dict[str, Any] = {
            "isDeleted": False,
            "metadata.createdAt": {"$gte": self.start_date, "$lt": self.end_date},
            "inspectionResult.aiResults": {"$elemMatch": element}
        }
        for key, value in self.filters.items():
            match[f"metadata.{to_camel(key)}"] = value

        return match

    def pipeline(self) -> list[dict[str, Any]]:
        """Flattens matching documents to one row per prediction of the analysed model, in chronological order

        The sort happens before any unwind so the createdAt index carries it, and the unwinds preserve the order.
        Detections stay nested in their prediction so an image row arrives together with its boxes.
        """
        entry_match: dict[str, Any] = {"inspectionResult.aiResults.aiModel": self.model}
        if self.task:
            entry_match["inspectionResult.aiResults.task"] = self.task

        return [
            {"$match": self.match()},
            {"$sort": {"metadata.createdAt": 1}},
            {"$project": {
                "metadata": 1,
                "dataSpec": 1,
                "schemaVersion": 1,
                "inspectionResult.aiResults": 1
            }},
            {"$unwind": {"path": "$inspectionResult.aiResults", "includeArrayIndex": "entryIndex"}},
            {"$match": entry_match},
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
