from datetime import timedelta

from mongodb_mcp.drift import DriftQuery
from tests.drift.synthetic import START

END = START + timedelta(days=7)


class TestDriftQuery:
    def test_match_filters(self):
        query = DriftQuery(
            "MetalCls/1.0", START, END, task="cls", filters={"gbm": "SEV", "equipment_id": "EQ-01", "mode": "production"}
        )
        assert query.match() == {
            "isDeleted": False,
            "metadata.createdAt": {"$gte": START, "$lt": END},
            "inspectionResult.aiResults": {"$elemMatch": {"aiModel": "MetalCls/1.0", "task": "cls"}},
            "metadata.gbm": "SEV",
            "metadata.equipmentId": "EQ-01",
            "metadata.mode": "production"
        }

    def test_match_without_optional_filters(self):
        match = DriftQuery("MetalCls/1.0", START, END, filters={"gbm": None, "mode": None}).match()
        assert set(match) == {"isDeleted", "metadata.createdAt", "inspectionResult.aiResults"}
        assert match["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0"}}

        match = DriftQuery("MetalCls/1.0", START, END).match()
        assert set(match) == {"isDeleted", "metadata.createdAt", "inspectionResult.aiResults"}

    def test_pipeline_shape(self):
        query = DriftQuery("MetalCls/1.0", START, END, task="cls")
        pipeline = query.pipeline()
        stages = [next(iter(stage)) for stage in pipeline]

        assert stages == ["$match", "$sort", "$project", "$unwind", "$match", "$unwind", "$project", "$unset"]
        assert pipeline[0]["$match"] == query.match()
        assert pipeline[1]["$sort"] == {"metadata.createdAt": 1}
        assert pipeline[3]["$unwind"]["includeArrayIndex"] == "entryIndex"
        assert pipeline[4]["$match"] == {
            "inspectionResult.aiResults.aiModel": "MetalCls/1.0", "inspectionResult.aiResults.task": "cls"
        }
        final = pipeline[6]["$project"]
        assert final["prediction"] == "$inspectionResult.aiResults.predictions"
        assert final["createdAt"] == "$metadata.createdAt"
        assert "dataSpec" in final and "classes" in final and "backend" in final
        assert pipeline[7]["$unset"] == ["prediction.feedbacks", "prediction.detections.feedbacks"]

    def test_pipeline_without_task(self):
        pipeline = DriftQuery("MetalCls/1.0", START, END).pipeline()
        assert pipeline[4]["$match"] == {"inspectionResult.aiResults.aiModel": "MetalCls/1.0"}
