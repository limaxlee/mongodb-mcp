from datetime import timedelta

from mongodb_mcp.drift import DriftQueryBuilder
from tests.drift.synthetic import START

END = START + timedelta(days=7)
ANY_TASK = {"$in": ["cls", "det"]}


class TestDriftQueryBuilder:
    def test_build_filters(self):
        builder = DriftQueryBuilder(
            "MetalCls/1.0", START, END, task="cls", filters={"gbm": "SEV", "equipment_id": "EQ-01", "mode": "production"}
        )
        assert builder.build() == {
            "isDeleted": False,
            "metadata.createdAt": {"$gte": START, "$lte": END},
            "inspectionResult.aiResults": {"$elemMatch": {"aiModel": "MetalCls/1.0", "task": "cls"}},
            "metadata.gbm": "SEV",
            "metadata.equipmentId": "EQ-01",
            "metadata.mode": "production"
        }

    def test_build_without_optional_filters(self):
        query = DriftQueryBuilder("MetalCls/1.0", START, END, filters={"gbm": None, "mode": None}).build()
        assert set(query) == {"isDeleted", "metadata.createdAt", "inspectionResult.aiResults"}
        assert query["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0", "task": ANY_TASK}}

        query = DriftQueryBuilder("MetalCls/1.0", START, END).build()
        assert set(query) == {"isDeleted", "metadata.createdAt", "inspectionResult.aiResults"}

    def test_both_boundaries_are_inclusive(self):
        query = DriftQueryBuilder("MetalCls/1.0", START, END).build()
        assert query["metadata.createdAt"] == {"$gte": START, "$lte": END}

    def test_pipeline_shape(self):
        builder = DriftQueryBuilder("MetalCls/1.0", START, END, task="cls")
        pipeline = builder.build_pipeline()
        stages = [next(iter(stage)) for stage in pipeline]

        assert stages == ["$match", "$sort", "$project", "$unwind", "$match", "$unwind", "$project", "$unset"]
        assert pipeline[0]["$match"] == builder.build()
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
        pipeline = DriftQueryBuilder("MetalCls/1.0", START, END).build_pipeline()
        assert pipeline[4]["$match"] == {"inspectionResult.aiResults.aiModel": "MetalCls/1.0"}
