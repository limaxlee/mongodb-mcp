from datetime import timedelta

from common.constants import Granularity
from mongodb_mcp.drift import StatisticsQuery
from tests.drift.synthetic import START

END = START + timedelta(days=7)


class TestStatisticsQuery:
    def test_match_with_every_filter(self):
        query = StatisticsQuery(
            "MetalCls", "1.0", START, END, Granularity.DAILY, task="cls", gbm="SEV", process="SMD",
            mode="production", equipment_id="EQ-01", product_id="PR-01"
        )
        match = query.build_match()
        assert match == {
            "modelName": "MetalCls",
            "modelVersion": "1.0",
            "task": "cls",
            "mode": "production",
            "gbm": "SEV",
            "process": "SMD",
            "granularity": "daily",
            "startDate": {"$gte": START, "$lt": END},
            "productId": "PR-01"
        }

    def test_match_without_optional_filters(self):
        match = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.HOURLY).build_match()
        assert match["task"] == {"$in": ["cls", "det"]}
        assert match["granularity"] == "hourly"
        assert "mode" not in match and "gbm" not in match and "process" not in match and "productId" not in match

    def test_match_follows_the_unique_index_order(self):
        match = StatisticsQuery(
            "MetalCls", "1.0", START, END, Granularity.DAILY, task="cls", mode="production", gbm="SEV", process="SMD"
        ).build_match()
        assert list(match) == ["modelName", "modelVersion", "task", "mode", "gbm", "process", "granularity", "startDate"]

    def test_end_of_window_is_exclusive(self):
        match = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY).build_match()
        assert match["startDate"] == {"$gte": START, "$lt": END}

    def test_block_is_total_by_default(self):
        assert StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY).build_block() == "$total"

    def test_block_is_the_equipment_entry(self):
        block = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY, equipment_id="EQ-01").build_block()
        assert block["$arrayElemAt"][1] == 0
        condition = block["$arrayElemAt"][0]["$filter"]["cond"]
        assert condition == {"$eq": ["$$item.equipmentId", "EQ-01"]}
        pipeline = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY, equipment_id="EQ-01").build_pipeline()
        assert pipeline[2]["$addFields"]["configBackends"] == ["$block.backend"]
        assert "configThresholds" not in pipeline[2]["$addFields"]

    def test_pipeline_shape(self):
        query = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY, task="cls")
        pipeline = query.build_pipeline()
        stages = [next(iter(stage)) for stage in pipeline]
        assert stages == ["$match", "$project", "$addFields", "$unwind", "$group", "$addFields", "$project", "$sort"]
        assert pipeline[0]["$match"] == query.build_match()
        assert pipeline[1]["$project"]["block"] == "$total"
        assert pipeline[3]["$unwind"] == "$entries"
        assert pipeline[2]["$addFields"]["configBackends"] == "$equipmentBackends"

        group = pipeline[4]["$group"]
        assert group["_id"] == {"date": "$startDate", "kind": "$entries.kind", "k": "$entries.k"}
        assert group["predictionCount"] == {"$sum": "$block.predictionCount"}
        assert group["histograms"] == {"$push": "$entries.v.histogram"}
        assert group["classCount"] == {"$sum": "$entries.v.classCount"}
        assert group["documentCount"]["$sum"]["$cond"][1:] == [1, 0]
        assert group["missingBlockCount"]["$sum"]["$cond"][1:] == [0, 1]

        assert "histogram" in pipeline[5]["$addFields"] and "boxesPerImageHistogram" in pipeline[5]["$addFields"]
        assert pipeline[6]["$project"] == {"histograms": 0, "boxesPerImageHistograms": 0}
        assert pipeline[7]["$sort"] == {"_id.date": 1, "_id.kind": 1, "_id.k": 1}

    def test_entries_cover_total_classes_and_counts(self):
        pipeline = StatisticsQuery("MetalCls", "1.0", START, END, Granularity.DAILY).build_pipeline()
        parts = pipeline[2]["$addFields"]["entries"]["$concatArrays"]
        assert parts[0][0]["kind"] == "total" and parts[0][0]["k"] is None
        assert parts[1]["$map"]["in"]["kind"] == "class"
        assert parts[2]["$map"]["in"]["kind"] == "count"
        assert parts[2]["$map"]["in"]["v"] == {"classCount": "$$item.v"}
