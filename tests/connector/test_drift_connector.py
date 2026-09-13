import pytest
from datetime import datetime, timedelta, timezone

from common.constants import DBCollections
from mongodb_mcp.connector.drift_query import build_drift_match, build_drift_pipeline
from mongodb_mcp.schemas import DriftAnalysisResult
from tests.conftest import DRIFT_ROWS, AsyncRowCursor
from tests.drift.synthetic import START

END = START + timedelta(days=7)


class TestDriftQuery:
    def test_match_filters(self):
        match = build_drift_match("MetalCls/1.0", START, END, task="cls", gbm="SEV", equipment_id="EQ-01",
                                  mode="production")
        assert match == {
            "isDeleted": False,
            "metadata.createdAt": {"$gte": START, "$lt": END},
            "inspectionResult.aiResults": {"$elemMatch": {"aiModel": "MetalCls/1.0", "task": "cls"}},
            "metadata.gbm": "SEV",
            "metadata.equipmentId": "EQ-01",
            "metadata.mode": "production"
        }

    def test_match_without_optional_filters(self):
        match = build_drift_match("MetalCls/1.0", START, END, mode=None)
        assert set(match) == {"isDeleted", "metadata.createdAt", "inspectionResult.aiResults"}
        assert match["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0"}}

    def test_pipeline_shape(self):
        match = build_drift_match("MetalCls/1.0", START, END)
        pipeline = build_drift_pipeline(match, "MetalCls/1.0", task="cls")
        stages = [next(iter(stage)) for stage in pipeline]

        assert stages == ["$match", "$sort", "$project", "$unwind", "$match", "$unwind", "$project"]
        assert pipeline[0]["$match"] is match
        assert pipeline[1]["$sort"] == {"metadata.createdAt": 1}
        assert pipeline[3]["$unwind"]["includeArrayIndex"] == "entryIndex"
        assert pipeline[4]["$match"] == {
            "inspectionResult.aiResults.aiModel": "MetalCls/1.0", "inspectionResult.aiResults.task": "cls"
        }
        final = pipeline[6]["$project"]
        assert final["prediction"] == "$inspectionResult.aiResults.predictions"
        assert final["createdAt"] == "$metadata.createdAt"
        assert "dataSpec" in final and "classes" in final and "backend" in final


class TestAnalyzeDataDrift:
    @pytest.fixture
    def drift_connector(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS]
        )
        return mock_connector

    @pytest.mark.asyncio
    async def test_range_mode(self, drift_connector):
        result = await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, gbm="SEV")

        assert isinstance(result, DriftAnalysisResult)
        assert result.model_name == "MetalCls" and result.model_version == "1.0"
        assert result.task == "cls" and result.mode == "range"
        assert result.data_quality.n_docs_scanned == len(DRIFT_ROWS)
        assert result.data_quality.n_records == len(DRIFT_ROWS)
        assert "CONFIDENCE_SHIFT" in result.flags
        assert result.filters == {"gbm": "SEV", "process": None, "location": None, "equipment_id": None,
                                  "mode": "production"}

        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]
        match = collection.count_documents.call_args[0][0]
        assert match["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0"}}
        assert match["metadata.gbm"] == "SEV" and match["metadata.mode"] == "production"
        assert match["metadata.createdAt"] == {"$gte": START, "$lt": END}
        assert collection.aggregate.call_args.kwargs == {"allowDiskUse": True}

    @pytest.mark.asyncio
    async def test_naive_dates_are_treated_as_utc(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", datetime(2026, 9, 1), datetime(2026, 9, 8)
        )
        assert result.range.start == START
        assert result.range.start.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_comparison_mode_queries_both_windows(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, reference_start=START - timedelta(days=7), reference_end=START
        )
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]

        assert result.mode == "comparison"
        assert collection.aggregate.call_count == 2
        assert result.data_quality.n_docs_scanned == 2 * len(DRIFT_ROWS)
        assert result.comparison is not None

    @pytest.mark.asyncio
    async def test_mode_none_includes_every_mode(self, drift_connector):
        await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, mode=None)
        match = drift_connector._mock_db[DBCollections.INSPECTIONS].count_documents.call_args[0][0]
        assert "metadata.mode" not in match

    @pytest.mark.asyncio
    async def test_no_matching_documents(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]
        collection.count_documents = mocker.AsyncMock(return_value=0)
        collection.aggregate = mocker.MagicMock(return_value=AsyncRowCursor([]))

        result = await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, task="det")
        assert result.pre_verdict == "undetermined"
        assert result.data_quality.n_docs_matched == 0
        assert any("No inspection results found" in message for message in result.warnings)

    @pytest.mark.asyncio
    async def test_validation_errors(self, drift_connector):
        with pytest.raises(ValueError, match="before end_date"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", END, START)
        with pytest.raises(ValueError, match="maximum is 30"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, START + timedelta(days=31))
        with pytest.raises(ValueError, match="not supported"):
            await drift_connector.analyze_data_drift("MetalSeg", "1.0", START, END, task="seg")
        with pytest.raises(ValueError, match="Unknown bucket"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, bucket="2d")
        with pytest.raises(ValueError, match="Unknown detail"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, detail="tiny")

    @pytest.mark.asyncio
    async def test_missing_collection(self, mock_connector):
        with pytest.raises(ValueError, match="doesn't exist"):
            await mock_connector.analyze_data_drift("MetalCls", "1.0", START, END)
