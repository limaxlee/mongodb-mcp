import pytest
from datetime import datetime, timedelta, timezone

from common.constants import DBCollections, DriftFlag, PreVerdict
from mongodb_mcp.schemas import DriftAnalysisResult
from tests.conftest import DRIFT_ROWS, AsyncRowCursor
from tests.drift.synthetic import START

END = START + timedelta(days=7)


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
        assert result.data_quality.scanned_document_count == len(DRIFT_ROWS)
        assert result.data_quality.record_count == len(DRIFT_ROWS)
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert result.filters == {"gbm": "SEV", "process": None, "location": None, "equipment_id": None,
                                  "mode": "production"}

        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]
        match = collection.count_documents.call_args[0][0]
        assert match["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0"}}
        assert match["metadata.gbm"] == "SEV" and match["metadata.mode"] == "production"
        assert match["metadata.createdAt"] == {"$gte": START, "$lt": END}
        assert collection.aggregate.call_args.kwargs == {"allowDiskUse": True}
        assert collection.aggregate.call_args[0][0][0]["$match"] == match

    @pytest.mark.asyncio
    async def test_naive_dates_are_treated_as_utc(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", datetime(2026, 9, 1), datetime(2026, 9, 8)
        )
        assert result.range.start_date == START
        assert result.range.start_date.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_comparison_mode_queries_both_windows(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, reference_start_date=START - timedelta(days=7), reference_end_date=START
        )
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]

        assert result.mode == "comparison"
        assert collection.aggregate.call_count == 2
        assert collection.count_documents.await_count == 2
        first_match = collection.count_documents.await_args_list[0][0][0]
        assert first_match["metadata.createdAt"] == {"$gte": START - timedelta(days=7), "$lt": START}
        assert result.data_quality.scanned_document_count == 2 * len(DRIFT_ROWS)
        assert result.data_quality.record_count == 2 * len(DRIFT_ROWS)
        assert result.comparison is not None and result.change_point is None

    @pytest.mark.asyncio
    async def test_mode_none_includes_every_mode(self, drift_connector):
        await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, mode=None)
        match = drift_connector._mock_db[DBCollections.INSPECTIONS].count_documents.call_args[0][0]
        assert "metadata.mode" not in match

    @pytest.mark.asyncio
    async def test_task_is_passed_to_the_query(self, drift_connector):
        await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, task="cls")
        match = drift_connector._mock_db[DBCollections.INSPECTIONS].count_documents.call_args[0][0]
        assert match["inspectionResult.aiResults"] == {"$elemMatch": {"aiModel": "MetalCls/1.0", "task": "cls"}}

    @pytest.mark.asyncio
    async def test_no_matching_documents(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]
        collection.count_documents = mocker.AsyncMock(return_value=0)
        collection.aggregate = mocker.MagicMock(return_value=AsyncRowCursor([]))

        result = await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, task="det")
        assert result.pre_verdict == PreVerdict.UNDETERMINED
        assert result.task == "det"
        assert result.data_quality.matched_document_count == 0
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

        drift_connector._mock_db[DBCollections.INSPECTIONS].count_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_collection(self, mock_connector):
        with pytest.raises(ValueError, match="doesn't exist"):
            await mock_connector.analyze_data_drift("MetalCls", "1.0", START, END)

    @pytest.mark.asyncio
    async def test_database_errors_propagate(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS]
        collection.count_documents = mocker.AsyncMock(side_effect=RuntimeError("Connector error"))
        with pytest.raises(RuntimeError, match="Connector error"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END)
