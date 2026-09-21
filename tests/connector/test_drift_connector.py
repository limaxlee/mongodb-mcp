import pytest
from datetime import datetime, timedelta, timezone

from common.constants import DBCollections, DriftFlag, Granularity
from mongodb_mcp.schemas import DriftAnalysisResult
from tests.conftest import DRIFT_ROWS, DRIFT_DOCUMENTS, AsyncRowCursor
from tests.drift.synthetic import START, aggregate

END = START + timedelta(days=7)


class TestAnalyzeDataDrift:
    @pytest.fixture
    def drift_connector(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_STATISTICS]
        )
        return mock_connector

    @pytest.mark.asyncio
    async def test_range_mode(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, gbm="SEV", mode="production", granularity="daily"
        )

        assert isinstance(result, DriftAnalysisResult)
        assert result.model_name == "MetalCls" and result.model_version == "1.0"
        assert result.task == "cls" and result.analysis_mode == "range" and result.mode == "production"
        assert result.granularity == Granularity.DAILY
        assert result.data_quality.document_count == len(DRIFT_DOCUMENTS)
        assert result.status.period_count == 7
        assert DriftFlag.CONFIDENCE_SHIFT in result.flags
        assert result.filters.model_dump() == {"gbm": "SEV", "process": None, "equipment_id": None, "product_id": None}

        collection = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS]
        assert collection.aggregate.call_args.kwargs == {"allowDiskUse": True}
        match = collection.aggregate.call_args[0][0][0]["$match"]
        assert match["modelName"] == "MetalCls" and match["modelVersion"] == "1.0"
        assert match["task"] == {"$in": ["cls", "det"]}
        assert match["gbm"] == "SEV" and match["mode"] == "production" and match["granularity"] == "daily"
        assert match["startDate"] == {"$gte": START, "$lt": END}

    @pytest.mark.asyncio
    async def test_auto_granularity_is_resolved_before_the_query(self, drift_connector):
        result = await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END)
        # 7 days: hourly would be 168 periods, shift 14, so shift is chosen; the mock returns daily rows regardless
        assert result.granularity == Granularity.SHIFT
        match = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS].aggregate.call_args[0][0][0]["$match"]
        assert match["granularity"] == "shift"

    @pytest.mark.asyncio
    async def test_naive_dates_are_treated_as_utc(self, drift_connector):
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", datetime(2026, 9, 1), datetime(2026, 9, 8), granularity="daily"
        )
        assert result.range.start_date == START
        assert result.range.start_date.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_comparison_mode_queries_both_windows(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS]
        reference_rows = [
            dict(row, _id=dict(row["_id"], date=row["_id"]["date"] - timedelta(days=7)),
                 endDate=row["endDate"] - timedelta(days=7))
            for row in DRIFT_ROWS
        ]
        collection.aggregate = mocker.MagicMock(
            side_effect=[AsyncRowCursor(reference_rows), AsyncRowCursor(DRIFT_ROWS)]
        )

        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, granularity="daily",
            reference_start_date=START - timedelta(days=7), reference_end_date=START
        )

        assert result.analysis_mode == "comparison"
        assert collection.aggregate.call_count == 2
        first_match = collection.aggregate.call_args_list[0][0][0][0]["$match"]
        assert first_match["startDate"] == {"$gte": START - timedelta(days=7), "$lt": START}
        assert result.status.reference_period_count == 7 and result.status.period_count == 7
        assert result.data_quality.document_count == 2 * len(DRIFT_DOCUMENTS)
        assert result.comparison is not None and result.change_point is None

    @pytest.mark.asyncio
    async def test_filters_reach_the_pipeline(self, drift_connector):
        await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, task="cls", gbm="SEV", process="SMD", equipment_id="EQ-01",
            product_id="PR-000-000", mode=None, granularity="daily"
        )
        pipeline = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS].aggregate.call_args[0][0]
        match = pipeline[0]["$match"]
        assert match["task"] == "cls" and match["process"] == "SMD" and match["productId"] == "PR-000-000"
        assert "mode" not in match
        assert pipeline[1]["$project"]["block"] != "$total"

    @pytest.mark.asyncio
    async def test_equipment_filter_uses_the_entry(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS]
        collection.aggregate = mocker.MagicMock(
            side_effect=lambda pipeline, **kwargs: AsyncRowCursor(aggregate(DRIFT_DOCUMENTS, "EQ-01"))
        )
        result = await drift_connector.analyze_data_drift(
            "MetalCls", "1.0", START, END, equipment_id="EQ-01", granularity="daily"
        )
        assert result.sites_seen.equipment_ids == ["EQ-01"] and result.status.period_count == 7

    @pytest.mark.asyncio
    async def test_no_documents(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS]
        collection.aggregate = mocker.MagicMock(side_effect=lambda pipeline, **kwargs: AsyncRowCursor([]))
        with pytest.raises(ValueError, match="No inspection statistics found"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, granularity="daily")

    @pytest.mark.asyncio
    async def test_ambiguous_task(self, mocker, drift_connector):
        collection = drift_connector._mock_db[DBCollections.INSPECTIONS_STATISTICS]
        mixed = [dict(row, tasks=["cls", "det"]) for row in DRIFT_ROWS]
        collection.aggregate = mocker.MagicMock(side_effect=lambda pipeline, **kwargs: AsyncRowCursor(mixed))
        with pytest.raises(ValueError, match="several tasks"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, granularity="daily")

    @pytest.mark.asyncio
    async def test_argument_validation(self, drift_connector):
        with pytest.raises(ValueError, match="Unsupported task"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, task="seg")
        with pytest.raises(ValueError, match="Unsupported detail"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, detail="verbose")
        with pytest.raises(ValueError, match="Unsupported granularity"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END, granularity="1d")
        with pytest.raises(ValueError, match="before end date"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", END, START)

    @pytest.mark.asyncio
    async def test_missing_collection(self, mocker, drift_connector):
        drift_connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=["collection_a"])
        with pytest.raises(ValueError, match="doesn't exist"):
            await drift_connector.analyze_data_drift("MetalCls", "1.0", START, END)
