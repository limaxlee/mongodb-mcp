import pytest
import pytest_asyncio
from datetime import timedelta
from fastmcp.exceptions import ToolError

from common.constants import DBCollections, PreVerdict
from mongodb_mcp.tools import tools_mcp
from mongodb_mcp.schemas import DriftAnalysisResult
from tests.drift.synthetic import START


@pytest_asyncio.fixture
async def drift_tool():
    tools = await tools_mcp.list_tools()
    return {tool.name: tool.fn for tool in tools}["mongodb_analyze_data_drift"]


class TestAnalyzeDataDriftTool:
    @pytest.mark.asyncio
    async def test_tool_returns_analysis(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=[DBCollections.INSPECTIONS])

        result = await drift_tool(mock_context, "MetalCls", "1.0", START, START + timedelta(days=7))

        assert isinstance(result, DriftAnalysisResult)
        assert result.model_name == "MetalCls"
        assert result.pre_verdict in list(PreVerdict)
        assert result.detail == "full" and result.mode == "range"

    @pytest.mark.asyncio
    async def test_tool_passes_arguments_through(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector.analyze_data_drift = mocker.AsyncMock(return_value=mocker.MagicMock())

        await drift_tool(
            mock_context, "MetalDet", "2.1", START, START + timedelta(days=7),
            task="det", gbm="SEV", process="SMD", location="Line_01", equipment_id="EQ-01", mode=None,
            bucket="1d", detail="compact", reference_start_date=START - timedelta(days=7), reference_end_date=START
        )
        kwargs = connector.analyze_data_drift.call_args.kwargs
        assert kwargs["model_name"] == "MetalDet" and kwargs["model_version"] == "2.1"
        assert kwargs["start_date"] == START and kwargs["end_date"] == START + timedelta(days=7)
        assert kwargs["task"] == "det" and kwargs["mode"] is None and kwargs["detail"] == "compact"
        assert kwargs["gbm"] == "SEV" and kwargs["process"] == "SMD" and kwargs["location"] == "Line_01"
        assert kwargs["equipment_id"] == "EQ-01" and kwargs["bucket"] == "1d"
        assert kwargs["reference_start_date"] == START - timedelta(days=7)
        assert kwargs["reference_end_date"] == START

    @pytest.mark.asyncio
    async def test_tool_defaults(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector.analyze_data_drift = mocker.AsyncMock(return_value=mocker.MagicMock())

        await drift_tool(mock_context, "MetalCls", "1.0", START, START + timedelta(days=7))
        kwargs = connector.analyze_data_drift.call_args.kwargs
        assert kwargs["mode"] == "production" and kwargs["bucket"] == "auto" and kwargs["detail"] == "full"
        assert kwargs["task"] is None and kwargs["reference_start_date"] is None

    @pytest.mark.asyncio
    async def test_tool_wraps_errors(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector.analyze_data_drift = mocker.AsyncMock(side_effect=ValueError("start_date must be before"))

        with pytest.raises(ToolError, match="start_date must be before"):
            await drift_tool(mock_context, "MetalCls", "1.0", START, START + timedelta(days=7))

        connector.analyze_data_drift = mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await drift_tool(mock_context, "MetalCls", "1.0", START, START + timedelta(days=7))
