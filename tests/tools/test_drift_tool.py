import pytest
import pytest_asyncio
from datetime import timedelta
from fastmcp.exceptions import ToolError

from common.constants import DBCollections
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
        assert result.pre_verdict in ("stable", "suspicious", "drift_likely", "undetermined")
        assert result.detail == "full"

    @pytest.mark.asyncio
    async def test_tool_passes_arguments_through(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector.analyze_data_drift = mocker.AsyncMock(return_value=mocker.MagicMock())

        await drift_tool(
            mock_context, "MetalDet", "2.1", START, START + timedelta(days=7),
            task="det", gbm="SEV", mode=None, bucket="1d", detail="compact", defect_classes=["NG"],
            reference_start=START - timedelta(days=7), reference_end=START
        )
        kwargs = connector.analyze_data_drift.call_args.kwargs
        assert kwargs["model_name"] == "MetalDet" and kwargs["model_version"] == "2.1"
        assert kwargs["task"] == "det" and kwargs["mode"] is None and kwargs["detail"] == "compact"
        assert kwargs["reference_start"] == START - timedelta(days=7)

    @pytest.mark.asyncio
    async def test_tool_wraps_errors(self, mocker, drift_tool, mock_context):
        connector = mock_context.request_context.lifespan_context.connector
        connector.analyze_data_drift = mocker.AsyncMock(side_effect=ValueError("start_date must be before"))

        with pytest.raises(ToolError, match="start_date must be before"):
            await drift_tool(mock_context, "MetalCls", "1.0", START, START + timedelta(days=7))
