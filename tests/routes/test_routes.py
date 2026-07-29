import json
import pytest
from starlette import status
from pymongo.errors import ConnectionFailure

from mongodb_mcp.routes.routes import download_logs, check_health


class TestCustomRoutes:
    @pytest.mark.asyncio
    async def test_download_logs(self, mocker):
        request = mocker.MagicMock()

        mocker.patch("mongodb_mcp.routes.routes.get_logs_zip_file", return_value=b'Log zip file')
        response = await download_logs(request)
        assert response.status_code == status.HTTP_200_OK
        assert response.media_type == "application/zip"
        assert response.body == b'Log zip file'

        mocker.patch("mongodb_mcp.routes.routes.get_logs_zip_file", return_value=None)
        with pytest.raises(Exception, match="No log files found"):
            await download_logs(request)

        mocker.patch("mongodb_mcp.routes.routes.get_logs_zip_file", side_effect=Exception("MCP Error"))
        with pytest.raises(Exception, match="MCP Error"):
            await download_logs(request)

    @pytest.mark.asyncio
    async def test_check_health(self, mocker):
        request = mocker.MagicMock()

        connector = mocker.patch("mongodb_mcp.routes.routes.MongoDBConnector").return_value
        connector.ping_database = mocker.AsyncMock(return_value={"ok": 1})
        connector.close = mocker.AsyncMock()

        response = await check_health(request)
        assert response.status_code == status.HTTP_200_OK
        body = json.loads(response.body)
        assert body["MCP Server status"] == "healthy"
        assert body["DB status"] == "healthy"
        connector.close.assert_awaited_once()

        connector.ping_database = mocker.AsyncMock(return_value={"ok": 0})
        response = await check_health(request)
        body = json.loads(response.body)
        assert body["DB status"] == "unhealthy"

        connector.ping_database = mocker.AsyncMock(side_effect=ConnectionFailure("No connection"))
        response = await check_health(request)
        body = json.loads(response.body)
        assert body["MCP Server status"] == "healthy"
        assert body["DB status"] == "unhealthy"

        connector.ping_database = mocker.AsyncMock(side_effect=Exception("MCP Error"))
        with pytest.raises(Exception, match="MCP Error"):
            await check_health(request)
