import logging
import datetime
from fastmcp.server import FastMCP
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from pymongo.errors import ConnectionFailure

from mongodb_mcp.connector import MongoDBConnector
from mongodb_mcp.utils import get_logs_zip_file

logger = logging.getLogger(__name__)

custom_mcp = FastMCP(name="custom_route")


@custom_mcp.custom_route("/logs", methods=["GET"])
async def download_logs(request: Request) -> Response:
    try:
        zip_name = f"{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%SZ')}.zip"
        zip_bytes = await get_logs_zip_file()
        if not zip_bytes:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No log files found")

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@custom_mcp.custom_route("/health", methods=["GET"])
async def check_health(request: Request) -> JSONResponse:
    connector = MongoDBConnector()

    try:
        health_status = {"MCP Server status": "healthy"}

        ping_result = await connector.ping_database()
        health_status["DB status"] = "healthy" if ping_result.get("ok", 0) == 1 else "unhealthy"

        logger.info(f"Checked server health status: {health_status}")
        return JSONResponse(health_status)
    except ConnectionFailure as e:
        logger.warning(f"Failed to check MongoDB health: {str(e)}")
        return JSONResponse({"MCP Server status": "healthy", "DB status": "unhealthy"})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        await connector.close()
