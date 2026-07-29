import uvicorn
from typing import AsyncIterator
from fastmcp.server import FastMCP
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from common.config import SETTINGS
from mongodb_mcp.connector import MongoDBContext, MongoDBConnector
from mongodb_mcp.middleware import LoggingMiddleware
from mongodb_mcp.routes import custom_mcp
from mongodb_mcp.tools import tools_mcp
from mongodb_mcp.utils import initialize_logger


@asynccontextmanager
async def server_lifespan(mcp: FastMCP) -> AsyncIterator[MongoDBContext]:
    connector = MongoDBConnector()

    try:
        yield MongoDBContext(connector)
    finally:
        await connector.close()


mcp = FastMCP(name="MongoDB MCP Server", lifespan=server_lifespan)
mcp.mount(tools_mcp, prefix="mcp")
mcp.mount(custom_mcp)
mcp.add_middleware(LoggingMiddleware())

app = mcp.http_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"]
)

if __name__ == "__main__":
    logger = initialize_logger("mongodb_mcp_server.log")
    logger.info("Starting MongoDB MCP Server")

    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.server_port)
