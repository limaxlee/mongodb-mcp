import logging
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConnectionInfo(BaseModel):
    current: int | None = None
    available: int | None = None
    total_created: int | None = Field(None, alias="totalCreated")


class ExtraInfo(BaseModel):
    note: str | None = None
    heap_usage_bytes: int | None = None
    page_faults: int | None = None


class ServerStatus(BaseModel):
    model_config = {"populate_by_name": True}

    host: str | None = None
    version: str | None = None
    process: str | None = None
    pid: int | None = None
    uptime: float | None = None
    uptime_millis: float | None = Field(None, alias="uptimeMillis")
    local_time: datetime | None = Field(None, alias="localTime")
    connections: ConnectionInfo = ConnectionInfo()
    extra_info: ExtraInfo = Field(default_factory=ExtraInfo, alias="extra_info")
