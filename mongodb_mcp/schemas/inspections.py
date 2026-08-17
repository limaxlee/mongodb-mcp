from datetime import datetime
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class InspectionModelInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    model_name: str
    model_version: str
    process: str
    task: str
    gbm: str
    mode: str
    date: datetime


class FindInspectionModelsResult(BaseModel):
    models: list[InspectionModelInfo] = []


class ValueStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0


class InspectionStatistics(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    data_count: dict[str, int] = {}
    confidence: dict[str, ValueStats] = {}
    elapsed_time: ValueStats = ValueStats()


class InspectionSummaryInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    schema_version: str = "1.0"
    model_name: str
    model_version: str
    gbm: str
    process: str
    mode: str | None = None
    date: datetime
    location: str
    equipment_id: str
    product_id: str | None = None
    local_timezone: str | None = None
    inspection_ids: list[str] = []
    task: str
    classes: list[str] = []
    conclusion: str | None = None
    threshold: float | None = None
    statistics: InspectionStatistics = InspectionStatistics()


class FindInspectionSummariesResult(BaseModel):
    summaries: list[InspectionSummaryInfo] = []
