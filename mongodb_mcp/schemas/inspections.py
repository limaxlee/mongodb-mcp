from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, NonNegativeInt, NonNegativeFloat
from pydantic.alias_generators import to_camel

from common.constants import ModelTasks, InspectionMode


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

    avg: NonNegativeFloat = 0.0
    min: NonNegativeFloat = 0.0
    max: NonNegativeFloat = 0.0
    sum: NonNegativeFloat = 0.0


class Statistics(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    data_count: dict[str, NonNegativeInt] = {}
    confidence: dict[str, ValueStats] = {}
    elapsed_time: ValueStats = ValueStats()


class InspectionSummaryDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    model_name: str
    model_version: str
    gbm: str
    process: str
    mode: InspectionMode | None = None
    date: datetime
    location: str
    equipment_id: str
    product_id: str | None = None
    local_timezone: str | None = None
    task: ModelTasks
    classes: list[str] = []
    threshold: NonNegativeFloat | None = None
    statistics: Statistics = Statistics()
    samples: dict[str, Any] = {}


class FindInspectionSummaryDocumentsResult(BaseModel):
    summaries: list[InspectionSummaryDocument] = []
