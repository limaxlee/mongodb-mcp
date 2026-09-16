from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt
from pydantic.alias_generators import to_camel

from common.constants import ModelTasks


class AccessControl(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    groups: list[str] = []
    users: list[str] = []


class FamilyMember(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    dataset_id: str
    version: str
    description: str = ""


class DatasetFamilyDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    family_id: str | None = Field(None, alias="_id")
    dataset_family_name: str
    task: ModelTasks
    members: list[FamilyMember] = []
    access_control: AccessControl = AccessControl()
    created_at: datetime


class FindDatasetFamilyDocumentsResult(BaseModel):
    families: list[DatasetFamilyDocument] = []


class LabelAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    count: NonNegativeInt = 0
    color: str = "#ff0000"
    shape: str | None = None


class TrainingRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    ai_model: str
    version: str
    start_time: datetime
    end_time: datetime
    status: str
    training_info: str = ""


class DatasetDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    document_id: str | None = Field(None, alias="_id")
    name: str
    version: str
    task: ModelTasks
    description: str = ""
    created_by: str
    projects: list[str] = []
    access_control: AccessControl = AccessControl()
    family_id: str | None = None
    schema_version: str = "1.0"
    created_at: datetime
    modified_at: datetime
    is_finalized: bool = False
    finalized_at: datetime | None = None
    is_used: bool = False
    download_count: NonNegativeInt = 0
    download_uri: str = ""
    data_count: NonNegativeInt = 0
    classes: dict[str, LabelAttributes] = {}
    training_records: list[TrainingRecord] = []


class FindDatasetDocumentsResult(BaseModel):
    datasets: list[DatasetDocument] = []
