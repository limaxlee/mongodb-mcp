from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ListDatabasesResult(BaseModel):
    databases: list[str] = []


class GetDatabaseStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    db: str
    collections: int
    views: int
    objects: int
    indexes: int
    index_size: int
    total_size: int
    scale_factor: int
    fs_used_size: int
    fs_total_size: int
    ok: int


class PingDatabaseResult(BaseModel):
    ok: int
