from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ListCollectionsResult(BaseModel):
    collections: list[str] = []


class GetCollectionStatsResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    ns: str
    size: int
    count: int
    storage_size: int
    free_storage_size: int
    avg_obj_size: int
    num_orphan_docs: int = 0
    capped: bool
    wired_tiger: dict[str, Any] = {}
    index_builds: list[Any] = []
    total_index_size: int = 0
    index_sizes: dict[str, Any] = {}
    total_size: int = 0
    scale_factor: int = 0
    ok: int = 1


class CreateCollectionResult(BaseModel):
    collection_name: str
    collection_created: bool = True


class DropCollectionResult(BaseModel):
    collection_name: str
    collection_dropped: bool = True


class RenameCollectionResult(BaseModel):
    collection_name: str
    new_collection_name: str
    collection_renamed: bool = True


class IndexInfo(BaseModel):
    v: int
    key: dict[str, Any] = {}
    name: str


class GetIndicesResult(BaseModel):
    collection_name: str
    count: int
    indices: list[IndexInfo] = []


class CreateIndexResult(BaseModel):
    collection_name: str
    index: str
    index_created: bool = True


class DropIndexInfoResult(BaseModel):
    collection_name: str
    index: str
    index_dropped: bool = True
