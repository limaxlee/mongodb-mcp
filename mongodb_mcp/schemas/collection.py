from typing import Any

from pydantic import BaseModel


class ListCollectionsResult(BaseModel):
    collections: list[str] = []


class GetCollectionInfoResult(BaseModel):
    collection_name: str
    collection_id: int
    auto_id: bool
    description: str
    num_shards: int
    num_partitions: int = 0
    enable_namespace: bool
    enable_dynamic_field: bool
    aliases: list[Any] = []
    classes: dict[str, int] = {}
    fields: list[dict[Any, Any]] = []
    functions: list[Any] = []
    consistency_level: int = 0
    properties: dict[str, Any] = {}


class GetCollectionStatsResult(BaseModel):
    collection_name: str
    row_count: int


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


class LoadCollectionResult(BaseModel):
    collection_name: str
    collection_loaded: bool = True
    replica_number: int = 1


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
