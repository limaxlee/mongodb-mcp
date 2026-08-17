from typing import Any
from pydantic import BaseModel


class InsertDocumentResult(BaseModel):
    collection_name: str
    document_id: str
    document_inserted: bool = True


class InsertManyDocumentsResult(BaseModel):
    collection_name: str
    document_ids: list[str]
    documents_inserted: bool = True


class FindDocumentsResult(BaseModel):
    collection_name: str
    documents: list[dict[str, Any]]


class CountDocumentsResult(BaseModel):
    collection_name: str
    document_count: int


class UpdateDocumentsResult(BaseModel):
    collection_name: str
    updated_document_count: int


class ReplaceDocumentResult(BaseModel):
    collection_name: str
    replaced_document_count: int


class DeleteDocumentsResult(BaseModel):
    collection_name: str
    deleted_document_count: int


class AggregateDocumentsResult(BaseModel):
    collection_name: str
    documents: list[dict[str, Any]]
