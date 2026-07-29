import logging
import asyncio
from typing import Any
from functools import wraps
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import TEXT, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

from common.config import SETTINGS
from common.constants import LIMIT, CONNECTION_RETRIES, CONNECTION_DELAY, DBCollections
from mongodb_mcp.utils import preprocess_operation, process_query_result, generate_normalized_regex
from mongodb_mcp.schemas import ServerStatus

logger = logging.getLogger(__name__)


def ensure_connection(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        for attempt in range(1, CONNECTION_RETRIES + 1):
            try:
                return await func(self, *args, **kwargs)
            except ConnectionFailure as e:
                if attempt == CONNECTION_RETRIES:
                    logger.error(f"MongoDB connection failed after {CONNECTION_RETRIES} attempts: {e}")
                    raise

                logger.warning(f"Connection attempt {attempt}/{CONNECTION_RETRIES}. Retrying in {CONNECTION_DELAY}s")
                await asyncio.sleep(CONNECTION_DELAY)
    return wrapper


class MongoDBConnector:
    def __init__(self, db_name: str = SETTINGS.mongodb_db_name):
        self.uri = f"mongodb://{SETTINGS.mongodb_host}:{SETTINGS.mongodb_port}"
        self._client = AsyncIOMotorClient(self.uri)
        self._db = self._client[db_name]

    @ensure_connection
    async def list_databases(self) -> list[str]:
        try:
            return await self._client.list_database_names()
        except Exception as e:
            logger.exception(f"Failed to list databases: {str(e)}")
            raise

    @ensure_connection
    async def list_collections(self) -> list[str]:
        try:
            return await self._db.list_collection_names()
        except Exception as e:
            logger.exception(f"Failed to list collections: {str(e)}")
            raise

    @ensure_connection
    async def create_collection(self, collection_name: str, **kwargs: Any):
        try:
            await self._db.create_collection(name=collection_name, **kwargs)
            logger.info(f"Created collection {collection_name}")
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def drop_collection(self, collection_name: str):
        try:
            await self._db.drop_collection(collection_name)
            logger.info(f"Dropped collection {collection_name}")
        except Exception as e:
            logger.exception(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def rename_collection(self, collection_name: str, new_collection_name: str):
        try:
            collections = await self._db.list_collection_names()
            if collection_name not in collections:
                raise ValueError(f"Collection {collection_name} doesn't exist")
            if new_collection_name in collections:
                raise ValueError(f"Collection {new_collection_name} already exists")

            await self._db[collection_name].rename(new_collection_name)
            logger.info(f"Renamed collection {collection_name} to {new_collection_name}")
        except Exception as e:
            logger.exception(f"Failed to rename collection {collection_name} to {new_collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            return await self._db.command("collStats", collection_name)
        except Exception as e:
            logger.exception(f"Failed to get statistics information for a collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def get_database_stats(self) -> dict[str, Any]:
        try:
            return await self._db.command(command="dbStats")
        except Exception as e:
            logger.exception(f"Failed to get statistics information for a database: {str(e)}")
            raise

    @ensure_connection
    async def list_indices(self, collection_name: str) -> list[Any]:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            indices = await self._db[collection_name].list_indexes().to_list(length=None)
            logger.info(f"Listed {len(indices)} indices for {collection_name} collection")

            return indices
        except Exception as e:
            logger.exception(f"Failed to list the indices for {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def create_index(self, collection_name: str, keys: dict[str, Any], **kwargs: Any) -> str:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            index_keys = []
            for field, direction in keys.items():
                if direction == 1 or direction == "asc" or direction == "ascending":
                    index_keys.append((field, ASCENDING))
                elif direction == -1 or direction == "desc" or direction == "descending":
                    index_keys.append((field, DESCENDING))
                elif direction == "text":
                    index_keys.append((field, TEXT))
                else:
                    index_keys.append((field, direction))

            index_name = await self._db[collection_name].create_index(index_keys, **kwargs)
            logger.info(f"Created index {index_name} for {collection_name} collection")

            return index_name
        except Exception as e:
            logger.exception(f"Failed to create index for {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def drop_index(self, collection_name: str, index_name: str):
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")
            if index_name == "_id_":
                raise ValueError(f"Cannot drop the default _id index from {collection_name} collection")

            await self._db[collection_name].drop_index(index_name)
            logger.info(f"Dropped index {index_name} from {collection_name} collection")
        except Exception as e:
            logger.exception(f"Failed to drop index {index_name} from {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def get_server_status(self) -> ServerStatus:
        try:
            status = await self._client.admin.command("serverStatus")
            server_status = ServerStatus.model_validate(status)

            logger.info(f"Retrieved server status: {server_status}")
            return server_status
        except Exception as e:
            logger.exception(f"Failed to get server status information: {str(e)}")
            raise

    @ensure_connection
    async def ping_database(self) -> dict[str, Any]:
        try:
            ping_result = await self._db.command(command="ping")
            logger.info(f"Pinged MongoDB database: {ping_result}")

            return ping_result
        except Exception as e:
            logger.exception(f"Failed to ping MongoDB database: {str(e)}")
            raise

    @ensure_connection
    async def insert_document(self, collection_name: str, document: dict[str, Any]) -> str:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            document = preprocess_operation(document)
            result = await self._db[collection_name].insert_one(document)
            if result.acknowledged:
                logger.info(f"Inserted document with id {str(result.inserted_id)} into {collection_name} collection")
            else:
                logger.error(f"Failed to insert document into {collection_name} collection")
                raise RuntimeError(f"Failed to insert document into {collection_name} collection")

            return str(result.inserted_id)
        except Exception as e:
            logger.exception(f"Failed to insert document into {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def insert_many_documents(
            self,
            collection_name: str,
            documents: list[dict[str, Any]],
            ordered: bool = True
    ) -> list[str]:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            documents = [preprocess_operation(document) for document in documents]
            result = await self._db[collection_name].insert_many(documents, ordered=ordered)
            if result.acknowledged:
                logger.info(f"Inserted {len(documents)} documents into {collection_name} collection")
            else:
                logger.error(f"Failed to insert documents into {collection_name} collection")
                raise RuntimeError(f"Failed to insert documents into {collection_name} collection")

            return [str(oid) for oid in result.inserted_ids]
        except Exception as e:
            logger.exception(f"Failed to insert documents into {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def find_documents(
            self,
            collection_name: str,
            query: dict[str, Any],
            projection: dict[str, Any] | None = None,
            limit: int = LIMIT,
            sort_field: str | None = None,
            sort_order: int = ASCENDING
    ) -> list[dict[str, Any]]:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            cursor = self._db[collection_name].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} documents in {collection_name} collection using query {query}")

            return [process_query_result(document) for document in documents]
        except Exception as e:
            logger.exception(f"Failed to find documents in {collection_name} collection using query {query}: {str(e)}")
            raise

    @ensure_connection
    async def count_documents(self, collection_name: str, query: dict[str, Any]) -> int:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            document_count = await self._db[collection_name].count_documents(query)
            logger.info(f"Counted {document_count} documents in {collection_name} collection using query {query}")

            return document_count
        except Exception as e:
            logger.exception(f"Failed to count documents in {collection_name} collection using query {query}: {str(e)}")
            raise

    @ensure_connection
    async def update_documents(
            self,
            collection_name: str,
            query: dict[str, Any],
            update_operation: dict[str, Any],
            upsert: bool = False
    ) -> int:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            update_operation = preprocess_operation(update_operation)
            result = await self._db[collection_name].update_many(query, update_operation, upsert=upsert)
            update_str = str(update_operation)

            if result.acknowledged:
                logger.info(f"Updated documents in {collection_name} by query`` {query} with: {update_str[:20]}")
            else:
                error_message = f"Failed to update document in {collection_name} by query {query} with"
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result.modified_count
        except Exception as e:
            logger.exception(f"Failed to update documents in {collection_name} collection by query {query}: {str(e)}")
            raise

    @ensure_connection
    async def replace_document(
            self,
            collection_name: str,
            query: dict[str, Any],
            document: dict[str, Any],
            upsert: bool = False
    ) -> int:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            document = preprocess_operation(document)
            result = await self._db[collection_name].replace_one(query, document, upsert=upsert)

            if result.acknowledged:
                logger.error(f"Replaced document in {collection_name} using query {query} with {document}")
            else:
                error_message = f"Failed to replace document in {collection_name} using query {query} with {document}"
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result.modified_count
        except Exception as e:
            logger.exception(f"Failed to replace document in {collection_name} collection by query {query}: {str(e)}")
            raise

    @ensure_connection
    async def delete_documents(self, collection_name: str, query: dict[str, Any]) -> int:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            result = await self._db[collection_name].delete_many(query)
            if result.acknowledged:
                logger.info(f"Deleted {result.deleted_count} documents from {collection_name} collection")
            else:
                error_message = f"Failed to delete documents from {collection_name} collection using query {query}"
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result.deleted_count
        except Exception as e:
            logger.exception(f"Failed to delete document in {collection_name} collection using query {query}: {str(e)}")
            raise

    @ensure_connection
    async def aggregate_documents(
            self,
            collection_name: str,
            pipeline: list[dict[str, Any]],
            options: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            pipeline = [preprocess_operation(item) for item in pipeline]
            if options:
                cursor = self._db[collection_name].aggregate(pipeline, **options)
            else:
                cursor = self._db[collection_name].aggregate(pipeline)

            documents = await cursor.to_list(length=None)
            logger.info(f"Executed aggregation pipeline with {len(documents)} results in {collection_name} collection")

            return [process_query_result(document) for document in documents]
        except Exception as e:
            logger.exception(f"Failed to execute aggregation pipeline on {collection_name} collection: {str(e)}")
            raise

    def _build_inspection_query(
            self,
            model_name: str | None = None,
            model_version: str | None = None,
            gbm: str | None = None,
            task: str | None = None,
            mode: str | None = None,
            process: str | None = None,
            start_date: datetime | None = None,
            end_date: datetime | None = None
    ) -> dict[str, Any]:
        query = {}

        if model_name:
            query["modelName"] = generate_normalized_regex(model_name)
        if model_version:
            query["modelVersion"] = generate_normalized_regex(model_version)
        if gbm:
            query["gbm"] = gbm
        if task:
            query["task"] = task
        if mode:
            query["mode"] = mode
        if process:
            query["process"] = generate_normalized_regex(process)
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["date"] = date_filter

        return query

    @ensure_connection
    async def find_inspection_models(
            self,
            model_name: str | None = None,
            model_version: str | None = None,
            gbm: str | None = None,
            task: str | None = None,
            mode: str | None = None,
            process: str | None = None,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
            projection: dict[str, Any] | None = None,
            limit: int = LIMIT,
            sort_field: str | None = None,
            sort_order: int = ASCENDING
    ) -> list[dict[str, Any]]:
        try:
            query = self._build_inspection_query(
                model_name=model_name,
                model_version=model_version,
                gbm=gbm,
                task=task,
                mode=mode,
                process=process,
                start_date=start_date,
                end_date=end_date
            )

            if DBCollections.DAILY_MODELS not in await self._db.list_collection_names():
                raise ValueError(f"Collection {DBCollections.DAILY_MODELS} doesn't exist")

            cursor = self._db[DBCollections.DAILY_MODELS].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} model documents using query {query}")

            return [process_query_result(document) for document in documents]
        except Exception as e:
            logger.exception(f"Failed to find inspection models: {str(e)}")
            raise

    @ensure_connection
    async def close(self):
        try:
            self._client.close()
            logger.info("Closed MongoDB client")
        except Exception as e:
            logger.exception(f"Failed to close MongoDB client: {str(e)}")
            raise
