import logging
import asyncio
from typing import Any
from functools import wraps
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import TEXT, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

from common.config import SETTINGS
from common.constants import (
    LIMIT, CONNECTION_RETRIES, CONNECTION_DELAY, DBCollections, DATASET_EXCLUDED_FIELDS, AUTO_BUCKET,
    BucketSize, DriftTask, DriftDetail, DriftWindow
)
from mongodb_mcp.utils import preprocess_operation, process_query_result, generate_normalized_regex
from mongodb_mcp.schemas import *
from mongodb_mcp.drift import DriftAnalyzer, DriftQuery, RecordExtractor, UnsupportedTaskError

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
    async def list_databases(self) -> ListDatabasesResult:
        try:
            databases = await self._client.list_database_names()
            return ListDatabasesResult(databases=databases)
        except Exception as e:
            logger.exception(f"Failed to list databases: {str(e)}")
            raise

    @ensure_connection
    async def list_collections(self) -> ListCollectionsResult:
        try:
            collections = await self._db.list_collection_names()
            return ListCollectionsResult(collections=collections)
        except Exception as e:
            logger.exception(f"Failed to list collections: {str(e)}")
            raise

    @ensure_connection
    async def create_collection(self, collection_name: str, **kwargs: Any) -> CreateCollectionResult:
        try:
            await self._db.create_collection(name=collection_name, **kwargs)
            logger.info(f"Created collection {collection_name}")

            return CreateCollectionResult(collection_name=collection_name)
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def drop_collection(self, collection_name: str) -> DropCollectionResult:
        try:
            await self._db.drop_collection(collection_name)
            logger.info(f"Dropped collection {collection_name}")

            return DropCollectionResult(collection_name=collection_name)
        except Exception as e:
            logger.exception(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def rename_collection(self, collection_name: str, new_collection_name: str) -> RenameCollectionResult:
        try:
            collections = await self._db.list_collection_names()
            if collection_name not in collections:
                raise ValueError(f"Collection {collection_name} doesn't exist")
            if new_collection_name in collections:
                raise ValueError(f"Collection {new_collection_name} already exists")

            await self._db[collection_name].rename(new_collection_name)
            logger.info(f"Renamed collection {collection_name} to {new_collection_name}")

            return RenameCollectionResult(collection_name=collection_name, new_collection_name=new_collection_name)
        except Exception as e:
            logger.exception(f"Failed to rename collection {collection_name} to {new_collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def get_collection_stats(self, collection_name: str) -> GetCollectionStatsResult:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            collection_stats = await self._db.command("collStats", collection_name)
            return GetCollectionStatsResult(**collection_stats)
        except Exception as e:
            logger.exception(f"Failed to get statistics information for a collection {collection_name}: {str(e)}")
            raise

    @ensure_connection
    async def get_database_stats(self) -> GetDatabaseStats:
        try:
            database_stats = await self._db.command(command="dbStats")
            return GetDatabaseStats(**database_stats)
        except Exception as e:
            logger.exception(f"Failed to get statistics information for a database: {str(e)}")
            raise

    @ensure_connection
    async def list_indices(self, collection_name: str) -> GetIndicesResult:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            indices = await self._db[collection_name].list_indexes().to_list(length=None)
            logger.info(f"Listed {len(indices)} indices for {collection_name} collection")

            return GetIndicesResult(
                collection_name=collection_name,
                count=len(indices),
                indices=[IndexInfo(**item) for item in indices]
            )
        except Exception as e:
            logger.exception(f"Failed to list the indices for {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def create_index(self, collection_name: str, keys: dict[str, Any], **kwargs: Any) -> CreateIndexResult:
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

            return CreateIndexResult(collection_name=collection_name, index=index_name)
        except Exception as e:
            logger.exception(f"Failed to create index for {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def drop_index(self, collection_name: str, index_name: str) -> DropIndexInfoResult:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")
            if index_name == "_id_":
                raise ValueError(f"Cannot drop the default _id index from {collection_name} collection")

            await self._db[collection_name].drop_index(index_name)
            logger.info(f"Dropped index {index_name} from {collection_name} collection")

            return DropIndexInfoResult(collection_name=collection_name, index=index_name)
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
    async def ping_database(self) -> PingDatabaseResult:
        try:
            ping_result = await self._db.command(command="ping")
            logger.info(f"Pinged MongoDB database: {ping_result}")

            return PingDatabaseResult(**ping_result)
        except Exception as e:
            logger.exception(f"Failed to ping MongoDB database: {str(e)}")
            raise

    @ensure_connection
    async def insert_document(self, collection_name: str, document: dict[str, Any]) -> InsertDocumentResult:
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

            return InsertDocumentResult(collection_name=collection_name, document_id=str(result.inserted_id))
        except Exception as e:
            logger.exception(f"Failed to insert document into {collection_name} collection: {str(e)}")
            raise

    @ensure_connection
    async def insert_many_documents(
            self,
            collection_name: str,
            documents: list[dict[str, Any]],
            ordered: bool = True
    ) -> InsertManyDocumentsResult:
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

            return InsertManyDocumentsResult(
                collection_name=collection_name,
                document_ids=[str(oid) for oid in result.inserted_ids]
            )
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
    ) -> FindDocumentsResult:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            cursor = self._db[collection_name].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} documents in {collection_name} collection using query {query}")

            return FindDocumentsResult(
                collection_name=collection_name,
                documents=[process_query_result(document) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to find documents in {collection_name} collection using query {query}: {str(e)}")
            raise

    @ensure_connection
    async def count_documents(self, collection_name: str, query: dict[str, Any]) -> CountDocumentsResult:
        try:
            if collection_name not in await self._db.list_collection_names():
                raise ValueError(f"Collection {collection_name} doesn't exist")

            query = preprocess_operation(query)
            document_count = await self._db[collection_name].count_documents(query)
            logger.info(f"Counted {document_count} documents in {collection_name} collection using query {query}")

            return CountDocumentsResult(collection_name=collection_name, document_count=document_count)
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
    ) -> UpdateDocumentsResult:
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

            return UpdateDocumentsResult(collection_name=collection_name, updated_document_count=result.modified_count)
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
    ) -> ReplaceDocumentResult:
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

            return ReplaceDocumentResult(collection_name=collection_name, replaced_document_count=result.modified_count)
        except Exception as e:
            logger.exception(f"Failed to replace document in {collection_name} collection by query {query}: {str(e)}")
            raise

    @ensure_connection
    async def delete_documents(self, collection_name: str, query: dict[str, Any]) -> DeleteDocumentsResult:
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

            return DeleteDocumentsResult(collection_name=collection_name, deleted_document_count=result.deleted_count)
        except Exception as e:
            logger.exception(f"Failed to delete document in {collection_name} collection using query {query}: {str(e)}")
            raise

    @ensure_connection
    async def aggregate_documents(
            self,
            collection_name: str,
            pipeline: list[dict[str, Any]],
            options: dict[str, Any] | None = None
    ) -> AggregateDocumentsResult:
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

            return AggregateDocumentsResult(
                collection_name=collection_name,
                documents=[process_query_result(document) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to execute aggregation pipeline on {collection_name} collection: {str(e)}")
            raise

    @staticmethod
    def _build_date_range_filter(start_date: datetime | None = None, end_date: datetime | None = None) -> dict[str, Any]:
        date_filter = {}

        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date

        return date_filter

    @staticmethod
    def _exclude_fields(projection: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
        projection = dict(projection) if projection else {}
        is_inclusion = any(value for key, value in projection.items() if key != "_id")

        for field in fields:
            if is_inclusion:
                projection.pop(field, None)
            else:
                projection[field] = 0

        # An inclusion projection that only named excluded fields would otherwise return every field
        if not projection:
            projection = {field: 0 for field in fields}

        return projection

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
            query["date"] = self._build_date_range_filter(start_date=start_date, end_date=end_date)

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
    ) -> FindInspectionModelsResult:
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

            return FindInspectionModelsResult(
                models=[InspectionModelInfo(**process_query_result(document)) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to find inspection models: {str(e)}")
            raise

    @ensure_connection
    async def find_inspection_summary_documents(
            self,
            model_name: str | None = None,
            model_version: str | None = None,
            gbm: str | None = None,
            task: str | None = None,
            mode: str | None = None,
            process: str | None = None,
            location: str | None = None,
            equipment_id: str | None = None,
            product_id: str | None = None,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
            projection: dict[str, Any] | None = None,
            limit: int = LIMIT,
            sort_field: str | None = None,
            sort_order: int = ASCENDING
    ) -> FindInspectionSummaryDocumentsResult:
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

            if location:
                query["location"] = generate_normalized_regex(location)
            if equipment_id:
                query["equipmentId"] = equipment_id
            if product_id:
                query["productId"] = product_id

            if DBCollections.INSPECTIONS_SUMMARY not in await self._db.list_collection_names():
                raise ValueError(f"Collection {DBCollections.INSPECTIONS_SUMMARY} doesn't exist")

            cursor = self._db[DBCollections.INSPECTIONS_SUMMARY].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} summary documents using query {query}")

            return FindInspectionSummaryDocumentsResult(
                summaries=[InspectionSummaryDocument(**process_query_result(document)) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to find inspection result summaries: {str(e)}")
            raise

    @ensure_connection
    async def find_dataset_family_documents(
            self,
            dataset_family_name: str | None = None,
            task: str | None = None,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
            projection: dict[str, Any] | None = None,
            limit: int = LIMIT,
            sort_field: str | None = None,
            sort_order: int = ASCENDING
    ) -> FindDatasetFamilyDocumentsResult:
        try:
            # Dataset families share the collection with dataset documents, only families carry a family name
            query = {"datasetFamilyName": {"$exists": True}}

            if dataset_family_name:
                query["datasetFamilyName"] = generate_normalized_regex(dataset_family_name)
            if task:
                query["task"] = task
            if start_date or end_date:
                query["createdAt"] = self._build_date_range_filter(start_date=start_date, end_date=end_date)

            if DBCollections.DATASETS not in await self._db.list_collection_names():
                raise ValueError(f"Collection {DBCollections.DATASETS} doesn't exist")

            cursor = self._db[DBCollections.DATASETS].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} dataset family documents using query {query}")

            return FindDatasetFamilyDocumentsResult(
                families=[DatasetFamilyDocument(**process_query_result(document)) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to find dataset families: {str(e)}")
            raise

    @ensure_connection
    async def find_dataset_documents(
            self,
            name: str | None = None,
            version: str | None = None,
            task: str | None = None,
            created_by: str | None = None,
            is_finalized: bool | None = None,
            is_used: bool | None = None,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
            projection: dict[str, Any] | None = None,
            limit: int = LIMIT,
            sort_field: str | None = None,
            sort_order: int = ASCENDING
    ) -> FindDatasetDocumentsResult:
        try:
            # Dataset documents share the collection with dataset families, only documents carry a schema version
            query = {"schemaVersion": {"$exists": True}}

            if name:
                query["name"] = generate_normalized_regex(name)
            if version:
                query["version"] = generate_normalized_regex(version)
            if task:
                query["task"] = task
            if created_by:
                query["createdBy"] = generate_normalized_regex(created_by)
            if is_finalized is not None:
                query["isFinalized"] = is_finalized
            if is_used is not None:
                query["isUsed"] = is_used
            if start_date or end_date:
                query["createdAt"] = self._build_date_range_filter(start_date=start_date, end_date=end_date)

            # The data map holds an entry per data sample and the last job is transient, both are never returned
            projection = self._exclude_fields(projection, DATASET_EXCLUDED_FIELDS)

            if DBCollections.DATASETS not in await self._db.list_collection_names():
                raise ValueError(f"Collection {DBCollections.DATASETS} doesn't exist")

            cursor = self._db[DBCollections.DATASETS].find(query, projection=projection)
            if sort_field:
                cursor = cursor.sort(sort_field, sort_order)

            documents = await cursor.to_list(length=limit)
            logger.info(f"Found {len(documents)} dataset documents using query {query}")

            return FindDatasetDocumentsResult(
                datasets=[DatasetDocument(**process_query_result(document)) for document in documents]
            )
        except Exception as e:
            logger.exception(f"Failed to find datasets: {str(e)}")
            raise

    @ensure_connection
    async def analyze_data_drift(
            self,
            model_name: str,
            model_version: str,
            start_date: datetime,
            end_date: datetime,
            task: str | None = None,
            gbm: str | None = None,
            process: str | None = None,
            location: str | None = None,
            equipment_id: str | None = None,
            mode: str | None = "production",
            bucket: str = AUTO_BUCKET,
            detail: str = DriftDetail.FULL,
            reference_start_date: datetime | None = None,
            reference_end_date: datetime | None = None
    ) -> DriftAnalysisResult:
        """Streams the flattened predictions of one model and runs the drift analysis over them"""
        try:
            windows = DriftWindows(
                start_date=start_date, end_date=end_date, reference_start_date=reference_start_date, reference_end_date=reference_end_date
            )
            if task is not None and task not in list(DriftTask):
                raise UnsupportedTaskError(task)
            if bucket != AUTO_BUCKET and bucket not in list(BucketSize):
                sizes = ", ".join(size.value for size in BucketSize)
                raise ValueError(f"Unknown bucket {bucket!r}, use {AUTO_BUCKET}, {sizes}")
            if detail not in list(DriftDetail):
                levels = ", ".join(level.value for level in DriftDetail)
                raise ValueError(f"Unknown detail {detail!r}, use {levels}")

            if DBCollections.INSPECTIONS not in await self._db.list_collection_names():
                raise ValueError(f"Collection {DBCollections.INSPECTIONS} doesn't exist")

            model = f"{model_name}/{model_version}"
            filters = {"gbm": gbm, "process": process, "location": location, "equipment_id": equipment_id, "mode": mode}

            extractor = RecordExtractor(task=task)
            if windows.comparison:
                await self._extract_drift_records(
                    extractor, DriftWindow.REFERENCE, model, windows.reference_start_date, windows.reference_end_date, task, filters
                )
            await self._extract_drift_records(
                extractor, DriftWindow.CURRENT, model, windows.start_date, windows.end_date, task, filters
            )

            resolved_task = extractor.resolve_task(task)
            if not extractor.records:
                extractor.warn_once(f"No inspection results found for model {model} in the requested range")

            analyzer = DriftAnalyzer(
                model_name=model_name,
                model_version=model_version,
                task=resolved_task,
                extractor=extractor,
                windows=windows,
                filters=filters,
                bucket=bucket,
                detail=detail
            )
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, analyzer.run)
            logger.info(
                f"Analysed data drift for {model}: {result.data_quality.record_count} records, "
                f"{result.status.bucket_count} buckets, verdict {result.pre_verdict}"
            )
            return result
        except Exception as e:
            logger.exception(f"Failed to analyse data drift for {model_name}/{model_version}: {str(e)}")
            raise

    async def _extract_drift_records(
            self,
            extractor: RecordExtractor,
            window: DriftWindow,
            model: str,
            start_date: datetime,
            end_date: datetime,
            task: str | None,
            filters: dict[str, str | None]
    ) -> None:
        """Streams the rows of one window into the extractor"""
        query = DriftQuery(model, start_date, end_date, task=task, filters=filters)
        collection = self._db[DBCollections.INSPECTIONS]
        extractor.quality.scanned_document_count += await collection.count_documents(query.match())

        before_count = len(extractor.records)
        cursor = collection.aggregate(query.pipeline(), allowDiskUse=True)
        async for row in cursor:
            extractor.add(row, window)

        logger.info(
            f"Extracted {len(extractor.records) - before_count} {window} drift records for {model} "
            f"between {start_date} and {end_date}"
        )

    @ensure_connection
    async def close(self):
        try:
            self._client.close()
            logger.info("Closed MongoDB client")
        except Exception as e:
            logger.exception(f"Failed to close MongoDB client: {str(e)}")
            raise
