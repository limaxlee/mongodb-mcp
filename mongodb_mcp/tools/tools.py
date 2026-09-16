import logging
from typing import Any
from pymongo import ASCENDING
from datetime import datetime
from fastmcp.exceptions import ToolError
from fastmcp.server import Context, FastMCP

from common.constants import LIMIT
from mongodb_mcp.schemas import *

logger = logging.getLogger(__name__)

tools_mcp = FastMCP(name="tools")


@tools_mcp.tool()
async def mongodb_list_databases(ctx: Context) -> ListDatabasesResult:
    """List all databases in the MongoDB

    Returns:
        Database list with the following fields:
            databases: Names of every database in the MongoDB
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.list_databases()
    except Exception as e:
        raise ToolError(f"Failed to list databases: {str(e)}")


@tools_mcp.tool()
async def mongodb_list_collections(ctx: Context) -> ListCollectionsResult:
    """List all collections in the database

    Returns:
        Collection list with the following fields:
            collections: Names of every collection in the configured database
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.list_collections()
    except Exception as e:
        raise ToolError(f"Failed to list collections: {str(e)}")


@tools_mcp.tool()
async def mongodb_create_collection(
        ctx: Context,
        collection_name: str,
        options: dict[str, Any] | None = None
) -> CreateCollectionResult:
    """Create a new collection with optional settings

    Args:
        collection_name: Name of the collection to create
        options: Additional keyword arguments for the collection creation

    Returns:
        Collection creation result with the following fields:
            collection_name: Name of the created collection
            collection_created: Always true, a failed creation raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.create_collection(
            collection_name=collection_name,
            **(options if options is not None else {})
        )
    except Exception as e:
        raise ToolError(f"Failed to create collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_drop_collection(ctx: Context, collection_name: str) -> DropCollectionResult:
    """Delete a collection from the database

    Args:
        collection_name: Name of the collection to delete

    Returns:
        Collection drop result with the following fields:
            collection_name: Name of the dropped collection
            collection_dropped: Always true, a failed drop raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.drop_collection(collection_name=collection_name)
    except Exception as e:
        raise ToolError(f"Failed to drop collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_rename_collection(
        ctx: Context,
        collection_name: str,
        new_collection_name: str
) -> RenameCollectionResult:
    """Rename a collection

    Args:
        collection_name: Current name of the collection
        new_collection_name: New name for the collection

    Returns:
        Collection rename result with the following fields:
            collection_name: Previous name of the collection
            new_collection_name: New name of the collection
            collection_renamed: Always true, a failed rename raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.rename_collection(
            collection_name=collection_name,
            new_collection_name=new_collection_name
        )
    except Exception as e:
        raise ToolError(f"Failed to rename collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_collection_stats(ctx: Context, collection_name: str) -> GetCollectionStatsResult:
    """Get statistics information for a collection

    Args:
        collection_name: Name of the collection

    Returns:
        Raw collStats document returned by MongoDB, whose fields vary by server version and storage engine,
        commonly including:
            ns: Namespace of the collection, in the form of database.collection
            count: Number of documents in the collection
            size: Total uncompressed size of the documents in bytes
            avgObjSize: Average uncompressed size of a document in bytes
            storageSize: Total size allocated for the collection on disk in bytes
            nindexes: Number of indexes on the collection
            totalIndexSize: Total size of all indexes on the collection in bytes
            indexSizes: Size of each index in bytes, keyed by index name
            ok: 1 when the command succeeded
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_collection_stats(collection_name=collection_name)
    except Exception as e:
        raise ToolError(f"Failed to get statistics information for a collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_database_stats(ctx: Context) -> GetDatabaseStats:
    """Get database statistics information

    Returns:
        Database statistics with the following fields:
            db: Name of the database
            collections: Number of collections in the database
            views: Number of views in the database
            objects: Number of documents across all collections in the database
            indexes: Number of indexes across all collections in the database
            indexSize: Total size of all indexes in bytes
            totalSize: Total size of all collections and indexes in bytes
            scaleFactor: Scale factor the reported sizes are divided by
            fsUsedSize: Used size of the filesystem the database is stored on in bytes
            fsTotalSize: Total size of the filesystem the database is stored on in bytes
            ok: 1 when the command succeeded
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_database_stats()
    except Exception as e:
        raise ToolError(f"Failed to get statistics information for a database: {str(e)}")


@tools_mcp.tool()
async def mongodb_list_indices(ctx: Context, collection_name: str) -> GetIndicesResult:
    """List all indices for the specified collection

    Args:
        collection_name: Name of the collection

    Returns:
        Index list with the following fields:
            collection_name: Name of the collection the indices belong to
            count: Number of indices on the collection
            indices: List of indices, each with the following fields:
                v: Version of the index
                key: Index key specification, mapping each indexed field to its direction or index type
                  (e.g. 1, -1, "text", "2dsphere", "hashed")
                name: Name of the index
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.list_indices(collection_name=collection_name)
    except Exception as e:
        raise ToolError(f"Failed to list the indices for {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_create_index(
        ctx: Context,
        collection_name: str,
        keys: dict[str, Any],
        options: dict[str, Any] | None = None
) -> CreateIndexResult:
    """Create an index on the specified collection

    Args:
        collection_name: Name of the collection to create
        keys: Index key specification (e.g., {"field": 1} for ascending)
        options: Additional keyword arguments for the index creation

    Returns:
        Index creation result with the following fields:
            collection_name: Name of the collection the index was created on
            index: Name of the created index
            index_created: Always true, a failed creation raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.create_index(
            collection_name=collection_name,
            keys=keys,
            **(options if options is not None else {})
        )
    except Exception as e:
        raise ToolError(f"Failed to create index for {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_drop_index(ctx: Context, collection_name: str, index_name: str) -> DropIndexInfoResult:
    """Drop an index from the specified collection

    Args:
        collection_name: Name of the collection
        index_name: Name of the index to drop

    Returns:
        Index drop result with the following fields:
            collection_name: Name of the collection the index was dropped from
            index: Name of the dropped index
            index_dropped: Always true, a failed drop raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.drop_index(collection_name=collection_name, index_name=index_name)
    except Exception as e:
        raise ToolError(f"Failed to drop index {index_name} from {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_server_status(ctx: Context) -> ServerStatus:
    """Get MongoDB server status information

    Returns:
        Server status with the following fields, each empty when the server did not report it:
            host: Hostname and port of the server
            version: Version of the server
            process: Process serving the request (e.g. "mongod", "mongos")
            pid: Process id of the server
            uptime: Uptime of the server in seconds
            uptimeMillis: Uptime of the server in milliseconds
            localTime: Current time of the server in UTC
            connections: Connection statistics with the following fields:
                current: Number of currently open incoming connections
                available: Number of incoming connections still available
                totalCreated: Number of connections created since the server started
            extra_info: Platform specific statistics with the following fields:
                note: Note the server attaches to the platform specific statistics
                heap_usage_bytes: Heap space used by the server process in bytes
                page_faults: Number of page faults since the server started
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_server_status()
    except Exception as e:
        raise ToolError(f"Failed to get server status information: {str(e)}")


@tools_mcp.tool()
async def mongodb_ping_database(ctx: Context) -> PingDatabaseResult:
    """Test database connection

    Returns:
        Ping result with the following fields:
            ok: 1 when the database answered the ping, an unreachable database raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.ping_database()
    except Exception as e:
        raise ToolError(f"Failed to ping MongoDB database: {str(e)}")


@tools_mcp.tool()
async def mongodb_insert_document(ctx: Context, collection_name: str, document: dict[str, Any]) -> InsertDocumentResult:
    """Insert a document into the specified collection

    Args:
        collection_name: Name of the collection
        document: Document to insert (JSON-compatible dictionary)

    Returns:
        Insert result with the following fields:
            collection_name: Name of the collection the document was inserted into
            document_id: Id assigned to the inserted document
            document_inserted: Always true, a failed insert raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.insert_document(collection_name=collection_name, document=document)
    except Exception as e:
        raise ToolError(f"Failed to insert document into {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_insert_many_documents(
        ctx: Context,
        collection_name: str,
        documents: list[dict[str, Any]],
        ordered: bool = True
) -> InsertManyDocumentsResult:
    """Insert multiple documents into the specified collection

    Args:
        collection_name: Name of the collection
        documents: List of documents to insert (list of JSON-compatible dictionary)
        ordered: Whether to perform ordered or unordered inserts

    Returns:
        Insert result with the following fields:
            collection_name: Name of the collection the documents were inserted into
            document_ids: Ids assigned to the inserted documents, in the order they were given
            documents_inserted: Always true, a failed insert raises error instead
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.insert_many_documents(
            collection_name=collection_name,
            documents=documents,
            ordered=ordered
        )
    except Exception as e:
        raise ToolError(f"Failed to insert documents into {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_find_documents(
        ctx: Context,
        collection_name: str,
        query: dict[str, Any],
        projection: dict[str, Any] | None = None,
        limit: int = LIMIT,
        sort_field: str | None = None,
        sort_order: int = ASCENDING
) -> FindDocumentsResult:
    """Find documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        projection: Projection (fields to include/exclude)
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order

    Returns:
        Find result with the following fields:
            collection_name: Name of the searched collection
            documents: Matching documents with their ids as strings, empty when the query matched nothing
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.find_documents(
            collection_name=collection_name,
            query=query,
            projection=projection,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    except Exception as e:
        raise ToolError(f"Failed to find documents in {collection_name} collection using query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_count_documents(ctx: Context, collection_name: str, query: dict[str, Any]) -> CountDocumentsResult:
    """Count documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter

    Returns:
        Count result with the following fields:
            collection_name: Name of the counted collection
            document_count: Number of documents matching the query
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.count_documents(collection_name=collection_name, query=query)
    except Exception as e:
        raise ToolError(f"Failed to count documents in {collection_name} collection using query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_update_documents(
        ctx: Context,
        collection_name: str,
        query: dict[str, Any],
        update_operation: dict[str, Any],
        upsert: bool = False
) -> UpdateDocumentsResult:
    """Update documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        update_operation: Update operation document (must include operators like $set)
        upsert: Whether to insert if no document matches the query

    Returns:
        Update result with the following fields:
            collection_name: Name of the updated collection
            updated_document_count: Number of documents modified by the update operation
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.update_documents(
            collection_name=collection_name,
            query=query,
            update_operation=update_operation,
            upsert=upsert
        )
    except Exception as e:
        raise ToolError(f"Failed to update documents in {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_replace_document(
        ctx: Context,
        collection_name: str,
        query: dict[str, Any],
        document: dict[str, Any],
        upsert: bool = False
) -> ReplaceDocumentResult:
    """Replace a single document in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        document: Replacement document (should not contain update operators)
        upsert: Whether to insert if no document matches the query

    Returns:
        Replace result with the following fields:
            collection_name: Name of the collection the document was replaced in
            replaced_document_count: Number of documents replaced by the operation
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.replace_document(
            collection_name=collection_name,
            query=query,
            document=document,
            upsert=upsert
        )
    except Exception as e:
        raise ToolError(f"Failed to replace document in {collection_name} collection by query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_delete_documents(ctx: Context, collection_name: str, query: dict[str, Any]) -> DeleteDocumentsResult:
    """Delete document from the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter

    Returns:
        Delete result with the following fields:
            collection_name: Name of the collection the documents were deleted from
            deleted_document_count: Number of documents deleted by the operation
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.delete_documents(collection_name=collection_name, query=query)
    except Exception as e:
        raise ToolError(f"Failed to delete documents in {collection_name} collection using query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_aggregate_documents(
        ctx: Context,
        collection_name: str,
        pipeline: list[dict[str, Any]],
        options: dict[str, Any] | None = None
) -> AggregateDocumentsResult:
    """Execute an aggregation pipeline on the specified collection

    Args:
        collection_name: Name of the collection
        pipeline: MongoDB aggregation pipeline (list of stage dictionaries)
        options: Aggregation options (allowDiskUse, maxTimeMS, etc.)

    Returns:
        Aggregation result with the following fields:
            collection_name: Name of the aggregated collection
            documents: Documents produced by the pipeline with their ids as strings, empty when the pipeline
              produced nothing
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.aggregate_documents(
            collection_name=collection_name,
            pipeline=pipeline,
            options=options
        )
    except Exception as e:
        raise ToolError(f"Failed to execute aggregation pipeline on {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_find_inspection_models(
        ctx: Context,
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
    """Find inspection models information with optional filters

    All filter arguments are optional. Omit any you don't want to filter on.

    Args:
        model_name: Name of the inspection model
        model_version: Version of the inspection model
        gbm: Manufacturing site where model was deployed
        task: Task the inspection model performs
        mode: Operating mode of the inspection model
        process: Process line where the model was deployed
        start_date: Lower range for model deployment date
        end_date: Upper range for model deployment date
        projection: Projection (fields to include/exclude)
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order

    Returns:
        Inspection model list with the following fields:
            models: Matching inspection models, empty when the filters matched nothing, each with the
              following fields:
                modelName: Name of the inspection model
                modelVersion: Version of the inspection model
                process: Process line where the model was deployed
                task: Task the inspection model performs
                gbm: Manufacturing site where the model was deployed
                mode: Operating mode of the inspection model
                date: Deployment date of the inspection model
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.find_inspection_models(
            model_name=model_name,
            model_version=model_version,
            gbm=gbm,
            task=task,
            mode=mode,
            process=process,
            start_date=start_date,
            end_date=end_date,
            projection=projection,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    except Exception as e:
        raise ToolError(f"Failed to find inspection models: {str(e)}")


@tools_mcp.tool()
async def mongodb_find_inspection_summary_documents(
        ctx: Context,
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
    """Find inspection results summaries with optional filters

    All filter arguments are optional. Omit any you don't want to filter on. The per data sample details of an
    inspection are left out of the results, only the statistics aggregated over them are returned.

    Args:
        model_name: Name of the inspection model the summary was produced by
        model_version: Version of the inspection model the summary was produced by
        gbm: Manufacturing site where the inspection ran
        task: Task the inspection model performs
        mode: Operating mode the inspection ran in
        process: Process line where the inspection ran
        location: Location within the process line where the inspection ran
        equipment_id: Id of the equipment the inspection ran on
        product_id: Id of the inspected product
        start_date: Lower range for inspection date
        end_date: Upper range for inspection date
        projection: Projection (fields to include/exclude), samples are excluded when omitted
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order

    Returns:
        Inspection summary list with the following fields:
            summaries: Matching inspection summaries, empty when the filters matched nothing, each with the
              following fields:
                modelName: Name of the inspection model
                modelVersion: Version of the inspection model
                gbm: Manufacturing site where the inspection ran
                process: Process line where the inspection ran
                mode: Operating mode the inspection ran in
                date: Date the inspection ran on
                location: Location within the process line where the inspection ran
                equipmentId: Id of the equipment the inspection ran on
                productId: Id of the inspected product
                localTimezone: Timezone of the site the inspection ran on
                task: Task the inspection model performs
                classes: Prediction classes the inspection model can output
                threshold: Confidence threshold the inspection decided with
                statistics: Statistics aggregated over the inspection results, with the following fields:
                    dataCount: Number of inspected data samples per prediction class
                    confidence: Confidence statistics per prediction class, each with the following fields:
                        avg: Average confidence of the class
                        min: Lowest confidence of the class
                        max: Highest confidence of the class
                        sum: Total confidence of the class
                    elapsedTime: Inference time statistics in seconds, with the same fields as confidence
                samples: Sample inspection result for each prediction class.
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.find_inspection_summary_documents(
            model_name=model_name,
            model_version=model_version,
            gbm=gbm,
            task=task,
            mode=mode,
            process=process,
            location=location,
            equipment_id=equipment_id,
            product_id=product_id,
            start_date=start_date,
            end_date=end_date,
            projection=projection,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    except Exception as e:
        raise ToolError(f"Failed to find inspection summaries: {str(e)}")


@tools_mcp.tool()
async def mongodb_find_dataset_families_documents(
        ctx: Context,
        dataset_family_name: str | None = None,
        task: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        projection: dict[str, Any] | None = None,
        limit: int = LIMIT,
        sort_field: str | None = None,
        sort_order: int = ASCENDING
) -> FindDatasetFamilyDocumentsResult:
    """Find dataset families with optional filters

    All filter arguments are optional. Omit any you don't want to filter on. A dataset family groups every version
    of a dataset, each version is listed as a member together with the id of its dataset document.

    Args:
        dataset_family_name: Name of the dataset family
        task: Task the datasets of the family are made for
        start_date: Lower range for the family creation date
        end_date: Upper range for the family creation date
        projection: Projection (fields to include/exclude)
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order

    Returns:
        Dataset family list with the following fields:
            families: Matching dataset families, empty when the filters matched nothing, each with the
              following fields:
                familyId: Id of the dataset family
                datasetFamilyName: Name of the dataset family
                task: Task the datasets of the family are made for
                members: Dataset versions belonging to the family, each with the following fields:
                    datasetId: Id of the dataset document of the version
                    version: Version of the dataset
                    description: Description of the dataset version
                accessControl: Who can access the family, with the following fields:
                    groups: Ids of the groups allowed to access the family
                    users: Ids of the users allowed to access the family
                createdAt: Date the family was created on
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.find_dataset_family_documents(
            dataset_family_name=dataset_family_name,
            task=task,
            start_date=start_date,
            end_date=end_date,
            projection=projection,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    except Exception as e:
        raise ToolError(f"Failed to find dataset families: {str(e)}")


@tools_mcp.tool()
async def mongodb_find_dataset_documents(
        ctx: Context,
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
    """Find dataset documents with optional filters

    All filter arguments are optional. Omit any you don't want to filter on. The per data sample map of a dataset
    and its last job are always left out of the results, only the summary fields are returned. Each dataset
    document is one version of a dataset family.

    Args:
        name: Name of the dataset
        version: Version of the dataset
        task: Task the dataset is made for
        created_by: User who created the dataset
        is_finalized: Whether the dataset is finalized
        is_used: Whether the dataset has been used
        start_date: Lower range for the dataset creation date
        end_date: Upper range for the dataset creation date
        projection: Projection (fields to include/exclude), dataMap and lastJob are always excluded
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order

    Returns:
        Dataset list with the following fields:
            datasets: Matching datasets, empty when the filters matched nothing, each with the following fields:
                documentId: Id of the dataset document
                name: Name of the dataset
                version: Version of the dataset
                task: Task the dataset is made for
                description: Description of the dataset
                createdBy: User who created the dataset
                projects: Projects the dataset is used in
                accessControl: Who can access the dataset, with the following fields:
                    groups: Ids of the groups allowed to access the dataset
                    users: Ids of the users allowed to access the dataset
                familyId: Id of the dataset family the dataset belongs to
                schemaVersion: Version of the dataset document schema
                createdAt: Date the dataset was created on
                modifiedAt: Date the dataset was last modified on
                isFinalized: Whether the dataset is finalized
                finalizedAt: Date the dataset was finalized on
                isUsed: Whether the dataset has been used
                downloadCount: Number of times the dataset was downloaded
                downloadUri: Uri of the downloadable dataset archive
                dataCount: Number of data samples in the dataset
                classes: Label classes of the dataset keyed by class name, each with the following fields:
                    count: Number of labels of the class
                    color: Display color of the class
                    shape: Label shape of the class
                trainingRecords: Trainings the dataset was used in, each with the following fields:
                    aiModel: Name of the trained model
                    version: Version of the trained model
                    startTime: Time the training started at
                    endTime: Time the training ended at
                    status: Final status of the training
                    trainingInfo: Additional training information
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.find_dataset_documents(
            name=name,
            version=version,
            task=task,
            created_by=created_by,
            is_finalized=is_finalized,
            is_used=is_used,
            start_date=start_date,
            end_date=end_date,
            projection=projection,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    except Exception as e:
        raise ToolError(f"Failed to find datasets: {str(e)}")


@tools_mcp.tool()
async def mongodb_analyze_data_drift(
        ctx: Context,
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
        bucket: str = "auto",
        detail: str = "full",
        reference_start_date: datetime | None = None,
        reference_end_date: datetime | None = None
) -> DriftAnalysisResult:
    """Compute the statistics needed to decide whether an inspection model's input data or behaviour drifted

    Works on the raw inspection results of one classification (cls) or object detection (det) model. Segmentation
    models are not supported. Two modes exist:
      - Range mode (default): analyses start_date to end_date, splits it into time buckets, searches for the moment
        the output distribution changed the most (change point), for gradual trends, and for outlier buckets.
      - Comparison mode: when reference_start_date and reference_end_date are also given, the reference window is compared
        against the current window at a fixed split instead of searching for one.
    The windows together may cover at most 30 days. Statistics are computed over every prediction of the model, a
    document contributes one record per matching aiResults entry and prediction. For detection models the
    confidence and class statistics are computed over bounding boxes, a list of confidences is reduced to its
    maximum. Nothing here is a verdict; the flags and pre-verdict are deterministic rules to be confirmed or
    overruled by reading the statistics.

    Args:
        model_name: Name of the inspection model, matched exactly together with the version as name/version
        model_version: Version of the inspection model
        start_date: Start of the analysed (current) window, inclusive, UTC like every date in and out of this tool
        end_date: End of the analysed (current) window, inclusive
        task: Task of the model, cls or det, required only when the model string is used for both tasks
        gbm: Manufacturing site the inspections ran at, exact match
        process: Process the inspections ran in, exact match
        location: Location within the process, exact match
        equipment_id: Id of the equipment the inspections ran on, exact match
        mode: Operating mode of the inspections, production by default, pass null to include every mode
        bucket: Time bucket size, auto (default), 1h, 1d or 1w; auto picks by range length and volume
        detail: full (default) keeps every histogram and quantile per bucket, compact keeps only the scalar series
        reference_start_date: Start of the reference window for comparison mode, inclusive
        reference_end_date: End of the reference window for comparison mode, inclusive, must not be after start_date

    Returns:
        Drift analysis with the following fields:
            modelName, modelVersion, task: The analysed model
            mode: range or comparison
            filters: The metadata filters that were applied
            range, referenceRange: Analysed windows with startDate, endDate and length in days
            bucket: The resolved bucket size
            status: What could be computed, with the following fields:
                analysisPossible: Whether a change point or comparison was computed
                changePointRan, comparisonRan, trendRan, outlierRan: Which analyses ran
                bucketCount: Number of buckets after merging small ones
            dataQuality: Volume and problems of the scanned data, with the following fields:
                scannedDocumentCount: Documents matching the filters
                matchedDocumentCount, matchedEntryCount: Documents and aiResults entries that produced records
                recordCount: Predictions analysed (images for detection)
                boxCount: Bounding boxes analysed, detection only
                missingConfidenceCount, missingImageSpecCount, parseErrorCount, parseErrorExamples: Skipped or degraded data
                mergedBuckets: Start times of buckets merged into a neighbour for being too small
            classes: Every class the model output
            buckets: Chronological per bucket statistics, each with the following fields:
                startDate, endDate: UTC boundaries, window: current or reference
                recordCount: Records in the bucket, mergedFrom: How many raw buckets were merged into it
                classDistribution: Share of every class
                medianConfidence, meanConfidence, stdConfidence, confidenceHistogram, confidenceQuantiles: Confidence statistics, confidenceHistogram uses fixed
                  bins [0,0.1) ... [0.9,1.0] so buckets are comparable
                belowThresholdRate: Share of predictions below their threshold, thresholdValues: Thresholds seen
                imageSpecs: Distinct (width, height, channels) seen, medianElapsedTime: Median inference time
                nearThresholdRate: Classification only
                boxCount, meanBoxesPerImage, stdBoxesPerImage, boxesPerImageHistogram, noBoxRate, boxesByClassPerImage,
                  box: Detection only, box holds normalised geometry quantiles and a log10 area histogram
            changePoint: Range mode, the split with the largest divergence, searched over the buckets that are not
              outliers so that a transient bucket does not read as a persistent shift, with the following fields:
                bucketIndex, date: Where the after side starts, score: Sum of PSI values used for the search
                candidateCount: Split positions the search tried; the p-values below are corrected for that choice
                beforeCount, afterCount, sidesSufficient: Sample sizes and whether both sides are large enough to trust
                psiConfidence, jsConfidence, ksConfidence: Confidence divergence (PSI < 0.1 none, 0.1-0.25 moderate, > 0.25 large)
                psiClass, chi2Class, maxClassProportionChange: Class distribution divergence
                psiBoxesPerImage, ksNormalizedArea, ksNormalizedCx, ksNormalizedCy: Detection geometry and box count divergence
                before, after: Summary of each side including its histograms, so the direction of a move is visible
            secondaryChangePoints: Further splits found on either side of the primary one
            comparison: Comparison mode, reference (before) versus current (after) with the changePoint fields
            maxPairwise: Largest PSI between any two buckets for confidence and classes, with the pair
            outlierBuckets: Buckets whose value in a series is far from the others (robust z-score above 3.5)
            trend: Per series Kendall tau, p-value, Theil-Sen slope per bucket, first and last value, and
              whether the trend is meaningful; the series are the scalar bucket fields plus class_share_<class>
              for every class, no class is treated as good or defect
            hardBreaks: Silent configuration changes: image_spec, threshold, backend, classes, elapsed_time; a
              detection threshold is tracked per class and the break carries the className
            flags: Deterministic rule results such as CONFIDENCE_SHIFT, CLASS_SHIFT, BOX_GEOMETRY_SHIFT,
              THRESHOLD_PRESSURE, TREND_<SERIES>, TREND_CLASS_SHARE, TRANSIENT_OUTLIER, HARD_BREAK,
              INSUFFICIENT_DATA, INSUFFICIENT_BUCKETS
            preVerdict: stable, suspicious, drift_likely or undetermined, derived from the flags; trends count per
              family (confidence, class share, box count, box geometry), not per series
            config: Every threshold used, for reproducibility
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.analyze_data_drift(
            model_name=model_name,
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
            task=task,
            gbm=gbm,
            process=process,
            location=location,
            equipment_id=equipment_id,
            mode=mode,
            bucket=bucket,
            detail=detail,
            reference_start_date=reference_start_date,
            reference_end_date=reference_end_date
        )
    except Exception as e:
        raise ToolError(f"Failed to analyse data drift: {str(e)}")
