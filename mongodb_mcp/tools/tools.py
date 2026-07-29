import logging
from typing import Any
from pymongo import ASCENDING
from datetime import datetime
from fastmcp.exceptions import ToolError
from fastmcp.server import Context, FastMCP

from common.constants import LIMIT
from mongodb_mcp.schemas import ServerStatus

logger = logging.getLogger(__name__)

tools_mcp = FastMCP(name="tools")


@tools_mcp.tool()
async def mongodb_list_databases(ctx: Context) -> str:
    """List all databases in the MongoDB"""
    try:
        connector = ctx.request_context.lifespan_context.connector
        databases = await connector.list_databases()
        return f"Databases in MongoDB:\n{', '.join(databases)}"
    except Exception as e:
        raise ToolError(f"Failed to list databases: {str(e)}")


@tools_mcp.tool()
async def mongodb_list_collections(ctx: Context) -> str:
    """List all collections in the database"""
    try:
        connector = ctx.request_context.lifespan_context.connector
        collections = await connector.list_collections()
        return f"Collections in database:\n{', '.join(collections)}"
    except Exception as e:
        raise ToolError(f"Failed to list collections: {str(e)}")


@tools_mcp.tool()
async def mongodb_create_collection(
        ctx: Context,
        collection_name: str,
        options: dict[str, Any] | None = None
) -> str:
    """Create a new collection with optional settings

    Args:
        collection_name: Name of the collection to create
        options: Additional keyword arguments for the collection creation
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.create_collection(
            collection_name=collection_name,
            **(options if options is not None else {})
        )

        return f"Collection {collection_name} created successfully"
    except Exception as e:
        raise ToolError(f"Failed to create collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_drop_collection(ctx: Context, collection_name: str) -> str:
    """Delete a collection from the database

    Args:
        collection_name: Name of the collection to delete
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.drop_collection(collection_name=collection_name)

        return f"Collection {collection_name} dropped successfully"
    except Exception as e:
        raise ToolError(f"Failed to drop collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_rename_collection(ctx: Context, collection_name: str, new_collection_name: str) -> str:
    """Rename a collection

    Args:
        collection_name: Current name of the collection
        new_collection_name: New name for the collection
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.rename_collection(collection_name=collection_name, new_collection_name=new_collection_name)

        return f"Successfully renamed collection {collection_name} to {new_collection_name}"
    except Exception as e:
        raise ToolError(f"Failed to rename collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_collection_stats(ctx: Context, collection_name: str) -> dict[str, Any]:
    """Get statistics information for a collection

    Args:
        collection_name: Name of the collection
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_collection_stats(collection_name=collection_name)
    except Exception as e:
        raise ToolError(f"Failed to get statistics information for a collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_database_stats(ctx: Context) -> dict[str, Any]:
    """Get statistics information for a database"""
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_database_stats()
    except Exception as e:
        raise ToolError(f"Failed to get statistics information for a database: {str(e)}")


@tools_mcp.tool()
async def mongodb_list_indices(ctx: Context, collection_name: str) -> dict[str, Any]:
    """List all indices for the specified collection

    Args:
        collection_name: Name of the collection
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.list_indices(collection_name=collection_name)

        return {
            "collection": collection_name,
            "count": len(result),
            "indices": result
        }
    except Exception as e:
        raise ToolError(f"Failed to list the indices for {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_create_index(
        ctx: Context,
        collection_name: str,
        keys: dict[str, Any],
        options: dict[str, Any] | None = None
) -> str:
    """Create an index on the specified collection

    Args:
        collection_name: Name of the collection to create
        keys: Index key specification (e.g., {"field": 1} for ascending)
        options: Additional keyword arguments for the index creation
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.create_index(
            collection_name=collection_name,
            keys=keys,
            **(options if options is not None else {})
        )

        return f"Created index {result} for {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to create index for {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_drop_index(ctx: Context, collection_name: str, index_name: str) -> str:
    """Drop an index from the specified collection

    Args:
        collection_name: Name of the collection
        index_name: Name of the index to drop
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.drop_index(collection_name=collection_name, index_name=index_name)

        return f"Dropped index {index_name} from {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to drop index {index_name} from {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_get_server_status(ctx: Context) -> ServerStatus:
    """Get MongoDB server status information"""
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_server_status()
    except Exception as e:
        raise ToolError(f"Failed to get server status information: {str(e)}")


@tools_mcp.tool()
async def mongodb_ping_database(ctx: Context) -> dict[str, Any]:
    """Test database connection"""
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.ping_database()
    except Exception as e:
        raise ToolError(f"Failed to ping MongoDB database: {str(e)}")


@tools_mcp.tool()
async def mongodb_insert_document(ctx: Context, collection_name: str, document: dict[str, Any]) -> str:
    """Insert a document into the specified collection

    Args:
        collection_name: Name of the collection
        document: Document to insert (JSON-compatible dictionary)
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.insert_document(collection_name=collection_name, document=document)

        return f"Inserted document with id {result} into {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to insert document into {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_insert_many_documents(
        ctx: Context,
        collection_name: str,
        documents: list[dict[str, Any]],
        ordered: bool = True
) -> str:
    """Insert multiple documents into the specified collection

    Args:
        collection_name: Name of the collection
        documents: List of documents to insert (list of JSON-compatible dictionary)
        ordered: Whether to perform ordered or unordered inserts
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.insert_many_documents(
            collection_name=collection_name,
            documents=documents,
            ordered=ordered
        )

        return f"Inserted documents into {collection_name} collection with ids: {result}"
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
) -> list[dict[str, Any]]:
    """Find documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        projection: Projection (fields to include/exclude)
        limit: Maximum number of documents to return (5 is default)
        sort_field: Field name to sort results based on
        sort_order: Result sorting order
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
async def mongodb_count_documents(ctx: Context, collection_name: str, query: dict[str, Any]) -> str:
    """Count documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.count_documents(collection_name=collection_name, query=query)

        return f"Counted {result} documents in {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to count documents in {collection_name} collection using query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_update_documents(
        ctx: Context,
        collection_name: str,
        query: dict[str, Any],
        update_operation: dict[str, Any],
        upsert: bool = False
) -> str:
    """Update documents in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        update_operation: Update operation document (must include operators like $set)
        upsert: Whether to insert if no document matches the query
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.update_documents(
            collection_name=collection_name,
            query=query,
            update_operation=update_operation,
            upsert=upsert
        )

        return f"Updated {result} documents in {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to update documents in {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def mongodb_replace_document(
        ctx: Context,
        collection_name: str,
        query: dict[str, Any],
        document: dict[str, Any],
        upsert: bool = False
) -> str:
    """Replace a single document in the specified collection matching the query

    Args:
        collection_name: Name of the collection
        query: Document query filter
        document: Replacement document (should not contain update operators)
        upsert: Whether to insert if no document matches the query
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.replace_document(
            collection_name=collection_name,
            query=query,
            document=document,
            upsert=upsert
        )

        return f"Replaced {result} documents in {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to replace document in {collection_name} collection by query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_delete_documents(ctx: Context, collection_name: str, query: dict[str, Any]) -> str:
    """Delete document from the specified collection matching the query
    Args:
        collection_name: Name of the collection
        query: Document query filter
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.delete_documents(collection_name=collection_name, query=query)

        return f"Deleted {result} documents in {collection_name} collection"
    except Exception as e:
        raise ToolError(f"Failed to delete documents in {collection_name} collection using query {query}: {str(e)}")


@tools_mcp.tool()
async def mongodb_aggregate_documents(
        ctx: Context,
        collection_name: str,
        pipeline: list[dict[str, Any]],
        options: dict[str, Any] | None = None
):
    """Execute an aggregation pipeline on the specified collection

    Args:
        collection_name: Name of the collection
        pipeline: MongoDB aggregation pipeline (list of stage dictionaries)
        options: Aggregation options (allowDiskUse, maxTimeMS, etc.)
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.aggregate_documents(
            collection_name=collection_name,
            pipeline=pipeline,
            options=options
        )

        return result
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
) -> list[dict[str, Any]]:
    """Find inspection models with optional filters

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
