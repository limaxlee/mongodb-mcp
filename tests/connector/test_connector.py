import pytest
from datetime import datetime
from pymongo.errors import AutoReconnect

from common.constants import LIMIT, EXCLUDE_SAMPLES, DBCollections
from mongodb_mcp.connector import MongoDBConnector


class TestMongoDBConnector:
    @pytest.mark.asyncio
    async def test_ensure_connection(self, mock_connector, mocker):
        mocker.patch("asyncio.sleep", new=mocker.AsyncMock())
        mock_connector._mock_client.list_database_names.side_effect = [
            AutoReconnect("Server timeout"),
            ["default", "test_db"]
        ]
        result = await mock_connector.list_databases()
        assert result.databases == ["default", "test_db"]
        assert mock_connector._mock_client.list_database_names.await_count == 2

    def test_client_init(self, mocker):
        mock_client = mocker.patch("mongodb_mcp.connector.connector.AsyncIOMotorClient")
        connector = MongoDBConnector()
        assert connector._client is mock_client.return_value

    @pytest.mark.asyncio
    async def test_list_databases(self, mock_connector):
        result = await mock_connector.list_databases()
        assert result.databases == ["default", "test_db"]

        mock_connector._mock_client.list_database_names.assert_awaited_once()

        mock_connector._mock_client.list_database_names.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_databases()

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_connector):
        result = await mock_connector.list_collections()
        assert result.collections == ["collection_a", "collection_b"]

        mock_connector._mock_db.list_collection_names.assert_awaited_once()

        mock_connector._mock_db.list_collection_names.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_collections()

    @pytest.mark.asyncio
    async def test_create_collection(self, mock_connector):
        result = await mock_connector.create_collection("collection_c")
        assert result.collection_name == "collection_c"
        assert result.collection_created is True
        assert mock_connector._mock_db.create_collection.await_count == 1

        mock_connector._mock_db.create_collection.side_effect = ValueError("Collection collection_a already exists")
        with pytest.raises(ValueError, match="Collection collection_a already exists"):
            await mock_connector.create_collection("collection_a")

        mock_connector._mock_db.create_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.create_collection("collection_d")

    @pytest.mark.asyncio
    async def test_drop_collection(self, mock_connector):
        result = await mock_connector.drop_collection("collection_c")
        assert result.collection_name == "collection_c"
        assert result.collection_dropped is True
        assert mock_connector._mock_db.drop_collection.await_count == 1

        mock_connector._mock_db.drop_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.drop_collection("collection_d")

    @pytest.mark.asyncio
    async def test_rename_collection(self, mock_connector):
        result = await mock_connector.rename_collection("collection_a", "collection_renamed")
        assert result.collection_name == "collection_a"
        assert result.new_collection_name == "collection_renamed"
        assert result.collection_renamed is True
        assert mock_connector._mock_db["collection_a"].rename.await_count == 1

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.rename_collection("collection_x", "collection_renamed")

        with pytest.raises(ValueError, match="Collection collection_b already exists"):
            await mock_connector.rename_collection("collection_a", "collection_b")

        mock_connector._mock_db["collection_a"].rename.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.rename_collection("collection_a", "collection_renamed")

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, mock_connector):
        result = await mock_connector.get_collection_stats("collection_a")
        assert result["db"] == "testDb"
        assert result["collections"] == 10

        mock_connector._mock_db.command.assert_awaited()

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.get_collection_stats("collection_x")

        mock_connector._mock_db.command.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_collection_stats("collection_a")

    @pytest.mark.asyncio
    async def test_get_database_stats(self, mock_connector):
        result = await mock_connector.get_database_stats()
        assert result.db == "testDb"
        assert result.collections == 10
        assert result.index_size == 170393
        assert result.fs_total_size == 7654321

        mock_connector._mock_db.command.assert_awaited_once()

        mock_connector._mock_db.command.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_database_stats()

    @pytest.mark.asyncio
    async def test_list_indices(self, mock_connector):
        result = await mock_connector.list_indices("collection_a")
        assert result.collection_name == "collection_a"
        assert result.count == 2
        assert len(result.indices) == 2
        assert result.indices[0].name == "_id_"
        assert result.indices[0].key == {"_id": 1}

        mock_connector._mock_db["collection_a"].list_indexes.assert_called_once()

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.list_indices("collection_x")

        mock_connector._mock_db["collection_a"].list_indexes.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_indices("collection_a")

    @pytest.mark.asyncio
    async def test_list_indices_with_non_numeric_keys(self, mock_connector):
        cursor = mock_connector._mock_db["collection_a"].list_indexes.return_value
        cursor.to_list.return_value = [
            {"v": 2, "key": {"_fts": "text", "_ftsx": 1}, "name": "field_text"},
            {"v": 2, "key": {"location": "2dsphere"}, "name": "location_2dsphere"},
            {"v": 2, "key": {"field": "hashed"}, "name": "field_hashed"}
        ]

        result = await mock_connector.list_indices("collection_a")
        assert result.count == 3
        assert result.indices[0].key == {"_fts": "text", "_ftsx": 1}
        assert result.indices[1].key == {"location": "2dsphere"}
        assert result.indices[2].key == {"field": "hashed"}

    @pytest.mark.asyncio
    async def test_create_index(self, mock_connector):
        result = await mock_connector.create_index("collection_a", {"field": 1})
        assert result.collection_name == "collection_a"
        assert result.index == "field_1"
        assert result.index_created is True
        assert mock_connector._mock_db["collection_a"].create_index.await_count == 1

        await mock_connector.create_index("collection_a", {"field": "asc"})
        await mock_connector.create_index("collection_a", {"field": "ascending"})
        await mock_connector.create_index("collection_a", {"field": -1})
        await mock_connector.create_index("collection_a", {"field": "desc"})
        await mock_connector.create_index("collection_a", {"field": "descending"})
        await mock_connector.create_index("collection_a", {"field": "text"})
        await mock_connector.create_index("collection_a", {"field": "2dsphere"})

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.create_index("collection_x", {"field": 1})

        mock_connector._mock_db["collection_a"].create_index.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.create_index("collection_a", {"field": 1})

    @pytest.mark.asyncio
    async def test_drop_index(self, mock_connector):
        result = await mock_connector.drop_index("collection_a", "field_1")
        assert result.collection_name == "collection_a"
        assert result.index == "field_1"
        assert result.index_dropped is True
        assert mock_connector._mock_db["collection_a"].drop_index.await_count == 1

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.drop_index("collection_x", "field_1")

        with pytest.raises(ValueError, match="Cannot drop the default _id index"):
            await mock_connector.drop_index("collection_a", "_id_")

        mock_connector._mock_db["collection_a"].drop_index.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.drop_index("collection_a", "field_1")

    @pytest.mark.asyncio
    async def test_get_server_status_handles_missing_fields(self, mocker, mock_connector):
        mock_connector._mock_client.admin.command = mocker.AsyncMock(return_value={
            "host": "localhost:27017",
            "version": "7.0.0"
        })
        result = await mock_connector.get_server_status()

        assert result.host == "localhost:27017"
        assert result.connections.current is None
        assert result.extra_info.note is None

    @pytest.mark.asyncio
    async def test_ping_database(self, mocker, mock_connector):
        mock_connector._mock_db.command = mocker.AsyncMock(return_value={"ok": 1})
        result = await mock_connector.ping_database()
        assert result.ok == 1

        mock_connector._mock_db.command.assert_awaited_once_with(command="ping")

        mock_connector._mock_db.command.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.ping_database()

    @pytest.mark.asyncio
    async def test_insert_document(self, mock_connector):
        result = await mock_connector.insert_document("collection_a", {"name": "test"})
        assert result.collection_name == "collection_a"
        assert result.document_id == "doc_id_1"
        assert result.document_inserted is True
        assert mock_connector._mock_db["collection_a"].insert_one.await_count == 1

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.insert_document("collection_x", {"name": "test"})

        mock_connector._mock_db["collection_a"].insert_one.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.insert_document("collection_a", {"name": "test"})

    @pytest.mark.asyncio
    async def test_insert_many_documents(self, mock_connector):
        docs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        result = await mock_connector.insert_many_documents("collection_a", docs)
        assert result.collection_name == "collection_a"
        assert result.document_ids == ["doc_id_1", "doc_id_2", "doc_id_3"]
        assert result.documents_inserted is True
        assert mock_connector._mock_db["collection_a"].insert_many.await_count == 1

        mock_connector._mock_db["collection_a"].insert_many.assert_awaited_with(docs, ordered=True)

        await mock_connector.insert_many_documents("collection_a", docs, ordered=False)
        mock_connector._mock_db["collection_a"].insert_many.assert_awaited_with(docs, ordered=False)

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.insert_many_documents("collection_x", docs)

        mock_connector._mock_db["collection_a"].insert_many.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.insert_many_documents("collection_a", docs)

    @pytest.mark.asyncio
    async def test_find_documents(self, mocker, mock_connector):
        result = await mock_connector.find_documents("collection_a", {"name": "alice"})
        assert result.collection_name == "collection_a"
        assert len(result.documents) == 2
        assert result.documents[0]["_id"] == "507f1f77bcf86cd799439011"

        mock_connector._mock_db["collection_a"].find.assert_called_once()

        cursor = mock_connector._mock_db["collection_a"].find.return_value
        await mock_connector.find_documents("collection_a", {"name": "alice"}, sort_field="name")
        cursor.sort.assert_called_once_with("name", mocker.ANY)

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.find_documents("collection_x", {})

        cursor.to_list.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.find_documents("collection_a", {})

    @pytest.mark.asyncio
    async def test_count_documents(self, mock_connector):
        result = await mock_connector.count_documents("collection_a", {"name": "alice"})
        assert result.collection_name == "collection_a"
        assert result.document_count == 2

        mock_connector._mock_db["collection_a"].count_documents.assert_awaited_once()

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.count_documents("collection_x", {})

        mock_connector._mock_db["collection_a"].count_documents.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.count_documents("collection_a", {})

    @pytest.mark.asyncio
    async def test_update_documents(self, mock_connector):
        result = await mock_connector.update_documents("collection_a", {"name": "nela"}, {"$set": {"name": "alen"}})
        assert result.collection_name == "collection_a"
        assert result.updated_document_count == 2

        mock_connector._mock_db["collection_a"].update_many.assert_awaited_once()

        await mock_connector.update_documents("collection_a", {"name": "nela"}, {"$set": {"name": "alen"}}, upsert=True)
        args, kwargs = mock_connector._mock_db["collection_a"].update_many.call_args
        assert kwargs.get("upsert") is True

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.update_documents("collection_x", {}, {"$set": {}})

        mock_connector._mock_db["collection_a"].update_many.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.update_documents("collection_a", {}, {"$set": {}})

    @pytest.mark.asyncio
    async def test_replace_document(self, mock_connector):
        result = await mock_connector.replace_document("collection_a", {"name": "nela"}, {"name": "alen", "age": 30})
        assert result.collection_name == "collection_a"
        assert result.replaced_document_count == 1

        mock_connector._mock_db["collection_a"].replace_one.assert_awaited_once()

        await mock_connector.replace_document("collection_a", {"name": "nela"}, {"name": "alen"}, upsert=True)
        args, kwargs = mock_connector._mock_db["collection_a"].replace_one.call_args
        assert kwargs.get("upsert") is True

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.replace_document("collection_x", {}, {})

        mock_connector._mock_db["collection_a"].replace_one.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.replace_document("collection_a", {}, {})

    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_connector):
        result = await mock_connector.delete_documents("collection_a", {"name": "alice"})
        assert result.collection_name == "collection_a"
        assert result.deleted_document_count == 2

        mock_connector._mock_db["collection_a"].delete_many.assert_awaited_once()

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.delete_documents("collection_x", {})

        mock_connector._mock_db["collection_a"].delete_many.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.delete_documents("collection_a", {})

    @pytest.mark.asyncio
    async def test_aggregate_documents(self, mocker, mock_connector):
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ]
        result = await mock_connector.aggregate_documents("collection_a", pipeline)

        assert result.collection_name == "collection_a"
        assert len(result.documents) == 2
        assert result.documents[0]["_id"] == "group_a"
        assert result.documents[0]["count"] == 5

        mock_connector._mock_db["collection_a"].aggregate.assert_called_once()

        options = {"allowDiskUse": True}
        await mock_connector.aggregate_documents("collection_a", pipeline, options=options)
        mock_connector._mock_db["collection_a"].aggregate.assert_called_with(mocker.ANY, allowDiskUse=True)

        with pytest.raises(ValueError, match="Collection collection_x doesn't exist"):
            await mock_connector.aggregate_documents("collection_x", pipeline)

        mock_connector._mock_db["collection_a"].aggregate.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.aggregate_documents("collection_a", pipeline)

    @pytest.mark.asyncio
    async def test_find_inspection_models(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", "collection_b", DBCollections.DAILY_MODELS]
        )

        result = await mock_connector.find_inspection_models(
            model_name="EpoxyModel",
            model_version="v1",
            gbm="SEV",
            task="cls",
            mode="test",
            process="ActiveAlign"
        )
        assert len(result.models) == 2
        assert result.models[0].model_name == "EpoxyModel"
        assert result.models[0].model_version == "v1"
        assert result.models[0].process == "ActiveAlign"

        mock_connector._mock_db[DBCollections.DAILY_MODELS].find.assert_called_once()

        query = mock_connector._mock_db[DBCollections.DAILY_MODELS].find.call_args[0][0]
        assert query["gbm"] == "SEV"
        assert query["task"] == "cls"
        assert query["mode"] == "test"

        cursor = mock_connector._mock_db[DBCollections.DAILY_MODELS].find.return_value
        await mock_connector.find_inspection_models(sort_field="modelName")
        cursor.sort.assert_called_once_with("modelName", mocker.ANY)

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=["collection_a"])
        with pytest.raises(ValueError):
            await mock_connector.find_inspection_models()

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.DAILY_MODELS]
        )
        cursor.to_list.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.find_inspection_models()

    @pytest.mark.asyncio
    async def test_find_inspection_summaries(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_SUMMARY]
        )

        result = await mock_connector.find_inspection_summaries(
            model_name="EpoxyModel",
            gbm="SEV",
            location="Line 1",
            equipment_id="EQ-01",
            product_id="PR-01",
            conclusion="Good"
        )
        assert len(result.summaries) == 2

        summary = result.summaries[0]
        assert summary.model_name == "EpoxyModel"
        assert summary.equipment_id == "EQ-01"
        assert summary.inspection_ids == ["insp_1", "insp_2"]
        assert summary.threshold == 0.7
        assert summary.statistics.data_count == {"Good": 8, "Bad": 2}
        assert summary.statistics.confidence["Good"].avg == 0.9
        assert summary.statistics.elapsed_time.max == 0.12

        collection = mock_connector._mock_db[DBCollections.INSPECTIONS_SUMMARY]
        collection.find.assert_called_once()

        query, kwargs = collection.find.call_args[0][0], collection.find.call_args[1]
        assert query["gbm"] == "SEV"
        assert query["equipmentId"] == "EQ-01"
        assert query["productId"] == "PR-01"
        assert query["conclusion"] == "Good"
        assert "$regex" in query["location"]
        assert "$regex" in query["modelName"]
        assert kwargs["projection"] == EXCLUDE_SAMPLES

        collection.find.return_value.to_list.assert_awaited_with(length=LIMIT)

    @pytest.mark.asyncio
    async def test_find_inspection_summaries_date_range(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_SUMMARY]
        )
        collection = mock_connector._mock_db[DBCollections.INSPECTIONS_SUMMARY]

        start_date, end_date = datetime(2026, 1, 1), datetime(2026, 2, 1)
        await mock_connector.find_inspection_summaries(start_date=start_date, end_date=end_date, limit=2)

        query = collection.find.call_args[0][0]
        assert query == {"date": {"$gte": start_date, "$lte": end_date}}
        collection.find.return_value.to_list.assert_awaited_with(length=2)

        await mock_connector.find_inspection_summaries()
        assert collection.find.call_args[0][0] == {}

    @pytest.mark.asyncio
    async def test_find_inspection_summaries_optional_fields(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_SUMMARY]
        )

        result = await mock_connector.find_inspection_summaries()
        summary = result.summaries[1]

        assert summary.schema_version == "1.0"
        assert summary.mode is None
        assert summary.product_id is None
        assert summary.local_timezone is None
        assert summary.conclusion is None
        assert summary.threshold is None
        assert summary.inspection_ids == []
        assert summary.classes == []
        assert summary.statistics.data_count == {}
        assert summary.statistics.elapsed_time.avg == 0.0

    @pytest.mark.asyncio
    async def test_find_inspection_summaries_errors(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=["collection_a"])
        with pytest.raises(ValueError, match=f"Collection {DBCollections.INSPECTIONS_SUMMARY} doesn't exist"):
            await mock_connector.find_inspection_summaries()

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_SUMMARY]
        )
        collection = mock_connector._mock_db[DBCollections.INSPECTIONS_SUMMARY]

        await mock_connector.find_inspection_summaries(sort_field="date", projection={"statistics": 0})
        collection.find.return_value.sort.assert_called_once_with("date", mocker.ANY)
        assert collection.find.call_args[1]["projection"] == {"statistics": 0}

        collection.find.return_value.to_list.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.find_inspection_summaries()

    @pytest.mark.asyncio
    async def test_close(self, mock_connector):
        await mock_connector.close()
        mock_connector._mock_client.close.assert_called_once()

        mock_connector._mock_client.close.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.close()
