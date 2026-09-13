import pytest
from datetime import datetime
from pymongo.errors import AutoReconnect

from common.constants import DBCollections
from mongodb_mcp.connector import MongoDBConnector
from mongodb_mcp.utils import generate_normalized_regex


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
        assert result.ns == "testDb"
        assert result.size == 111

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
    async def test_find_inspection_summary_documents(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.INSPECTIONS_SUMMARY]
        )

        result = await mock_connector.find_inspection_summary_documents(
            model_name="EpoxyModel",
            gbm="SEV",
            location="Line 1",
            equipment_id="EQ-01",
            product_id="PR-01"
        )
        assert len(result.summaries) == 2

        summary = result.summaries[0]
        assert summary.model_name == "EpoxyModel"
        assert summary.equipment_id == "EQ-01"
        assert summary.threshold == 0.7

        collection = mock_connector._mock_db[DBCollections.INSPECTIONS_SUMMARY]

        start_date, end_date = datetime(2026, 1, 1), datetime(2026, 2, 1)
        await mock_connector.find_inspection_summary_documents(start_date=start_date, end_date=end_date, limit=2)

        query = collection.find.call_args[0][0]
        assert query == {"date": {"$gte": start_date, "$lte": end_date}}
        collection.find.return_value.to_list.assert_awaited_with(length=2)

        await mock_connector.find_inspection_summary_documents()
        assert collection.find.call_args[0][0] == {}

    @pytest.mark.asyncio
    async def test_find_dataset_family_documents(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.DATASETS]
        )

        result = await mock_connector.find_dataset_family_documents(dataset_family_name="hqehleddisplay", task="det")
        assert len(result.families) == 2

        family = result.families[0]
        assert family.family_id == "69fd14e65b804eb1b8b60be4"
        assert family.dataset_family_name == "hqehleddisplay"
        assert family.task == "det"
        assert len(family.members) == 2
        assert family.members[0].dataset_id == "69fd14e65b804eb1b8b60be5"
        assert family.members[0].version == "1.0"
        assert family.access_control.groups == ["52", "65"]
        assert result.families[1].members == []

        collection = mock_connector._mock_db[DBCollections.DATASETS]
        collection.find.assert_called_once()

        query = collection.find.call_args[0][0]
        assert query["datasetFamilyName"] == generate_normalized_regex("hqehleddisplay")
        assert query["task"] == "det"

        start_date, end_date = datetime(2026, 1, 1), datetime(2026, 2, 1)
        await mock_connector.find_dataset_family_documents(start_date=start_date, end_date=end_date, limit=2)

        query = collection.find.call_args[0][0]
        assert query == {
            "datasetFamilyName": {"$exists": True},
            "createdAt": {"$gte": start_date, "$lte": end_date}
        }
        collection.find({"datasetFamilyName": {"$exists": True}}).to_list.assert_awaited_with(length=2)

        await mock_connector.find_dataset_family_documents()
        assert collection.find.call_args[0][0] == {"datasetFamilyName": {"$exists": True}}

        cursor = collection.find({"datasetFamilyName": {"$exists": True}})
        await mock_connector.find_dataset_family_documents(sort_field="createdAt")
        cursor.sort.assert_called_once_with("createdAt", mocker.ANY)

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=["collection_a"])
        with pytest.raises(ValueError):
            await mock_connector.find_dataset_family_documents()

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.DATASETS]
        )
        cursor.to_list.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.find_dataset_family_documents()

    def test_exclude_fields(self, mock_connector):
        excluded = ("dataMap", "lastJob")

        assert mock_connector._exclude_fields(None, excluded) == {"dataMap": 0, "lastJob": 0}
        assert mock_connector._exclude_fields({}, excluded) == {"dataMap": 0, "lastJob": 0}
        assert mock_connector._exclude_fields({"classes": 0}, excluded) == {"classes": 0, "dataMap": 0, "lastJob": 0}
        assert mock_connector._exclude_fields({"dataMap": 1, "name": 1}, excluded) == {"name": 1}
        assert mock_connector._exclude_fields({"_id": 0, "name": 1, "lastJob": 1}, excluded) == {"_id": 0, "name": 1}
        assert mock_connector._exclude_fields({"dataMap": 1}, excluded) == {"dataMap": 0, "lastJob": 0}

    @pytest.mark.asyncio
    async def test_find_dataset_documents(self, mocker, mock_connector):
        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.DATASETS]
        )

        result = await mock_connector.find_dataset_documents(
            name="hqehleddisplay",
            version="1.0",
            task="det",
            created_by="minchang.kim",
            is_finalized=True,
            is_used=True
        )
        assert len(result.datasets) == 2

        dataset = result.datasets[0]
        assert dataset.document_id == "69fd14e65b804eb1b8b60be5"
        assert dataset.name == "hqehleddisplay"
        assert dataset.version == "1.0"
        assert dataset.family_id == "69fd14e65b804eb1b8b60be4"
        assert dataset.is_finalized is True
        assert dataset.data_count == 198
        assert dataset.classes["STEP 1"].count == 50
        assert dataset.classes["STEP 1"].shape == "rectangle"
        assert dataset.training_records[0].ai_model == "EpoxyModel"
        assert dataset.training_records[0].status == "completed"
        assert not hasattr(dataset, "data_map")
        assert not hasattr(dataset, "last_job")
        assert result.datasets[1].classes == {}

        collection = mock_connector._mock_db[DBCollections.DATASETS]
        collection.find.assert_called_once()

        query = collection.find.call_args[0][0]
        assert query["schemaVersion"] == {"$exists": True}
        assert query["name"] == generate_normalized_regex("hqehleddisplay")
        assert query["version"] == generate_normalized_regex("1.0")
        assert query["task"] == "det"
        assert query["createdBy"] == generate_normalized_regex("minchang.kim")
        assert query["isFinalized"] is True
        assert query["isUsed"] is True
        assert collection.find.call_args[1]["projection"] == {"dataMap": 0, "lastJob": 0}

        start_date, end_date = datetime(2026, 1, 1), datetime(2026, 2, 1)
        await mock_connector.find_dataset_documents(
            start_date=start_date,
            end_date=end_date,
            projection={"classes": 0},
            limit=2
        )

        query = collection.find.call_args[0][0]
        assert query == {
            "schemaVersion": {"$exists": True},
            "createdAt": {"$gte": start_date, "$lte": end_date}
        }
        assert collection.find.call_args[1]["projection"] == {"classes": 0, "dataMap": 0, "lastJob": 0}

        cursor = collection.find({"schemaVersion": {"$exists": True}})
        cursor.to_list.assert_awaited_with(length=2)

        await mock_connector.find_dataset_documents(is_used=False, projection={"name": 1, "dataMap": 1})
        assert collection.find.call_args[0][0] == {"schemaVersion": {"$exists": True}, "isUsed": False}
        assert collection.find.call_args[1]["projection"] == {"name": 1}

        await mock_connector.find_dataset_documents(sort_field="createdAt")
        cursor.sort.assert_called_once_with("createdAt", mocker.ANY)

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(return_value=["collection_a"])
        with pytest.raises(ValueError):
            await mock_connector.find_dataset_documents()

        mock_connector._mock_db.list_collection_names = mocker.AsyncMock(
            return_value=["collection_a", DBCollections.DATASETS]
        )
        cursor.to_list.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.find_dataset_documents()

    @pytest.mark.asyncio
    async def test_close(self, mock_connector):
        await mock_connector.close()
        mock_connector._mock_client.close.assert_called_once()

        mock_connector._mock_client.close.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.close()
