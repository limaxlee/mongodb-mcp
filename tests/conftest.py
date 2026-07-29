import sys
import yaml
import pytest
import pathlib

from mongodb_mcp.connector.connector import MongoDBConnector

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    cfg = ROOT_DIR / "config.yaml"
    if not cfg.is_file():
        raise pytest.UsageError("Missing configuration file")


@pytest.fixture
def sample_config_dict():
    return {
        "server_port": 8443,
        "mongodb_host": "localhost",
        "mongodb_port": 27017,
        "mongodb_db_name": "default"
    }


@pytest.fixture
def tmp_config_file(tmp_path, sample_config_dict):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(sample_config_dict))
    return cfg_path


@pytest.fixture
def mock_async_mongodb_client(mocker, mock_async_mongodb_db):
    client = mocker.MagicMock(name="AsyncIOMotorClient")
    client.list_database_names = mocker.AsyncMock(return_value=["default", "test_db"])
    client.admin = mocker.MagicMock(name="AdminDatabase")
    client.admin.command = mocker.AsyncMock(return_value={
        "host": "localhost:27017",
        "version": "7.0.0",
        "process": "mongod",
        "pid": 1234,
        "uptime": 3600,
        "uptimeMillis": 3600000,
        "localTime": "2026-01-01T00:00:00Z",
        "connections": {
            "current": 5,
            "available": 995,
            "totalCreated": 100,
        },
        "extra_info": {
            "note": "fields vary by platform",
            "heap_usage_bytes": 12345678,
            "page_faults": 0,
        },
        "ok": 1
    })

    return client


@pytest.fixture
def mock_async_mongodb_db(mocker):
    db = mocker.MagicMock(name="AsyncIOMotorClient")
    db.list_collection_names = mocker.AsyncMock(return_value=["collection_a", "collection_b"])
    db.create_collection = mocker.AsyncMock(return_value=None)
    db.drop_collection = mocker.AsyncMock(return_value=None)
    db.command = mocker.AsyncMock(return_value={
        "db": "testDb",
        "collections": 10,
        "objects": 9000,
        "avObjSize": 327.23,
        "dataSize": 300144,
        "storageSize": 1265664,
        "indexes": 25,
        "indexSize": 170393,
        "totalSize": 2969600,
        "scaleFactor": 1,
        "fsUsedSize": 1234567,
        "fsTotalSize": 7654321,
        "ok": 1
    })

    list_indexes_cursor = mocker.MagicMock(name="IndexCursor")
    list_indexes_cursor.to_list = mocker.AsyncMock(return_value=[
        {"v": 2, "key": {"_id": 1}, "name": "_id_"},
        {"v": 2, "key": {"field": 1}, "name": "field_1"}
    ])

    find_cursor = mocker.MagicMock(name="FindCursor")
    find_cursor.sort = mocker.MagicMock(return_value=find_cursor)
    find_cursor.to_list = mocker.AsyncMock(return_value=[
        {"_id": "507f1f77bcf86cd799439011", "name": "alen"},
        {"_id": "507f1f77bcf86cd799439012", "name": "nela"}
    ])

    aggregate_cursor = mocker.MagicMock(name="AggregateCursor")
    aggregate_cursor.to_list = mocker.AsyncMock(return_value=[
        {"_id": "group_a", "count": 5},
        {"_id": "group_b", "count": 3}
    ])

    insert_one_result = mocker.MagicMock(name="InsertOneResult")
    insert_one_result.inserted_id = "doc_id_1"
    insert_many_result = mocker.MagicMock(name="InsertManyResult")
    insert_many_result.inserted_ids = ["doc_id_1", "doc_id_2", "doc_id_3"]

    update_result = mocker.MagicMock(name="UpdateResult")
    update_result.acknowledged = True
    update_result.modified_count = 2

    replace_result = mocker.MagicMock(name="ReplaceResult")
    replace_result.acknowledged = True
    replace_result.modified_count = 1

    delete_result = mocker.MagicMock(name="DeleteResult")
    delete_result.acknowledged = True
    delete_result.deleted_count = 2

    collection_a = mocker.MagicMock(name="CollectionMocker")
    collection_a.rename = mocker.AsyncMock(return_value=None)
    collection_a.create_index = mocker.AsyncMock(return_value="field_1")
    collection_a.drop_index = mocker.AsyncMock(return_value=None)
    collection_a.list_indexes = mocker.MagicMock(return_value=list_indexes_cursor)
    collection_a.database = mocker.MagicMock(name="CollectionDatabase")
    collection_a.database.command = mocker.AsyncMock(return_value={"ok": 1})
    collection_a.insert_one = mocker.AsyncMock(return_value=insert_one_result)
    collection_a.insert_many = mocker.AsyncMock(return_value=insert_many_result)
    collection_a.find = mocker.MagicMock(return_value=find_cursor)
    collection_a.count_documents = mocker.AsyncMock(return_value=2)
    collection_a.update_many = mocker.AsyncMock(return_value=update_result)
    collection_a.replace_one = mocker.AsyncMock(return_value=replace_result)
    collection_a.delete_many = mocker.AsyncMock(return_value=delete_result)
    collection_a.aggregate = mocker.MagicMock(return_value=aggregate_cursor)
    db.__getitem__.side_effect = lambda name: collection_a

    return db


@pytest.fixture
def mock_connector(mocker, mock_async_mongodb_client, mock_async_mongodb_db):
    mocker.patch("mongodb_mcp.connector.connector.AsyncIOMotorClient", return_value=mock_async_mongodb_client)
    connector = MongoDBConnector()
    connector._db = mock_async_mongodb_db
    connector._mock_client = mock_async_mongodb_client
    connector._mock_db = mock_async_mongodb_db

    yield connector


@pytest.fixture
def mock_context(mocker, mock_connector):
    ctx = mocker.MagicMock()
    ctx.request_context.lifespan_context.connector = mock_connector
    return ctx
