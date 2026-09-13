import sys
import yaml
import pytest
import pathlib
from datetime import datetime

from common.constants import DBCollections
from mongodb_mcp.connector.connector import MongoDBConnector

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.drift.synthetic import generate_rows, cls_prediction  # noqa: E402

# Seven days of classification rows with a confidence drop on day four, as the drift pipeline would emit them
DRIFT_ROWS = generate_rows("cls", 7, 300, lambda rng, day: cls_prediction(rng, 0.93 if day < 4 else 0.84, 0.02))


class AsyncRowCursor:
    """Stands in for a motor aggregation cursor, which is iterated with async for"""

    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for row in self.rows:
            yield row


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
        "ns": "testDb",
        "collections": 10,
        "size": 111,
        "count": 3,
        "freeStorageSize": 16384,
        "avgObjSize": 37,
        "capped": True,
        "views": 0,
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

    daily_models_cursor = mocker.MagicMock(name="DailyModelsCursor")
    daily_models_cursor.sort = mocker.MagicMock(return_value=daily_models_cursor)
    daily_models_cursor.to_list = mocker.AsyncMock(return_value=[
        {
            "_id": "507f1f77bcf86cd799439011",
            "modelName": "EpoxyModel",
            "modelVersion": "v1",
            "process": "ActiveAlign",
            "task": "cls",
            "gbm": "SEV",
            "mode": "test",
            "date": datetime(2026, 1, 1)
        },
        {
            "_id": "507f1f77bcf86cd799439012",
            "modelName": "EpoxyModel",
            "modelVersion": "v2",
            "process": "ActiveAlign",
            "task": "cls",
            "gbm": "SEV",
            "mode": "test",
            "date": datetime(2026, 2, 1)
        }
    ])

    daily_models = mocker.MagicMock(name="DailyModelsCollectionMocker")
    daily_models.find = mocker.MagicMock(return_value=daily_models_cursor)

    summaries_cursor = mocker.MagicMock(name="InspectionsSummaryMocker")
    summaries_cursor.sort = mocker.MagicMock(return_value=summaries_cursor)
    summaries_cursor.to_list = mocker.AsyncMock(return_value=[
        {
            "_id": "507f1f77bcf86cd799439021",
            "modelName": "EpoxyModel",
            "modelVersion": "v1",
            "gbm": "SEV",
            "process": "ActiveAlign",
            "mode": "test",
            "date": datetime(2026, 1, 1),
            "location": "Line 1",
            "equipmentId": "EQ-01",
            "productId": "PR-01",
            "localTimezone": "Asia/Seoul",
            "inspectionIds": ["inspection-id1", "inspection-id2"],
            "task": "cls",
            "classes": ["Good", "Bad"],
            "threshold": 0.7,
            "statistics": {
                "dataCount": {"Good": 8, "Bad": 2},
                "confidence": {
                    "Good": {"avg": 0.9, "min": 0.8, "max": 0.99, "sum": 7.2},
                    "Bad": {"avg": 0.75, "min": 0.7, "max": 0.8, "sum": 1.5}
                },
                "elapsedTime": {"avg": 0.05, "min": 0.01, "max": 0.12, "sum": 0.5}
            },
            "samples": {}
        },
        {
            "_id": "507f1f77bcf86cd799439022",
            "modelName": "EpoxyModel",
            "modelVersion": "v2",
            "gbm": "SEV",
            "process": "ActiveAlign",
            "date": datetime(2026, 2, 1),
            "location": "Line 2",
            "equipmentId": "EQ-02",
            "task": "cls"
        }
    ])

    summaries = mocker.MagicMock(name="InspectionsSummaryCollectionMocker")
    summaries.find = mocker.MagicMock(return_value=summaries_cursor)

    dataset_families_cursor = mocker.MagicMock(name="DatasetFamiliesCursor")
    dataset_families_cursor.sort = mocker.MagicMock(return_value=dataset_families_cursor)
    dataset_families_cursor.to_list = mocker.AsyncMock(return_value=[
        {
            "_id": "69fd14e65b804eb1b8b60be4",
            "datasetFamilyName": "hqehleddisplay",
            "task": "det",
            "members": [
                {"datasetId": "69fd14e65b804eb1b8b60be5", "version": "1.0", "description": "SEC_AC_EH_LED Display"},
                {"datasetId": "6a03d7085b804eb1b8b6b297", "version": "1.1", "description": ""}
            ],
            "accessControl": {"groups": ["52", "65"], "users": []},
            "createdAt": datetime(2026, 5, 7)
        },
        {
            "_id": "69fd14e65b804eb1b8b60be6",
            "datasetFamilyName": "epoxyinspection",
            "task": "cls",
            "createdAt": datetime(2026, 6, 1)
        }
    ])

    dataset_documents_cursor = mocker.MagicMock(name="DatasetDocumentsCursor")
    dataset_documents_cursor.sort = mocker.MagicMock(return_value=dataset_documents_cursor)
    dataset_documents_cursor.to_list = mocker.AsyncMock(return_value=[
        {
            "_id": "69fd14e65b804eb1b8b60be5",
            "name": "hqehleddisplay",
            "version": "1.0",
            "task": "det",
            "description": "SEC_AC_EH_LED Display",
            "createdBy": "minchang.kim",
            "projects": ["hqehleddisplay/detarea"],
            "accessControl": {"groups": ["52", "65"], "users": []},
            "familyId": "69fd14e65b804eb1b8b60be4",
            "schemaVersion": "1.0",
            "createdAt": datetime(2026, 5, 7),
            "modifiedAt": datetime(2026, 5, 7),
            "isFinalized": True,
            "finalizedAt": datetime(2026, 5, 7),
            "isUsed": True,
            "downloadCount": 1,
            "downloadUri": "datasets/69fd14e65b804eb1b8b60be5/hqehleddisplay_1.0_20260507-224142Z.zip",
            "dataCount": 198,
            "classes": {
                "STEP 0": {"count": 48, "color": "#ff0000", "shape": "rectangle"},
                "STEP 1": {"count": 50, "color": "#ff0000", "shape": "rectangle"}
            },
            "trainingRecords": [
                {
                    "aiModel": "EpoxyModel",
                    "version": "v1",
                    "startTime": datetime(2026, 5, 8),
                    "endTime": datetime(2026, 5, 9),
                    "status": "completed",
                    "trainingInfo": "epochs=10"
                }
            ]
        },
        {
            "_id": "6a03d7085b804eb1b8b6b297",
            "name": "hqehleddisplay",
            "version": "1.1",
            "task": "det",
            "createdBy": "minchang.kim",
            "familyId": "69fd14e65b804eb1b8b60be4",
            "schemaVersion": "1.0",
            "createdAt": datetime(2026, 5, 10),
            "modifiedAt": datetime(2026, 5, 10)
        }
    ])

    datasets = mocker.MagicMock(name="DatasetsCollectionMocker")
    datasets.find = mocker.MagicMock(
        side_effect=lambda query, projection=None:
            dataset_families_cursor if "datasetFamilyName" in query else dataset_documents_cursor
    )

    inspections = mocker.MagicMock(name="InspectionsCollectionMocker")
    inspections.count_documents = mocker.AsyncMock(return_value=len(DRIFT_ROWS))
    inspections.aggregate = mocker.MagicMock(side_effect=lambda pipeline, **kwargs: AsyncRowCursor(DRIFT_ROWS))

    collections_by_name = {
        DBCollections.DAILY_MODELS: daily_models,
        DBCollections.INSPECTIONS_SUMMARY: summaries,
        DBCollections.DATASETS: datasets,
        DBCollections.INSPECTIONS: inspections
    }

    db.__getitem__.side_effect = lambda name: collections_by_name.get(name, collection_a)

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
