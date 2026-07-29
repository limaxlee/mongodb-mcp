from datetime import datetime
from bson.objectid import ObjectId

from mongodb_mcp.utils.db_helpers import preprocess_operation, process_query_result


class TestDBHelpers:
    def test_process_operation(self):
        document_id = "507f1f77bcf86cd799439011"
        result = preprocess_operation({"_id": document_id})
        assert isinstance(result["_id"], ObjectId)
        assert str(result["_id"]) == document_id

        result = preprocess_operation({"created": "2026-01-15 10:30:00"})
        assert isinstance(result["created"], datetime)
        assert result["created"] == datetime(2026, 1, 15, 10, 30, 0)

        result = preprocess_operation({"name": "alen"})
        assert result["name"] == "alen"

        result = preprocess_operation({"age": 30, "active": True, "score": 9.5})
        assert result == {"age": 30, "active": True, "score": 9.5}

        document_id = "507f1f77bcf86cd799439011"
        result = preprocess_operation({"meta": {"_id": document_id, "tag": "x"}})
        assert isinstance(result["meta"]["_id"], ObjectId)
        assert result["meta"]["tag"] == "x"

        document_id = "507f1f77bcf86cd799439011"
        result = preprocess_operation({"_id": {"$in": [document_id]}})
        assert isinstance(result["_id"]["$in"][0], ObjectId)

    def test_process_query_result(self):
        document_id = ObjectId("507f1f77bcf86cd799439011")
        result = process_query_result({"_id": document_id})
        assert result["_id"] == "507f1f77bcf86cd799439011"
        assert isinstance(result["_id"], str)

        result = process_query_result({"name": "alen", "age": 30})
        assert result == {"name": "alen", "age": 30}

        document_id = ObjectId("507f1f77bcf86cd799439011")
        result = process_query_result({"meta": {"_id": document_id, "tag": "x"}})
        assert result["meta"]["_id"] == "507f1f77bcf86cd799439011"
        assert result["meta"]["tag"] == "x"

        document_id = ObjectId("507f1f77bcf86cd799439011")
        result = process_query_result({"refs": [document_id, "plain", 7]})
        assert result["refs"][0] == "507f1f77bcf86cd799439011"
        assert result["refs"][1] == "plain"
        assert result["refs"][2] == 7

        document_id = "507f1f77bcf86cd799439011"
        queried = preprocess_operation({"_id": document_id})
        restored = process_query_result(queried)
        assert restored["_id"] == document_id
