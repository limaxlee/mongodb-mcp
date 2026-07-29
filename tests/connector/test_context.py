from mongodb_mcp.connector import MongoDBContext


class TestMongoDBContext:
    def test_context_stores_connector(self, mocker):
        connector = mocker.MagicMock(name="connector")
        ctx = MongoDBContext(connector)

        assert ctx.connector is connector
