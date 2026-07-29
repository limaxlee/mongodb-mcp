from mongodb_mcp.connector.connector import MongoDBConnector


class MongoDBContext:
    def __init__(self, connector: MongoDBConnector):
        self.connector = connector
