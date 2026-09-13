import pytest
import pytest_asyncio
from datetime import datetime
from fastmcp.exceptions import ToolError

from mongodb_mcp.tools import tools_mcp
from mongodb_mcp.schemas import (
    GetIndicesResult, PingDatabaseResult, FindInspectionModelsResult, Statistics,
    InspectionModelInfo, FindInspectionSummaryDocumentsResult, InspectionSummaryDocument,
    FindDatasetFamilyDocumentsResult, DatasetFamilyDocument, FamilyMember,
    FindDatasetDocumentsResult, DatasetDocument, LabelAttributes
)


@pytest_asyncio.fixture
async def tools_by_name():
    tools = await tools_mcp.list_tools()
    return {tool.name: tool.fn for tool in tools}


class TestMongoDBMCPTools:
    @pytest.mark.asyncio
    async def test_mongodb_list_databases(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_list_databases"]
        result = await tool(mock_context)

        assert "default" in result.databases
        assert "test_db" in result.databases

        mock_context.request_context.lifespan_context.connector.list_databases = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_mongodb_list_collections(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_list_collections"]
        result = await tool(mock_context)

        assert "collection_a" in result.collections
        assert "collection_b" in result.collections

        mock_context.request_context.lifespan_context.connector.list_collections = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_mongodb_create_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_create_collection"]
        result = await tool(mock_context, "collection_c")

        assert result.collection_name == "collection_c"
        assert result.collection_created is True

        result = await tool(mock_context, "collection_d", {"capped": True, "size": 1024})
        assert result.collection_name == "collection_d"

        mock_context.request_context.lifespan_context.connector.create_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_c")

    @pytest.mark.asyncio
    async def test_mongodb_drop_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_drop_collection"]
        result = await tool(mock_context, "collection_a")

        assert result.collection_name == "collection_a"
        assert result.collection_dropped is True

        mock_context.request_context.lifespan_context.connector.drop_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a")

    @pytest.mark.asyncio
    async def test_mongodb_rename_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_rename_collection"]
        result = await tool(mock_context, "collection_a", "collection_renamed")

        assert result.collection_name == "collection_a"
        assert result.new_collection_name == "collection_renamed"
        assert result.collection_renamed is True

        mock_context.request_context.lifespan_context.connector.rename_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "collection_renamed")

    @pytest.mark.asyncio
    async def test_mongodb_get_collection_stats(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_get_collection_stats"]
        result = await tool(mock_context, "collection_a")

        assert result.ns == "testDb"
        assert result.size == 111

        mock_context.request_context.lifespan_context.connector.get_collection_stats = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a")

    @pytest.mark.asyncio
    async def test_mongodb_get_database_stats(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_get_database_stats"]
        result = await tool(mock_context)

        assert result.db == "testDb"
        assert result.collections == 10

        mock_context.request_context.lifespan_context.connector.get_database_stats = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_mongodb_list_indices(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_list_indices"]
        result = await tool(mock_context, "collection_a")
        assert result.collection_name == "collection_a"
        assert result.count == 2

        mock_context.request_context.lifespan_context.connector.list_indices = \
            mocker.AsyncMock(return_value=GetIndicesResult(collection_name="collection_a", count=0))
        result = await tool(mock_context, "collection_a")
        assert result.count == 0
        assert result.indices == []

        mock_context.request_context.lifespan_context.connector.list_indices = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a")

    @pytest.mark.asyncio
    async def test_mongodb_create_index(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_create_index"]
        result = await tool(mock_context, "collection_a", {"field": 1})

        assert result.collection_name == "collection_a"
        assert result.index == "field_1"
        assert result.index_created is True

        result = await tool(mock_context, "collection_a", {"field": 1}, {"unique": True})
        assert result.collection_name == "collection_a"

        mock_context.request_context.lifespan_context.connector.create_index = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"field": 1})

    @pytest.mark.asyncio
    async def test_mongodb_drop_index(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_drop_index"]
        result = await tool(mock_context, "collection_a", "field_1")

        assert result.collection_name == "collection_a"
        assert result.index == "field_1"
        assert result.index_dropped is True

        mock_context.request_context.lifespan_context.connector.drop_index = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "field_1")

    @pytest.mark.asyncio
    async def test_mongodb_get_server_status(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_get_server_status"]
        result = await tool(mock_context)

        assert result.host == "localhost:27017"
        assert result.version == "7.0.0"

        mock_context.request_context.lifespan_context.connector.get_server_status = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_mongodb_ping_database(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_ping_database"]
        mock_context.request_context.lifespan_context.connector.ping_database = \
            mocker.AsyncMock(return_value=PingDatabaseResult(ok=1))
        result = await tool(mock_context)
        assert result.ok == 1

        mock_context.request_context.lifespan_context.connector.ping_database = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_mongodb_insert_document(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_insert_document"]
        result = await tool(mock_context, "collection_a", {"name": "test"})

        assert result.collection_name == "collection_a"
        assert result.document_id == "doc_id_1"
        assert result.document_inserted is True

        mock_context.request_context.lifespan_context.connector.insert_document = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "test"})

    @pytest.mark.asyncio
    async def test_mongodb_insert_many_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_insert_many_documents"]
        docs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        result = await tool(mock_context, "collection_a", docs)

        assert result.collection_name == "collection_a"
        assert result.document_ids == ["doc_id_1", "doc_id_2", "doc_id_3"]
        assert result.documents_inserted is True

        mock_context.request_context.lifespan_context.connector.insert_many_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", docs)

    @pytest.mark.asyncio
    async def test_mongodb_find_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_find_documents"]
        result = await tool(mock_context, "collection_a", {"name": "alice"})

        assert result.collection_name == "collection_a"
        assert len(result.documents) == 2

        result = await tool(
            mock_context, "collection_a", {"name": "alice"},
            projection={"name": 1}, limit=10, sort_field="name", sort_order=-1
        )
        assert len(result.documents) == 2

        mock_context.request_context.lifespan_context.connector.find_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "alice"})

    @pytest.mark.asyncio
    async def test_mongodb_count_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_count_documents"]
        result = await tool(mock_context, "collection_a", {"name": "alice"})

        assert result.collection_name == "collection_a"
        assert result.document_count == 2

        mock_context.request_context.lifespan_context.connector.count_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "alice"})

    @pytest.mark.asyncio
    async def test_mongodb_update_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_update_documents"]
        result = await tool(mock_context, "collection_a", {"name": "alice"}, {"$set": {"name": "alen"}})

        assert result.collection_name == "collection_a"
        assert result.updated_document_count == 2

        result = await tool(mock_context, "collection_a", {"name": "alice"}, {"$set": {"name": "alen"}}, upsert=True)
        assert result.collection_name == "collection_a"

        mock_context.request_context.lifespan_context.connector.update_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "alice"}, {"$set": {"name": "alen"}})

    @pytest.mark.asyncio
    async def test_mongodb_replace_document(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_replace_document"]
        result = await tool(mock_context, "collection_a", {"name": "alice"}, {"name": "alen", "age": 30})

        assert result.collection_name == "collection_a"
        assert result.replaced_document_count == 1

        result = await tool(mock_context, "collection_a", {"name": "alice"}, {"name": "alen"}, upsert=True)
        assert result.collection_name == "collection_a"

        mock_context.request_context.lifespan_context.connector.replace_document = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "alice"}, {"name": "alen"})

    @pytest.mark.asyncio
    async def test_mongodb_delete_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_delete_documents"]
        result = await tool(mock_context, "collection_a", {"name": "alice"})

        assert result.collection_name == "collection_a"
        assert result.deleted_document_count == 2

        mock_context.request_context.lifespan_context.connector.delete_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", {"name": "alice"})

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_aggregate_documents"]
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ]
        result = await tool(mock_context, "collection_a", pipeline)

        assert result.collection_name == "collection_a"
        assert len(result.documents) == 2
        assert result.documents[0]["_id"] == "group_a"
        assert result.documents[0]["count"] == 5

        result = await tool(mock_context, "collection_a", pipeline, {"allowDiskUse": True})
        assert len(result.documents) == 2

        mock_context.request_context.lifespan_context.connector.aggregate_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", pipeline)

    @pytest.mark.asyncio
    async def test_mongodb_find_inspection_models(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_find_inspection_models"]
        mock_context.request_context.lifespan_context.connector.find_inspection_models = \
            mocker.AsyncMock(return_value=FindInspectionModelsResult(models=[
                InspectionModelInfo(
                    model_name="EpoxyInspector",
                    model_version="v1",
                    process="ActiveAlign",
                    task="cls",
                    gbm="SEV",
                    mode="test",
                    date=datetime(2026, 1, 1)
                )
            ]))

        result = await tool(mock_context, model_name="EpoxyInspector")
        assert len(result.models) == 1
        assert result.models[0].model_name == "EpoxyInspector"

        mock_context.request_context.lifespan_context.connector.find_inspection_models.assert_awaited_once()

        mock_context.request_context.lifespan_context.connector.find_inspection_models = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, model_name="EpoxyInspector")

    @pytest.mark.asyncio
    async def test_mongodb_find_inspection_summary_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_find_inspection_summary_documents"]
        mock_context.request_context.lifespan_context.connector.find_inspection_summary_documents = \
            mocker.AsyncMock(return_value=FindInspectionSummaryDocumentsResult(summaries=[
                InspectionSummaryDocument(
                    model_name="EpoxyInspector",
                    model_version="v1",
                    gbm="SEV",
                    process="ActiveAlign",
                    date=datetime(2026, 1, 1),
                    location="Line 1",
                    equipment_id="EQ-01",
                    task="cls",
                    statistics=Statistics(data_count={"Good": 8})
                )
            ]))

        result = await tool(mock_context, model_name="EpoxyInspector", equipment_id="EQ-01")
        assert len(result.summaries) == 1
        assert result.summaries[0].model_name == "EpoxyInspector"
        assert result.summaries[0].equipment_id == "EQ-01"
        assert result.summaries[0].statistics.data_count == {"Good": 8}

        connector = mock_context.request_context.lifespan_context.connector
        connector.find_inspection_summary_documents.assert_awaited_once()
        assert connector.find_inspection_summary_documents.await_args.kwargs["equipment_id"] == "EQ-01"

        mock_context.request_context.lifespan_context.connector.find_inspection_summary_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, model_name="EpoxyInspector")

    @pytest.mark.asyncio
    async def test_mongodb_find_dataset_families_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_find_dataset_families_documents"]
        mock_context.request_context.lifespan_context.connector.find_dataset_family_documents =             mocker.AsyncMock(return_value=FindDatasetFamilyDocumentsResult(families=[
                DatasetFamilyDocument(
                    family_id="69fd14e65b804eb1b8b60be4",
                    dataset_family_name="hqehleddisplay",
                    task="det",
                    members=[FamilyMember(dataset_id="69fd14e65b804eb1b8b60be5", version="1.0")],
                    created_at=datetime(2026, 5, 7)
                )
            ]))

        result = await tool(mock_context, dataset_family_name="hqehleddisplay", task="det")
        assert len(result.families) == 1
        assert result.families[0].dataset_family_name == "hqehleddisplay"
        assert result.families[0].members[0].dataset_id == "69fd14e65b804eb1b8b60be5"

        connector = mock_context.request_context.lifespan_context.connector
        connector.find_dataset_family_documents.assert_awaited_once()
        assert connector.find_dataset_family_documents.await_args.kwargs["task"] == "det"

        mock_context.request_context.lifespan_context.connector.find_dataset_family_documents =             mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, dataset_family_name="hqehleddisplay")

    @pytest.mark.asyncio
    async def test_mongodb_find_dataset_documents(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["mongodb_find_dataset_documents"]
        mock_context.request_context.lifespan_context.connector.find_dataset_documents = \
            mocker.AsyncMock(return_value=FindDatasetDocumentsResult(datasets=[
                DatasetDocument(
                    document_id="69fd14e65b804eb1b8b60be5",
                    name="hqehleddisplay",
                    version="1.0",
                    task="det",
                    created_by="minchang.kim",
                    family_id="69fd14e65b804eb1b8b60be4",
                    created_at=datetime(2026, 5, 7),
                    modified_at=datetime(2026, 5, 7),
                    is_used=True,
                    data_count=198,
                    classes={"STEP 1": LabelAttributes(count=50, shape="rectangle")}
                )
            ]))

        result = await tool(mock_context, name="hqehleddisplay", is_used=True)
        assert len(result.datasets) == 1
        assert result.datasets[0].name == "hqehleddisplay"
        assert result.datasets[0].data_count == 198
        assert result.datasets[0].classes["STEP 1"].count == 50

        connector = mock_context.request_context.lifespan_context.connector
        connector.find_dataset_documents.assert_awaited_once()
        assert connector.find_dataset_documents.await_args.kwargs["is_used"] is True

        mock_context.request_context.lifespan_context.connector.find_dataset_documents = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, name="hqehleddisplay")
