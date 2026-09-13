# mongodb-mcp
## Data drift analysis

The `mongodb_analyze_data_drift` tool streams the raw predictions of one model from the `inspections` collection with an aggregation pipeline that sorts on `metadata.createdAt` before unwinding. Create this index on the collection so the sort and the model filter are served from the index:

```javascript
db.inspections.createIndex({ "metadata.createdAt": 1, "inspectionResult.aiResults.aiModel": 1 })
```
