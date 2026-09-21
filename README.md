# mongodb-mcp
## Data drift analysis

The `mongodb_analyze_data_drift` tool reads the pre-aggregated `inspectionStatistics` collection (one document per model, version, task, mode, site, product and period) and never the raw `inspections` collection. The documents of a period are summed inside MongoDB with an aggregation pipeline, so the tool transfers one small row per period and class whatever the number of products. The pipeline matches on the collection's unique index:

```javascript
db.inspectionStatistics.createIndex(
  { modelName: 1, modelVersion: 1, task: 1, mode: 1, gbm: 1, process: 1, granularity: 1, startDate: 1, productId: 1 },
  { unique: true }
)
```

The collection is written by the Data Service. Its schema is described in [docs/inspection_statistics.md](docs/inspection_statistics.md) and the tool in [DATA_DRIFT.md](DATA_DRIFT.md).
