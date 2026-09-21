# Inspection Document → Inspection Statistics Conversion (for Data Drift Analysis)

**Korean version:** [confluence_inspection_statistics.ko.md](confluence_inspection_statistics.ko.md)

## Purpose

- Build a document conversion pipeline so that the Data Agent can reliably analyse **data drift** of AI inspection models (the model's input data or behaviour changing over time)
- Instead of raw inspection documents, create a **collection of statistics pre-aggregated per period** (`inspectionStatistics`)
- Remove the calculation from the MCP server (the tool the Data Agent calls); the Data Service computes the statistics when it stores the data

## Current state

- The Data Service stores inspection result documents in the MongoDB `inspections` collection
- The current data drift tool reads raw documents straight from that collection, flattens every single prediction, and computes every statistic inside the tool
- One inspection document can hold the results of several inspection models (`inspectionResults.aiResults`), and a document can become very large depending on the number of predictions

## Problems

Analysing drift this way causes the following problems.

- **Query cost**: one analysis reads and flattens thousands to tens of thousands of raw documents. The longer the analysed range, the slower it gets, in proportion.
- **Range limit and sampling**: because of that cost the range was capped at 30 days, and beyond 15,000 records the tool had to sample. Long-term trends are hard to see, and sampling lowers the accuracy of the result.
- **Complex logic**: automatic bucket choice, merging of small buckets and several statistical tests lived inside the tool, which made it hard to maintain and gave the LLM too many fields to read.
- **Dependence on raw documents**: inspection documents expire after about 30 days, so a comparison with anything older is simply impossible.

## Solution

### 1. Conversion

- The Data Service **aggregates inspection documents per period** and stores them in a separate collection, `inspectionStatistics`
- Four period types (`granularity`): `hourly` (1 hour), `shift` (12 hours, 00:00–12:00 / 12:00–24:00), `daily` (1 day), `weekly` (Monday to Monday). Every boundary is UTC
- Conversion unit: one statistics document per **(model name × model version × task × mode × gbm × process × productId × period type × period start)**
- One document holds the statistics of every equipment plus a `total` that sums all equipments
- If a period has no inspection document, no statistics document is created
- Target tasks are `cls` (classification) and `det` (detection). `seg` is not analysed for drift and is left out
- Retention: about one year (raw documents last 30 days, so long-term comparison becomes possible only through this collection)

### 2. What is stored

Drift analysis does not need "every single prediction"; it needs **"the distribution of how the model answered during this period"**. So individual predictions are not stored, only the numbers below. All of them are **additive** (counts, sums, histograms), so adding hourly documents gives exact daily and weekly values.

| Item | Description |
|---|---|
| Data counts | number of inspection documents, predictions (images), and for detection the number of bounding boxes |
| Counts per class | e.g. OK 2,190, NG 170 → class ratio |
| Confidence distribution | a **histogram**: the confidence values divided into 10 fixed bins with the count of predictions in each bin. Stored for the total and per class |
| Confidence summary | sum, sum of squares (→ mean and standard deviation), min/max, quantiles (p05–p95) |
| Threshold counts | predictions with confidence below the threshold, and predictions near the threshold (±0.05) |
| Boxes per image (det) | histogram of how many images had 0, 1, 2, ... boxes, and the number of images without a box |
| Inference time | count and sum of elapsed time |
| Runtime configuration | backend and threshold values (to detect configuration changes) |
| Quality | predictions with a missing confidence or a parse failure |

**Why the confidence histogram is the key item.** Put the histograms of two periods side by side and "the model is less sure than it used to be" shows up directly as counts moving between bins. And because histograms are additive, the exact distribution of any combination of periods can be rebuilt. Storing only an average would make this comparison impossible.

### 3. Statistics document schema (classification example, abbreviated)

```json
{
  "modelName": "sidetopsideu8000",
  "modelVersion": "sidetopsideu8000_260316_260316081905",
  "task": "cls",
  "mode": "production",
  "gbm": "SEHC",
  "process": "Side",
  "productId": "12HJ3NGL601106K",
  "granularity": "hourly",
  "startDate": "2026-06-20T00:00:00Z",
  "endDate":   "2026-06-20T01:00:00Z",
  "classes": ["OK", "NG"],
  "localTimezone": "Asia/Bangkok",
  "bins": {
    "confidenceEdges": [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0],
    "nearThresholdMargin": 0.05,
    "boxesPerImageMax": 5
  },
  "computedAt": "2026-06-20T01:05:12Z",
  "equipmentCount": 2,
  "total": { "...same structure as an equipment entry, the sum of every equipment..." },
  "equipments": [
    {
      "equipmentId": "SEHC_Side_VM07",
      "location": "VM07",
      "backend": "ts",
      "threshold": 0.5,
      "inspectionCount": 118,
      "predictionCount": 2360,
      "quality": { "missingConfidenceCount": 0, "parseErrorCount": 0 },
      "elapsedTime": { "count": 2360, "sum": 472.0, "mean": 0.2, "min": 0.1, "max": 0.31, "p50": 0.2, "p90": 0.24 },
      "classCounts": { "OK": 2190, "NG": 170 },
      "confidence": {
        "count": 2360, "sum": 2276.4, "sumSq": 2199.1, "mean": 0.9646, "std": 0.041,
        "min": 0.5012, "max": 0.9999,
        "quantiles": { "p05": 0.88, "p10": 0.91, "p25": 0.95, "p50": 0.975, "p75": 0.99, "p90": 0.995, "p95": 0.998 },
        "histogram": [0, 0, 0, 0, 3, 12, 41, 210, 620, 1474],
        "belowThresholdCount": 3,
        "nearThresholdCount": 15
      },
      "perClass": {
        "OK": { "...same structure as confidence, only predictions predicted as OK..." },
        "NG": { "...only predictions predicted as NG..." }
      }
    },
    { "equipmentId": "SEHC_Side_VM08", "location": "VM08", "..." : "..." }
  ]
}
```

- `bins` defines the histogram bins. The analysis tool reads them from the document, so the bins can be changed later without touching the tool. Documents with different bins are never compared with each other.
- For detection models `confidence`, `classCounts` and `perClass` are over bounding boxes, and `boxCount` plus an `images` block (histogram of boxes per image, number of images without a box) are added. The threshold is per class inside `perClass`.
- Indexes: one unique index on `(modelName, modelVersion, task, mode, gbm, process, granularity, startDate, productId)` and a TTL index on `startDate` for retention

### 4. How the drift analysis works

The new tool reads statistics documents only and does **nothing beyond additions and divisions**. The result is read in this order.

**(1) When did it change — the period series and consecutive comparison**

The requested range is listed period by period (for example the last 30 days, one day each) with the mean confidence, class ratio, below-threshold rate and so on of each period. Next to it comes the **PSI of every period against the period before it**. The day the PSI jumps is the moment of the change.

**(2) How much did it change — PSI (Population Stability Index)**

PSI is a single number that says "how different are two distributions". It compares the confidence histograms of two periods bin by bin and adds up how much of the share moved in each bin. It has been used for a long time in credit scoring and model monitoring, with a conventional reading scale.

| PSI | Reading |
|---|---|
| below 0.10 | no meaningful change |
| 0.10 to 0.25 | moderate change, worth checking |
| 0.25 and above | large change |

For example, if in the first week of June 95% of the confidences sat in the bins above 0.95, and in the third week that share dropped to 60% while the 0.8–0.9 bins grew, PSI goes above 0.25 and the meaning is "the model is less sure than it used to be". This is the earliest signal when the images changed (lighting, camera contamination, a part change).

The same PSI is computed on the **class ratio** (did the OK/NG ratio change) and, for detection, on the **boxes-per-image distribution** (is the detector finding fewer or more boxes than before). For the class ratio a statistical test (chi-square) is attached as a check, so that a tiny change is not exaggerated when the data volume is very large.

**(3) Where did it split — the change point**

Over the whole range the tool finds "the point where the PSI sum between the earlier part and the later part is largest", and returns a before/after summary around that point (histograms, class ratios, mean confidence and below-threshold rate of each side). When a reference range is given explicitly (comparison mode), the reference and the current range are compared directly. Per-class PSI says which class moved.

**(4) Was it a configuration change — hard breaks**

When the backend, the threshold or the class list differs between consecutive periods, that is reported separately as a "configuration change". A configuration change is an operational event, not drift, and needs confirmation on the line before any statistical conclusion.

**(5) The judgement — flags and preVerdict**

Fixed rules are applied to the numbers above to raise flags (`CONFIDENCE_SHIFT`, `CLASS_SHIFT`, `BOX_COUNT_SHIFT`, `THRESHOLD_PRESSURE`, `HARD_BREAK`, insufficient-data flags, ...), which are combined into a `preVerdict` (stable / suspicious / drift_likely / undetermined). The final decision is made by the LLM or the owner, who confirms or overrules it by reading the numbers. When either side has less data than the minimum (500 predictions for classification, 300 for detection) no flag is raised and the verdict stays undetermined.

### 5. Comparison with the current approach

| Item | Current (reads inspections directly) | New (inspectionStatistics) |
|---|---|---|
| Data read | thousands to tens of thousands of raw documents | a few small rows per period |
| Analysable range | at most 30 days, before the raw data expires | up to about 400 days, one year of retention |
| Sampling | above 15,000 records | none, always everything |
| Where the computation happens | inside the MCP tool | distributions and counts in the Data Service, only comparisons in the tool |
| Result fields | buckets, trends, outliers, KS, box geometry, ... | period series, consecutive PSI, change point, hard breaks, flags |
| Query by product | not possible | possible by productId |

The items removed (KS test, trend tests, outlier scores, box size and position statistics, image size change detection) either need raw values or are covered well enough by the period series and the consecutive PSI.

## Use cases

- **Model monitoring**: confidence distribution, class ratio, below-threshold rate and inference time over time, per model, line and equipment
- **Drift detection**: the Data Agent reads the tool result of section 4 and explains "when, how much, which class, and whether it was a configuration change"
- **Automated alerts**: `preVerdict` and the flags alone are enough for an alert policy (drift_likely → alert, suspicious → log, hard break → notify the owner)
- **Long-term comparison**: a reference period from months ago can still be compared with today after the raw data has expired

## Considerations

- **How documents are produced**: like the existing summary collection, on-ingest (updated as data arrives) plus an API endpoint for backfill. Because every stored item is additive, a late-arriving inspection document can be added into its period document. Quantiles and min/max are not additive, so they are recomputed once the period closes or kept as approximations.
- **Histogram bins**: a healthy model's confidences pile up above 0.9, so **finer bins above 0.9** (for example 0.9, 0.95, 0.98, 0.99, 1.0) are recommended instead of uniform 0.1 steps. Confirm against a real confidence sample from a model before fixing them, and do not change them afterwards (different bins cannot be compared with the past).
- **Cost of productId in the key**: if productId is a product barcode (one unit), a period gets as many documents as barcodes. A line inspecting 300 units an hour produces about 7,000 documents a day and roughly 2.5 million a year per model, and the tool has to sum them inside the database on every query. In that case, **also storing one "all products" document per period with productId null**, and reading only that document when no product is requested, should be considered.
- **Human feedback**: not included in the statistics.
- **Image spec, box geometry**: not stored. A camera or resolution change shows up only indirectly as a confidence change.
- **Retention**: a one-year TTL index on `startDate`. A separate `expiresAt` field can replace it later if the retention should differ per granularity.
