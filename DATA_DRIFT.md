# Data Drift Analysis — Implementation Guide

This document describes the `mongodb_analyze_data_drift` MCP tool as it is implemented in this repository: what it
computes, how the code is organised, how each step works, what the output looks like, and how the output is meant to
be handed to an LLM for the final drift decision. The original design specification is in `data_drift.md`; section
11 lists where the implementation deliberately departs from it.

---

## 1. Purpose

An inspection model is trained on images collected at one point in time. Once deployed, the images it receives can
change: a camera moves, lighting changes, a new supplier's parts arrive, a lens gets dirty. The model keeps
answering, but its answers become less reliable, and nobody notices because there is no ground truth at inference
time. This is **data drift**.

The database holds no pixels and almost no true labels. It does hold everything the model *output*: the predicted
class, the confidence, the bounding boxes, the threshold that was applied, the human feedback that was registered.
The tool therefore monitors the model's outputs over time and reports every statistic needed to decide whether the
inputs or the behaviour of the model changed. It does **not** decide on its own. It produces deterministic flags
and a pre-verdict, and an LLM (or a human) confirms or overrules them by reading the numbers.

Three kinds of drift are distinguished and each leaves a different signature in the output:

| Kind | Meaning | Signature in the output |
|---|---|---|
| Covariate shift | The images changed, the meaning of the labels did not | Confidence histogram moves to lower bins, below-threshold rate rises, class distribution barely moves |
| Prior / label shift | The proportion of classes changed | Class distribution moves, confidence histogram stays |
| Concept drift | The meaning of a label changed | Human feedback disagrees with predictions more often |

Only classification (`cls`) and object detection (`det`) models are supported. Segmentation output is a mask file,
which carries nothing the tool can compare.

---

## 2. Code layout

```
common/constants.py                  DBCollections.INSPECTIONS = "inspections"
mongodb_mcp/drift/
  config.py        DriftConfig: every threshold and minimum, echoed in the result
  extract.py       RecordExtractor: flattened rows -> Record / BoxRecord, confidence normalisation, feedback,
                   local time, data quality counters
  bucketing.py     bucket boundaries (1h, shift, 1d, 1w), automatic choice, bucket cap, small bucket merging
  summarize.py     per bucket and per side statistics, additive count arrays
  stats.py         PSI, Jensen-Shannon, KS, chi-square, Kendall tau, Theil-Sen, robust z, quantiles, histograms
  changepoint.py   before/after comparison of two record groups, best split search, secondary splits
  flags.py         rule table -> flags, flags -> pre-verdict
  analyze.py       orchestration: bucketing, summaries, splits, trends, outliers, hard breaks, result assembly
mongodb_mcp/connector/drift_query.py build_drift_match() and build_drift_pipeline()
mongodb_mcp/connector/connector.py   MongoDBConnector.analyze_data_drift(): streams rows, runs the analysis
mongodb_mcp/schemas/drift.py         pydantic output models (DriftAnalysisResult and its parts)
mongodb_mcp/tools/tools.py           the MCP tool mongodb_analyze_data_drift
tests/drift/                         synthetic row generator and unit tests for every module
tests/connector/test_drift_connector.py, tests/tools/test_drift_tool.py
```

Everything under `mongodb_mcp/drift/` is synchronous, pure Python with no third party numeric library. The only
runtime dependency added is `tzdata`, so that `zoneinfo` can resolve factory timezones on systems without a system
timezone database (Windows).

---

## 3. Tool interface

Registered name on the server: `mcp_mongodb_analyze_data_drift` (the tools server is mounted with the `mcp`
prefix; inside the code the function is `mongodb_analyze_data_drift`).

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model_name` | str | required | Model name, e.g. `MetalDet` |
| `model_version` | str | required | Model version, e.g. `1.0` |
| `start_date` | datetime | required | Start of the analysed (current) window, inclusive |
| `end_date` | datetime | required | End of the analysed (current) window, exclusive |
| `task` | `cls` / `det` / null | null | Only needed when the same model string is used for both tasks |
| `gbm`, `process`, `location`, `equipment_id` | str / null | null | Exact-match filters on `metadata.*` |
| `mode` | str / null | `production` | Operating mode. Pass `null` explicitly to include rework and test data |
| `bucket` | `auto`, `1h`, `shift`, `1d`, `1w` | `auto` | Time bucket size |
| `detail` | `full` / `compact` | `full` | Whether buckets carry histograms and quantiles or only scalar series |
| `defect_classes` | list[str] / null | null | Classes counted in the defect rate; every class except `Good` when omitted |
| `reference_start`, `reference_end` | datetime / null | null | Reference window; giving both switches to comparison mode |

The model string matched in the database is `f"{model_name}/{model_version}"`, compared exactly and
case-sensitively. A regex match (as the summary tools use) would blend `MetalDet/1.0` and `MetalDet/1.01` into one
series, which would defeat the purpose.

Naive datetimes are treated as UTC, which is how MongoDB stores `metadata.createdAt`.

### 3.1 Limits and validation

- `start_date < end_date`, and the same for the reference window.
- The reference window must end before the current window starts. Both must be given together.
- The windows together may cover at most **30 days** (`max_total_days`). Under this cap the automatic bucket choice
  produces daily buckets, or hourly ones for ranges of two days or less, so the result never exceeds a few dozen
  buckets.
- Never more than **60 buckets** (`max_buckets`). An explicit `bucket="1h"` over 30 days is coarsened step by step
  with a warning rather than emitting 720 buckets.
- Unknown `bucket` or `detail` values, an unsupported `task`, and a missing `inspections` collection raise a
  `ValueError` which the tool wraps in a `ToolError`.

---

## 4. Step 1 — Query and flattening (MongoDB side)

`build_drift_match()` produces the document-level filter:

```python
{
  "isDeleted": False,
  "metadata.createdAt": {"$gte": start_date, "$lt": end_date},
  "inspectionResult.aiResults": {"$elemMatch": {"aiModel": "MetalDet/1.0", "task": task}},   # task only if given
  "metadata.gbm": ..., "metadata.process": ..., "metadata.location": ..., "metadata.equipmentId": ...,
  "metadata.mode": "production"                                                             # omitted when mode=None
}
```

`build_drift_pipeline()` turns matching documents into **one row per prediction of the analysed model**:

| Stage | Purpose |
|---|---|
| `$match` | the filter above; uses the index on `metadata.createdAt` + `inspectionResult.aiResults.aiModel` |
| `$sort {metadata.createdAt: 1}` | done *before* any `$unwind` so the index carries the sort and the unwinds keep the order |
| `$project` | keeps only `metadata`, `dataSpec`, `schemaVersion`, `inspectionResult.conclusion`, `inspectionResult.aiResults` |
| `$unwind aiResults` with `includeArrayIndex: entryIndex` | one row per model entry; the index lets the extractor count entries per document |
| `$match aiModel (+ task)` | drops the other models in the document (e.g. the segmentation entry) |
| `$unwind predictions` | one row per prediction |
| `$project` | flat row: `_id, createdAt, localTimezone, gbm, process, location, equipmentId, productId, mode, conclusion, schemaVersion, dataSpec, entryIndex, backend, aiModel, task, classes, prediction` |

Detections stay nested inside `prediction` so a detection image arrives together with its boxes. `dataSpec` is
projected whole (it is a tiny list) and the image spec is picked in Python by `fileIndex`, which avoids
`$arrayElemAt` edge cases with null indices.

The connector runs `count_documents(match)` for `nDocsScanned`, then iterates the aggregation cursor with
`allowDiskUse=True` and feeds every row to a `RecordExtractor`. In comparison mode this happens once per window.
The rows are never loaded into a list; only the extracted records are kept in memory.

**Why not stream full documents?** Inspection documents are one per product or image; a two-week range can be
hundreds of thousands of documents, each carrying every model's output and additional data. Flattening in the
database ships only the fields the analysis needs.

**Documents versus records.** The document is not the unit of analysis; the prediction is. A document in which the
model appears twice yields two records. Five hundred documents in one day yield five hundred records in that day's
bucket. After extraction the document boundary disappears and all records are pooled by time. `inspection_id` and
`prediction_id` stay on every record for traceability, and `nDocsMatched`, `nEntriesMatched` and `nRecords` in the
result show the ratio.

---

## 5. Step 2 — Extraction (`extract.py`)

`RecordExtractor.add(row)` converts a row into a `Record`. It **never raises on a malformed row**: the row is
counted in `nParseErrors` (with up to ten example ids) and skipped. The only exception that propagates is
`UnsupportedTaskError` for a segmentation entry, because that is a caller mistake, not bad data.

### 5.1 Common fields

| Record field | Source | Notes |
|---|---|---|
| `inspection_id`, `prediction_id` | `_id`, `prediction.predictionId` | traceability |
| `created_at` | `metadata.createdAt` | normalised to UTC |
| `local_created_at` | `created_at` converted with `metadata.localTimezone` | used for bucket boundaries; UTC with a warning if the timezone is missing or unknown |
| `gbm, process, location, equipment_id, product_id, mode, conclusion, backend, classes` | metadata / entry | kept for investigation and hard break scanning |
| `threshold` | `prediction.threshold` (cls) / first box threshold (det) | may be null |
| `elapsed_time`, `is_patch` | prediction | |
| `patch_w`, `patch_h` | `patchSpec.x2 - x1`, `y2 - y1` | only when `isPatch` |
| `image_w`, `image_h`, `image_c` | `dataSpec[fileIndex]` | null and counted in `nMissingImageSpec` when the index is out of range or the keys are missing |
| `feedback_label`, `feedback_mismatch`, `feedback_other` | `feedbacks` | see 5.3 |

### 5.2 Confidence normalisation

`normalize_confidence(conf)` returns `(conf_max, margin, entropy)`:

- A **list** collapses to its **maximum** (`conf_max`). This is the rule requested by the product owner and applies
  everywhere a confidence appears, in classification predictions and in detection boxes alike. When the list has
  at least two entries the extractor also keeps `margin` (top minus second) and the Shannon `entropy` of the
  renormalised vector; both are more sensitive than the maximum alone.
- A **scalar** is used as is, with `margin` and `entropy` null.
- An empty list, a missing value or a non-numeric value gives null and is counted in `nMissingConfidence`.

Derived per classification record: `below_threshold = conf_max < threshold`, `near_threshold = |conf_max −
threshold| < 0.05`, `decision_differs = decision != prediction`. Both threshold flags are null when the threshold
is null.

### 5.3 Human feedback

`feedbacks` is a list of free-text entries. The most recent by `registeredAt` wins. If its text, stripped and
case-folded, equals one of the model's classes it is a **corrected label**: `feedback_label` is that class and
`feedback_mismatch` says whether it differs from the prediction. Any other text is a **comment**
(`feedback_other = true`) and does not count as a label. Feedback is the only proxy for ground truth in the data,
and it is sparse and biased towards suspicious cases, so it is always reported with its own sample count and never
used alone to declare drift.

### 5.4 Detection boxes

Each box becomes a `BoxRecord` with `prediction`, `conf_max` (maximum of the list, see above), `threshold`,
`below_threshold`, the raw geometry `x1 y1 x2 y2 w h area aspect cx cy`, and the normalised geometry `w_norm h_norm
area_norm cx_norm cy_norm` divided by the image size. Normalised geometry exists so that a camera resolution change
does not look like a box size change; it is null when the image size is unknown. Per image the extractor keeps
`n_boxes`, `n_boxes_by_class`, `n_below_threshold`, `mean_conf`, `min_conf`. A box with a malformed `bbox` is
skipped and counted as a parse error.

### 5.5 Bookkeeping

The extractor also tracks the set of tasks seen (for task resolution), the ordered union of `classes` across
entries, the timezones seen (a warning is added when several are mixed), and one-time warnings for an unknown
`schemaVersion`, a prediction class not listed in `classes`, or a timezone problem.

### 5.6 Task resolution

If `task` was given it is used (and validated). Otherwise the task is the single task seen in the extracted rows; if
the model string was used for both `cls` and `det` the tool raises `"The model has multiple tasks [...], specify
task"`. With no rows at all the tool still returns a result (`undetermined`, with a warning) rather than failing.

---

## 6. Step 3 — Bucketing (`bucketing.py`)

Statistics are computed per time bucket so the result is a *series* and the reader can see *when* something changed.
Buckets are cut on **factory local time** so that days and shifts line up with real operations.

| `bucket` | Boundary |
|---|---|
| `1h` | top of each hour |
| `shift` | 06:00, 14:00, 22:00 (`shift_hours`); a record before 06:00 belongs to the previous day's night shift |
| `1d` | local midnight |
| `1w` | Monday 00:00 local |

A UTC range therefore usually produces one more local bucket than calendar days (for example, UTC midnight is 09:00
in Seoul, so 14 UTC days spill into 15 partial-or-full Seoul days). The partial first and last buckets are merged if
they are too small.

**Automatic choice (`bucket="auto"`)**, with `MIN_N` = 200 classification records / 100 detection images per bucket:

1. Start with `1d`; use `1h` if the range is ≤ 2 days, `1w` if > 60 days (unreachable under the 30-day cap).
2. Build the buckets. If more than 30 % of them hold fewer than `MIN_N` records, coarsen one step
   (`1h → shift → 1d → 1w`) and repeat.
3. Whatever was chosen (automatically or explicitly), coarsen further while the bucket count exceeds 60, adding a
   warning each time.
4. Merge every remaining bucket with fewer than `MIN_N` records into its next neighbour (previous one for the last
   bucket). Merged buckets are listed in `dataQuality.mergedBuckets` and the surviving bucket's `mergedFrom` counts
   how many raw buckets it absorbed.

In comparison mode the bucket size is chosen once on both windows together, then the windows are bucketed and merged
separately so that a merge never crosses the gap between them.

---

## 7. Step 4 — Per bucket summaries (`summarize.py`)

Every bucket gets the same summary structure. `n` is always present so the reader can judge reliability.

**Important convention for detection:** an image row has no confidence of its own, so for `det` the confidence
statistics, the class distribution, the threshold statistics and the feedback statistics of a bucket are computed
**over the boxes** in that bucket. The `box` block then carries only geometry. `n` is the number of images and
`nBoxes` the number of boxes.

| Key | Definition |
|---|---|
| `bucketStart`, `bucketEnd`, `window` | local time with offset; `current` or `reference` |
| `n`, `mergedFrom` | records in the bucket, raw buckets merged into it |
| `classDist` | share of every class (all classes present, 0.0 included) |
| `defectRate` | sum of `classDist` over `defectClasses` |
| `confP50`, `confMean`, `confStd` | over `conf_max` |
| `confHist` | proportions over the **fixed** bins `[0,0.1) … [0.9,1.0]`; fixed edges make buckets comparable |
| `confQuantiles` | p05 p10 p25 p50 p75 p90 p95 |
| `belowThresholdRate` | over records/boxes that have a threshold; null if none has one |
| `thresholdValues` | distinct thresholds seen (should be exactly one) |
| `imageSpecs` | distinct `[width, height, channels]` seen |
| `elapsedTimeP50` | median inference time |
| `nFeedback`, `feedbackMismatchRate`, `nFeedbackOther` | labelled feedback count, mismatch rate over labelled feedback, comment count |
| *cls only* `marginQuantiles`, `entropyQuantiles` | p10 p50 p90; null when confidence vectors were not stored |
| *cls only* `decisionDiffersRate`, `patchWP50`, `patchHP50`, `nearThresholdRate` | |
| *det only* `nBoxes`, `boxesPerImageMean`, `boxesPerImageStd` | |
| *det only* `boxesPerImageHist` | proportions for 0, 1, 2, 3, 4, 5+ boxes |
| *det only* `noBoxRate`, `boxesByClassPerImage` | share of images with no box; mean box count per class per image |
| *det only* `box` | `nBoxes`, p10/p50/p90 of `areaNorm`, `wNorm`, `hNorm`, `aspect`, `cxNorm`, `cyNorm`; `areaNormHist` over fixed log10 bins `[-5,-4) … [-1,0]` |

With `detail="compact"` a bucket keeps only `bucketStart, bucketEnd, window, n, mergedFrom, classDist, defectRate,
confP50, confMean, belowThresholdRate, nFeedback, feedbackMismatchRate, nFeedbackOther, nearThresholdRate,
boxesPerImageMean, noBoxRate`. That is every scalar the trend and outlier rules use, so the flags are identical in
both forms. Everything outside `buckets` is unaffected by `detail`.

Internally every bucket also keeps its raw **count** arrays (confidence bin counts, class counts, boxes-per-image
counts). Counts are additive, so the histogram of any contiguous group of buckets is the exact sum of the counts,
never an average of proportions. This is what makes the change-point search cheap and exact.

---

## 8. Step 5 — Statistics across the range

### 8.1 Distance measures (`stats.py`)

All implemented in pure Python and unit tested against known values.

| Measure | Used for | Notes |
|---|---|---|
| **PSI** `Σ (q−p)·ln(q/p)` | histograms: confidence, classes, boxes per image | zeros replaced by 1e-4 and renormalised. < 0.10 nothing, 0.10–0.25 moderate, > 0.25 significant |
| **Jensen–Shannon** (base 2, in [0,1]) | second opinion on the confidence histogram | |
| **Kolmogorov–Smirnov** two-sample | continuous variables: confidence, margin, entropy, normalised area / cx / cy | statistic `D` (0 identical, 1 disjoint) and asymptotic p-value. Always read `D`: with large samples a tiny `D` is "significant" but irrelevant |
| **Chi-square** of independence | before/after class count table | p-value via the regularised incomplete gamma function; degenerate tables give (0, 1, 0) |
| **Kendall tau-b** + normal approximation | monotonic trend of a bucket series | tie corrected |
| **Theil–Sen slope** | robust slope per bucket | |
| **Robust z-score** `0.6745·(x−median)/MAD` | outlier buckets, leave-one-out | falls back to the mean absolute deviation when the MAD is 0, and to a capped ±99 when every other point is identical |

### 8.2 Change point — range mode (`changepoint.py`)

There is no predefined reference window, so the tool finds the split inside the range itself:

```
for k in [2, K−2]:                       # at least 2 buckets on each side
    before = buckets[:k], after = buckets[k:]       (count arrays summed via prefix sums)
    score_k = PSI(conf_hist) + PSI(class_dist) + (det) PSI(boxes_per_image_hist)
best = argmax score_k among splits where both sides have enough records
       (MIN_SIDE: 500 cls records / 300 det images and 300 boxes)
```

If no split has enough records on both sides, the best split among all of them is still reported with
`sidesSufficient=false`, so the reader sees where the largest movement is, but the shift flags stay silent.

For the winning split the full report is computed **from the records of the two sides**: `psiConf`, `jsConf`,
`ksConf`, `psiClass`, `chi2Class`, `maxClassPropChange`, and for detection `psiBoxesPerImage`, `ksAreaNorm`,
`ksCxNorm`, `ksCyNorm`, for classification `ksMargin`, `ksEntropy`. `before` and `after` carry a side summary each
(`n`, `classDist`, `confHist`, `confQuantiles`, `belowThresholdRate`, `boxesPerImageHist`, feedback, elapsed time)
so the *direction* of the move is visible, not only its size.

After the primary split, the same search runs once on each side that still holds at least 4 buckets; a secondary
split is reported only when its sides are sufficient (`secondaryChangePoints`).

### 8.3 Comparison mode

When `reference_start`/`reference_end` are given, the change-point search is skipped and its place is taken by a
**fixed** comparison: reference records are the `before` side, current records the `after` side, and `comparison`
carries exactly the fields of a change point (with `bucketIndex` null and `date` = start of the current window).
`changePoint` is null in this mode. Trend and outlier detection run over the combined chronological bucket sequence
(reference buckets first); Kendall tau is rank based, so the gap between the windows does not invalidate it.
Insufficient data is checked per window.

### 8.4 Trends (`analyze.py::_trends`)

For every series in `conf_p50, conf_mean, below_threshold_rate, defect_rate, boxes_per_image_mean, no_box_rate,
margin_p50, entropy_p50, area_norm_p50, cx_norm_p50, cy_norm_p50` with at least 6 non-null bucket values:
Kendall tau and p-value against the bucket index, Theil–Sen slope per bucket, first and last value. A trend is
`meaningful` when `p < 0.05`, `|tau| ≥ 0.5`, and the move from first to last is practically relevant:

- values (confidence, margin, entropy, normalised geometry): absolute change ≥ 0.03
- rates (below threshold, defect, no box): absolute change ≥ 0.02 **or** relative change ≥ 50 %
- counts (boxes per image): relative change ≥ 20 %

Note that a sharp step also produces a high |tau|, so a step change usually carries both `CONFIDENCE_SHIFT` and one
or more `TREND_*` flags. That is expected.

### 8.5 Outlier buckets (`analyze.py::_outliers`)

For `conf_p50, below_threshold_rate, defect_rate, boxes_per_image_mean, no_box_rate`, with at least 6 buckets, every
bucket gets a leave-one-out robust z-score. It is an outlier when `|z| > 3.5` **and** the move against the median of
the other buckets is practically relevant (for rates: absolute ≥ 0.02 **and** relative ≥ 50 %). The second condition
was added because on a very stable series the MAD is tiny and ordinary noise produced z-scores above 3.5.

An outlier whose neighbours are normal is a **transient** (one bad lot, a lamp off for a shift), not drift.

### 8.6 Hard breaks (`analyze.py::_hard_breaks`)

The ordered records (both windows in comparison mode) are scanned for changes of values that should never change
silently: `image_spec` (camera or preprocessing changed), `threshold` (model configuration changed), `backend`
(runtime changed), `classes` (output space changed). Each change is reported with the date, the inspection id, the
old and the new value. Additionally, when the median `elapsed_time` after the split is more than 1.5× or less than
0.67× the median before, an `elapsed_time` break is added; it is an infrastructure signal, not drift, but it often
co-occurs with a resolution change.

### 8.7 Pairwise maximum

`maxPairwise` holds the largest PSI between any two buckets for the confidence histogram and for the class
distribution, with the pair of bucket start times. It shows the worst disagreement in the range even when no clean
split exists.

---

## 9. Step 6 — Flags and pre-verdict (`flags.py`)

| Flag | Condition |
|---|---|
| `INSUFFICIENT_DATA` | fewer than `MIN_SIDE` records overall (per window in comparison mode) |
| `INSUFFICIENT_BUCKETS` | fewer than 3 buckets after merging (range mode) |
| `HARD_BREAK` | any hard break other than `elapsed_time` |
| `CONFIDENCE_SHIFT` | `psiConf ≥ 0.25` with both sides sufficient |
| `CONFIDENCE_SHIFT_MODERATE` | `0.10 ≤ psiConf < 0.25` |
| `CLASS_SHIFT` | `psiClass ≥ 0.25`, or `chi2Class.p < 0.001` with `maxClassPropChange ≥ 0.05` |
| `CLASS_SHIFT_MODERATE` | `0.10 ≤ psiClass < 0.25` |
| `BOX_COUNT_SHIFT` | det: `psiBoxesPerImage ≥ 0.25` |
| `BOX_GEOMETRY_SHIFT` | det: any of `ksAreaNorm`, `ksCxNorm`, `ksCyNorm` with `D ≥ 0.15` and `p < 0.01` |
| `THRESHOLD_PRESSURE` | `belowThresholdRate` after ≥ 2× before and absolute increase ≥ 0.02 |
| `TREND_<SERIES>` | one per meaningful trend, e.g. `TREND_CONF_P50`, `TREND_NO_BOX_RATE` |
| `TRANSIENT_OUTLIER` | outlier bucket(s) present but no shift flag and no trend flag |
| `FEEDBACK_DEGRADATION` | ≥ 30 labelled feedbacks on both sides and mismatch rate after ≥ 1.5× before |

Split-based flags only fire when `sidesSufficient` is true.

```
undetermined  if any INSUFFICIENT_* flag
drift_likely  if HARD_BREAK, CONFIDENCE_SHIFT, CLASS_SHIFT, BOX_COUNT_SHIFT, BOX_GEOMETRY_SHIFT,
              FEEDBACK_DEGRADATION, or two or more TREND_* flags
suspicious    if only *_MODERATE, THRESHOLD_PRESSURE, TRANSIENT_OUTLIER, or exactly one TREND_* flag
stable        otherwise
```

---

## 10. Output

The tool returns a `DriftAnalysisResult`. On the wire every key is camelCase, floats are rounded to 4 decimals,
timestamps are ISO 8601 with offset (bucket times in factory local time, range and hard break times in UTC), and
nothing is a Python-only type.

```jsonc
{
  "modelName": "MetalDet", "modelVersion": "1.0", "task": "det",
  "mode": "range",                                   // or "comparison"
  "filters": {"gbm": "SEV", "process": null, "location": null, "equipment_id": null, "mode": "production"},
  "range": {"start": "2026-09-01T00:00:00Z", "end": "2026-09-15T00:00:00Z", "days": 14.0},
  "referenceRange": null,                            // filled in comparison mode
  "bucket": "1d",                                    // the resolved size, never "auto"
  "detail": "full",
  "status": {"analysisPossible": true, "changePointRan": true, "comparisonRan": false,
             "trendRan": true, "outlierRan": true, "nBuckets": 15},
  "dataQuality": {"nDocsScanned": 4200, "nDocsMatched": 4200, "nEntriesMatched": 4200, "nRecords": 4200,
                  "nBoxes": 8290, "nMissingConfidence": 0, "nMissingImageSpec": 0, "nParseErrors": 0,
                  "parseErrorExamples": [], "mergedBuckets": ["2026-09-15T00:00:00+09:00"]},
  "classes": ["Good", "NG"], "defectClasses": ["NG"],
  "warnings": [],
  "buckets": [ { /* section 7 */ }, ... ],
  "changePoint": {
    "bucketIndex": 7, "date": "2026-09-08T00:00:00+09:00", "score": 0.61,
    "nBefore": 2100, "nAfter": 2100, "sidesSufficient": true,
    "psiConf": 0.48, "jsConf": 0.09, "ksConf": {"d": 0.31, "p": 0.0, "nBefore": 4140, "nAfter": 4150},
    "psiClass": 0.01, "chi2Class": {"stat": 0.8, "p": 0.37, "dof": 1}, "maxClassPropChange": 0.006,
    "psiBoxesPerImage": 0.02, "ksAreaNorm": {...}, "ksCxNorm": {...}, "ksCyNorm": {...},
    "before": {"n": 2100, "nBoxes": 4140, "classDist": {...}, "confHist": [...], "confQuantiles": {...},
               "belowThresholdRate": 0.03, "boxesPerImageHist": [...], "nFeedback": 4,
               "feedbackMismatchRate": 0.25, "elapsedTimeP50": 0.008},
    "after":  { /* same keys */ }
  },
  "secondaryChangePoints": [],
  "comparison": null,                                // filled in comparison mode, same shape as changePoint
  "maxPairwise": {"psiConf": {"value": 0.62, "pair": ["2026-09-02T00:00:00+09:00", "2026-09-11T00:00:00+09:00"]},
                  "psiClass": {"value": 0.04, "pair": [...]}},
  "outlierBuckets": [],
  "trend": {"conf_p50": {"tau": -0.71, "p": 0.0004, "slopePerBucket": -0.004, "first": 0.94, "last": 0.89,
                         "nPoints": 15, "meaningful": true}, ...},
  "hardBreaks": [],
  "flags": ["CONFIDENCE_SHIFT", "THRESHOLD_PRESSURE", "TREND_CONF_P50"],
  "preVerdict": "drift_likely",
  "config": { /* every DriftConfig value */ }
}
```

**Null means "not computable", never zero.** A null `belowThresholdRate` means no record had a threshold; a null
`marginQuantiles` means confidence vectors were not stored; a null `changePoint` means the search did not run.

### 10.1 Size

Measured on synthetic data with compact JSON:

| Scenario | Buckets | Size | Approx. tokens |
|---|---|---|---|
| 14 days daily, detection, full | 15 | 26 KB | 7k |
| 30 days daily, detection, full | 31 | 46 KB | 13k |
| 30 days daily, classification, full | 31 | ~40 KB | 11k |
| any of the above, compact | | 10–25 KB | 3–7k |

A detection bucket is about 1.3 KB in full form, a classification bucket about 1.1 KB, the fixed part (change
point with both sides, trends, quality, config) 6–8 KB. Under the 30-day cap the largest possible result is 48
hourly detection buckets at roughly 75 KB.

---

## 11. Deviations from `data_drift.md` and why

| Spec | Implementation | Reason |
|---|---|---|
| Parse each document with `InspectionDocumentV1.model_validate` | Extract from raw dicts with tolerant access | That model lives in `cosmo_data`, not in this repo; tolerant access also matches "never crash on one malformed document" |
| Stream full documents with a projection | Aggregation pipeline flattens to one row per prediction | Far less data over the wire; the sort uses the index |
| `pandas` and `scipy` | Pure Python `stats.py` | No numeric dependency in the project; the statistics are simple and the counts are small; Docker image stays lean |
| Trend and outlier analysis run whenever ≥ 3 buckets | Require ≥ 6 buckets | Kendall tau and a median-based z-score are meaningless on 3–5 points |
| Outlier = `|z| > 3.5` | Also requires a practically relevant move | On a stable series the MAD is tiny and noise alone exceeded 3.5 |
| `model` as one string in the output | `modelName` and `modelVersion` | Requested; the combined string is only used in the query |
| Seven-point quantiles everywhere | Seven points for confidence, three (p10/p50/p90) for margin, entropy and box geometry | Keeps a detection bucket near 1.3 KB |
| No range limit | 30 days over both windows, 60 buckets maximum | Bounds result size and query cost |
| Single mode | Range mode plus comparison mode with a reference window | Needed for "September 1–7 versus October 1–7" questions |
| `detail` not specified | `full` (default) or `compact` | Lets a caller ask for just the series |
| Feedback counted per record | Detection feedback counted per box | Detection predictions carry feedback on the boxes, not on the image |
| Change point only reported with sufficient sides | Best insufficient split reported with `sidesSufficient=false` | The reader still sees where the movement is; flags stay silent |

---

## 12. Configuration (`DriftConfig`)

Every constant is a field of the frozen dataclass `DriftConfig` and is echoed in `result.config`. Defaults:

| Field | Value | Field | Value |
|---|---|---|---|
| `max_total_days` | 30 | `psi_moderate` / `psi_significant` | 0.10 / 0.25 |
| `max_buckets` | 60 | `ks_d_geometry` / `ks_p_geometry` | 0.15 / 0.01 |
| `min_bucket_records_cls` / `_det` | 200 / 100 | `chi2_p` / `class_prop_change` | 0.001 / 0.05 |
| `small_bucket_share` | 0.3 | `threshold_pressure_ratio` / `_abs` | 2.0 / 0.02 |
| `min_side_records_cls` / `_det` | 500 / 300 | `trend_p` / `trend_tau` | 0.05 / 0.5 |
| `min_side_boxes` | 300 | `trend_value_change` | 0.03 |
| `min_buckets_changepoint` | 3 | `trend_rate_abs` / `trend_rate_rel` | 0.02 / 0.5 |
| `min_buckets_trend` | 6 | `trend_count_rel` | 0.2 |
| `min_segment_buckets` | 2 | `robust_z` | 3.5 |
| `shift_hours` | (6, 14, 22) | `feedback_min` / `feedback_ratio` | 30 / 1.5 |
| `conf_bin_edges` | 0.0 … 1.0 step 0.1 | `near_threshold_margin` | 0.05 |
| `area_log_bin_edges` | −5 … 0 | `elapsed_ratio_high` / `_low` | 1.5 / 0.67 |
| `boxes_per_image_max_bin` | 5 | `decimals` | 4 |
| `eps` | 1e-4 | `good_class` | "Good" |

The connector uses `DEFAULT_CONFIG`; a different `DriftConfig` can be passed to `analyze_records()` directly.

---

## 13. Edge cases handled

- Model not found in the range: result with `analysisPossible=false`, `nDocsMatched=0`, flag `INSUFFICIENT_DATA`, a
  warning, no exception.
- Same model twice in one document: both entries processed, `nEntriesMatched > nDocsMatched`.
- Empty or missing confidence: record kept for class statistics, excluded from confidence statistics, counted.
- Null threshold: threshold flags null; rates computed over records that have a threshold; `THRESHOLD_PRESSURE`
  cannot fire when nobody has one.
- `fileIndex` beyond `dataSpec`: normalised geometry null, counted in `nMissingImageSpec`.
- Missing or unknown `localTimezone`: UTC used, warning added. Several timezones in one range: warning added.
- `classes` differ between entries: `classes` hard break, union used for the class distribution.
- Prediction class not in `classes`: included in the distribution, warning added.
- Unknown `schemaVersion`: extracted on a best effort basis, warning added.
- All records in one bucket, or fewer than 3 buckets: descriptive summary returned, `INSUFFICIENT_BUCKETS`.
- Explicit hourly buckets over a long range: coarsened with a warning instead of exceeding 60 buckets.

---

## 14. Tests

`tests/drift/synthetic.py` generates flattened rows exactly as the pipeline would emit them, with controllable
confidence mean, defect rate, box count, box size and position, image size, feedback and timezone.

| File | Covers |
|---|---|
| `tests/drift/test_stats.py` | every helper against known values (PSI symmetry and zero handling, KS on identical/disjoint samples, chi-square on a known table, chi-square survival at textbook critical values, Kendall on monotonic and flat series, Theil–Sen, robust z fallbacks) |
| `tests/drift/test_extract.py` | list confidence maximum, margin and entropy, thresholds, patch and image geometry, box normalisation, feedback label vs comment and latest-wins, repeated model entries, malformed rows counted not raised, segmentation rejected, timezone fallbacks, warnings |
| `tests/drift/test_bucketing.py` | hour/day/week/shift boundaries, local time bucketing, automatic choice and coarsening, bucket cap, small bucket merging |
| `tests/drift/test_analyze.py` | the eight scenarios from the spec (no drift → stable; step on day 8 → `CONFIDENCE_SHIFT` at the right date; gradual trend → `TREND_CONF_P50`; one bad day → `TRANSIENT_OUTLIER` only; resolution change → `HARD_BREAK` without a false geometry shift; list confidence; repeated model; too little data), plus comparison mode, compact detail, JSON serialisation, size bound, defect class override, threshold hard break, config echo, window validation, task resolution |
| `tests/connector/test_drift_connector.py` | match and pipeline shape, range and comparison mode against a mocked async aggregation cursor, naive dates as UTC, mode null, empty result, validation errors, missing collection |
| `tests/tools/test_drift_tool.py` | tool registration, argument pass-through, error wrapping in `ToolError` |

Run with `python -m pytest -q`.

### 14.1 Performance

Synthetic 30-day detection range, 150 000 images, 296 000 boxes: extraction 2.5 s, analysis 5.2 s, 31 buckets,
46 KB result (Python 3.11, no numeric libraries). The analysis runs in a thread via `run_in_executor` so other tool
calls are not blocked.

---

## 15. Operations

- **Index** on the inspections collection, so the sort and the model filter are served from the index:
  ```javascript
  db.inspections.createIndex({ "metadata.createdAt": 1, "inspectionResult.aiResults.aiModel": 1 })
  ```
  The tool does not create it; index creation is an operator decision.
- **Dependency:** `tzdata` in both requirement files. Linux images carry a system timezone database and ignore it.
- **Logging:** the connector logs the record count per window and the final record count, bucket count and
  pre-verdict per call. Malformed rows are logged at debug level with their id.

---

## 16. Handing the result to an LLM

### 16.1 How the result reaches the LLM

The tool is an ordinary MCP tool, so the flow is the standard one:

1. The LLM client (an agent built on the Claude API, Claude Code with this server configured, or any MCP host)
   lists the server's tools and sees `mcp_mongodb_analyze_data_drift` with the docstring above as its description
   and the JSON schema of its arguments.
2. When a user asks something like *"did MetalDet 1.0 drift on the SEV SMD line in the first two weeks of
   September?"*, the LLM decides to call the tool and fills the arguments (`model_name="MetalDet"`,
   `model_version="1.0"`, `start_date="2026-09-01T00:00:00Z"`, `end_date="2026-09-15T00:00:00Z"`, `gbm="SEV"`,
   `process="SMD"`). For *"compare September 1–7 with October 1–7"* it also fills `reference_start` and
   `reference_end`.
3. The server runs the query and the analysis and returns the `DriftAnalysisResult`. FastMCP serialises it with the
   camelCase aliases into the tool result (structured content plus its JSON text), which the host places into the
   model's context as the tool call's result.
4. The LLM reads the JSON and answers. It computes nothing; it interprets.

The result is 10–75 KB depending on range, task and `detail`, which is a small fraction of any current context
window. Ask for `detail="compact"` when the consumer only needs the series and the change point; ask for `full`
when it should reason about histogram shapes per bucket.

The instruction in 16.2 belongs in the **system prompt** of the consuming agent (or in the prompt of a dedicated
"drift analyst" sub-agent that receives only the tool result). It is written so that the LLM's role is to **confirm
or overrule `preVerdict` with an explanation**, and to return a fixed JSON structure that the calling service can
store next to the tool result.

### 16.2 Instruction for the consuming LLM

Copy the block below verbatim into the system prompt.

````text
You are an AI inspection model monitoring analyst. You receive the JSON result of the `mcp_mongodb_analyze_data_drift`
tool for one inspection model over one time range (mode "range") or over a reference window and a current window
(mode "comparison"). Your job is to decide whether the model's input data or behaviour drifted, to say when and
how, to name the most likely cause, and to recommend an action. You do not compute statistics; every number you
need is already in the JSON. You confirm or overrule the `preVerdict` field and explain why.

## How to read the result

- `status` says what could be computed. If `analysisPossible` is false, or `flags` contain INSUFFICIENT_DATA or
  INSUFFICIENT_BUCKETS, answer "undetermined" and explain what is missing (too few records, too short a range).
- `dataQuality` tells you how much data the numbers rest on. `nRecords` is the number of predictions (images for
  detection), `nBoxes` the number of boxes. Mention any non-zero `nParseErrors`, `nMissingConfidence` or
  `nMissingImageSpec` that is large relative to `nRecords`. Read `warnings`.
- `buckets` is the chronological series. Each bucket has `n`; treat buckets with small `n` as unreliable. In
  comparison mode `window` tells you whether a bucket belongs to the reference or the current period.
- `changePoint` (range mode) or `comparison` (comparison mode) is the main before/after evidence. `date` is where
  the "after" side starts. `sidesSufficient` must be true for the divergence values to be trusted. `before` and
  `after` contain the histograms and quantiles of each side so you can see the direction of a move.
- `trend` gives, per series, Kendall tau (−1..1), its p-value, the slope per bucket, and the first and last value.
  `meaningful` is true only when the trend is significant and the move is large enough to matter.
- `outlierBuckets` lists single buckets that stand far from all others. An outlier with normal neighbours and no
  change point or trend is a transient event, not drift.
- `hardBreaks` lists configuration changes found in the data: image_spec (camera or preprocessing changed),
  threshold (model configuration changed), backend (runtime changed), classes (output space changed),
  elapsed_time (infrastructure changed). A hard break must be confirmed with the line owner before any
  statistical conclusion; it usually explains everything that changed after it.
- `flags` and `preVerdict` are deterministic rule outputs computed from thresholds listed in `config`.

For detection models the confidence, class and threshold statistics are computed over bounding boxes, and the
`box` block of each bucket carries normalised geometry (size and position relative to the image).

## Reading the divergence numbers

- PSI (psiConf, psiClass, psiBoxesPerImage): below 0.10 no meaningful change; 0.10 to 0.25 moderate, worth a
  look; above 0.25 significant.
- KS statistic D (ksConf, ksAreaNorm, ksCxNorm, ksCyNorm, ksMargin, ksEntropy): 0 identical, 1 completely
  separate. Judge by D, not by the p-value: with thousands of samples a D of 0.03 has a tiny p-value and means
  nothing. Treat D below 0.05 as no change, 0.05 to 0.15 as small, above 0.15 as a real move.
- chi2Class: use together with maxClassPropChange; a significant p with a proportion change under 0.05 is not a
  real class shift.
- jsConf is a second opinion on psiConf in the range 0..1; use it to confirm, not to decide.

## Signatures and their most likely explanation

| Observation | Most likely explanation |
|---|---|
| confHist moves to lower bins, belowThresholdRate and nearThresholdRate rise, classDist barely moves | Covariate shift: the images changed and the model is less sure. Earliest and most reliable drift signature. |
| classDist changes strongly while confHist stays the same | Prior/label shift: the product mix or the defect rate really changed. The model may be fine, production may not be. Report as "production change, model behaviour stable". |
| Both confidence and class distribution move together | Genuine drift or a new defect type. Check feedback. |
| ksAreaNorm, ksCxNorm or ksCyNorm large | Camera moved, zoom changed, fixture or part variant changed. Almost always a physical cause. |
| boxesPerImage histogram or noBoxRate moves | Detector missing objects (lighting, contamination) or seeing extra objects (debris, new part). |
| patchWP50 or patchHP50 moves (classification) | The upstream detector changed its output, so the classifier receives different crops. |
| hardBreaks non-empty | Configuration change. Confirm with the line owner first. |
| One entry in outlierBuckets, no change point flag, no trend flag | Transient event (bad lot, a shift with a lamp off). Not drift, but worth logging. |
| feedbackMismatchRate rises while nFeedback is adequate on both sides | Strongest available evidence of real accuracy degradation (concept drift or severe covariate shift). |
| Meaningful trend, no change point flag | Gradual degradation (lens contamination, wear, slow process change). |
| elapsed_time hard break together with an image_spec break | Resolution or preprocessing change. Infrastructure, not data. |

A change point answers WHEN, PSI and KS answer HOW MUCH, the before and after histograms answer IN WHICH DIRECTION,
the flags give a consistent first decision. A step change usually also produces TREND flags because a step is
monotonic; do not count that as two independent signals.

## Caution rules

1. Distrust any comparison where a side has fewer records than min_side_records in `config`, or where
   `sidesSufficient` is false.
2. Ignore statistically significant but tiny effects: KS D below 0.05, PSI below 0.10.
3. Treat isolated outlier buckets as transients unless a change point or trend supports them.
4. Treat hard breaks as a configuration event first and a drift second.
5. Feedback is sparse and biased towards suspicious cases; never declare drift on feedback alone, and never
   ignore a feedback degradation when nFeedback is adequate.
6. Do not invent numbers. Quote the values from the JSON, rounded sensibly, and name the bucket dates you rely on.
7. When you disagree with `preVerdict`, say so explicitly and give the numbers that made you overrule it.

## Required answer

Return first a JSON object with exactly these fields, then a short plain-language explanation of at most ten
sentences for the line owner.

{
  "drift_detected": true | false | null,            // null when undetermined
  "confidence": "high" | "medium" | "low",
  "drift_type": "covariate_shift" | "label_shift" | "concept_drift" | "configuration_change" | "transient" | "none" | "undetermined",
  "start_date": "YYYY-MM-DD" | null,                // bucket date the change starts, or the change point date
  "signals": ["...", "..."],                        // 2 to 6 short statements with numbers, e.g. "confidence median fell from 0.94 to 0.89 on 2026-09-08"
  "likely_cause": "...",
  "recommended_action": "...",
  "agrees_with_pre_verdict": true | false,
  "pre_verdict": "<copy of preVerdict>"
}

Mapping guidance: drift_likely with covariate signatures -> drift_detected true, high; suspicious -> usually
drift_detected false with medium or low confidence and a "monitor" action, unless the numbers convince you
otherwise; stable -> false, high; undetermined -> null, low.
````

### 16.3 Example answer

For the example output in section 10 the consuming LLM is expected to answer roughly:

```json
{
  "drift_detected": true,
  "confidence": "high",
  "drift_type": "covariate_shift",
  "start_date": "2026-09-08",
  "signals": [
    "confidence PSI between the sides of the change point is 0.48, far above 0.25",
    "confidence median fell from 0.94 before 2026-09-08 to 0.89 after",
    "below-threshold rate rose from 0.03 to 0.07",
    "class distribution unchanged (PSI 0.01, NG share moved by 0.006)",
    "box geometry unchanged (KS D below 0.05 on area and position)"
  ],
  "likely_cause": "Images changed while the product mix did not: most likely lighting or camera contamination on the line rather than a production change.",
  "recommended_action": "Inspect camera and lighting on Line_01, collect and label samples from 2026-09-08 onward, and compare them with the training set before retraining.",
  "agrees_with_pre_verdict": true,
  "pre_verdict": "drift_likely"
}
```

followed by the plain-language explanation.

### 16.4 Using the result without an LLM

`preVerdict`, `flags` and `changePoint.date` (or `comparison`) are already usable as an automated alert.
Suggested policy: alert on `drift_likely`, log on `suspicious`, ignore `stable`, and treat `undetermined` as a
data availability problem rather than a model problem.
