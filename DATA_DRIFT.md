# Data Drift Analysis

This document describes the `mongodb_analyze_data_drift` MCP tool as implemented in this repository: why it exists,
where the code lives, how a call flows through it, how every step works and why it was built that way, every
configuration value, and every field of the result with the way it is meant to be read.

---

## 1. Purpose

An inspection model is trained on images collected at one point in time. Once deployed, the images it receives can
change: a camera moves, lighting changes, a new supplier's parts arrive, a lens gets dirty. The model keeps answering,
but its answers become less reliable, and nobody notices because there is no ground truth at inference time. This is
**data drift**.

The database holds no pixels and no true labels. It does hold everything the model *output*: the predicted class,
the confidence, the bounding boxes, the threshold that was applied. The tool therefore monitors the model's outputs
over time and reports every statistic needed to decide whether the inputs or the behaviour of the model changed.
It does **not** decide on its own. It produces deterministic flags and a pre-verdict, and an LLM (or a human) confirms
or overrules them by reading the numbers.

Three kinds of drift are distinguished and each leaves a different signature in the output:

| Kind | Meaning | Signature in the output |
|---|---|---|
| Covariate shift | The images changed, the meaning of the labels did not | Confidence histogram moves to lower bins, below-threshold rate rises, class distribution barely moves |
| Prior / label shift | The proportion of classes changed | Class distribution moves, confidence histogram stays |
| Configuration change | Camera, threshold, runtime or class list changed | A hard break, usually followed by one of the two above |

Only classification (`cls`) and object detection (`det`) models are supported. Segmentation output is a mask file,
which carries nothing the tool can compare.

---

## 2. Files and folders

```
common/constants.py                  every drift enum and constant: DriftTask, BucketSize, DriftDetail, DriftMode,
                                     DriftWindowMode, DriftFlag, PreVerdict, FLAG_VERDICT, TrendSeries,
                                     OutlierSeries, TrendFamily, TREND_FLAG_FAMILY, SeriesKind, HardBreakKind,
                                     BoxGeometry, AUTO_BUCKET, CLASS_SHARE_PREFIX
common/config.py                     DataDriftConfig: every threshold and minimum, filled from the data_drift
                                     section of config.yaml, reachable as SETTINGS.data_drift
config.yaml                          the data_drift section with the values used in production

mongodb_mcp/drift/                   the analysis, one class per module, plain Python, synchronous
  window.py        DriftWindow       the analysed windows: UTC normalisation, validation, date ranges, total days
  query.py         DriftQueryBuilder build() document filter and build_pipeline() flattening aggregation
  extractor.py     RecordExtractor   flattened rows -> Record / BoxRecord, data quality counters
  buckets.py       BucketBuilder     bucket boundaries, automatic size choice, bucket cap, small bucket merging
  summaries.py     Summarizer        per bucket and per side statistics, additive side counts, series access
  splits.py        SplitFinder       before/after divergence, best split search, secondary splits
  series.py        SeriesAnalyzer    trends and outlier buckets over the scalar bucket series
  rules.py         DriftRules        statistics -> flags, flags -> pre-verdict
  drift.py         DriftAnalyzer     orchestration, hard breaks, pairwise maximum, result assembly

mongodb_mcp/utils/stats.py           PSI, Jensen-Shannon, KS, chi-square, Kendall tau, Theil-Sen, robust z,
                                     quantiles, histograms; plain functions over plain sequences
mongodb_mcp/utils/datetime.py        to_utc(), convert_to_utc_datetime(), get_elapsed_days()
mongodb_mcp/schemas/drift.py         pydantic models: the internal Record, BoxRecord, Bucket, SideCounts, and the
                                     result DriftAnalysisResult with all its parts
mongodb_mcp/connector/connector.py   MongoDBConnector.analyze_data_drift(): validates, streams rows, runs the analysis
mongodb_mcp/tools/tools.py           the MCP tool mongodb_analyze_data_drift
tests/drift/                         synthetic row generator and one test module per drift module
tests/utils/test_stats.py            the statistics against textbook values
tests/connector/test_drift_connector.py, tests/tools/test_drift_tool.py
```

No third party numeric library is used. Every statistic is implemented in `utils/stats.py` over plain Python lists,
which keeps the Docker image lean and makes every function unit testable with synthetic data.

---

## 3. Tool interface

Registered name on the server: `mcp_mongodb_analyze_data_drift` (the tools server is mounted with the `mcp` prefix).

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model_name` | str | required | Model name, e.g. `MetalDet`; matched exactly together with the version as `name/version` |
| `model_version` | str | required | Model version, e.g. `1.0` |
| `start_date` | datetime | required | Start of the analysed (current) window, UTC |
| `end_date` | datetime | required | End of the current window |
| `task` | `cls` / `det` / null | null | Only needed when the same model string is used for both tasks |
| `gbm`, `process`, `location`, `equipment_id` | str / null | null | Exact-match filters on `metadata.*` |
| `mode` | str / null | `production` | Operating mode. Pass `null` explicitly to include rework and test data |
| `bucket` | `auto`, `1h`, `1d`, `1w` | `auto` | Time bucket size |
| `detail` | `full` / `compact` | `full` | Whether buckets carry histograms and quantiles or only the scalar series |
| `reference_start_date`, `reference_end_date` | datetime / null | null | Reference window; giving both switches to comparison mode |

Two modes exist:

- **Range mode** (default): analyses `start_date` to `end_date`, cuts it into time buckets, searches for the moment the
  output distribution changed the most (change point), for gradual trends and for outlier buckets.
- **Comparison mode**: when the reference window is also given, the reference is compared against the current window
  at the fixed boundary between them instead of searching for a split.

The windows together may cover at most `max_total_days` (30). The reference window must end before the current one
starts. All dates are UTC, naive datetimes are treated as UTC, which is how MongoDB stores them. Both boundaries of a
window are inclusive in the query (`$gte` start, `$lte` end).

**When a model runs on several cameras, pass `equipment_id`.** Hard breaks (section 6.7) compare consecutive records,
and two cameras with different image sizes interleaved in time look like a camera that keeps changing.

---

## 4. Flow of one call

```
tool argument validation (fastmcp)
  -> MongoDBConnector.analyze_data_drift
       argument checks         task, bucket and detail must be supported values, the collection must exist
       DriftWindow             UTC-normalises the dates, validate() checks order and total length
       DriftQueryBuilder       build() + build_pipeline(), once per window
       RecordExtractor         one instance, rows of the reference window first, then the current window
       no records at all       ValueError "No inspection results found ..."
       DriftAnalyzer.run()     in a thread so the server stays responsive
            BucketBuilder      choose the bucket size on all records, build and merge per window
            Summarizer         one summary per bucket
            SeriesAnalyzer     trends and outliers over the summaries
            SplitFinder        change point (range mode, isolated outliers left aside) or fixed comparison
            hard breaks        scan of the ordered records
            DriftRules         flags and pre-verdict
            DriftAnalysisResult (rounded to `decimals`)
```

Every class reads `SETTINGS.data_drift` once when it is created. `Summarizer` holds the task and the class list, and
`SplitFinder` and `SeriesAnalyzer` are built on top of it, so the task never has to be threaded through the calls.

---

## 5. Step by step

### 5.1 Windows (`DriftWindow`)

A plain class holding the four dates. The constructor converts each one to UTC (a naive datetime is stamped as UTC,
an aware one is converted). The connector then calls `validate()`, which raises a `ValueError` when:

- the start date is not before the end date,
- only one of the two reference dates is given,
- the reference start is not before the reference end,
- the reference window ends after the current window starts,
- the two windows together cover more than `max_total_days`.

The class also exposes `comparison` (true when a reference window exists), `current` and `reference` as `DateRange`
values (start, end, length in days) that go straight into the result, and `total_days`, which the bucket chooser
uses.

*Why a plain class and not a pydantic model:* the object is built exactly once from already typed arguments, never
parsed from JSON, and never serialised. Validation on construction would hide the check inside a framework hook; an
explicit `validate()` call in the connector makes the order of checks visible where the request is handled.

### 5.2 Query and flattening (`DriftQueryBuilder`)

`build()` produces the document-level filter:

```javascript
{
  isDeleted: false,
  "metadata.createdAt": { $gte: start_date, $lte: end_date },
  "inspectionResult.aiResults": { $elemMatch: { aiModel: "MetalDet/1.0", task: "det" } },   // task: $in [cls, det] when not given
  "metadata.gbm": ..., "metadata.equipmentId": ...                                            // only the given filters
}
```

`build_pipeline()` turns the matching documents into **one row per prediction of the analysed model**: match, sort by
`metadata.createdAt` (before any unwind so the index carries the sort), project the needed fields, unwind
`aiResults` keeping the entry index, keep only entries of the model (and task, when given), unwind `predictions`,
project a flat row, and drop feedback arrays. Detections stay nested inside their prediction so an image row arrives
together with its boxes.

The connector runs `count_documents(filter)` for `scannedDocumentCount`, then iterates the aggregation cursor with
`allowDiskUse=True` and feeds every row into the extractor. Rows are never loaded into a list; only the extracted
records are kept in memory.

Create this index once on the `inspections` collection so both the sort and the model filter are served from it:

```javascript
db.inspections.createIndex({ "metadata.createdAt": 1, "inspectionResult.aiResults.aiModel": 1 })
```

### 5.3 Extraction (`RecordExtractor`)

`add_record(row, window)` turns a row into a `Record` tagged with its window (`current` or `reference`). A row whose
`task` differs from the requested one is ignored. When no task was requested, the first row's task becomes the task
of the analysis. The extractor **never raises on a malformed row**: any exception during extraction is caught, the
row is counted in `parseErrorCount` (with up to `max_parse_error_examples` ids), logged, and skipped.

| Record field | Source | Notes |
|---|---|---|
| `inspection_id`, `prediction_id` | `_id`, `prediction.predictionId` | traceability |
| `created_at` | `metadata.createdAt` | UTC |
| `gbm, process, location, equipment_id, product_id, mode, backend, classes` | metadata / entry | kept for hard break scanning and investigation |
| `threshold` | `prediction.threshold` (cls); the prediction threshold or else the first box threshold (det) | may be null |
| `elapsed_time` | `prediction.elapsedTime` | inference time |
| `image_width`, `image_height`, `image_channels` | `dataSpec[fileIndex]` | null and counted in `missingImageSpecCount` when the index is out of range or the keys are missing |

**Confidence normalisation.** A list of confidences collapses to its **maximum** (`max_confidence`); the other
entries are ignored. A scalar is used as is. An empty list, a missing or a non-numeric value gives null and is
counted in `missingConfidenceCount`; the record is kept and simply contributes nothing to the confidence statistics.
Derived per classification record, when both a threshold and a confidence exist:
`below_threshold = max_confidence < threshold`, `near_threshold = |max_confidence - threshold| < near_threshold_margin`.
The `decision` field of a prediction is not read: it carries nothing about the input distribution.

**Detection boxes.** Each box becomes a `BoxRecord` with `prediction`, `max_confidence`, `threshold`,
`below_threshold`, the raw geometry `x1 y1 x2 y2 width height area aspect cx cy`, and the normalised geometry
`normalized_width normalized_height normalized_area normalized_cx normalized_cy` divided by the image size.
Normalised geometry exists so that a camera resolution change does not look like a box size change; it is null when
the image size is unknown. A box without a confidence is counted in `missingConfidenceCount`. Per image the record
keeps `box_count`, `box_count_by_class`, `below_threshold_count`, `mean_confidence`, `min_confidence`. A box with a
malformed `bbox` is skipped and counted as a parse error.

**Bookkeeping.** The extractor tracks the ordered union of `classes` across entries and the matched document and
entry ids. Every prediction is identified by document id, entry index and prediction id, and a prediction seen a
second time is ignored. Both window bounds are inclusive, so a document stamped exactly on the boundary between the
reference and the current window is returned by both queries; the copy from the reference window, extracted first,
is the one that counts. When neither window produced a single record, the connector raises
`"No inspection results found for model ... in the requested time window"` instead of returning an empty result.

### 5.4 Bucketing (`BucketBuilder`)

Statistics are computed per time bucket so the result is a *series* and the reader can see *when* something changed.
Buckets are cut on UTC boundaries: top of the hour (`1h`), midnight (`1d`), Monday midnight (`1w`).

**A bucket exists only where records exist.** `build()` groups records by the floor of their timestamp; a day without
inspections produces no bucket at all. Daily buckets over a month of sparse production are therefore not contiguous,
and the number of buckets is the number of distinct periods with data, not the number of periods in the window.

#### 5.4.1 Why a minimum bucket size

Every bucket becomes a set of statistics: class shares, a confidence histogram, quantiles. The change point search
and the outlier detection compare those statistics bucket to bucket with PSI and chi-square. On a bucket with 30
records those numbers are mostly sampling noise: a class share of 0.40 versus 0.55 between two such buckets is
normal variation, not drift, yet it would score as a large move and could pull the change point next to it. Every
bucket must therefore hold at least `MIN_N` records, `min_bucket_records_cls` (200) for classification or
`min_bucket_records_det` (100) for detection. Two mechanisms enforce it: the automatic size choice tries to pick a
grid where buckets are naturally large enough, and the merge folds the leftovers into a neighbour.

#### 5.4.2 Automatic bucket size (`bucket="auto"`)

`choose()` is a guess-and-check in three steps:

1. **Initial guess from the window length.** Total analysed days (current plus reference) picks the start:
   `1h` when the total is at most 2 days, `1w` when it is over 60 days, `1d` otherwise. With `max_total_days` at 30
   the weekly branch is unreachable, so auto effectively starts hourly for a short window and daily for everything
   else.
2. **Coarsen while too many buckets are thin.** Build raw buckets at that size over all records and count how many
   hold fewer than `MIN_N`. If more than `small_bucket_share` (30 %) of them are small, move one step coarser
   (`1h -> 1d -> 1w`) and try again. The ladder stops at `1w`.
3. **Hard cap, in auto and explicit mode alike.** While the chosen size would produce more than `max_buckets` (60)
   buckets, coarsen once more. With a 30 day maximum this only bites for hourly buckets over more than two and a half
   days.

*Why not always the finest grid:* finer buckets locate a change more precisely, but only if each bucket is big
enough to trust. If a third of the buckets would be merged away anyway, the resolution is too fine for the data
volume; a coarser grid gives fewer, more honest points. *Why the share is counted over buckets, not records:* it is a
cheap rule that is easy to reason about. Its known weakness is that two nearly empty days out of five trigger
coarsening even though they carry almost none of the records.

An explicit `bucket` skips steps 1 and 2; only the cap applies.

#### 5.4.3 Merging small buckets (`merge_small`)

After the size is chosen, the raw buckets of one window are built and every bucket under `MIN_N` is folded into a
neighbour:

- The merger walks the list left to right. A small bucket is merged **forward** into the next bucket, which takes
  over the small bucket's start date.
- If the small bucket is the **last** one, there is no next bucket, so it is merged **backward** into the previous
  one, which extends its end date.
- The absorbing bucket's `mergedFrom` grows by the absorbed bucket's `mergedFrom`, so it counts raw buckets, not merge
  operations. The start dates of every absorbed raw bucket are listed in `dataQuality.mergedBuckets`.
- Merging happens **per window**. In comparison mode the reference buckets and the current buckets are merged
  separately, so a merge never crosses the boundary between the two windows.

Merging is repeated until every bucket in the window holds at least `MIN_N` records or only one bucket is left.
Because a merged bucket can itself still be small, a run of thin buckets collapses into one.

*Why forward rather than into the larger neighbour:* a rule that always merges in one direction is deterministic and
keeps the chronological order trivially intact. Merging into whichever neighbour is larger would make the resulting
bucket boundaries depend on volumes, so the same window could be cut differently after a few more records arrive.
The single exception for the last bucket exists only because there is nothing after it.

*Why merging instead of dropping:* the records of a thin bucket are real traffic. Dropping them would make
`recordCount` disagree with what was scanned and would silently delete the very periods where something may have
gone wrong (a line that produced little because it was stopping for a problem).

*What to watch for:* a merged bucket's date span covers the whole distance between the first and the last absorbed
bucket, including any empty days in between. A daily bucket that reads `2026-07-16` to `2026-07-25` with
`mergedFrom = 2` holds two days of data with a week-long gap, not nine days. `mergedFrom` is the only hint.

#### 5.4.4 Worked example

A classification model, 30 day range, `bucket="1d"`, `MIN_N = 200`. Only five days carry inspections:

| Raw daily bucket | Records | Fate |
|---|---|---|
| 07-15 | 1094 | kept as is |
| 07-16 | under 200 | merged forward into 07-24 |
| 07-24 | the rest of 1867 | absorbed 07-16, now spans 07-16 to 07-25, `mergedFrom = 2` |
| 07-29 | the rest of 2029 | absorbed 07-31, now spans 07-29 to 08-01, `mergedFrom = 2` |
| 07-31 | under 200 | last bucket, merged backward into 07-29 |

Three buckets remain, `mergedBuckets = [07-16, 07-31]`, and because the change point needs four buckets the result
carries `INSUFFICIENT_BUCKETS` and an `undetermined` verdict. With `bucket="auto"` the same data would have moved to
weekly buckets (two of five raw buckets are small, 40 % is above the 30 % share), landing in three calendar weeks
with no merge needed, and still three buckets. The real problem in such a case is sparsity, not bucketing; a
comparison against an earlier reference window is the mode that can still produce a verdict.

### 5.5 Summaries (`Summarizer`)

Every bucket gets the same summary (section 8.4). **Convention for detection:** an image row has no confidence of
its own, so for `det` the confidence statistics, the class distribution and the threshold statistics of a bucket are
computed **over the boxes** in that bucket. The `box` block then carries only geometry. `recordCount` is the number
of images and `boxCount` the number of boxes.

Internally every group of records also yields its raw **count** arrays (`SideCounts`: confidence bin counts, class
counts, boxes-per-image counts). Counts are additive and subtractable, so the histogram of any contiguous group of
buckets is the exact sum of the counts, never an average of proportions. This is what makes the change point search
cheap and exact.

The quantile points come from the schema itself: `ConfidenceQuantiles` has the fields `p05 p10 p25 p50 p75 p90 p95`,
`Quantiles` has `p10 p50 p90`, and the summarizer reads the points from those names.

### 5.6 Series analysis (`SeriesAnalyzer`)

Runs before the split search so that the split can leave transients aside. Both analyses need at least
`min_buckets_trend` (6) buckets with a value for the series; below that nothing is computed and `status.trendRan`
is false.

**Trends.** For every series in `TrendSeries` (`median_confidence, mean_confidence, below_threshold_rate,
mean_boxes_per_image, no_box_rate, median_normalized_area, median_normalized_cx, median_normalized_cy`) plus one
`class_share_<class>` series per class: Kendall tau and its p-value against the bucket index, Theil-Sen slope per
bucket, first and last value. A trend is `meaningful` when `p < trend_p`, `|tau| >= trend_tau`, and the move from
first to last is practically relevant (section 6.6).

**Outliers.** For every series in `OutlierSeries` (`median_confidence, below_threshold_rate, mean_boxes_per_image,
no_box_rate`) plus the class shares, every bucket gets a leave-one-out robust z-score. It is an outlier when
`|z| > robust_z` (3.5) **and** the move against the median of the other buckets is practically relevant (for rates
both the absolute and the relative threshold must hold).

### 5.7 Change point and comparison (`SplitFinder`)

The question a change point answers is: *if the range were cut into a before and an after, where would the cut be so
that the two sides differ the most?*

**Range mode, step 1: leave isolated outliers aside.** A bucket flagged as an outlier whose neighbours are not
flagged is removed from the search. A single anomalous bucket would otherwise pull the best split next to itself and
read as a persistent shift, when it is really a one-day event. Two or more flagged neighbours in a row are a level
change, not a transient, so they stay in. The removed buckets are still reported in `buckets` and `outlierBuckets`.

**Step 2: score every cut.** Over the remaining `K` buckets, every position with at least `min_segment_buckets` (2)
buckets on each side is a candidate, so there are `K - 3` candidates:

```
for k in [2, K - 2]:
    before = buckets[:k], after = buckets[k:]                      # count arrays via prefix sums, exact and cheap
    score_k = PSI(confidence_hist) + PSI(class_dist) + (det) PSI(boxes_per_image_hist)
```

*Why PSI as the score:* it is the same quantity the flags read, so the split the search picks is the one the flags
will judge. *Why a sum of several PSIs:* a covariate shift moves the confidence histogram, a label shift moves the
class distribution, and a detector losing objects moves the boxes-per-image histogram; summing them lets the search
find whichever kind of change is largest without knowing in advance which one to look for.

**Step 3: prefer a cut with enough data on both sides.** The best score among cuts where both sides are
*sufficient* wins: `min_side_records_cls` (500) records or `min_side_records_det` (300) images on each side, plus
`min_side_boxes` (300) boxes per side for detection. If no cut is sufficient, the best cut overall is still reported
with `sidesSufficient = false`, so the reader sees where the largest movement is, but no shift flag fires and
`INSUFFICIENT_DATA` is raised instead. `bucketIndex` refers to the full bucket sequence, including the buckets that
were left aside.

*Why an explicit sufficiency rule:* PSI, chi-square and KS all become more excitable as samples shrink. Reporting a
divergence value computed on 80 records as if it were as trustworthy as one on 8000 would mislead the reader; the
flag stays silent and the number is marked as indicative.

**Step 4: the full report.** For the winning cut the complete comparison (section 8.5) is computed from the records
of the two sides: PSI, Jensen-Shannon and KS on the confidences, PSI and chi-square on the classes, the largest
change of any class share, and for detection PSI on boxes per image and KS on the normalised geometry. The p-values
of the tests are multiplied by `candidateCount` (section 6.9). Both sides' summaries are attached so the direction of
the move is visible.

**Step 5: secondary cuts.** The same search runs once on each side of the primary cut that still holds at least
`2 * min_segment_buckets` buckets. A secondary cut is reported only when its sides are sufficient. This is one level
of binary segmentation: enough to reveal a second event in the range without turning the result into a list of
every wobble.

**Comparison mode.** The search is skipped. The reference records are the `before` side, the current records the
`after` side, and `comparison` carries exactly the fields of a change point with `bucketIndex` null, `date` equal to
the start of the current window and `candidateCount` 1 (no correction needed, the cut was not chosen by the data).
Trend and outlier detection still run over the combined chronological bucket sequence; Kendall tau is rank based, so
the gap between the windows does not invalidate it.

### 5.8 Hard breaks and pairwise maximum (`DriftAnalyzer`)

The ordered records (both windows in comparison mode) are scanned for values that should never change silently
(section 6.7). Additionally, when the median `elapsed_time` after the split is more than `elapsed_ratio_high` (1.5x)
or less than `elapsed_ratio_low` (0.67x) the median before, an `elapsed_time` break is added.

`maxPairwise` holds the largest PSI between any two buckets for the confidence histogram and for the class
distribution, with the pair of bucket start dates. It shows the worst disagreement in the range even when no clean
split exists, for example two short episodes that cancel out in a before/after view.

### 5.9 Rules (`DriftRules`)

Flags and the pre-verdict are pure functions of the computed statistics and the thresholds in `config` (section 9).

---

## 6. Techniques and why they were chosen

The tool never sees images or labels. Everything it can measure is a *distribution of outputs over time*. The
techniques below are the standard toolbox for comparing distributions and detecting change in a series, chosen so
that each one answers one question, they need no fitted model and no numeric library, and they behave well on the
sample sizes a production line produces (hundreds to thousands of predictions per bucket).

### 6.1 Fixed-bin histograms and additive counts

Every confidence is put into one of ten fixed bins `[0,0.1) ... [0.9,1.0]` (`confidence_bin_edges`), every image
into a boxes-per-image bin `0 .. 5+`, every box area into a fixed `log10` bin. **Fixed** edges are essential: two
histograms can only be compared bin by bin when the bins are the same, and a bucket from last week must be comparable
with one from today. Counts, unlike proportions, add up, so any group of buckets has an exact histogram.

### 6.2 Population Stability Index (PSI)

```
PSI(p, q) = sum_i (q_i - p_i) * ln(q_i / p_i)        zeros replaced by eps = 1e-4, then renormalised
```

*What it measures:* how much a distribution over fixed bins moved. It is symmetric, unbounded, and 0 for identical
distributions. Each bin contributes according to both how much its share changed and by what ratio, so a bin that
goes from 1 % to 5 % counts more than one going from 40 % to 44 %. *Why:* it is the industry standard for scoring
stability of model inputs and outputs, has a widely used reading scale (`< 0.10` nothing, `0.10 - 0.25` moderate,
`> 0.25` significant, the `psi_moderate` and `psi_significant` thresholds), and works directly on the additive
counts. *Where:* confidence histogram, class distribution, boxes-per-image histogram, the split search score, the
pairwise maximum. *Limits:* a bin that is empty on one side and populated on the other is punished hard because of
the logarithm, which is why the split search leaves isolated outlier buckets aside and why empty bins are smoothed
with `eps`. PSI has no p-value: it is a size, not a significance.

### 6.3 Jensen-Shannon divergence

```
JS(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m),   m = (p + q) / 2,   base 2, bounded in [0, 1]
```

*What:* a bounded, symmetric relative of the KL divergence. *Why:* PSI is unbounded and jumps on sparse bins; JS is
bounded and smoother, so it is a good **second opinion** on the confidence histogram. *Where:* `jsConfidence` only.
It never drives a flag.

### 6.4 Two-sample Kolmogorov-Smirnov test

```
D = max_x | F_before(x) - F_after(x) |,   p from the asymptotic Kolmogorov distribution
```

*What:* the largest gap between the two empirical cumulative distributions of a **continuous** variable, `D` in
`[0, 1]`, with a p-value. *Why:* it needs no binning, so it complements PSI on the raw confidences, and it is the
natural test for the normalised geometry (area, centre x, centre y), which is continuous and has no meaningful bins.
*Where:* `ksConfidence`, `ksNormalizedArea`, `ksNormalizedCx`, `ksNormalizedCy`; the geometry tests drive
`BOX_GEOMETRY_SHIFT` with `D >= ks_d_geometry` (0.15) **and** `p < ks_p_geometry` (0.01). *Limits:* with thousands of
samples a `D` of 0.03 is "significant" and means nothing, so the reader must judge by `D`; at a searched split the
p-value is corrected (6.9).

### 6.5 Chi-square test of independence

*What:* tests whether the class counts before and after come from the same distribution, on a 2 x classes table,
with the p-value from the regularised incomplete gamma function. *Why:* the class distribution is categorical and
counts are exact, so a contingency test is the textbook tool; it is paired with `maxClassProportionChange` so that a
tiny but "significant" change on large samples does not count. *Where:* `chi2Class`; `CLASS_SHIFT` fires on
`psiClass >= 0.25` **or** (`chi2 p < chi2_p` **and** `maxClassProportionChange >= class_prop_change`). Rows or
columns that are entirely zero are dropped before the test, degenerate tables give `(0, 1, 0)`.

### 6.6 Kendall tau, Theil-Sen slope and practical relevance (trends)

*What:* Kendall tau-b measures whether a series is monotonic (rank correlation with the bucket index, `-1 .. 1`, tie
corrected, p-value from the normal approximation with continuity correction). The Theil-Sen slope is the median of
all pairwise slopes, a robust "how much per bucket". *Why:* both are rank based and robust to single bad buckets and
to the uneven spacing that merged buckets or the gap between windows produce; a linear regression would not be.
*Where:* every series in `trend`. A trend is `meaningful` only when it is significant (`p < trend_p`,
`|tau| >= trend_tau`) **and** the move from the first to the last bucket is practically relevant:

| Series kind | Decided by the name | Relevant when |
|---|---|---|
| value (confidence, geometry) | everything else | absolute change `>= trend_value_change` (0.03) |
| rate (`*_rate`) and class shares | suffix `_rate` or prefix `class_share_` | absolute `>= trend_rate_abs` (0.02) **or** relative `>= trend_rate_rel` (50 %) for trends, **and** for outliers |
| count (`mean_boxes_per_image`) | the name | relative change `>= trend_count_rel` (20 %) |

*Why a relevance test on top of significance:* with many records per bucket, a decline of 0.005 in median confidence
can be statistically certain and operationally irrelevant. The relevance thresholds keep the flags about changes
someone would act on.

Note that a sharp step also produces a high `|tau|`, so a step change usually carries both a shift flag and trend
flags. The pre-verdict counts trends per **family**, not per series (6.10).

### 6.7 Robust z-score (outlier buckets) and hard breaks

*What:* for every bucket, `0.6745 * (x - median(others)) / MAD(others)`, the modified z-score of Iglewicz and Hoaglin,
leave-one-out; falls back to the mean absolute deviation when the MAD is zero and to a capped value when every other
point is identical. *Why:* the median and MAD are not pulled by the outlier itself, which the mean and standard
deviation would be, and with 6 to 60 buckets there is no room for a fitted model. *Where:* `outlierBuckets`, the
`TRANSIENT_OUTLIER` flag, and the exclusion of isolated outliers from the split search.

**Hard breaks** are not a statistic but a scan: values that should never change silently are compared with the last
value seen. `image_spec` (camera or preprocessing changed), `backend` (runtime changed), `classes` (output space
changed), and `threshold` (model configuration changed). A detection threshold belongs to a class, since every class
may carry its own, so it is tracked **per class** and the break names the `className`. A hard break usually explains
everything that changed after it, which is why it alone makes the pre-verdict `drift_likely`.

### 6.8 Change point by exhaustive split search

*What:* one level of binary segmentation. Every position with at least `min_segment_buckets` buckets on each side is
scored, the best is the change point, then the same search runs once on each side. *Why:* with at most 60 buckets
the exhaustive search is cheap thanks to prefix sums of the additive counts, it needs no assumption about the shape of
the change, and the score (a sum of PSI values) is the same quantity the flags read. *Why isolated outliers are left
aside:* a split test only asks whether before and after differ overall; one bad day inside the after side makes them
differ without any persistent change. Removing lone outlier buckets from the search, but not runs of them, lets a
transient stay a transient and a real step stay a step.

### 6.9 Multiple comparisons at the chosen split

The split is the **maximum** over `candidateCount` positions, so the divergence there is biased upwards and the
p-values at it are too small. The p-values of `ksConfidence`, `chi2Class` and the geometry KS tests are therefore
multiplied by `candidateCount` (Bonferroni, capped at 1). PSI values are not adjusted: they are sizes, not tests, and
the reading scale already carries the margin.

### 6.10 Sufficiency and trend families (the rule design)

Two design rules keep the verdict honest:

- **Insufficiency is judged by what the analysis needed.** `INSUFFICIENT_DATA` fires when the current window is below
  the side minimum **or** when the primary split or comparison has `sidesSufficient=false` (records, and boxes for
  detection). `INSUFFICIENT_BUCKETS` fires below `min_buckets_changepoint` or when no split could be searched. The
  verdict is then `undetermined`, never a reassuring `stable` on data that was not tested.
- **Trends are counted per family.** `median_confidence`, `mean_confidence` and `below_threshold_rate` are three
  views of the same decline, so the pre-verdict counts families (confidence, class share, box count, box geometry)
  instead of series. Every individual `TREND_*` flag is still listed.

---

## 7. Configuration

Every value below is a field of `DataDriftConfig` in `common/config.py`, filled from the `data_drift` section of
`config.yaml` (defaults apply for anything not listed there), read as `SETTINGS.data_drift`, and echoed in
`result.config` so that a result is reproducible.

| Field | Default | Used by |
|---|---|---|
| `max_total_days` | 30 | `DriftWindow`: both windows together may not exceed it |
| `max_buckets` | 60 | `BucketBuilder`: coarsen while more buckets would result |
| `min_bucket_records_cls` / `_det` | 200 / 100 | `BucketBuilder`: automatic size choice and small bucket merging |
| `small_bucket_share` | 0.3 | `BucketBuilder`: share of small buckets that triggers coarsening |
| `min_side_records_cls` / `_det` | 500 / 300 | `SplitFinder`: records each side of a split needs |
| `min_side_boxes` | 300 | `SplitFinder`: boxes each side needs for detection |
| `min_buckets_changepoint` | 4 | `DriftAnalyzer`: below it `INSUFFICIENT_BUCKETS`; equals `2 * min_segment_buckets` |
| `min_buckets_trend` | 6 | `SeriesAnalyzer`: series shorter than this are not tested |
| `min_segment_buckets` | 2 | `SplitFinder`: minimum buckets on each side of a split |
| `confidence_bin_edges` | 0.0 ... 1.0 step 0.1 | `Summarizer`: confidence histogram |
| `area_log_bin_edges` | -5 ... 0 | `Summarizer`: log10 normalised area histogram |
| `boxes_per_image_max_bin` | 5 | `Summarizer`: last boxes-per-image bin is "5 or more" |
| `eps` | 1e-4 | PSI and JS smoothing of empty bins |
| `psi_moderate` / `psi_significant` | 0.10 / 0.25 | `DriftRules`: `*_SHIFT_MODERATE` / `*_SHIFT` |
| `ks_d_geometry` / `ks_p_geometry` | 0.15 / 0.01 | `DriftRules`: `BOX_GEOMETRY_SHIFT` |
| `chi2_p` / `class_prop_change` | 0.001 / 0.05 | `DriftRules`: second route to `CLASS_SHIFT` |
| `threshold_pressure_ratio` / `_abs` | 2.0 / 0.02 | `DriftRules`: `THRESHOLD_PRESSURE` |
| `trend_p` / `trend_tau` | 0.05 / 0.5 | `SeriesAnalyzer`: significance of a trend |
| `trend_value_change` | 0.03 | `SeriesAnalyzer`: relevance of a value series move |
| `trend_rate_abs` / `trend_rate_rel` | 0.02 / 0.5 | `SeriesAnalyzer`: relevance of a rate or class share move |
| `trend_count_rel` | 0.2 | `SeriesAnalyzer`: relevance of a count series move |
| `robust_z` | 3.5 | `SeriesAnalyzer`: outlier threshold |
| `near_threshold_margin` | 0.05 | `RecordExtractor`: `near_threshold` of a classification record |
| `elapsed_ratio_high` / `_low` | 1.5 / 0.67 | `DriftAnalyzer`: `elapsed_time` hard break |
| `decimals` | 4 | rounding of every float in the result |
| `max_parse_error_examples` | 10 | `RecordExtractor`: ids kept in `parseErrorExamples` |

---

## 8. The result, field by field

The result is a `DriftAnalysisResult`, serialised with camelCase aliases. Every float is rounded to `decimals`.

### 8.1 Identity and request echo

| Field | Meaning | Use |
|---|---|---|
| `modelName`, `modelVersion`, `task` | the analysed model; `task` is `cls` or `det` | tells the reader which convention applies (records vs boxes) |
| `mode` | `range` or `comparison` | decides whether to read `changePoint` or `comparison` |
| `filters` | the metadata filters that were applied (`gbm, process, location, equipment_id, mode`) | shows what population the numbers describe |
| `range`, `referenceRange` | `startDate`, `endDate`, `days` of each window; `referenceRange` null in range mode | context for every date below |
| `bucket` | the resolved bucket size `1h`, `1d` or `1w` | how far apart consecutive buckets are |
| `detail` | `full` or `compact` | whether the buckets carry histograms |

### 8.2 `status`: what could be computed

| Field | Meaning |
|---|---|
| `analysisPossible` | a change point or a comparison was computed |
| `changePointRan`, `comparisonRan` | which of the two ran |
| `trendRan`, `outlierRan` | the series analysis ran (at least `min_buckets_trend` buckets) |
| `bucketCount` | buckets after merging, reference and current together |

Read this first. When `analysisPossible` is false the rest is descriptive only.

### 8.3 `dataQuality`: how much the numbers rest on

| Field | Meaning |
|---|---|
| `scannedDocumentCount` | documents matching the filters, both windows |
| `matchedDocumentCount`, `matchedEntryCount` | documents and `aiResults` entries that produced records |
| `recordCount` | predictions analysed (images for detection) |
| `boxCount` | boxes analysed, detection only |
| `missingConfidenceCount`, `missingImageSpecCount` | predictions or boxes without a confidence, records without an image size |
| `parseErrorCount`, `parseErrorExamples` | rows or boxes skipped as malformed, with example ids |
| `mergedBuckets` | start dates of the raw buckets that were merged into a neighbour for being too small (section 5.4.3) |

A large `parseErrorCount` or `missingConfidenceCount` relative to `recordCount` means the statistics describe a
subset of the traffic. `classes` (every class the model output, in order of appearance) sits next to it.

### 8.4 `buckets`: the chronological series

One entry per bucket, reference buckets first in comparison mode. With `detail="compact"` only the first block below
is kept; the flags are identical in both forms because the rules only read the scalar series.

Compact block:

| Field | Meaning |
|---|---|
| `startDate`, `endDate`, `window` | UTC boundaries; `current` or `reference`. After a merge the span covers every absorbed raw bucket, gaps included |
| `recordCount`, `mergedFrom` | records in the bucket; raw buckets merged into it (1 when untouched) |
| `classDistribution` | share of every class (every class present, 0.0 included); over boxes for detection |
| `medianConfidence`, `meanConfidence` | over `max_confidence` |
| `belowThresholdRate` | share of predictions (boxes) below their threshold, over those that have one; null if none has |
| `nearThresholdRate` | classification only, share within `near_threshold_margin` of the threshold |
| `meanBoxesPerImage`, `noBoxRate` | detection only, mean box count per image and share of images without a box |

Full detail adds:

| Field | Meaning |
|---|---|
| `confidenceHistogram` | proportions over the fixed bins `[0,0.1) ... [0.9,1.0]`, comparable across buckets |
| `confidenceQuantiles` | `p05 p10 p25 p50 p75 p90 p95` |
| `stdConfidence` | standard deviation of the confidence |
| `thresholdValues` | distinct thresholds seen (should be one per class) |
| `imageSpecs` | distinct `[width, height, channels]` seen (should be one) |
| `medianElapsedTime` | median inference time |
| `boxCount`, `stdBoxesPerImage`, `boxesPerImageHistogram`, `boxesByClassPerImage` | detection only: box count, spread of boxes per image, proportions for 0, 1, 2, 3, 4, 5+ boxes, mean box count per class per image |
| `box` | detection only: `boxCount`, `p10/p50/p90` of `normalizedArea`, `normalizedWidth`, `normalizedHeight`, `aspect`, `normalizedCx`, `normalizedCy`, and `normalizedAreaHistogram` over the fixed log10 bins |

How to use it: scan `medianConfidence` and `belowThresholdRate` down the list to see *when* a move happened, look at
`recordCount` to judge how reliable each bucket is, and look at the histograms of the buckets around a change point
to see the *shape* of the move (a whole distribution sliding down versus a second mode appearing).

### 8.5 `changePoint`, `secondaryChangePoints`, `comparison`

`changePoint` is filled in range mode, `comparison` in comparison mode; both have the same shape. `secondaryChangePoints`
lists at most one further sufficient split on each side of the primary one.

| Field | Meaning | Use |
|---|---|---|
| `bucketIndex`, `date` | index into `buckets` and start of the `after` side; index null in comparison mode | answers **when** |
| `candidateCount` | split positions the search tried (1 for a fixed comparison) | the p-values below are corrected by it |
| `score` | sum of the PSI values the search maximised | ranks splits, not a threshold |
| `beforeCount`, `afterCount`, `sidesSufficient` | records on each side and whether both meet the minimums | when false, no shift flag fires and the values are indicative only |
| `psiConfidence`, `jsConfidence` | confidence histogram divergence | **how much**; PSI on the standard scale, JS as second opinion |
| `ksConfidence` | `d`, `p`, `beforeCount`, `afterCount` of the KS test on the raw confidences | judge by `d` |
| `psiClass`, `chi2Class` (`stat`, `p`, `dof`), `maxClassProportionChange` | class distribution divergence | label shift evidence |
| `psiBoxesPerImage` | detection: boxes-per-image histogram divergence | detector missing or inventing objects |
| `ksNormalizedArea`, `ksNormalizedCx`, `ksNormalizedCy` | detection: geometry KS tests, null when a side has no geometry | camera, zoom or fixture change |
| `before`, `after` | side summaries: `recordCount`, `boxCount`, `classDistribution`, `confidenceHistogram`, `confidenceQuantiles`, `belowThresholdRate`, `boxesPerImageHistogram`, `medianElapsedTime` | answers **in which direction** |

### 8.6 `maxPairwise`

`psi_confidence` and `psi_class`, each with `value` and `pair` (the start dates of the two buckets that disagree
most). Shows the worst disagreement in the range even when no clean split exists, for example two short episodes
that cancel out in a before/after view.

### 8.7 `trend`

One entry per tested series (`median_confidence`, `mean_confidence`, `below_threshold_rate`, `mean_boxes_per_image`,
`no_box_rate`, `median_normalized_area`, `median_normalized_cx`, `median_normalized_cy`, and `class_share_<class>`
per class), each with `tau`, `p`, `slopePerBucket`, `first`, `last`, `pointCount`, `meaningful`. Read `meaningful`
first, then `first` and `last` for the size and direction, then `slopePerBucket` for the pace. Empty when fewer than
`min_buckets_trend` buckets exist.

### 8.8 `outlierBuckets`

Entries with `bucketIndex`, `startDate`, `series`, `z`, `value`. An outlier whose neighbours are normal and that is
not backed by a shift or trend flag is a **transient** (one bad lot, a lamp off for a shift), not drift. On a step
series the leave-one-out score can flag many buckets; only the isolated ones are transients.

### 8.9 `hardBreaks`

Entries with `kind` (`image_spec`, `threshold`, `backend`, `classes`, `elapsed_time`), `className` (the class whose
detection threshold changed, null otherwise), `date`, `inspectionId` (null for `elapsed_time`), `from`, `to`. A hard
break is a configuration event first and drift second: confirm it with the line owner before any statistical
conclusion.

### 8.10 `flags`, `preVerdict`, `config`

`flags` lists every rule that fired (section 9), `preVerdict` is the deterministic conclusion, `config` echoes every
threshold that produced them. An automated consumer can act on `preVerdict` alone; an LLM or a person should use it
as a starting point and confirm or overrule it with the numbers above.

---

## 9. Flags and pre-verdict

| Flag | Condition |
|---|---|
| `INSUFFICIENT_DATA` | fewer than `min_side_records` records in the current window, or the primary split / comparison has `sidesSufficient=false` |
| `INSUFFICIENT_BUCKETS` | fewer than `min_buckets_changepoint` buckets after merging, or no split could be searched (range mode only) |
| `HARD_BREAK` | any hard break other than `elapsed_time` |
| `CONFIDENCE_SHIFT` | `psiConfidence >= psi_significant` with both sides sufficient |
| `CONFIDENCE_SHIFT_MODERATE` | `psi_moderate <= psiConfidence < psi_significant` |
| `CLASS_SHIFT` | `psiClass >= psi_significant`, or `chi2Class.p < chi2_p` with `maxClassProportionChange >= class_prop_change` |
| `CLASS_SHIFT_MODERATE` | `psi_moderate <= psiClass < psi_significant` |
| `BOX_COUNT_SHIFT` | det: `psiBoxesPerImage >= psi_significant` |
| `BOX_GEOMETRY_SHIFT` | det: any of the geometry KS tests with `d >= ks_d_geometry` and `p < ks_p_geometry` |
| `THRESHOLD_PRESSURE` | `belowThresholdRate` after `>= threshold_pressure_ratio` times before and increase `>= threshold_pressure_abs` |
| `TREND_<SERIES>` | one per meaningful trend, e.g. `TREND_MEDIAN_CONFIDENCE`, `TREND_NO_BOX_RATE`; every class share trend collapses into `TREND_CLASS_SHARE` |
| `TRANSIENT_OUTLIER` | outlier bucket(s) present but no shift flag and no trend flag |

Split-based flags only fire when `sidesSufficient` is true.

```
undetermined  if any INSUFFICIENT_* flag
drift_likely  if HARD_BREAK, CONFIDENCE_SHIFT, CLASS_SHIFT, BOX_COUNT_SHIFT, BOX_GEOMETRY_SHIFT,
              or TREND_* flags from two or more families
suspicious    if only *_MODERATE, THRESHOLD_PRESSURE, TRANSIENT_OUTLIER, or TREND_* flags from one family
stable        otherwise
```

Trend families: confidence (`median_confidence`, `mean_confidence`, `below_threshold_rate`), class share, box count
(`mean_boxes_per_image`, `no_box_rate`), box geometry (`median_normalized_area`, `median_normalized_cx`,
`median_normalized_cy`).

---

## 10. Behaviour on edge cases

- Model not found in the windows: `ValueError` "No inspection results found ...", no result is produced.
- Invalid windows, unsupported task, bucket or detail, missing collection: `ValueError` before any query runs.
- Enough records overall but no split with enough records on each side: the best split is reported with
  `sidesSufficient=false`, `INSUFFICIENT_DATA`, verdict `undetermined`.
- Fewer than 4 buckets after merging: descriptive summary returned, `INSUFFICIENT_BUCKETS`, verdict `undetermined`.
  Sparse production (few distinct days with data) is the usual cause, see section 5.4.4.
- Fewer than 6 buckets: no trend and no outlier analysis, `trend` empty, `outlierBuckets` empty.
- One anomalous bucket among normal ones: `TRANSIENT_OUTLIER` only, verdict `suspicious`; the change point is searched
  without that bucket.
- Two or more consecutive anomalous buckets: treated as a level change and reported as a shift.
- Camera resolution change with unchanged scene: `HARD_BREAK` on `image_spec`, no geometry flag, because geometry is
  normalised by the image size.
- Per-class detection thresholds: no threshold break, because thresholds are tracked per class.
- Same model string used for both tasks without `task`: the first row's task is used and rows of the other task are
  ignored. Pass `task` to make the choice explicit.
- Malformed rows: skipped and counted, never fatal. A prediction without a usable confidence is not malformed: it is
  kept with a null confidence and counted in `missingConfidenceCount`.

Known limitations, deliberately left as they are:

- Hard breaks compare consecutive records regardless of equipment. A model that runs on several cameras must be
  analysed with `equipment_id` set, otherwise the image size flips between records and `HARD_BREAK` is meaningless.
- The relevance thresholds of trends and outliers are fixed numbers, not scaled with the sample size of a bucket, so
  on low base rates (a below-threshold rate around 2 %) ordinary sampling noise can produce `TRANSIENT_OUTLIER` and a
  `suspicious` verdict. It never hides real drift.
- The automatic bucket choice counts small buckets, not the records they hold, so a few nearly empty periods can push
  the grid one step coarser than the bulk of the data would need.
- A merged bucket's date span hides gaps between the absorbed periods; `mergedFrom` is the only indication.

---

## 11. Using the result

### 11.1 With an LLM

The tool is an ordinary MCP tool. The LLM client lists the server's tools, sees `mcp_mongodb_analyze_data_drift`
with the docstring as its description and the JSON schema of its arguments, fills the arguments from the user's
question, receives the `DriftAnalysisResult` as the tool result, and interprets it. It computes nothing.

The result is 10 to 75 KB depending on range, task and `detail`. Ask for `detail="compact"` when the consumer only
needs the series and the change point; ask for `full` when it should reason about histogram shapes per bucket.

The instruction below belongs in the **system prompt** of the consuming agent, or in the prompt of a dedicated
drift analyst sub-agent that receives only the tool result. Its role is to confirm or overrule `preVerdict` with an
explanation and to return a fixed JSON structure the calling service can store next to the tool result.

````text
You are an AI inspection model monitoring analyst. You receive the JSON result of the `mcp_mongodb_analyze_data_drift`
tool for one inspection model over one time range (mode "range") or over a reference window and a current window
(mode "comparison"). Your job is to decide whether the model's input data or behaviour drifted, to say when and
how, to name the most likely cause, and to recommend an action. You do not compute statistics; every number you
need is already in the JSON. You confirm or overrule the `preVerdict` field and explain why.

## How to read the result

- `status` says what could be computed. If `analysisPossible` is false, or `flags` contain INSUFFICIENT_DATA or
  INSUFFICIENT_BUCKETS, answer "undetermined" and explain what is missing (too few records, too few periods with
  data, too short a range).
- `dataQuality` tells you how much data the numbers rest on. `recordCount` is the number of predictions (images for
  detection), `boxCount` the number of boxes. Mention any `parseErrorCount`, `missingConfidenceCount` or
  `missingImageSpecCount` that is large relative to `recordCount`. `mergedBuckets` lists periods that were too thin
  to stand on their own and were folded into a neighbour.
- `buckets` is the chronological series. Each bucket has `recordCount`; treat buckets with a small `recordCount` as
  unreliable. `mergedFrom` above 1 means the bucket spans several raw periods, possibly with gaps between them. In
  comparison mode `window` tells you whether a bucket belongs to the reference or the current period.
- `changePoint` (range mode) or `comparison` (comparison mode) is the main before/after evidence. `date` is where
  the "after" side starts. `sidesSufficient` must be true for the divergence values to be trusted. `before` and
  `after` contain the histograms and quantiles of each side so you can see the direction of a move.
- `trend` gives, per series, Kendall tau (-1..1), its p-value, the slope per bucket, and the first and last value.
  `meaningful` is true only when the trend is significant and the move is large enough to matter.
- `outlierBuckets` lists buckets that stand far from the others. An outlier with normal neighbours and no change
  point or trend is a transient event, not drift.
- `hardBreaks` lists configuration changes found in the data: image_spec (camera or preprocessing changed),
  threshold (model configuration changed, `className` says which class for detection), backend (runtime changed),
  classes (output space changed), elapsed_time (infrastructure changed). A hard break must be confirmed with the
  line owner before any statistical conclusion; it usually explains everything that changed after it.
- `flags` and `preVerdict` are deterministic rule outputs computed from the thresholds listed in `config`.

For detection models the confidence, class and threshold statistics are computed over bounding boxes, and the
`box` block of each bucket carries normalised geometry (size and position relative to the image).

## Reading the divergence numbers

- PSI (psiConfidence, psiClass, psiBoxesPerImage): below 0.10 no meaningful change; 0.10 to 0.25 moderate, worth a
  look; above 0.25 significant.
- KS statistic d (ksConfidence, ksNormalizedArea, ksNormalizedCx, ksNormalizedCy): 0 identical, 1 completely
  separate. Judge by d, not by the p-value: with thousands of samples a d of 0.03 has a tiny p-value and means
  nothing. Treat d below 0.05 as no change, 0.05 to 0.15 as small, above 0.15 as a real move. The p-values at a
  change point are already corrected for the number of split positions tried (candidateCount).
- chi2Class: use together with maxClassProportionChange; a significant p with a proportion change under 0.05 is
  not a real class shift.
- jsConfidence is a second opinion on psiConfidence in the range 0..1; use it to confirm, not to decide.

## Signatures and their most likely explanation

| Observation | Most likely explanation |
|---|---|
| confidenceHistogram moves to lower bins, belowThresholdRate and nearThresholdRate rise, classDistribution barely moves | Covariate shift: the images changed and the model is less sure. Earliest and most reliable drift signature. |
| classDistribution changes strongly while confidenceHistogram stays the same | Prior/label shift: the product mix or the defect rate really changed. The model may be fine, production may not be. Report as "production change, model behaviour stable". |
| Both confidence and class distribution move together | Genuine drift or a new defect type. |
| ksNormalizedArea, ksNormalizedCx or ksNormalizedCy large | Camera moved, zoom changed, fixture or part variant changed. Almost always a physical cause. |
| boxesPerImageHistogram or noBoxRate moves | Detector missing objects (lighting, contamination) or seeing extra objects (debris, new part). |
| hardBreaks non-empty | Configuration change. Confirm with the line owner first. |
| One entry in outlierBuckets, no shift flag, no trend flag | Transient event (bad lot, a shift with a lamp off). Not drift, but worth logging. |
| Meaningful trend, no shift flag | Gradual degradation (lens contamination, wear, slow process change). |
| elapsed_time hard break together with an image_spec break | Resolution or preprocessing change. Infrastructure, not data. |
| INSUFFICIENT_BUCKETS with a long range | Production ran on few distinct days. Suggest comparison mode against an earlier reference window. |

A change point answers WHEN, PSI and KS answer HOW MUCH, the before and after histograms answer IN WHICH DIRECTION,
the flags give a consistent first decision. A step change usually also produces TREND flags because a step is
monotonic; do not count that as two independent signals.

## Caution rules

1. Distrust any comparison where `sidesSufficient` is false.
2. Ignore statistically significant but tiny effects: KS d below 0.05, PSI below 0.10.
3. Treat isolated outlier buckets as transients unless a change point or trend supports them.
4. Treat hard breaks as a configuration event first and a drift second.
5. Do not invent numbers. Quote the values from the JSON, rounded sensibly, and name the bucket dates you rely on.
6. When you disagree with `preVerdict`, say so explicitly and give the numbers that made you overrule it.

## Required answer

Return first a JSON object with exactly these fields, then a short plain-language explanation of at most ten
sentences for the line owner.

{
  "drift_detected": true | false | null,            // null when undetermined
  "confidence": "high" | "medium" | "low",
  "drift_type": "covariate_shift" | "label_shift" | "configuration_change" | "transient" | "none" | "undetermined",
  "start_date": "YYYY-MM-DD" | null,                // bucket date the change starts, or the change point date
  "signals": ["...", "..."],                        // 2 to 6 short statements with numbers
  "likely_cause": "...",
  "recommended_action": "...",
  "agrees_with_pre_verdict": true | false,
  "pre_verdict": "<copy of preVerdict>"
}

Mapping guidance: drift_likely with covariate signatures -> drift_detected true, high; suspicious -> usually
drift_detected false with medium or low confidence and a "monitor" action, unless the numbers convince you
otherwise; stable -> false, high; undetermined -> null, low.
````

### 11.2 Example answer

For a classification model whose confidence dropped on 2026-09-08 the consuming LLM is expected to answer roughly:

```json
{
  "drift_detected": true,
  "confidence": "high",
  "drift_type": "covariate_shift",
  "start_date": "2026-09-08",
  "signals": [
    "confidence PSI between the sides of the change point is 0.48, far above 0.25",
    "median confidence fell from 0.94 before 2026-09-08 to 0.89 after",
    "below-threshold rate rose from 0.03 to 0.07",
    "class distribution unchanged (PSI 0.01, NG share moved by 0.006)"
  ],
  "likely_cause": "Images changed while the product mix did not: most likely lighting or camera contamination on the line rather than a production change.",
  "recommended_action": "Inspect camera and lighting on Line_01, collect and label samples from 2026-09-08 onward, and compare them with the training set before retraining.",
  "agrees_with_pre_verdict": true,
  "pre_verdict": "drift_likely"
}
```

### 11.3 Without an LLM

`preVerdict`, `flags` and `changePoint.date` (or `comparison`) are already usable as an automated alert. Suggested
policy: alert on `drift_likely`, log on `suspicious`, ignore `stable`, and treat `undetermined` as a data
availability problem rather than a model problem. `hardBreaks` deserve their own notification to the line owner
regardless of the verdict.

---

## 12. Operations

- The index from section 5.2 must exist on the inspections collection; the tool does not create it.
- The analysis runs in a thread via `run_in_executor`, so other tool calls are not blocked.
- The connector logs the record count per window and the final record count per call. Malformed rows are logged at
  warning level with their id.
- Every threshold is tunable in `config.yaml` without a code change, and every result carries the values it was
  produced with.
