# Data Drift Analysis

This document describes the `mongodb_analyze_data_drift` MCP tool as implemented in this repository: why it exists,
where the code lives, how a call flows through it, what it reads, what it computes, every configuration value, and
every field of the result with the way it is meant to be read.

The tool reads the pre-aggregated `inspectionStatistics` collection described in
[docs/inspection_statistics.md](docs/inspection_statistics.md). It never reads raw inspection documents. The design
discussion that led here is in [docs/data_drift_v2.md](docs/data_drift_v2.md).

---

## 1. Purpose

An inspection model is trained on images collected at one point in time. Once deployed, the images it receives can
change: a camera moves, lighting changes, a new supplier's parts arrive, a lens gets dirty. The model keeps answering,
but its answers become less reliable, and nobody notices because there is no ground truth at inference time. This is
**data drift**.

The database holds no pixels and no true labels. It does hold, per period, everything the model *output*: how many
predictions of each class, the distribution of its confidence, how many fell below the threshold, how many boxes it
drew per image, and the runtime configuration it ran with. The tool reads those per-period statistics and reports
every number needed to decide whether the inputs or the behaviour of the model changed. It does **not** decide on
its own. It produces deterministic flags and a pre-verdict, and an LLM (or a human) confirms or overrules them by
reading the numbers.

Three kinds of drift are distinguished and each leaves a different signature in the output:

| Kind | Meaning | Signature in the output |
|---|---|---|
| Covariate shift | The images changed, the meaning of the labels did not | Confidence histogram moves to lower bins, below-threshold rate rises, class distribution barely moves |
| Prior / label shift | The proportion of classes changed | Class distribution moves, confidence histogram stays |
| Configuration change | Threshold, runtime or class list changed | A hard break, usually followed by one of the two above |

Only classification (`cls`) and object detection (`det`) models are supported. Segmentation produces no statistics
document.

---

## 2. Files and folders

| Path | Role |
|---|---|
| `mongodb_mcp/tools/tools.py` | `mongodb_analyze_data_drift`: the MCP tool, argument documentation, error wrapping |
| `mongodb_mcp/connector/connector.py` | `analyze_data_drift`: validation, one pipeline per window, task resolution, runs the analyzer in a thread |
| `mongodb_mcp/drift/window.py` | `DriftWindow`: UTC windows, validation, automatic granularity |
| `mongodb_mcp/drift/query.py` | `StatisticsQuery`: the aggregation pipeline over `inspectionStatistics` |
| `mongodb_mcp/drift/period.py` | `build_periods` (pipeline rows to `PeriodCounts`), `sum_periods`, `PeriodSummarizer` |
| `mongodb_mcp/drift/splits.py` | `SplitFinder`: divergence between two groups of periods, best split search |
| `mongodb_mcp/drift/rules.py` | `DriftRules`: flags and pre-verdict |
| `mongodb_mcp/drift/drift.py` | `DriftAnalyzer`: orchestration, consecutive divergence, hard breaks, result assembly |
| `mongodb_mcp/schemas/drift.py` | `PeriodCounts` / `ConfidenceCounts` (internal, additive) and the response models |
| `mongodb_mcp/utils/stats.py` | Pure helpers: proportions, PSI, JS, chi-square, histogram quantile |
| `common/config.py`, `config.yaml` | `DataDriftConfig`, the `data_drift` section |
| `common/constants.py` | `Granularity`, `DriftFlag`, `PreVerdict`, `HardBreakKind`, `MedianSource`, ... |
| `tests/drift/synthetic.py` | Synthetic statistics documents and a Python emulation of the pipeline |
| `tests/drift/`, `tests/connector/test_drift_connector.py`, `tests/tools/test_drift_tool.py` | Tests |

---

## 3. Tool interface

Registered name on the server: `mcp_mongodb_analyze_data_drift` (the tools server is mounted with the `mcp` prefix).

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model_name`, `model_version` | str | required | Exact match on the statistics documents |
| `start_date`, `end_date` | datetime | required | Current window, UTC. A period belongs to the window when its `startDate` lies in `[start_date, end_date)` |
| `task` | `cls` / `det` / null | null | Only needed when the model has statistics for both tasks; otherwise resolved from the documents |
| `gbm`, `process` | str / null | null | Exact match; null sums every site or process the model ran at |
| `equipment_id` | str / null | null | Null analyses the `total` block; a value analyses that entry of `equipments[]` |
| `product_id` | str / null | null | Null sums every product of a period |
| `mode` | str / null | `production` | Operating mode. Pass `null` explicitly to sum every mode |
| `granularity` | `auto`, `hourly`, `shift`, `daily`, `weekly` | `auto` | Period type of the documents to read |
| `detail` | `full` / `compact` | `full` | Whether periods carry histograms, quantiles and per-class blocks or only scalars |
| `reference_start_date`, `reference_end_date` | datetime / null | null | Reference window; giving both switches to comparison mode |

Two modes exist:

- **Range mode** (default): the current window is a series of periods. The tool reports every period, the
  divergence of every period against the previous one, and searches the split where the output distribution
  changed the most (change point).
- **Comparison mode**: when the reference window is also given, the reference periods are compared against the
  current periods at the fixed boundary between them instead of searching for a split.

The windows together may cover at most `max_total_days` (400). The reference window must end before the current one
starts. All dates are UTC, naive datetimes are treated as UTC. `auto` granularity picks the finest period type for
which both windows together hold at most `max_periods` (60) periods: hourly up to 2.5 days, shift up to 30 days,
daily up to 60 days, weekly beyond.

---

## 4. Flow of one call

```
tool argument validation (fastmcp)
  -> MongoDBConnector.analyze_data_drift
       argument checks         task and detail must be supported values, the collection must exist
       DriftWindow             UTC-normalises the dates, validate() checks order and total length,
                               resolve_granularity() turns auto into a period type
       StatisticsQuery         build_pipeline(), once per window, reference first
       build_periods           pipeline rows -> one PeriodCounts per period, tagged with its window
       no data at all          ValueError "No inspection statistics found ..."
       task resolution         the task of the documents; an error when the model has both and none was requested
       DriftAnalyzer.run()     in a thread so the server stays responsive
            PeriodSummarizer   one summary per period
            consecutive PSI    period i against period i-1
            SplitFinder        best split (range mode) or fixed comparison (comparison mode)
            hard breaks        backend, threshold and classes between consecutive periods
            DriftRules         flags and pre-verdict
            DriftAnalysisResult (rounded to `decimals`)
```

---

## 5. Step by step

### 5.1 The query (`StatisticsQuery`)

One aggregation pipeline per window, run with `allowDiskUse`:

1. `$match` on the unique index prefix of the collection: `modelName, modelVersion, task, mode, gbm, process,
   granularity, startDate range`, plus `productId` when given. `task` is `{$in: [cls, det]}` when not given.
2. `$project` the identity fields, `bins`, `classes`, the equipment ids, backends and thresholds of `equipments[]`,
   and `block`: `$total`, or the `equipments[]` entry whose `equipmentId` matches (missing when absent).
3. `$addFields` an `entries` array: one `total` entry with `block.confidence`, one `class` entry per key of
   `block.perClass`, one `count` entry per key of `block.classCounts`; plus the backend and threshold values the
   analysis should see (every equipment for the total, the requested one otherwise).
4. `$unwind` the entries and `$group` by `(startDate, kind, class)`. Every additive number is `$sum`med, sets are
   `$addToSet`ed, histograms are `$push`ed, and a `present` flag records whether an optional field (box count,
   images block, threshold counts) ever existed in the group.
5. `$addFields` the element-wise sum of the pushed histograms with `$reduce` / `$zip`, drop the pushed arrays,
   `$sort` by date.

The result is one small row per period and entry, whatever the number of documents (products) behind a period. A
document whose `equipments[]` has no entry for the requested equipment still produces a row with
`documentCount 0` and `missingBlockCount 1`, so that the analysis can list the periods it had to leave out.

### 5.2 Periods (`build_periods`, `PeriodCounts`)

The rows of one window are grouped by period start into `PeriodCounts`: document, product, inspection, prediction
and box counts, the sets seen (sites, modes, equipments, classes, bins, backends, thresholds), elapsed time sums,
the images block for detection, the class counts, and a `ConfidenceCounts` for the total and for every predicted
class (`count, sum, sumSq, min, max, histogram, belowThresholdCount, nearThresholdCount, thresholds, quantiles`).
`quantiles` is kept only when the period is a single document, because quantiles are not additive.

`PeriodCounts` and `ConfidenceCounts` define `+`, so the counts of any group of periods (a side of a split, a whole
window) are an exact sum. Optional counts stay `None` when no document carried them.

### 5.3 Summaries (`PeriodSummarizer`)

Every period gets the same summary (section 8.5). Shares are counts divided by their total, means are `sum / count`,
standard deviations come from `sumSq`, rates are `belowThresholdCount / count`. The median is the stored `p50` when
the period is one document (`medianSource: exact`) and otherwise interpolated inside the histogram
(`medianSource: histogram`). For detection the confidence and class blocks are over boxes, boxes per image and the
no-box rate over images. The summarizer reads the bin edges once, from the first document, so that every histogram
in the result is proportions over the same bins.

### 5.4 Consecutive divergence

For every period after the first: PSI of its confidence histogram, class counts and boxes-per-image histogram
against the previous period, and whether both periods meet the sufficiency minimums. PSI on confidence is `null`
when the two periods carry different bin edges. This series answers **when** with no search at all.

### 5.5 Split and comparison (`SplitFinder`)

**Range mode.** Every split position with at least `min_segment_periods` (2) periods on each side is tried. The
counts of the sides are prefix sums, so each candidate costs a few hundred additions. The score is
`psiConfidence + psiClass (+ psiBoxesPerImage for detection)`. A split whose sides are both sufficient is preferred
over a higher-scoring split whose sides are not. The search needs at least `min_periods_changepoint` (4) periods.

**Comparison mode.** The reference periods are one side and the current periods the other, no search.

The chosen split reports PSI and JS on the confidence histogram, PSI per predicted class, PSI and chi-square on the
class counts with the largest proportion change, PSI on boxes per image, the deltas of mean confidence and
below-threshold rate, the elapsed-time ratio, and a summary of each side. The chi-square p-value is multiplied by
`candidateCount` (Bonferroni) because the split was chosen to maximise divergence.

**Sufficiency.** A side is sufficient with at least `min_side_records_cls` (500) predictions for classification, or
`min_side_records_det` (300) images and `min_side_boxes` (300) boxes for detection. Shift flags fire only when both
sides are sufficient.

### 5.6 Hard breaks

`backend`, `threshold` (per predicted class for detection) and `classes` are compared between consecutive periods.
A change is a break dated at the later period. A period that itself carries several values (two products in the
same hour with different backends) is a break to the list of values, dated at that period. The `total` block has
no backend or threshold, so those come from the `equipments[]` entries of the documents.

### 5.7 Bins

Every document carries the bin edges its histograms were built with. When the periods read carry more than one set
of edges, the histograms cannot be compared: no split runs, the crossing consecutive PSI is `null`,
`dataQuality.incompatibleBins` is true and `INCOMPATIBLE_BINS` fires. The series itself is still returned.

### 5.8 Rules (`DriftRules`)

See section 9.

---

## 6. Techniques

### 6.1 Additive counts

Everything the analysis needs is a count, a sum or a fixed-bin histogram, all stored per period by the Data
Service. Sums of periods are exact, so the histogram of a side is never an average of proportions. Quantiles,
minimum and maximum are the only non-additive numbers: min and max survive a sum, quantiles are reported only for
single-document periods and otherwise replaced by a histogram-interpolated median.

### 6.2 Population Stability Index (PSI)

`PSI = sum_i (q_i - p_i) * ln(q_i / p_i)` over the bins, after smoothing every proportion to at least `eps`.
Conventional reading: below 0.10 no meaningful change, 0.10 to 0.25 moderate, above 0.25 significant. It is used on
the confidence histogram (total and per class), on the class counts, and on the boxes-per-image histogram.

### 6.3 Jensen-Shannon divergence

Base-2 JS divergence between the same two smoothed histograms, bounded in `[0, 1]`. A second opinion on the
confidence PSI, symmetric and bounded, useful when PSI is inflated by a nearly empty bin.

### 6.4 Chi-square test of independence

On the `2 x classes` table of class counts before and after. Reported with `maxClassProportionChange` because with
tens of thousands of predictions a tiny move is significant; the class shift rule needs both the test and a
proportion change of at least `class_prop_change` (0.05).

### 6.5 Histogram median

The quantile of a fixed-bin histogram by linear interpolation inside the bin that crosses it. Exact at the bin
edges, approximate within a bin; the result says `medianSource: histogram` whenever it was used.

### 6.6 What was dropped from the previous implementation

The previous tool scanned raw inspection documents and computed everything itself. Moving to the statistics
collection removed: KS tests (need raw values), box geometry (not stored), image spec breaks (not stored), trend
tests and outlier scores (the period series and the consecutive PSI make them redundant for a reader), secondary
change points and the pairwise maximum, the record budget, automatic bucket size by volume and small-bucket merging
(the caller picks the granularity and a period is small whatever the volume).

---

## 7. Configuration

Section `data_drift` of `config.yaml`, loaded into `DataDriftConfig`; every value is echoed in `result.config`.

| Key | Default | Used by |
|---|---|---|
| `max_total_days` | 400 | `DriftWindow.validate` |
| `max_periods` | 60 | automatic granularity |
| `min_side_records_cls`, `min_side_records_det`, `min_side_boxes` | 500, 300, 300 | sufficiency of sides and of consecutive pairs |
| `min_periods_changepoint` | 4 | range mode needs this many current periods |
| `min_segment_periods` | 2 | periods on each side of a candidate split |
| `eps` | 1e-4 | histogram smoothing for PSI and JS |
| `psi_moderate`, `psi_significant` | 0.10, 0.25 | shift flags |
| `chi2_p`, `class_prop_change` | 0.001, 0.05 | class shift by test |
| `threshold_pressure_ratio`, `threshold_pressure_abs` | 2.0, 0.02 | threshold pressure flag |
| `decimals` | 4 | rounding of every float in the result |

The bin edges are **not** configuration: they are read from the documents.

---

## 8. The result, field by field

### 8.1 Identity and request echo

`modelName`, `modelVersion`, `task` (resolved from the documents when not requested), `mode` (the operating mode
filter, `null` when every mode was summed), `analysisMode` (`range` or `comparison`), `filters` (`gbm`, `process`,
`equipmentId`, `productId` as applied), `sitesSeen` (distinct `gbms`, `processes`, `modes`, `equipmentIds` that
went into the numbers), `granularity` (resolved), `detail`, `range` and `referenceRange` (`startDate`, `endDate`,
`days`), `classes` (union over the documents), `bins` (`confidenceEdges`, `nearThresholdMargin`,
`boxesPerImageMax` as read).

### 8.2 `status`

`analysisPossible` (a split or comparison with sufficient sides was computed), `splitRan`, `comparisonRan`,
`periodCount`, `referencePeriodCount`.

### 8.3 `dataQuality`

`documentCount` (statistics documents read over both windows), `productCount` (distinct products),
`inspectionCount`, `predictionCount` (images), `boxCount` (detection), `missingConfidenceCount`, `parseErrorCount`
(as recorded by the Data Service), `partialPeriods` (start dates of periods whose end is in the future),
`periodsMissingEquipment` (periods left out because the requested equipment had no entry), `incompatibleBins`.

### 8.4 `periods`

One entry per period, reference periods first in comparison mode.

| Field | Meaning |
|---|---|
| `startDate`, `endDate`, `window`, `partial` | UTC boundaries; `current` or `reference`; whether the period is still running |
| `documentCount`, `productCount`, `inspectionCount`, `predictionCount`, `boxCount` | volumes behind the period |
| `classDistribution` | share of every class (every class present, 0.0 included); over boxes for detection |
| `meanConfidence`, `stdConfidence`, `medianConfidence`, `medianSource` | over the confidence of the predicted class; `medianSource` is `exact` or `histogram` |
| `belowThresholdRate`, `nearThresholdRate` | share of predictions below / within the margin of the threshold; null when the model has no threshold |
| `meanBoxesPerImage`, `noBoxRate` | detection only |
| `meanElapsedTime` | mean inference time |
| `backends`, `thresholds`, `thresholdsByClass` | runtime configuration seen; more than one value inside a period is a configuration change inside it |

Full detail adds `confidenceHistogram` (proportions over `bins.confidenceEdges`), `confidenceQuantiles` (only when
`medianSource` is `exact`), `boxesPerImageHistogram` (detection, bins `0, 1, ..., max+`), and `perClass`: per
predicted class `count`, `share`, `meanConfidence`, `stdConfidence`, `belowThresholdRate`, `nearThresholdRate`,
`threshold` (detection) and `confidenceHistogram`.

### 8.5 `consecutive`

One entry per period after the first: `periodIndex`, `date`, `psiConfidence`, `psiClass`, `psiBoxesPerImage`,
`sufficient`. A spike is the moment of a step.

### 8.6 `changePoint`, `comparison`

`changePoint` is filled in range mode, `comparison` in comparison mode; both have the same shape.

| Field | Meaning |
|---|---|
| `periodIndex`, `date` | index into `periods` and start of the `after` side; index null in comparison mode |
| `candidateCount`, `score` | split positions the search tried (1 for a comparison); the PSI sum that was maximised |
| `beforeCount`, `afterCount`, `sidesSufficient` | predictions on each side and whether both meet the minimums |
| `psiConfidence`, `jsConfidence`, `psiConfidenceByClass` | confidence histogram divergence, total and per predicted class |
| `psiClass`, `chi2Class` (`stat`, `p`, `dof`), `maxClassProportionChange` | class distribution divergence |
| `psiBoxesPerImage` | detection |
| `meanConfidenceDelta`, `belowThresholdRateDelta`, `elapsedTimeRatio` | after minus before, after over before |
| `before`, `after` | side summaries: dates, `periodCount`, volumes, `classDistribution`, `meanConfidence`, `medianConfidence` (histogram), rates, `confidenceHistogram`, `boxesPerImageHistogram`, `perClass` |

### 8.7 `hardBreaks`

Entries with `kind` (`backend`, `threshold`, `classes`), `className` (detection thresholds), `date`, `from`, `to`.

### 8.8 `flags`, `preVerdict`, `config`

`flags` lists every rule that fired (section 9), `preVerdict` is the deterministic conclusion, `config` echoes every
threshold that produced them.

---

## 9. Flags and pre-verdict

| Flag | Rule | Verdict weight |
|---|---|---|
| `INSUFFICIENT_DATA` | fewer than `min_side_records` predictions in the current window, or the chosen split has an insufficient side, or a comparison side is missing | undetermined |
| `INSUFFICIENT_PERIODS` | fewer than `min_periods_changepoint` current periods, or no valid split position | undetermined |
| `INCOMPATIBLE_BINS` | bin edges differ across the documents read | undetermined |
| `HARD_BREAK` | any hard break | suspicious |
| `CONFIDENCE_SHIFT` / `CONFIDENCE_SHIFT_MODERATE` | split `psiConfidence >= psi_significant` / `>= psi_moderate` | drift likely / suspicious |
| `CLASS_SHIFT` / `CLASS_SHIFT_MODERATE` | `psiClass >= psi_significant`, or corrected chi-square `p < chi2_p` with `maxClassProportionChange >= class_prop_change` / `psiClass >= psi_moderate` | drift likely / suspicious |
| `BOX_COUNT_SHIFT` | detection, `psiBoxesPerImage >= psi_significant` | drift likely |
| `THRESHOLD_PRESSURE` | after rate at least `threshold_pressure_ratio` times the before rate and at least `threshold_pressure_abs` higher | suspicious |

Shift flags need `sidesSufficient`. Pre-verdict: any insufficiency flag gives `undetermined`; otherwise
`drift_likely` when any drift-likely flag fired, `suspicious` when any suspicious flag fired, else `stable`.

---

## 10. Behaviour on edge cases

| Situation | Behaviour |
|---|---|
| No document in either window | `ValueError` "No inspection statistics found ..." |
| The model has statistics for both tasks and none was requested | `ValueError` asking for `task` |
| Requested equipment absent in some periods | those periods are listed in `periodsMissingEquipment` and left out |
| Several products (documents) in a period | summed exactly in the pipeline; `medianSource` becomes `histogram` |
| A period still running | `partial` true, listed in `partialPeriods`; its numbers are used as they are |
| Different bin edges across periods | series kept, no split, crossing PSI null, `INCOMPATIBLE_BINS` |
| Model without threshold | threshold rates null, no threshold pressure, no threshold hard breaks |
| Fewer than 4 current periods in range mode | no split, `INSUFFICIENT_PERIODS` |
| Reference window without documents | no comparison, `INSUFFICIENT_DATA` |

---

## 11. Using the result

### 11.1 With an LLM

The tool is an ordinary MCP tool. The LLM client sees `mcp_mongodb_analyze_data_drift` with the docstring as its
description, fills the arguments from the user's question, receives the `DriftAnalysisResult` and interprets it. It
computes nothing. Ask for `detail="compact"` when only the series and the split matter; ask for `full` when the
consumer should reason about histogram shapes and per-class behaviour.

The instruction below belongs in the **system prompt** of the consuming agent, or of a dedicated drift analyst
sub-agent that receives only the tool result.

````text
You are an AI inspection model monitoring analyst. You receive the JSON result of the `mcp_mongodb_analyze_data_drift`
tool for one inspection model over one time range (analysisMode "range") or over a reference window and a current
window (analysisMode "comparison"). Your job is to decide whether the model's input data or behaviour drifted, to say
when and how, to name the most likely cause, and to recommend an action. You do not compute statistics; every number
you need is already in the JSON. You confirm or overrule the `preVerdict` field and explain why.

## How to read the result

- `status` says what could be computed. If `analysisPossible` is false, or `flags` contain INSUFFICIENT_DATA,
  INSUFFICIENT_PERIODS or INCOMPATIBLE_BINS, answer "undetermined" and explain what is missing.
- `dataQuality` tells you how much data the numbers rest on: `predictionCount` is the number of predictions (images
  for detection), `boxCount` the number of boxes, `documentCount` and `productCount` how many statistics documents
  and products were summed. Mention `partialPeriods` (still running) and `periodsMissingEquipment` when present.
- `periods` is the chronological series. Treat periods with a small `predictionCount` as unreliable. In comparison
  mode `window` says whether a period is reference or current. `medianSource` says whether the median is exact or
  interpolated from the histogram.
- `consecutive` is the PSI of every period against the previous one. A single spike is a step at that date; a run
  of moderate values is a gradual move; `sufficient` false means the pair is too small to trust.
- `changePoint` (range) or `comparison` (comparison) is the main before/after evidence. `date` is where the after
  side starts. `sidesSufficient` must be true for the divergence values to be trusted. `psiConfidenceByClass` says
  which class moved. `before` and `after` hold the histograms and class shares of each side.
- `hardBreaks` lists configuration changes found in the data: threshold (model configuration, `className` for
  detection), backend (runtime), classes (output space). A hard break must be confirmed with the line owner before
  any statistical conclusion; it usually explains everything that changed after it.
- `flags` and `preVerdict` are deterministic rule outputs computed from the thresholds listed in `config`.

For detection models the confidence, class and threshold statistics are over bounding boxes; boxes per image and
the no-box rate are over images.

## Reading the divergence numbers

- PSI (psiConfidence, psiClass, psiBoxesPerImage, psiConfidenceByClass): below 0.10 no meaningful change; 0.10 to
  0.25 moderate; above 0.25 significant.
- chi2Class: use together with maxClassProportionChange; a significant p with a proportion change under 0.05 is
  not a real class shift. The p-value at a change point is already corrected for candidateCount.
- jsConfidence is a second opinion on psiConfidence in the range 0..1; use it to confirm, not to decide.

## Signatures and their most likely explanation

| Observation | Most likely explanation |
|---|---|
| confidenceHistogram moves to lower bins, belowThresholdRate and nearThresholdRate rise, classDistribution barely moves | Covariate shift: the images changed and the model is less sure. Earliest and most reliable drift signature. |
| classDistribution changes strongly while confidenceHistogram stays the same | Prior/label shift: the product mix or the defect rate really changed. Report as "production change, model behaviour stable". |
| Both confidence and class distribution move together | Genuine drift or a new defect type. |
| psiConfidenceByClass large for one class only | That class's inputs changed; look at its perClass histograms on both sides. |
| boxesPerImageHistogram or noBoxRate moves | Detector missing objects (lighting, contamination) or seeing extra objects (debris, new part). |
| hardBreaks non-empty | Configuration change. Confirm with the line owner first. |
| One spike in consecutive, flat before and after, no shift flag | Transient event (bad lot, a shift with a lamp off). Not drift, but worth logging. |
| Consecutive PSI moderate for many periods in a row, mean confidence sliding | Gradual degradation (lens contamination, wear, slow process change). |
| INSUFFICIENT_PERIODS with a long range | Production ran on few distinct periods. Suggest comparison mode against an earlier reference window. |

The consecutive series and the change point answer WHEN, PSI answers HOW MUCH, the before and after histograms
answer IN WHICH DIRECTION, the flags give a consistent first decision.

## Caution rules

1. Distrust any comparison where `sidesSufficient` is false.
2. Ignore statistically significant but tiny effects: PSI below 0.10, proportion changes below 0.05.
3. Treat a single consecutive spike with flat neighbours as a transient unless the change point supports it.
4. Treat hard breaks as a configuration event first and a drift second.
5. Do not invent numbers. Quote the values from the JSON, rounded sensibly, and name the period dates you rely on.
6. When you disagree with `preVerdict`, say so explicitly and give the numbers that made you overrule it.

## Required answer

Return first a JSON object with exactly these fields, then a short plain-language explanation of at most ten
sentences for the line owner.

{
  "drift_detected": true | false | null,            // null when undetermined
  "confidence": "high" | "medium" | "low",
  "drift_type": "covariate_shift" | "label_shift" | "configuration_change" | "transient" | "none" | "undetermined",
  "start_date": "YYYY-MM-DD" | null,                // period date the change starts, or the change point date
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

### 11.2 Without an LLM

`preVerdict`, `flags` and `changePoint.date` (or `comparison`) are already usable as an automated alert. Suggested
policy: alert on `drift_likely`, log on `suspicious`, ignore `stable`, and treat `undetermined` as a data
availability problem rather than a model problem. `hardBreaks` deserve their own notification to the line owner
regardless of the verdict.

---

## 12. Operations

- The `inspectionStatistics` collection and its unique index (see `docs/inspection_statistics.md`, section 7) are
  created by the Data Service; the tool only reads them.
- The pipeline reads every document of the requested model and window. With one document per product and period
  that is many documents per period; the grouping happens in the database and only one row per period and entry
  is transferred.
- The analysis runs in a thread via `run_in_executor`, so other tool calls are not blocked.
- The connector logs the number of periods read per window and the volumes of the final result.
- Every threshold is tunable in `config.yaml` without a code change, and every result carries the values it was
  produced with. The bin edges are read from the documents, never configured here.
- `tests/drift/synthetic.py` holds a Python emulation of the pipeline's accumulator semantics. When the pipeline in
  `StatisticsQuery` changes, the emulation must change with it, otherwise the tests stop describing the server.
