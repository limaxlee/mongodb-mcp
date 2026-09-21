# Data Drift Tool on the Statistics Collection (proposal)

How `mongodb_analyze_data_drift` works on the `inspectionStatistics` collection described in
[inspection_statistics.md](inspection_statistics.md). This was the design proposal; it is implemented, and the
implementation reference is [DATA_DRIFT.md](../DATA_DRIFT.md). Where the two differ, `DATA_DRIFT.md` is right.

---

## 1. Principle

The tool never touches `inspections`. It reads statistics documents for the requested model and windows, sums them
per period inside MongoDB, and does only arithmetic over counts on the MCP side: proportions, PSI, JS, chi-square,
rates, ratios. There is no sampling, no bucket merging, no record budget, no quantile computation from raw values.

What the tool answers, in this order:

1. **When** did something change: a chronological series of periods with a few scalars each, plus the PSI between
   every period and the one before it.
2. **How much**: divergence at the single strongest split (range mode) or at the fixed reference/current
   boundary (comparison mode).
3. **In which direction**: the summed before and after sides with their histograms and class shares.
4. **Was it a configuration event**: backend, threshold or class list differing between consecutive periods.
5. **Deterministic flags and a pre-verdict** for an automated consumer; an LLM or a person confirms or overrules.

Dropped from the current tool: trend tests (Kendall tau, Theil-Sen), outlier z-scores, secondary change points,
pairwise maximum, KS tests, box geometry, image spec breaks, elapsed-time breaks, record budget, automatic bucket
choice by volume, small-bucket merging, parse-error examples.

---

## 2. Tool interface

Registered name unchanged: `mongodb_analyze_data_drift`.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model_name`, `model_version` | str | required | Exact match on the document key |
| `start_date`, `end_date` | datetime | required | Current window, UTC; `start_date <= startDate < end_date` on the documents |
| `task` | `cls` \| `det` \| null | null | Needed only when the model string exists for both tasks; otherwise resolved from the documents |
| `gbm`, `process` | str \| null | null | Exact match; when null, every site the model ran at is summed and the distinct values are reported |
| `mode` | str \| null | `production` | Null sums every mode |
| `equipment_id` | str \| null | null | Null uses the `total` block; a value uses that entry of `equipments[]` |
| `product_id` | str \| null | null | Null sums every product in the period |
| `granularity` | `auto` \| `hourly` \| `shift` \| `daily` \| `weekly` | `auto` | Period type of the documents to read; auto picks the finest that yields at most `max_periods` periods over both windows |
| `detail` | `full` \| `compact` | `full` | Whether periods carry histograms and per-class blocks or scalars only |
| `reference_start_date`, `reference_end_date` | datetime \| null | null | Both given switches to comparison mode |

`location` is dropped as a filter: it is a property of an equipment entry, so `equipment_id` covers it. `bucket`
is renamed to `granularity` to match the collection.

Validation, as today through `DriftWindow`: dates UTC-normalised, reference before current, total span at most
`max_total_days`. Because a year of daily documents is a few hundred small reads, `max_total_days` can be raised
from 30 to about 400 with `max_periods` 60 forcing weekly documents for long spans.

**Automatic granularity.** `auto` looks only at the total span of both windows, never at the data volume, and picks
the finest period type that yields at most `max_periods` periods:

| Total span of both windows | Periods it would produce | Chosen |
|---|---|---|
| up to 2.5 days | up to 60 hourly | `hourly` |
| 2.5 to 30 days | up to 60 shifts of 12 hours | `shift` |
| 30 to 60 days | up to 60 daily | `daily` |
| over 60 days | weekly, at most 57 for the 400-day maximum | `weekly` |

The choice is made before the query, so it names the documents that are read; there is no fallback to another
period type when that granularity was not written. Small periods on a quiet line are reported as insufficient rather
than merged, and the caller picks a coarser granularity.

---

## 3. Flow of one call

```
tool argument validation (fastmcp)
  -> MongoDBConnector.analyze_data_drift
       DriftWindow            validate windows, resolve granularity when auto
       StatisticsQuery        one aggregation pipeline per window (section 4)
       no periods at all      ValueError "No inspection statistics found for model ..."
       DriftAnalyzer.run()    in a thread, pure arithmetic (section 5)
            PeriodCounts       one per period, straight from the pipeline output
            PeriodSummary      scalars and proportions per period
            consecutive PSI    period i against period i-1
            SplitFinder        best split (range) or fixed boundary (comparison) on summed counts
            hard breaks        config equality between consecutive periods
            DriftRules         flags and pre-verdict
            DriftAnalysisResult
```

---

## 4. Query: one pipeline per window

The unique index prefix `(modelName, modelVersion, task, mode, gbm, process, granularity, startDate)` serves the
match. `productId` is the last key, so a range on `startDate` still uses the index when no product is given.

```
$match    modelName, modelVersion, task?, mode?, gbm?, process?, granularity,
          startDate: {$gte: start, $lt: end}, productId?
$project  startDate, endDate, gbm, process, mode, productId, classes, bins,
          block: equipment_id given
                   ? first element of $filter(equipments, equipmentId == equipment_id)   (null when absent)
                   : total
$group    by (startDate, entry kind, class):   one row per period and entry, documents without the block
                                            count in missingBlockCount instead of being dropped
            documentCount        $sum 1
            productIds           $addToSet productId          (count only is reported)
            gbms, processes, modes  $addToSet
            inspectionCount, predictionCount, boxCount              $sum
            confidence.count / sum / sumSq / belowThresholdCount / nearThresholdCount   $sum
            confidence.histogram                                    $push  -> element-wise $reduce/$zip sum
            classCounts                                             $objectToArray, unwind or $reduce into a map
            perClass[c].count / sum / sumSq / histogram / below / near   same, keyed by class
            images.noBoxCount, images.boxesPerImage.sum/sumSq/histogram   $sum / element-wise
            elapsedTime.count / sum                                 $sum
            backends, classLists, binsSeen                          $addToSet
            perClass[c].threshold                                   $addToSet, per class
            quality.*                                               $sum
            quantiles                                               $first, kept only when documentCount == 1
$sort     startDate
```

Everything the analysis needs is additive, so the sum over products (and over sites or modes when those filters are
null) is exact. Only `quantiles`, `min` and `max` are not additive; they are kept when a period is a single document
and otherwise replaced by a histogram-interpolated median, labelled as such (section 5.2).

**Size of what comes back.** One document per period, about 1 KB compact, a few KB with per-class blocks. Sixty
periods is well under 1 MB regardless of how many barcodes contributed.

**Bins.** Every document carries `bins`. Periods whose `bins.confidenceEdges` differ cannot be compared; the tool
reports `INCOMPATIBLE_BINS`, keeps the series, sets the crossing consecutive confidence PSI to null and runs no split.

**Backend and threshold.** The `total` block carries no backend, so the pipeline collects the backends from
`equipments[]`: every entry when the total is analysed, the requested entry otherwise. Thresholds are per predicted
class inside `perClass` for both tasks and come with the class entries.

---

## 5. Analysis

### 5.1 `PeriodCounts`

The raw additive numbers of one period, straight from the pipeline. It is the same idea as today's `SideCounts`
(`__add__` / `__sub__` defined), extended with per-class histograms, threshold counts, elapsed time and box counts.
Sums of `PeriodCounts` give the counts of any contiguous group of periods and of each side of a split.

### 5.2 `PeriodSummary` (the series)

Compact fields, one entry per period, reference periods first in comparison mode:

| Field | Derived from |
|---|---|
| `startDate`, `endDate`, `window` | document |
| `documentCount`, `productCount`, `inspectionCount`, `predictionCount`, `boxCount` | sums |
| `partial` | `endDate > now` (the period is still running) |
| `classDistribution` | `classCounts / sum` |
| `meanConfidence`, `stdConfidence` | `sum / count`, `sqrt(sumSq / count - mean^2)` |
| `medianConfidence` | `quantiles.p50` when one document, else linear interpolation inside the histogram, with `medianSource: exact | histogram` |
| `belowThresholdRate`, `nearThresholdRate` | `count / confidence.count`, null when the threshold counts are null |
| `meanBoxesPerImage`, `noBoxRate` | det only |
| `meanElapsedTime` | `elapsedTime.sum / count` |
| `backends`, `thresholdsByClass` | the sets seen; a set with more than one element is itself a hard break inside the period |

Full detail adds `confidenceHistogram` (proportions), `confidenceQuantiles` (when exact), and `perClass` with
`count`, `share`, `meanConfidence`, `belowThresholdRate`, `histogram`.

### 5.3 Consecutive divergence

For every period `i > 0`: `psiConfidence`, `psiClass` and, for det, `psiBoxesPerImage` against period `i-1`, with
`sufficient` true when both periods meet `min_side_records`. This is the simplest thing an LLM can read to answer
"when": a spike in the consecutive PSI series is the step.

### 5.4 Split (range mode) or comparison (comparison mode)

Range mode: exhaustive search over split positions with at least `min_segment_periods` on each side, maximising
`psiConfidence + psiClass` of the summed sides, as the current `SplitFinder` does but on `PeriodCounts` only. Sixty
periods means at most sixty sums of a few hundred integers.

Comparison mode: the reference periods are one side and the current periods the other, no search.

The chosen split reports:

| Field | Meaning |
|---|---|
| `periodIndex`, `date` | start of the after side; index null in comparison mode |
| `candidateCount` | positions tried, the chi-square p-value is Bonferroni-corrected by it |
| `beforeCount`, `afterCount`, `sidesSufficient` | predictions (boxes for det) on each side against `min_side_records_cls/det` |
| `psiConfidence`, `jsConfidence` | on the summed confidence histograms |
| `psiConfidenceByClass` | per predicted class, to say which class moved |
| `psiClass`, `chi2Class` (`stat`, `p`, `dof`), `maxClassProportionChange` | class mix |
| `psiBoxesPerImage` | det |
| `belowThresholdRateBefore/After`, `meanConfidenceBefore/After`, `meanElapsedTimeBefore/After` | direction |
| `before`, `after` | side summaries with the same fields as a `PeriodSummary` |

### 5.5 Hard breaks

Consecutive periods compared on `backend`, `threshold` (per predicted class) and `classes`. A period whose own set has
more than one value is also a break, dated at that period. Entries: `kind`, `className`, `date`, `from`, `to`.

### 5.6 Rules and pre-verdict

Kept flags and their rules, thresholds from `SETTINGS.data_drift`:

| Flag | Rule |
|---|---|
| `INSUFFICIENT_DATA` | fewer than `min_side_records` predictions in a window |
| `INSUFFICIENT_PERIODS` | fewer than `min_periods_changepoint` periods in range mode |
| `INCOMPATIBLE_BINS` | bin edges differ across the read documents |
| `HARD_BREAK` | any hard break |
| `CONFIDENCE_SHIFT` / `_MODERATE` | split `psiConfidence >= psi_significant` / `psi_moderate`, sides sufficient |
| `CLASS_SHIFT` / `_MODERATE` | `psiClass >= psi_significant`, or corrected `chi2 p < chi2_p` and `maxClassProportionChange >= class_prop_change` / `psiClass >= psi_moderate` |
| `BOX_COUNT_SHIFT` | det, `psiBoxesPerImage >= psi_significant` |
| `THRESHOLD_PRESSURE` | after rate at least `threshold_pressure_ratio` times before and `threshold_pressure_abs` higher |

Pre-verdict as today: `undetermined` on an insufficiency flag, `drift_likely` on any full shift flag, `suspicious` on a
moderate flag, threshold pressure or a hard break, otherwise `stable`. Trend and outlier flags are gone, so the
"two trend families" rule goes too.

---

## 6. Response schema

Pydantic models, serialised in camelCase like today's `DriftModel`. `mode` is the operating mode of the inspections
(as in the collection); `analysisMode` says whether a split was searched or a fixed comparison was made. Today's
result uses `mode` for the latter, this renames it to match the collection.

```python
class DriftModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class DateRange(DriftModel):
    start_date: datetime
    end_date: datetime
    days: float


class Filters(DriftModel):
    gbm: str | None = None
    process: str | None = None
    equipment_id: str | None = None
    product_id: str | None = None


class SitesSeen(DriftModel):                     # distinct values summed when a filter was null
    gbms: list[str] = []
    processes: list[str] = []
    modes: list[str] = []
    equipment_ids: list[str] = []                # from equipments[].equipmentId of the read documents


class Bins(DriftModel):                          # as read from the documents
    confidence_edges: list[float]
    near_threshold_margin: float | None = None
    boxes_per_image_max: int | None = None


class Status(DriftModel):
    analysis_possible: bool                      # a split or comparison was computed with sufficient sides
    split_ran: bool
    comparison_ran: bool
    period_count: int
    reference_period_count: int = 0


class DataQuality(DriftModel):
    document_count: int                          # statistics documents read over both windows
    product_count: int                           # distinct productId
    inspection_count: int
    prediction_count: int                        # images
    box_count: int | None = None                 # det only
    missing_confidence_count: int
    parse_error_count: int
    partial_periods: list[datetime] = []         # startDate of periods whose endDate is in the future
    periods_missing_equipment: list[datetime] = []   # equipment_id given but absent in these periods
    incompatible_bins: bool = False


class ConfidenceQuantiles(DriftModel):
    p05: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None


class ClassSummary(DriftModel):                  # one predicted class inside a period or a side
    count: int
    share: float
    mean_confidence: float | None = None
    std_confidence: float | None = None
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    threshold: float | None = None               # threshold of this predicted class
    confidence_histogram: list[float] | None = None   # proportions, full detail only


class CompactPeriodSummary(DriftModel):
    start_date: datetime
    end_date: datetime
    window: Literal["current", "reference"] = "current"
    partial: bool = False
    document_count: int
    product_count: int
    inspection_count: int
    prediction_count: int
    box_count: int | None = None                 # det only
    class_distribution: dict[str, float] = {}
    mean_confidence: float | None = None
    std_confidence: float | None = None
    median_confidence: float | None = None
    median_source: Literal["exact", "histogram"] | None = None
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    mean_boxes_per_image: float | None = None    # det only
    no_box_rate: float | None = None             # det only
    mean_elapsed_time: float | None = None
    backends: list[str] = []                     # more than one element is a break inside the period
    thresholds_by_class: dict[str, list[float]] = {}   # per predicted class, both tasks


class PeriodSummary(CompactPeriodSummary):
    confidence_histogram: list[float] | None = None    # proportions over bins.confidence_edges
    confidence_quantiles: ConfidenceQuantiles | None = None   # only when median_source == "exact"
    boxes_per_image_histogram: list[float] | None = None      # det only
    per_class: dict[str, ClassSummary] = {}


class ConsecutiveDivergence(DriftModel):         # period i against period i-1, i > 0
    period_index: int
    date: datetime                               # start of period i
    psi_confidence: float | None = None
    psi_class: float | None = None
    psi_boxes_per_image: float | None = None     # det only
    sufficient: bool                             # both periods meet min_side_records


class Chi2Result(DriftModel):
    stat: float
    p: float                                     # Bonferroni-corrected by candidate_count
    dof: int


class SideSummary(DriftModel):                   # a summed group of periods
    start_date: datetime
    end_date: datetime
    period_count: int
    document_count: int
    prediction_count: int
    box_count: int | None = None
    class_distribution: dict[str, float] = {}
    mean_confidence: float | None = None
    median_confidence: float | None = None       # always histogram-interpolated
    below_threshold_rate: float | None = None
    near_threshold_rate: float | None = None
    mean_boxes_per_image: float | None = None
    no_box_rate: float | None = None
    mean_elapsed_time: float | None = None
    confidence_histogram: list[float] | None = None
    boxes_per_image_histogram: list[float] | None = None
    per_class: dict[str, ClassSummary] = {}


class SplitResult(DriftModel):                   # change_point (range) or comparison (comparison mode)
    period_index: int | None = None              # index into periods of the first "after" period; null in comparison
    date: datetime                               # start of the after side
    candidate_count: int                         # 1 in comparison mode
    before_count: int
    after_count: int
    sides_sufficient: bool
    psi_confidence: float | None = None
    js_confidence: float | None = None
    psi_confidence_by_class: dict[str, float | None] = {}
    psi_class: float | None = None
    chi2_class: Chi2Result | None = None
    max_class_proportion_change: float | None = None
    psi_boxes_per_image: float | None = None     # det only
    mean_confidence_delta: float | None = None   # after - before
    below_threshold_rate_delta: float | None = None
    elapsed_time_ratio: float | None = None      # after / before
    before: SideSummary
    after: SideSummary


class HardBreak(DriftModel):
    kind: Literal["backend", "threshold", "classes"]
    class_name: str | None = None                # threshold breaks
    date: datetime                               # start of the period where the new value appears
    from_value: Any = Field(None, alias="from")
    to_value: Any = Field(None, alias="to")


class DriftAnalysisResult(DriftModel):
    model_name: str
    model_version: str
    task: Literal["cls", "det"]
    mode: str | None                             # operating mode filter, null when every mode was summed
    analysis_mode: Literal["range", "comparison"]
    filters: Filters
    sites_seen: SitesSeen
    granularity: Literal["hourly", "shift", "daily", "weekly"]
    range: DateRange
    reference_range: DateRange | None = None
    status: Status
    data_quality: DataQuality
    classes: list[str]
    bins: Bins
    periods: list[PeriodSummary] | list[CompactPeriodSummary]
    consecutive: list[ConsecutiveDivergence] = []
    change_point: SplitResult | None = None      # range mode
    comparison: SplitResult | None = None        # comparison mode
    hard_breaks: list[HardBreak] = []
    flags: list[DriftFlag] = []
    pre_verdict: PreVerdict
    config: dict[str, Any]                       # thresholds used, echoed from SETTINGS.data_drift
```

### 6.1 Example (compact, classification, range mode, abbreviated)

```json
{
  "modelName": "sidetopsideu8000",
  "modelVersion": "sidetopsideu8000_260316_260316081905",
  "task": "cls",
  "mode": "production",
  "analysisMode": "range",
  "filters": {"gbm": "SEHC", "process": "Side", "equipmentId": null, "productId": null},
  "sitesSeen": {"gbms": ["SEHC"], "processes": ["Side"], "modes": ["production"], "equipmentIds": ["SEHC_Side_VM07", "SEHC_Side_VM08"]},
  "granularity": "daily",
  "range": {"startDate": "2026-09-01T00:00:00Z", "endDate": "2026-09-22T00:00:00Z", "days": 21.0},
  "referenceRange": null,
  "status": {"analysisPossible": true, "splitRan": true, "comparisonRan": false, "periodCount": 19, "referencePeriodCount": 0},
  "dataQuality": {"documentCount": 4120, "productCount": 4120, "inspectionCount": 4120, "predictionCount": 82400,
                  "boxCount": null, "missingConfidenceCount": 0, "parseErrorCount": 0,
                  "partialPeriods": ["2026-09-21T00:00:00Z"], "periodsMissingEquipment": [], "incompatibleBins": false},
  "classes": ["OK", "NG"],
  "bins": {"confidenceEdges": [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0], "nearThresholdMargin": 0.05, "boxesPerImageMax": null},
  "periods": [
    {"startDate": "2026-09-01T00:00:00Z", "endDate": "2026-09-02T00:00:00Z", "window": "current", "partial": false,
     "documentCount": 220, "productCount": 220, "inspectionCount": 220, "predictionCount": 4400, "boxCount": null,
     "classDistribution": {"OK": 0.93, "NG": 0.07},
     "meanConfidence": 0.981, "stdConfidence": 0.03, "medianConfidence": 0.99, "medianSource": "histogram",
     "belowThresholdRate": 0.001, "nearThresholdRate": 0.004,
     "meanBoxesPerImage": null, "noBoxRate": null, "meanElapsedTime": 0.2,
     "backends": ["ts"], "thresholdsByClass": {"OK": [0.5], "NG": [0.5]}},
    "..."
  ],
  "consecutive": [
    {"periodIndex": 1, "date": "2026-09-02T00:00:00Z", "psiConfidence": 0.004, "psiClass": 0.001, "psiBoxesPerImage": null, "sufficient": true},
    "...",
    {"periodIndex": 12, "date": "2026-09-14T00:00:00Z", "psiConfidence": 0.31, "psiClass": 0.02, "psiBoxesPerImage": null, "sufficient": true}
  ],
  "changePoint": {
    "periodIndex": 12, "date": "2026-09-14T00:00:00Z", "candidateCount": 16,
    "beforeCount": 52800, "afterCount": 29600, "sidesSufficient": true,
    "psiConfidence": 0.34, "jsConfidence": 0.05,
    "psiConfidenceByClass": {"OK": 0.36, "NG": 0.08},
    "psiClass": 0.01, "chi2Class": {"stat": 3.1, "p": 1.0, "dof": 1}, "maxClassProportionChange": 0.012,
    "psiBoxesPerImage": null,
    "meanConfidenceDelta": -0.041, "belowThresholdRateDelta": 0.013, "elapsedTimeRatio": 1.0,
    "before": {"startDate": "2026-09-01T00:00:00Z", "endDate": "2026-09-14T00:00:00Z", "periodCount": 12, "...": "..."},
    "after":  {"startDate": "2026-09-14T00:00:00Z", "endDate": "2026-09-22T00:00:00Z", "periodCount": 7, "...": "..."}
  },
  "comparison": null,
  "hardBreaks": [],
  "flags": ["CONFIDENCE_SHIFT", "THRESHOLD_PRESSURE"],
  "preVerdict": "drift_likely",
  "config": {"psiModerate": 0.1, "psiSignificant": 0.25, "...": "..."}
}
```

Reading: the consecutive PSI jumps at 14 September, the split confirms it with PSI 0.34 on the confidence histogram
while the class mix did not move, the per-class PSI says it is the OK class whose confidence slid down, and the
below-threshold rate rose. That is the covariate-shift signature; no hard break explains it.

---

## 7. Code changes

| Area | Change |
|---|---|
| `drift/query.py` | Replace `DriftQueryBuilder` (inspections match + unwind) with `StatisticsQuery` building the section 4 pipeline |
| `drift/extractor.py`, `drift/buckets.py`, `drift/summaries.py`, `drift/series.py` | Removed. Bucketing, extraction, quantiles and trends no longer exist on this side |
| `drift/period.py` (new) | `PeriodCounts` (additive, from a pipeline row) and `PeriodSummary` |
| `drift/splits.py` | Keep the search, feed it `PeriodCounts`; drop KS and geometry inputs; add per-class PSI |
| `drift/drift.py` | Orchestration trimmed to section 3; consecutive PSI and hard breaks live here |
| `drift/rules.py` | Drop trend, outlier and geometry branches; add `INCOMPATIBLE_BINS`, rename `INSUFFICIENT_BUCKETS` |
| `drift/window.py` | Granularity resolution replaces bucket-size resolution |
| `schemas/drift.py` | Remove `Record`, `BoxRecord`, `Bucket`, geometry and trend models; add the section 6 shapes |
| `utils/stats.py` | Keep `proportions`, `psi`, `js_divergence`, `chi2_contingency`; add `histogram_median`; remove KS, Kendall, Theil-Sen, robust z |
| `common/constants.py` | `Granularity` enum, `DBCollections.INSPECTIONS_STATISTICS`, trimmed `DriftFlag`, remove trend/outlier/geometry enums |
| `config.yaml` | Remove record budget, bucket minimums, merge share, bin edges, KS, trend, outlier and elapsed-break settings; add `max_periods`, `min_periods_changepoint`, `min_segment_periods`; raise `max_total_days` |
| `connector.py` | `analyze_data_drift` and the extraction helper rewritten around the pipeline |
| tests | Synthetic generator produces statistics documents instead of inspection documents |

---

## 8. Risk: `productId` in the document key (still open)

With one document per barcode and period, the tool must sum documents per period in the pipeline, which the design
above does. The cost is on the database, not on the tool: a line inspecting 300 units an hour writes 300 hourly
documents an hour, roughly 7,000 a day and 2.5 million a year per model, each around 25 KB. The pipeline then reads
every one of them for the requested window before grouping. For a 30-day daily analysis that is over 200,000
documents scanned per call.

If barcodes are per unit, the practical fix is on the Data Service side: also write one document per period with
`productId: null` (the "all products" document), and let the tool read that unless `product_id` is given. The tool
design does not change, only the `$match` adds `productId: null` by default.
