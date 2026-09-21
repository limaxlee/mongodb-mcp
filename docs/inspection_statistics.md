# Inspection Statistics Collection (proposal)

Design proposal for the `inspectionStatistics` collection: pre-aggregated statistics of AI inspection results,
computed by the Data Service, that replace the raw `inspections` scan previously done by the data drift tool.
The MCP tool that reads this collection is implemented (see [DATA_DRIFT.md](../DATA_DRIFT.md)); the collection
itself is written by the Data Service and is not part of this repository.

---

## 1. Goal and scope

The current drift tool reads raw inspection documents, flattens every prediction into a record, cuts the records
into time buckets and computes everything itself (histograms, quantiles, PSI, KS, change point search, trends,
outliers, hard breaks, flags, pre-verdict). This proposal moves all **per-bucket** statistics into MongoDB so that the
MCP side only reads small documents and, at most, compares two histograms.

Decisions already taken:

| Topic | Decision |
|---|---|
| Who computes | Data Service; out of scope here |
| Divergence measures (PSI, JS, chi-square) | **Not stored.** They compare two buckets. Documents store fixed-bin histograms and counts; the tool computes PSI/JS between any two documents or groups of documents from those (a few lines of arithmetic over ten numbers) |
| KS test on raw confidences | Dropped. Raw values are not stored; PSI/JS on the histogram plus quantiles replace it |
| Level of statistics | Per **predicted class**, plus a total over all classes |
| Confidence of one prediction | `max(confidence)` when the field is a list, the value itself when it is a float |
| Task types | `cls` and `det` only; `seg` produces no statistics document |
| `productId` | Barcode of the inspected product; part of the document key (open point 5) |
| `mode` | Statistics are computed for every mode; `mode` is part of the document key next to model name and version |
| `location` | Lives inside each `equipments[]` entry |
| `classes`, `localTimezone` | Properties of the model version and the site, stored once at document level |
| `backend`, `threshold` | Scalars per equipment (last value seen in the period); a change is detected between periods, not inside one |
| Image specs, box geometry | Not stored, not used in the analysis |
| Time boundaries | Strictly UTC: hour, 00:00 / 12:00 shift, midnight, Monday midnight |
| Human feedback | Ignored completely |
| Existence | A document exists only for a period that has at least one inspection document |
| Retention | About one year |

---

## 2. Document identity

One document per **(modelName, modelVersion, task, mode, gbm, process, productId, granularity, startDate)**.
Inside it, one entry per equipment, plus a `total` block that sums every equipment.

| Field | Type | Meaning |
|---|---|---|
| `modelName`, `modelVersion` | str | Split of `aiResults[].aiModel` on the first `/` (as in the summary collection) |
| `task` | `cls` \| `det` | From `aiResults[].task` |
| `mode` | str | `metadata.mode` (`production`, `rework`, `test`, ...) |
| `gbm`, `process` | str | `metadata.gbm`, `metadata.process` |
| `productId` | str | `metadata.productId`, the barcode of the inspected product |
| `granularity` | `hourly` \| `shift` \| `daily` \| `weekly` | Period type (proposed name instead of `bucket`; `bucket` is also what the tool calls one time slice of an analysis, which would be confusing) |
| `startDate`, `endDate` | datetime (UTC) | Period boundaries, start inclusive, end exclusive. Consistent with the `metrics` collection. `startDate` is what other collections call `date` |

Period boundaries for `startDate` = 2026-09-21 04:00 UTC:

| granularity | startDate | endDate |
|---|---|---|
| hourly | 2026-09-21 04:00 | 2026-09-21 05:00 |
| shift | 2026-09-21 00:00 or 12:00 | +12 h |
| daily | 2026-09-21 00:00 | 2026-09-22 00:00 |
| weekly | Monday 2026-09-21 00:00 | Monday 2026-09-28 00:00 |

A document is selected by the `createdAt` of the inspection documents: `startDate <= metadata.createdAt < endDate`.

One inspection document contributes to **one statistics document per distinct `(aiModel, task)` in its
`aiResults`**. If the same `aiModel` appears twice in one inspection document, both entries count.

---

## 3. Full schema

Values are illustrative. Comments are not part of the document.

```jsonc
{
  "_id": ObjectId,

  // ---- identity (unique together) ----
  "modelName": "sidetopsideu8000",
  "modelVersion": "sidetopsideu8000_260316_260316081905",
  "task": "cls",
  "mode": "production",
  "gbm": "SEHC",
  "process": "Side",
  "productId": "12HJ3NGL601106K",
  "granularity": "hourly",
  "startDate": ISODate("2026-06-20T00:00:00Z"),
  "endDate":   ISODate("2026-06-20T01:00:00Z"),

  // ---- model and site properties ----
  "classes": ["OK", "NG"],                       // ordered union of aiResults[].classes seen in the period
  "localTimezone": "Asia/Bangkok",               // metadata.localTimezone of the site

  // ---- bin definitions used by every histogram in this document ----
  "bins": {
    "confidenceEdges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   // 10 bins, last one closed
    "nearThresholdMargin": 0.05,                                                  // |confidence - threshold| < margin
    "boxesPerImageMax": 5                                                         // bins 0,1,2,3,4,5+  (det only)
  },

  // ---- bookkeeping ----
  "computedAt": ISODate("2026-06-20T01:05:12Z"),   // last time this document was (re)computed
  "equipmentCount": 2,

  // ---- sum over every equipment; same shape as one equipment entry minus equipmentId/location/backend/threshold ----
  "total": { ...EquipmentStats },

  "equipments": [
    {
      "equipmentId": "SEHC_Side_VM07",
      "location": "VM07",
      ...EquipmentStats
    },
    {
      "equipmentId": "SEHC_Side_VM08",
      "location": "VM08",
      ...EquipmentStats
    }
  ]
}
```

### 3.1 `EquipmentStats` (classification)

```jsonc
{
  "inspectionCount": 118,                        // inspection documents that contributed
  "predictionCount": 2360,                       // images (cls) / images (det)

  // ---- runtime configuration, last value seen in the period ----
  "backend": "ts",
  "threshold": 0.5,                              // cls: prediction.threshold; null when the model has none

  // ---- data quality ----
  "quality": {
    "missingConfidenceCount": 0,                 // predictions with empty / missing / non-numeric confidence
    "parseErrorCount": 0
  },

  // ---- inference time over all predictions ----
  "elapsedTime": { "count": 2360, "sum": 472.0, "mean": 0.2, "min": 0.1, "max": 0.31, "p50": 0.2, "p90": 0.24 },

  // ---- class distribution as counts (additive; the tool derives shares) ----
  "classCounts": { "OK": 2190, "NG": 170 },

  // ---- confidence over all predictions ----
  "confidence": { ...ConfidenceStats },

  // ---- confidence per predicted class ----
  "perClass": {
    "OK": { ...ConfidenceStats },
    "NG": { ...ConfidenceStats }
  }
}
```

### 3.2 `ConfidenceStats`

Used for the total and for each predicted class. Every count is additive, so any set of hourly documents can be
combined exactly into a larger window. `mean` and `std` are stored for direct reading and can be recomputed from
`sum`, `sumSq` and `count` after combining.

```jsonc
{
  "count": 2190,                                  // predictions with a numeric confidence
  "sum": 2116.6,
  "sumSq": 2049.9,
  "mean": 0.9665,
  "std": 0.0412,
  "min": 0.5012,
  "max": 0.9999,
  "quantiles": { "p05": 0.88, "p10": 0.91, "p25": 0.95, "p50": 0.975, "p75": 0.99, "p90": 0.995, "p95": 0.998 },
  "histogram": [0, 0, 0, 0, 0, 3, 12, 41, 210, 1924],   // counts per bin of bins.confidenceEdges

  // threshold relation; null when the equipment (cls) or the class (det) has no threshold
  "belowThresholdCount": 3,                       // confidence < threshold, rate = belowThresholdCount / count
  "nearThresholdCount": 15                        // |confidence - threshold| < bins.nearThresholdMargin
}
```

### 3.3 Detection additions

For `det` the per-prediction unit is the **bounding box**. `confidence`, `classCounts` and `perClass` are computed over
boxes, exactly like the current tool. Image-level information goes into `images`. Box geometry (size, position)
is not stored and not used in the analysis.

```jsonc
{
  "predictionCount": 1180,                        // images
  "boxCount": 2955,                               // boxes over all images

  "backend": "trt",
  "threshold": null,                              // det: thresholds are per box class, see perClass[c].threshold

  // ---- image level ----
  "images": {
    "noBoxCount": 210,                            // images with zero boxes
    "boxesPerImage": { "sum": 2955, "sumSq": 9812, "mean": 2.5, "std": 1.4, "max": 9,
                       "histogram": [210, 320, 290, 200, 100, 60] },       // bins 0,1,2,3,4,5+
    "boxesPerImageByClass": { "scratch": 1.6, "dent": 0.7, "particle": 0.2 }   // mean per image
  },

  "classCounts": { "scratch": 1890, "dent": 830, "particle": 235 },
  "confidence": { ...ConfidenceStats },
  "perClass": {
    "scratch":  { "threshold": 0.4, ...ConfidenceStats },
    "dent":     { "threshold": 0.4, ...ConfidenceStats },
    "particle": { "threshold": 0.3, ...ConfidenceStats }
  }
}
```

---

## 4. What is kept, what is dropped, and why

### Kept (moved into the document)

| Current tool output | Where it lives now |
|---|---|
| `recordCount`, `boxCount` | `predictionCount`, `boxCount` |
| `classDistribution` (shares) | `classCounts` (counts, the share is `count / sum`) |
| `medianConfidence`, `meanConfidence`, `stdConfidence`, `confidenceQuantiles` | `confidence.quantiles.p50`, `mean`, `std`, `quantiles` |
| `confidenceHistogram` (proportions) | `confidence.histogram` (counts) |
| `belowThresholdRate`, `nearThresholdRate` | `belowThresholdCount / count`, same for near |
| `meanBoxesPerImage`, `stdBoxesPerImage`, `noBoxRate`, `boxesPerImageHistogram`, `boxesByClassPerImage` | `images.*` |
| `thresholdValues`, backend, classes (hard breaks) | `threshold`, `backend` per equipment, `classes` per document; compared between consecutive periods |
| `imageSpecs` (image_spec hard break) | Not stored (open point 4) |
| `medianElapsedTime` | `elapsedTime` |
| `missingConfidenceCount`, `parseErrorCount` | `quality.*` |

### New

| Field | Why |
|---|---|
| `perClass` | Confidence drift is usually visible in one class first (for example NG confidence collapsing while OK stays); the current tool only has a total |
| `sum`, `sumSq` | Exact mean/std over any combination of documents |
| `inspectionCount` | Throughput and reliability of the bucket without touching `inspections` |
| `total` | One read for "the model on this line" without summing the array in a query |
| `localTimezone` | Lets the reader translate UTC boundaries into a factory shift when explaining a result |

### Dropped from the storage side (tool concern, not storage)

| Current feature | Decision |
|---|---|
| PSI, JS, chi-square, `maxClassProportionChange` | Computed by the tool from two histograms / two count maps. Cheap and exact |
| KS on raw values (`ksConfidence`, `ksNormalizedArea/Cx/Cy`) | Dropped, no raw values stored. Quantile differences and histogram PSI cover the confidence signal |
| Box geometry (`box.*`, geometry KS tests and trends, `BOX_GEOMETRY_SHIFT`) | Dropped entirely. A camera or fixture move is not analysed |
| Auto bucket size, `merge_small`, record budget | Not needed. The caller picks the granularity, and a document is small whatever the volume |
| Change point search, secondary change points, `maxPairwise` | Recommended to drop. With per-period documents the LLM can see the series directly. If wanted later, it is a loop over the stored histograms, still without touching `inspections` |
| Trends (Kendall tau, Theil-Sen), outlier z-scores | Recommended to drop for the same reason. Can be re-added on the tool side later |
| `preVerdict`, most flags | Recommended to reduce to a small deterministic set (section 5) |
| Human feedback | Ignored |

---

## 5. What the drift tool can compute from these documents (for later)

Not part of this proposal's deliverable, listed to show the schema is sufficient.

| Question | Inputs from the documents | Computation |
|---|---|---|
| Did the confidence distribution move? | `confidence.histogram` of two periods (or two sums of periods) | PSI, JS, with `eps` smoothing |
| Did one class's confidence move? | `perClass[c].histogram` | Same |
| Did the class mix move? | `classCounts` | PSI over shares, chi-square from the two count maps |
| Threshold pressure? | `belowThresholdCount / count` over time | Ratio and absolute change |
| Detector missing or inventing objects? | `images.boxesPerImage.histogram`, `images.noBoxCount` | PSI, rate change |
| Configuration change (hard break)? | `backend`, `threshold`, `classes` compared between consecutive periods | Equality check |
| Slower inference? | `elapsedTime.p50` | Ratio |
| Is the number reliable? | `predictionCount`, `boxCount`, `quality.*` | Sufficiency thresholds |

Everything above is arithmetic over at most a few hundred numbers per period and needs no access to `inspections`.

---

## 6. Open points to confirm

1. **Quantiles per class.** The proposal stores the full `p05...p95` set in every `ConfidenceStats`. If document
   size matters for detection models with many classes, keep the full set only in `confidence` and `p10/p50/p90`
   in `perClass`.
2. **Threshold placement.** `threshold` is a float at equipment level for `cls` and a float inside each
   `perClass[c]` block for `det`, because detection thresholds are per box class. If one shape is preferred, put
   `threshold` inside `perClass` for both tasks and drop the equipment-level field.
3. **Rollups.** With `sum`, `sumSq`, `histogram` and every count additive, daily, shift and weekly documents are an
   exact sum of hourly documents except for `quantiles`, `min`, `max` and `std`, which are also derivable
   (min/max by union, std from sums, quantiles from the raw data). Whether the Data Service builds the coarser
   granularities from raw inspections or from hourly documents is its choice; the schema works for both.
4. **Image specs and box geometry are not stored.** A camera, resolution or fixture change therefore cannot be
   named by the tool; it will only show up indirectly as a confidence or box-count shift.
5. **`productId` in the key.** One document exists per barcode and period. If a barcode identifies a single
   inspected unit, an hourly document typically holds one inspection and the histograms are only meaningful once
   the tool sums documents across barcodes (every count is additive, so that is exact). If a barcode spans many
   inspections (a lot or panel), each document is already a usable sample.

---

## 7. Indexes

```python
DBCollections.INSPECTIONS_STATISTICS: [
    {
        # identity, one document per key
        "index_keys": ["modelName", "modelVersion", "task", "mode", "gbm", "process", "granularity", "startDate", "productId"],
        "keys": [
            ("modelName", pymongo.ASCENDING),
            ("modelVersion", pymongo.ASCENDING),
            ("task", pymongo.ASCENDING),
            ("mode", pymongo.ASCENDING),
            ("gbm", pymongo.ASCENDING),
            ("process", pymongo.ASCENDING),
            ("granularity", pymongo.ASCENDING),
            ("startDate", pymongo.ASCENDING),
            ("productId", pymongo.ASCENDING)
        ],
        "unique": True
    },
    {
        # site-level browsing: "which models ran on this line last week"
        "index_keys": ["gbm", "process", "granularity", "startDate"],
        "keys": [
            ("gbm", pymongo.ASCENDING),
            ("process", pymongo.ASCENDING),
            ("granularity", pymongo.ASCENDING),
            ("startDate", pymongo.ASCENDING)
        ],
        "unique": False
    },
    {
        # equipment lookup across models (multikey on the array)
        "index_keys": ["equipments.equipmentId", "granularity", "startDate"],
        "keys": [
            ("equipments.equipmentId", pymongo.ASCENDING),
            ("granularity", pymongo.ASCENDING),
            ("startDate", pymongo.ASCENDING)
        ],
        "unique": False
    },
    {
        # retention: one year after the period start, and a plain time scan
        "index_keys": ["startDate"],
        "keys": "startDate",
        "unique": False,
        "expireAfterSeconds": 31536000
    }
]
```

Notes:
- The drift tool's main query is `modelName, modelVersion, task, mode, gbm, process, granularity, startDate range`,
  which is exactly the unique index prefix. `productId` is the last key so that a range scan on `startDate` still
  uses the index when no product is requested; a product-only lookup is served by the `inspections` collection's
  `metadata.productId` index instead. `task` is in the key because the same `aiModel` string can appear with
  different tasks, the reason the current tool has a `task` argument.
- A TTL index expires a document `expireAfterSeconds` after `startDate`, so a weekly document for a week that started
  366 days ago disappears at the same moment as its hourly documents. If a different retention per granularity is
  wanted later, replace it with an `expiresAt` field set by the Data Service.
- The multikey index on `equipments.equipmentId` cannot be unique and is optional. It only matters if the LLM is
  expected to ask "what happened on equipment X" without knowing the model.

---

## 8. Pydantic sketch

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, NonNegativeInt, NonNegativeFloat


class Quantiles(BaseModel):
    p05: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None


class ValueStats(BaseModel):
    count: NonNegativeInt = 0
    sum: float = 0.0
    sumSq: float = 0.0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    quantiles: Quantiles = Quantiles()


class ConfidenceStats(ValueStats):
    histogram: list[NonNegativeInt] = []               # len == len(bins.confidenceEdges) - 1
    belowThresholdCount: NonNegativeInt | None = None  # None when no threshold applies
    nearThresholdCount: NonNegativeInt | None = None
    threshold: float | None = None                     # det only, per box class


class ElapsedTimeStats(BaseModel):
    count: NonNegativeInt = 0
    sum: float = 0.0
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    p50: float | None = None
    p90: float | None = None


class Quality(BaseModel):
    missingConfidenceCount: NonNegativeInt = 0
    parseErrorCount: NonNegativeInt = 0


class BoxesPerImage(BaseModel):
    sum: NonNegativeInt = 0
    sumSq: NonNegativeInt = 0
    mean: float | None = None
    std: float | None = None
    max: NonNegativeInt | None = None
    histogram: list[NonNegativeInt] = []               # bins 0..boxesPerImageMax, last is "max or more"


class ImageStats(BaseModel):                            # det only
    noBoxCount: NonNegativeInt = 0
    boxesPerImage: BoxesPerImage = BoxesPerImage()
    boxesPerImageByClass: dict[str, float] = {}


class EquipmentStats(BaseModel):
    inspectionCount: NonNegativeInt = 0
    predictionCount: NonNegativeInt = 0                # images
    boxCount: NonNegativeInt | None = None             # det only
    quality: Quality = Quality()
    elapsedTime: ElapsedTimeStats = ElapsedTimeStats()
    classCounts: dict[str, NonNegativeInt] = {}
    confidence: ConfidenceStats = ConfidenceStats()
    perClass: dict[str, ConfidenceStats] = {}
    images: ImageStats | None = None                   # det only


class EquipmentEntry(EquipmentStats):
    equipmentId: str
    location: str
    backend: str | None = None                         # last value seen in the period
    threshold: float | None = None                     # cls only; det thresholds live in perClass[c].threshold


class Bins(BaseModel):
    confidenceEdges: list[NonNegativeFloat] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nearThresholdMargin: NonNegativeFloat = 0.05
    boxesPerImageMax: NonNegativeInt = 5


class InspectionStatisticsDocument(BaseModel):
    modelName: str
    modelVersion: str
    task: Literal["cls", "det"]
    mode: str
    gbm: str
    process: str
    productId: str
    granularity: Literal["hourly", "shift", "daily", "weekly"]
    startDate: datetime
    endDate: datetime
    classes: list[str]                                 # ordered union of aiResults[].classes in the period
    localTimezone: str | None = None
    bins: Bins = Bins()
    computedAt: datetime
    equipmentCount: NonNegativeInt
    total: EquipmentStats
    equipments: list[EquipmentEntry]
```

---

## 9. Size estimate

Per `ConfidenceStats` block: about 25 numbers. Per equipment entry: `(classes + 1)` blocks plus roughly 30
bookkeeping numbers (about 45 for detection with the `images` block).

| Case | Numbers per equipment | Numbers per document (10 equipments) | Rough BSON size |
|---|---|---|---|
| cls, 2 classes | ~105 | ~1.2 k | ~25 KB |
| det, 10 classes | ~320 | ~3.5 k | ~80 KB |
| det, 30 classes | ~820 | ~9 k | ~200 KB |

Far below the 16 MB limit, but a detection model with many classes and many equipments is still too large to hand
to an LLM raw. The tool should project only `total` or one equipment, and only the fields it needs, before returning.
