"""Aggregation pipeline over the inspectionStatistics collection

One pipeline per window. It selects the statistics documents of one model, picks the total block or one equipment
entry, and sums every additive number per period inside MongoDB, so that one small row per (period, entry) comes
back whatever the number of documents (products) behind a period.

Row shape after the pipeline (one row per period and entry):

    _id:   {date, kind, k}      kind is "total" (k null), "class" (k = predicted class) or "count" (k = class)
    period level (read from the "total" row): endDate, documentCount, missingBlockCount, tasks, productIds, gbms,
        processes, modes, equipmentIdSets, classSets, bins, inspectionCount, predictionCount, boxCount,
        boxCountPresent, missingConfidenceCount, parseErrorCount, elapsedCount, elapsedSum, imagesPresent,
        noBoxCount, boxesPerImageSum, boxesPerImageSumSq, boxesPerImageHistogram, backendSets, thresholdSets
    entry level: count, sum, sumSq, min, max, belowThresholdCount, belowPresent, nearThresholdCount, nearPresent,
        histogram, entryThresholds, quantiles, classCount

A document whose equipments[] has no entry for the requested equipment keeps its period alive with
documentCount 0 and missingBlockCount > 0, so the analysis can report the periods it had to leave out.

The total block carries no backend or threshold, so those come from the equipments[] entries: all of them when the
total is analysed, the requested one otherwise.
"""
from typing import Any
from datetime import datetime

from common.constants import DriftTask, Granularity

ENTRY_TOTAL = "total"
ENTRY_CLASS = "class"
ENTRY_COUNT = "count"

BLOCK_PRESENT = {"$ne": [{"$ifNull": ["$block", None]}, None]}


def _present(field: str) -> dict[str, Any]:
    """1 when the field exists and is not null, so that $max over a group says whether it was ever present"""
    return {"$max": {"$cond": [{"$ne": [{"$ifNull": [field, None]}, None]}, 1, 0]}}


def _if_block(expression: Any) -> dict[str, Any]:
    """The expression for documents that carry the requested block, null for the others"""
    return {"$cond": [BLOCK_PRESENT, expression, None]}


def _sum_arrays(field: str) -> dict[str, Any]:
    """Element-wise sum of the arrays pushed into a group, arrays only, an empty list when none was pushed"""
    return {
        "$reduce": {
            "input": {"$filter": {"input": field, "as": "item", "cond": {"$isArray": "$$item"}}},
            "initialValue": [],
            "in": {
                "$cond": [
                    {"$eq": [{"$size": "$$value"}, 0]},
                    "$$this",
                    {
                        "$map": {
                            "input": {"$zip": {"inputs": ["$$value", "$$this"]}},
                            "as": "pair",
                            "in": {"$sum": "$$pair"}
                        }
                    }
                ]
            }
        }
    }


class StatisticsQuery:
    def __init__(
            self,
            model_name: str,
            model_version: str,
            start_date: datetime,
            end_date: datetime,
            granularity: Granularity,
            task: str | None = None,
            gbm: str | None = None,
            process: str | None = None,
            mode: str | None = None,
            equipment_id: str | None = None,
            product_id: str | None = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.start_date = start_date
        self.end_date = end_date
        self.granularity = granularity
        self.task = task
        self.gbm = gbm
        self.process = process
        self.mode = mode
        self.equipment_id = equipment_id
        self.product_id = product_id

    def build_match(self) -> dict[str, Any]:
        """Filter on the unique index prefix: periods whose start lies in [start_date, end_date)"""
        match: dict[str, Any] = {
            "modelName": self.model_name,
            "modelVersion": self.model_version,
            "task": self.task if self.task is not None else {"$in": [task.value for task in DriftTask]}
        }
        if self.mode is not None:
            match["mode"] = self.mode
        if self.gbm is not None:
            match["gbm"] = self.gbm
        if self.process is not None:
            match["process"] = self.process
        match["granularity"] = self.granularity.value
        match["startDate"] = {"$gte": self.start_date, "$lt": self.end_date}
        if self.product_id is not None:
            match["productId"] = self.product_id

        return match

    def build_block(self) -> Any:
        """The total block, or the equipments[] entry of the requested equipment (missing when absent)"""
        if self.equipment_id is None:
            return "$total"

        return {
            "$arrayElemAt": [
                {
                    "$filter": {
                        "input": {"$ifNull": ["$equipments", []]},
                        "as": "item",
                        "cond": {"$eq": ["$$item.equipmentId", self.equipment_id]}
                    }
                },
                0
            ]
        }

    def build_pipeline(self) -> list[dict[str, Any]]:
        entries = {
            "$concatArrays": [
                [{"kind": ENTRY_TOTAL, "k": None, "v": {"$ifNull": ["$block.confidence", {}]}}],
                {
                    "$map": {
                        "input": {"$objectToArray": {"$ifNull": ["$block.perClass", {}]}},
                        "as": "item",
                        "in": {"kind": ENTRY_CLASS, "k": "$$item.k", "v": "$$item.v"}
                    }
                },
                {
                    "$map": {
                        "input": {"$objectToArray": {"$ifNull": ["$block.classCounts", {}]}},
                        "as": "item",
                        "in": {"kind": ENTRY_COUNT, "k": "$$item.k", "v": {"classCount": "$$item.v"}}
                    }
                }
            ]
        }

        group = {
            "_id": {"date": "$startDate", "kind": "$entries.kind", "k": "$entries.k"},
            "endDate": {"$first": "$endDate"},
            "documentCount": {"$sum": {"$cond": [BLOCK_PRESENT, 1, 0]}},
            "missingBlockCount": {"$sum": {"$cond": [BLOCK_PRESENT, 0, 1]}},
            "tasks": {"$addToSet": "$task"},
            "productIds": {"$addToSet": _if_block("$productId")},
            "gbms": {"$addToSet": _if_block("$gbm")},
            "processes": {"$addToSet": _if_block("$process")},
            "modes": {"$addToSet": _if_block("$mode")},
            "equipmentIdSets": {"$addToSet": _if_block("$equipmentIds")},
            "classSets": {"$addToSet": _if_block("$classes")},
            "bins": {"$addToSet": _if_block("$bins")},
            "inspectionCount": {"$sum": "$block.inspectionCount"},
            "predictionCount": {"$sum": "$block.predictionCount"},
            "boxCount": {"$sum": "$block.boxCount"},
            "boxCountPresent": _present("$block.boxCount"),
            "missingConfidenceCount": {"$sum": "$block.quality.missingConfidenceCount"},
            "parseErrorCount": {"$sum": "$block.quality.parseErrorCount"},
            "elapsedCount": {"$sum": "$block.elapsedTime.count"},
            "elapsedSum": {"$sum": "$block.elapsedTime.sum"},
            "imagesPresent": _present("$block.images"),
            "noBoxCount": {"$sum": "$block.images.noBoxCount"},
            "boxesPerImageSum": {"$sum": "$block.images.boxesPerImage.sum"},
            "boxesPerImageSumSq": {"$sum": "$block.images.boxesPerImage.sumSq"},
            "boxesPerImageHistograms": {"$push": "$block.images.boxesPerImage.histogram"},
            "backendSets": {"$addToSet": _if_block("$configBackends")},
            "thresholdSets": {"$addToSet": _if_block("$configThresholds")},
            "count": {"$sum": "$entries.v.count"},
            "sum": {"$sum": "$entries.v.sum"},
            "sumSq": {"$sum": "$entries.v.sumSq"},
            "min": {"$min": "$entries.v.min"},
            "max": {"$max": "$entries.v.max"},
            "belowThresholdCount": {"$sum": "$entries.v.belowThresholdCount"},
            "belowPresent": _present("$entries.v.belowThresholdCount"),
            "nearThresholdCount": {"$sum": "$entries.v.nearThresholdCount"},
            "nearPresent": _present("$entries.v.nearThresholdCount"),
            "histograms": {"$push": "$entries.v.histogram"},
            "entryThresholds": {"$addToSet": "$entries.v.threshold"},
            "quantiles": {"$first": "$entries.v.quantiles"},
            "classCount": {"$sum": "$entries.v.classCount"}
        }

        return [
            {"$match": self.build_match()},
            {
                "$project": {
                    "startDate": 1,
                    "endDate": 1,
                    "task": 1,
                    "gbm": 1,
                    "process": 1,
                    "mode": 1,
                    "productId": 1,
                    "classes": 1,
                    "bins": 1,
                    "equipmentIds": {"$ifNull": ["$equipments.equipmentId", []]},
                    "equipmentBackends": {"$ifNull": ["$equipments.backend", []]},
                    "equipmentThresholds": {"$ifNull": ["$equipments.threshold", []]},
                    "block": self.build_block()
                }
            },
            {
                "$addFields": {
                    "entries": entries,
                    "configBackends": "$equipmentBackends" if self.equipment_id is None else ["$block.backend"],
                    "configThresholds": "$equipmentThresholds" if self.equipment_id is None
                    else ["$block.threshold"]
                }
            },
            {"$unwind": "$entries"},
            {"$group": group},
            {
                "$addFields": {
                    "histogram": _sum_arrays("$histograms"),
                    "boxesPerImageHistogram": _sum_arrays("$boxesPerImageHistograms")
                }
            },
            {"$project": {"histograms": 0, "boxesPerImageHistograms": 0}},
            {"$sort": {"_id.date": 1, "_id.kind": 1, "_id.k": 1}}
        ]
