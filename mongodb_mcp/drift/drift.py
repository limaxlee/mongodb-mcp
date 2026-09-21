from typing import Any, Sequence
from datetime import datetime, timezone

from common.config import SETTINGS
from common.constants import DriftDetail, DriftMode, DriftWindowMode, DriftTask, HardBreakKind, Granularity
from mongodb_mcp.schemas import PeriodCounts, DriftAnalysisResult, DriftFilters
from mongodb_mcp.drift.window import DriftWindow
from mongodb_mcp.drift.period import PeriodSummarizer, sum_periods
from mongodb_mcp.drift.splits import SplitFinder
from mongodb_mcp.drift.rules import DriftRules


class DriftAnalyzer:
    """Pure arithmetic over the per-period counts read from the statistics collection"""

    def __init__(
            self,
            model_name: str,
            model_version: str,
            task: str,
            mode: str | None,
            periods: Sequence[PeriodCounts],
            windows: DriftWindow,
            filters: DriftFilters,
            granularity: Granularity,
            detail: str = DriftDetail.FULL
    ):
        self.config = SETTINGS.data_drift
        self.model_name = model_name
        self.model_version = model_version
        self.task = DriftTask(task)
        self.mode = mode
        self.windows = windows
        self.filters = filters
        self.granularity = granularity
        self.detail = DriftDetail(detail)

        ordered = sorted(periods, key=lambda item: (item.window != DriftWindowMode.REFERENCE, item.start_date))
        self.missing_equipment = [item.start_date for item in ordered if item.document_count == 0]
        self.periods = [item for item in ordered if item.document_count > 0]
        self.current = [item for item in self.periods if item.window == DriftWindowMode.CURRENT]
        self.reference = [item for item in self.periods if item.window == DriftWindowMode.REFERENCE]

        self.classes = self._classes(self.periods)
        self.edge_sets = self._edge_sets(self.periods)
        self.incompatible_bins = len(self.edge_sets) > 1
        self.bins = next((item for period in self.periods for item in period.bins), {})

        self.summarizer = PeriodSummarizer(task, self.classes, self.edge_sets[0] if self.edge_sets else None, detail)
        self.splits = SplitFinder(self.summarizer)
        self.rules = DriftRules()

    def run(self) -> DriftAnalysisResult:
        config = self.config
        comparison_mode = self.windows.comparison
        now = datetime.now(timezone.utc)

        summaries = [self.summarizer.period(item, now) for item in self.periods]
        consecutive = self._consecutive(self.periods)

        change_point = None
        comparison = None
        if comparison_mode:
            if self.current and self.reference and not self.incompatible_bins:
                comparison = self.splits.compare(self.reference, self.current, self.windows.start_date)
            insufficient_data = comparison is None or not comparison["sides_sufficient"]
            insufficient_periods = False
            split = comparison
        else:
            insufficient_periods = len(self.current) < config.min_periods_changepoint
            if not insufficient_periods and not self.incompatible_bins:
                change_point = self.splits.primary(self.current)
                insufficient_periods = change_point is None
            total_predictions = sum(item.prediction_count for item in self.current)
            insufficient_data = total_predictions < self.splits.min_side_records or (
                change_point is not None and not change_point["sides_sufficient"]
            )
            split = change_point

        hard_breaks = self._hard_breaks(self.periods)
        flags = self.rules.flags(
            split, hard_breaks, insufficient_data, insufficient_periods, self.incompatible_bins
        )

        everything = sum_periods(self.periods) if self.periods else None
        reference_range = self.windows.reference

        result = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "mode": self.mode,
            "analysis_mode": DriftMode.COMPARISON if comparison_mode else DriftMode.RANGE,
            "filters": self.filters,
            "sites_seen": {
                "gbms": everything.gbms if everything else [],
                "processes": everything.processes if everything else [],
                "modes": everything.modes if everything else [],
                "equipment_ids": [self.filters.equipment_id] if self.filters.equipment_id
                else (everything.equipment_ids if everything else [])
            },
            "granularity": self.granularity,
            "detail": self.detail,
            "range": self.windows.current.model_dump(),
            "reference_range": reference_range.model_dump() if reference_range else None,
            "status": {
                "analysis_possible": (change_point is not None and change_point["sides_sufficient"])
                or (comparison is not None and comparison["sides_sufficient"]),
                "split_ran": change_point is not None,
                "comparison_ran": comparison is not None,
                "period_count": len(self.current),
                "reference_period_count": len(self.reference)
            },
            "data_quality": {
                "document_count": everything.document_count if everything else 0,
                "product_count": len(everything.product_ids) if everything else 0,
                "inspection_count": everything.inspection_count if everything else 0,
                "prediction_count": everything.prediction_count if everything else 0,
                "box_count": everything.box_count if everything and self.summarizer.detection else None,
                "missing_confidence_count": everything.missing_confidence_count if everything else 0,
                "parse_error_count": everything.parse_error_count if everything else 0,
                "partial_periods": [item.start_date for item in summaries if item.partial],
                "periods_missing_equipment": self.missing_equipment,
                "incompatible_bins": self.incompatible_bins
            },
            "classes": self.classes,
            "bins": {
                "confidence_edges": self.bins.get("confidenceEdges") or [],
                "near_threshold_margin": self.bins.get("nearThresholdMargin"),
                "boxes_per_image_max": self.bins.get("boxesPerImageMax")
            },
            "periods": summaries,
            "consecutive": consecutive,
            "change_point": change_point,
            "comparison": comparison,
            "hard_breaks": hard_breaks,
            "flags": flags,
            "pre_verdict": self.rules.pre_verdict(flags),
            "config": config.model_dump()
        }

        return DriftAnalysisResult.model_validate(self._round(result, config.decimals))

    @staticmethod
    def _classes(periods: Sequence[PeriodCounts]) -> list[str]:
        classes: list[str] = []
        for period in periods:
            for name in list(period.classes) + list(period.class_counts) + list(period.per_class):
                if name not in classes:
                    classes.append(name)
        return classes

    @staticmethod
    def _edge_sets(periods: Sequence[PeriodCounts]) -> list[list[float]]:
        edge_sets: list[list[float]] = []
        for period in periods:
            for item in period.bins:
                edges = item.get("confidenceEdges")
                if isinstance(edges, list) and edges and edges not in edge_sets:
                    edge_sets.append(list(edges))
        return edge_sets

    def _consecutive(self, periods: Sequence[PeriodCounts]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for index in range(1, len(periods)):
            before, after = periods[index - 1], periods[index]
            same_edges = before.confidence_edges == after.confidence_edges
            reports.append({
                "period_index": index,
                "date": after.start_date,
                "psi_confidence": self.splits.psi_confidence(before.confidence, after.confidence)
                if same_edges else None,
                "psi_class": self.splits.psi_class(before, after),
                "psi_boxes_per_image": self.splits.psi_boxes_per_image(before, after),
                "sufficient": self.splits.sides_sufficient(before, after)
            })
        return reports

    def _hard_breaks(self, periods: Sequence[PeriodCounts]) -> list[dict[str, Any]]:
        """Backend, threshold and class list compared between consecutive periods

        Several values inside one period are reported as a break to the list of values, dated at that period.
        """
        breaks: list[dict[str, Any]] = []
        last: dict[tuple[str, str | None], Any] = {}

        for period in periods:
            observed: list[tuple[HardBreakKind, str | None, Any]] = [
                (HardBreakKind.BACKEND, None, self._single_or_list(period.backends)),
                (HardBreakKind.CLASSES, None, list(period.classes) or None)
            ]
            observed.extend(
                (HardBreakKind.THRESHOLD, name, self._single_or_list(counts.thresholds))
                for name, counts in period.per_class.items()
            )

            for kind, class_name, value in observed:
                if value is None:
                    continue
                key = (kind, class_name)
                previous = last.get(key)
                if (key in last and previous != value) or (key not in last and isinstance(value, list)
                                                            and kind != HardBreakKind.CLASSES):
                    breaks.append({
                        "kind": kind,
                        "class_name": class_name,
                        "date": period.start_date,
                        "from": previous,
                        "to": value
                    })
                last[key] = value

        return breaks

    @staticmethod
    def _single_or_list(values: Sequence[Any]) -> Any:
        if not values:
            return None
        return values[0] if len(values) == 1 else list(values)

    @classmethod
    def _round(cls, value: Any, decimals: int) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, decimals)
        if isinstance(value, dict):
            return {key: cls._round(item, decimals) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._round(item, decimals) for item in value]
        if hasattr(value, "model_dump"):
            return cls._round(value.model_dump(by_alias=False), decimals)
        return value
