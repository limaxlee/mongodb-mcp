import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent

LIMIT = 5
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CONNECTION_RETRIES = 3
CONNECTION_DELAY = 1.0

AUTO_GRANULARITY = "auto"


class DBCollections(StrEnum):
    DAILY_MODELS = "dailyModels"
    DATASETS = "datasets"
    INSPECTIONS = "inspections"
    INSPECTIONS_STATISTICS = "inspectionStatistics"


class ModelTasks(StrEnum):
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


class Granularity(StrEnum):
    """Period type of an inspection statistics document, every boundary is UTC"""
    HOURLY = "hourly"
    SHIFT = "shift"
    DAILY = "daily"
    WEEKLY = "weekly"


GRANULARITY_HOURS = {
    Granularity.HOURLY: 1,
    Granularity.SHIFT: 12,
    Granularity.DAILY: 24,
    Granularity.WEEKLY: 168
}


class DriftTask(StrEnum):
    CLASSIFICATION = ModelTasks.CLASSIFICATION.value
    DETECTION = ModelTasks.DETECTION.value


class DriftDetail(StrEnum):
    FULL = "full"
    COMPACT = "compact"


class DriftMode(StrEnum):
    RANGE = "range"
    COMPARISON = "comparison"


class DriftWindowMode(StrEnum):
    CURRENT = "current"
    REFERENCE = "reference"


class MedianSource(StrEnum):
    EXACT = "exact"
    HISTOGRAM = "histogram"


class DriftFlag(StrEnum):
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_PERIODS = "INSUFFICIENT_PERIODS"
    INCOMPATIBLE_BINS = "INCOMPATIBLE_BINS"
    HARD_BREAK = "HARD_BREAK"
    CONFIDENCE_SHIFT = "CONFIDENCE_SHIFT"
    CONFIDENCE_SHIFT_MODERATE = "CONFIDENCE_SHIFT_MODERATE"
    CLASS_SHIFT = "CLASS_SHIFT"
    CLASS_SHIFT_MODERATE = "CLASS_SHIFT_MODERATE"
    BOX_COUNT_SHIFT = "BOX_COUNT_SHIFT"
    THRESHOLD_PRESSURE = "THRESHOLD_PRESSURE"


class PreVerdict(StrEnum):
    STABLE = "stable"
    SUSPICIOUS = "suspicious"
    DRIFT_LIKELY = "drift_likely"
    UNDETERMINED = "undetermined"


FLAG_VERDICT = {
    DriftFlag.CONFIDENCE_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.CLASS_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.BOX_COUNT_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.CONFIDENCE_SHIFT_MODERATE: PreVerdict.SUSPICIOUS,
    DriftFlag.CLASS_SHIFT_MODERATE: PreVerdict.SUSPICIOUS,
    DriftFlag.THRESHOLD_PRESSURE: PreVerdict.SUSPICIOUS,
    DriftFlag.HARD_BREAK: PreVerdict.SUSPICIOUS
}

INSUFFICIENCY_FLAGS = (DriftFlag.INSUFFICIENT_DATA, DriftFlag.INSUFFICIENT_PERIODS, DriftFlag.INCOMPATIBLE_BINS)


class HardBreakKind(StrEnum):
    BACKEND = "backend"
    THRESHOLD = "threshold"
    CLASSES = "classes"
