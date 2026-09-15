import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent

LIMIT = 15
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CONNECTION_RETRIES = 3
CONNECTION_DELAY = 1.0

DATASET_EXCLUDED_FIELDS = ("dataMap", "lastJob")

# Bucket request value that lets the drift analysis pick the bucket size itself
AUTO_BUCKET = "auto"
# Prefix of the per class share series of the drift analysis, one series per class so that no class is privileged
CLASS_SHARE_PREFIX = "class_share_"


class DBCollections(StrEnum):
    DAILY_MODELS = "dailyModels"
    INSPECTIONS_SUMMARY = "inspectionsSummary"
    DATASETS = "datasets"
    INSPECTIONS = "inspections"


class ModelTasks(StrEnum):
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


class DriftTask(StrEnum):
    """Tasks the drift analysis knows how to extract statistics for"""
    CLASSIFICATION = ModelTasks.CLASSIFICATION.value
    DETECTION = ModelTasks.DETECTION.value


class InspectionMode(StrEnum):
    PRODUCTION = "production"
    REWORK = "rework"
    TEST = "test"


class LabelShape(StrEnum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    LINESTRIP = "linestrip"


class TrainingStatus(StrEnum):
    TERMINATED = "terminated"
    COMPLETED = "completed"
    ERROR = "error"


class BucketSize(StrEnum):
    """Time bucket sizes of the drift analysis, from finest to coarsest; definition order is the coarsening order"""
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"


class DriftDetail(StrEnum):
    FULL = "full"
    COMPACT = "compact"


class DriftMode(StrEnum):
    RANGE = "range"
    COMPARISON = "comparison"


class DriftWindow(StrEnum):
    CURRENT = "current"
    REFERENCE = "reference"


class DriftFlag(StrEnum):
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_BUCKETS = "INSUFFICIENT_BUCKETS"
    HARD_BREAK = "HARD_BREAK"
    CONFIDENCE_SHIFT = "CONFIDENCE_SHIFT"
    CONFIDENCE_SHIFT_MODERATE = "CONFIDENCE_SHIFT_MODERATE"
    CLASS_SHIFT = "CLASS_SHIFT"
    CLASS_SHIFT_MODERATE = "CLASS_SHIFT_MODERATE"
    BOX_COUNT_SHIFT = "BOX_COUNT_SHIFT"
    BOX_GEOMETRY_SHIFT = "BOX_GEOMETRY_SHIFT"
    THRESHOLD_PRESSURE = "THRESHOLD_PRESSURE"
    TRANSIENT_OUTLIER = "TRANSIENT_OUTLIER"
    TREND_CLASS_SHARE = "TREND_CLASS_SHARE"
    TREND_MEDIAN_CONFIDENCE = "TREND_MEDIAN_CONFIDENCE"
    TREND_MEAN_CONFIDENCE = "TREND_MEAN_CONFIDENCE"
    TREND_BELOW_THRESHOLD_RATE = "TREND_BELOW_THRESHOLD_RATE"
    TREND_MEAN_BOXES_PER_IMAGE = "TREND_MEAN_BOXES_PER_IMAGE"
    TREND_NO_BOX_RATE = "TREND_NO_BOX_RATE"
    TREND_MEDIAN_NORMALIZED_AREA = "TREND_MEDIAN_NORMALIZED_AREA"
    TREND_MEDIAN_NORMALIZED_CX = "TREND_MEDIAN_NORMALIZED_CX"
    TREND_MEDIAN_NORMALIZED_CY = "TREND_MEDIAN_NORMALIZED_CY"


class PreVerdict(StrEnum):
    STABLE = "stable"
    SUSPICIOUS = "suspicious"
    DRIFT_LIKELY = "drift_likely"
    UNDETERMINED = "undetermined"


# The pre-verdict a single flag implies on its own; trend flags are counted instead and insufficiency overrides all
FLAG_VERDICT = {
    DriftFlag.HARD_BREAK: PreVerdict.DRIFT_LIKELY,
    DriftFlag.CONFIDENCE_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.CLASS_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.BOX_COUNT_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.BOX_GEOMETRY_SHIFT: PreVerdict.DRIFT_LIKELY,
    DriftFlag.CONFIDENCE_SHIFT_MODERATE: PreVerdict.SUSPICIOUS,
    DriftFlag.CLASS_SHIFT_MODERATE: PreVerdict.SUSPICIOUS,
    DriftFlag.THRESHOLD_PRESSURE: PreVerdict.SUSPICIOUS,
    DriftFlag.TRANSIENT_OUTLIER: PreVerdict.SUSPICIOUS
}


class TrendSeries(StrEnum):
    """Scalar bucket series checked for a monotonic trend; the class share series are added per class at run time"""
    MEDIAN_CONFIDENCE = "median_confidence"
    MEAN_CONFIDENCE = "mean_confidence"
    BELOW_THRESHOLD_RATE = "below_threshold_rate"
    MEAN_BOXES_PER_IMAGE = "mean_boxes_per_image"
    NO_BOX_RATE = "no_box_rate"
    MEDIAN_NORMALIZED_AREA = "median_normalized_area"
    MEDIAN_NORMALIZED_CX = "median_normalized_cx"
    MEDIAN_NORMALIZED_CY = "median_normalized_cy"


class OutlierSeries(StrEnum):
    """Scalar bucket series checked for transient outliers, a subset of the trend series"""
    MEDIAN_CONFIDENCE = "median_confidence"
    BELOW_THRESHOLD_RATE = "below_threshold_rate"
    MEAN_BOXES_PER_IMAGE = "mean_boxes_per_image"
    NO_BOX_RATE = "no_box_rate"


class TrendFamily(StrEnum):
    """What a trend series measures; the pre-verdict counts families, since the series of one family move together"""
    CONFIDENCE = "confidence"
    CLASS_SHARE = "class_share"
    BOX_COUNT = "box_count"
    BOX_GEOMETRY = "box_geometry"


TREND_FLAG_FAMILY = {
    DriftFlag.TREND_MEDIAN_CONFIDENCE: TrendFamily.CONFIDENCE,
    DriftFlag.TREND_MEAN_CONFIDENCE: TrendFamily.CONFIDENCE,
    DriftFlag.TREND_BELOW_THRESHOLD_RATE: TrendFamily.CONFIDENCE,
    DriftFlag.TREND_CLASS_SHARE: TrendFamily.CLASS_SHARE,
    DriftFlag.TREND_MEAN_BOXES_PER_IMAGE: TrendFamily.BOX_COUNT,
    DriftFlag.TREND_NO_BOX_RATE: TrendFamily.BOX_COUNT,
    DriftFlag.TREND_MEDIAN_NORMALIZED_AREA: TrendFamily.BOX_GEOMETRY,
    DriftFlag.TREND_MEDIAN_NORMALIZED_CX: TrendFamily.BOX_GEOMETRY,
    DriftFlag.TREND_MEDIAN_NORMALIZED_CY: TrendFamily.BOX_GEOMETRY
}


class SeriesKind(StrEnum):
    """Decides which practical relevance threshold applies to a move in a series"""
    VALUE = "value"
    RATE = "rate"
    COUNT = "count"
    CLASS_SHARE = "class_share"


class HardBreakKind(StrEnum):
    IMAGE_SPEC = "image_spec"
    THRESHOLD = "threshold"
    BACKEND = "backend"
    CLASSES = "classes"
    ELAPSED_TIME = "elapsed_time"


class BoxGeometry(StrEnum):
    """Normalised box measures compared between the two sides of a split"""
    NORMALIZED_AREA = "normalized_area"
    NORMALIZED_CX = "normalized_cx"
    NORMALIZED_CY = "normalized_cy"
