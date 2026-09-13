import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent

LIMIT = 15
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CONNECTION_RETRIES = 3
CONNECTION_DELAY = 1.0

DATASET_EXCLUDED_FIELDS = ("dataMap", "lastJob")


class DBCollections(StrEnum):
    DAILY_MODELS = "dailyModels"
    INSPECTIONS_SUMMARY = "inspectionsSummary"
    DATASETS = "datasets"
    INSPECTIONS = "inspections"


class ModelTasks(StrEnum):
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


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
