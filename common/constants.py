import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent

LIMIT = 5
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXCLUDE_SAMPLES = {"samples": 0}

CONNECTION_RETRIES = 3
CONNECTION_DELAY = 1.0


class DBCollections(StrEnum):
    DAILY_MODELS = "dailyModels"
    INSPECTIONS_SUMMARY = "inspectionsSummary"
