import re
from typing import Any
from datetime import datetime
from bson.objectid import ObjectId

from common.constants import DATE_FORMAT


def preprocess_operation(operation: dict[str, Any]) -> dict[str, Any]:
    converted_operation = {}

    for key, value in operation.items():
        if key == "_id" and ObjectId.is_valid(value):
            converted_operation[key] = ObjectId(value)
        elif isinstance(value, str) and _is_valid_datetime(value):
            converted_operation[key] = _convert_to_datetime(value)
        elif isinstance(value, dict):
            converted_operation[key] = preprocess_operation(value)
        elif isinstance(value, list):
            converted_operation[key] = [
                ObjectId(item) if isinstance(item, str) and ObjectId.is_valid(item) else item for item in value
            ]
        else:
            converted_operation[key] = value

    return converted_operation


def process_query_result(result: dict[str, Any]) -> dict[str, Any]:
    converted_result = {}

    for key, value in result.items():
        if isinstance(value, ObjectId):
            converted_result[key] = str(value)
        elif isinstance(value, dict):
            converted_result[key] = process_query_result(value)
        elif isinstance(value, list):
            converted_result[key] = [str(item) if isinstance(item, ObjectId) else item for item in value]
        else:
            converted_result[key] = value

    return converted_result


def _is_valid_datetime(date_string: str, date_format: str = DATE_FORMAT) -> bool:
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False


def _convert_to_datetime(date_string: str, date_format: str = DATE_FORMAT) -> datetime:
    return datetime.strptime(date_string, date_format)


def generate_normalized_regex(value: str) -> dict[str, str]:
    pattern = r"\s*".join(re.escape(item) for item in re.sub(r"\s+", "", value))
    return {"$regex": pattern, "$options": "i"}

