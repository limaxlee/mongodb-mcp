from typing import Any
from datetime import datetime, timezone


def get_elapsed_days(start_date: datetime | None, end_date: datetime | None) -> float:
    if start_date is None or end_date is None:
        return 0.0
    return (end_date - start_date).total_seconds() / 86400


def convert_to_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def to_utc(value: Any) -> datetime | None:
    """Treats naive datetimes as UTC, which is how MongoDB stores them; anything that is not a datetime becomes None"""
    if not isinstance(value, datetime):
        return None

    return convert_to_utc_datetime(value)
