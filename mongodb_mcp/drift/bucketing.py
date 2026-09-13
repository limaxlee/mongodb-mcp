"""Cuts the ordered record stream into time buckets on factory local time"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from common.constants import ModelTasks
from mongodb_mcp.drift.config import DriftConfig, DEFAULT_CONFIG
from mongodb_mcp.drift.extract import Record

BUCKET_ORDER = ("1h", "shift", "1d", "1w")


@dataclass
class Bucket:
    start: datetime
    end: datetime
    records: list[Record] = field(default_factory=list)
    merged_from: int = 1
    window: str = "current"

    @property
    def n(self) -> int:
        return len(self.records)


def bucket_floor(local_dt: datetime, bucket: str, config: DriftConfig = DEFAULT_CONFIG) -> datetime:
    base = local_dt.replace(minute=0, second=0, microsecond=0)
    if bucket == "1h":
        return base

    midnight = base.replace(hour=0)
    if bucket == "1d":
        return midnight
    if bucket == "1w":
        return midnight - timedelta(days=midnight.weekday())
    if bucket == "shift":
        hours = sorted(config.shift_hours)
        previous = [hour for hour in hours if hour <= base.hour]
        if previous:
            return midnight.replace(hour=previous[-1])
        # Before the first shift of the day, so it still belongs to the last shift of the previous day
        return (midnight - timedelta(days=1)).replace(hour=hours[-1])

    raise ValueError(f"Unknown bucket {bucket!r}")


def bucket_ceiling(start: datetime, bucket: str, config: DriftConfig = DEFAULT_CONFIG) -> datetime:
    if bucket == "1h":
        return start + timedelta(hours=1)
    if bucket == "1d":
        return start + timedelta(days=1)
    if bucket == "1w":
        return start + timedelta(days=7)
    if bucket == "shift":
        hours = sorted(config.shift_hours)
        following = [hour for hour in hours if hour > start.hour]
        if following:
            return start.replace(hour=following[0])
        return (start + timedelta(days=1)).replace(hour=hours[0])

    raise ValueError(f"Unknown bucket {bucket!r}")


def build_buckets(
        records: list[Record],
        bucket: str,
        window: str = "current",
        config: DriftConfig = DEFAULT_CONFIG
) -> list[Bucket]:
    """Groups records by the local wall-clock bucket they fall in, chronologically ordered"""
    grouped: dict[datetime, Bucket] = {}
    for record in records:
        start = bucket_floor(record.local_created_at, bucket, config)
        key = start.replace(tzinfo=None)
        if key not in grouped:
            grouped[key] = Bucket(start=start, end=bucket_ceiling(start, bucket, config), window=window)
        grouped[key].records.append(record)

    return [grouped[key] for key in sorted(grouped)]


def min_bucket_records(task: str, config: DriftConfig = DEFAULT_CONFIG) -> int:
    if task == ModelTasks.CLASSIFICATION.value:
        return config.min_bucket_records_cls
    return config.min_bucket_records_det


def coarsen(bucket: str) -> str | None:
    index = BUCKET_ORDER.index(bucket)
    return BUCKET_ORDER[index + 1] if index + 1 < len(BUCKET_ORDER) else None


def choose_bucket(
        records: list[Record],
        days: float,
        task: str,
        requested: str = "auto",
        config: DriftConfig = DEFAULT_CONFIG
) -> tuple[str, list[str]]:
    """Resolves the bucket size, either the requested one or the automatic rule, returns it with any warnings"""
    warnings: list[str] = []
    min_n = min_bucket_records(task, config)

    if requested != "auto":
        bucket = requested
    else:
        bucket = "1d"
        if days <= 2:
            bucket = "1h"
        elif days > 60:
            bucket = "1w"

        while True:
            buckets = build_buckets(records, bucket, config=config)
            if not buckets:
                break
            small = sum(1 for item in buckets if item.n < min_n)
            next_bucket = coarsen(bucket)
            if small / len(buckets) > config.small_bucket_share and next_bucket:
                bucket = next_bucket
                continue
            break

    # Never emit more buckets than the result can reasonably carry
    while len(build_buckets(records, bucket, config=config)) > config.max_buckets:
        next_bucket = coarsen(bucket)
        if not next_bucket:
            break
        warnings.append(f"Bucket {bucket} would exceed {config.max_buckets} buckets, coarsened to {next_bucket}")
        bucket = next_bucket

    return bucket, warnings


def merge_small_buckets(buckets: list[Bucket], min_n: int) -> tuple[list[Bucket], list[datetime]]:
    """Merges every bucket below the minimum into its next neighbour (previous for the last one)"""
    merged_starts: list[datetime] = []
    items = [Bucket(b.start, b.end, list(b.records), b.merged_from, b.window) for b in buckets]

    index = 0
    while index < len(items) and len(items) > 1:
        current = items[index]
        if current.n >= min_n:
            index += 1
            continue

        merged_starts.append(current.start)
        if index + 1 < len(items):
            target = items[index + 1]
            target.records = current.records + target.records
            target.start = current.start
            target.merged_from += current.merged_from
            items.pop(index)
        else:
            target = items[index - 1]
            target.records = target.records + current.records
            target.end = current.end
            target.merged_from += current.merged_from
            items.pop(index)
            index -= 1

    return items, merged_starts
