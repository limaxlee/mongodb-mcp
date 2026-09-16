from datetime import datetime, timedelta

from common.config import SETTINGS
from common.constants import AUTO_BUCKET, BucketSize, DriftWindowMode, DriftTask
from mongodb_mcp.schemas import Record, Bucket


class BucketBuilder:
    def __init__(self, task: str):
        self.config = SETTINGS.data_drift
        self.min_records = self.config.min_bucket_records_det if task == DriftTask.DETECTION \
            else self.config.min_bucket_records_cls
        self.merged_start_dates: list[datetime] = []

    @staticmethod
    def floor(moment: datetime, bucket: BucketSize) -> datetime:
        base = moment.replace(minute=0, second=0, microsecond=0)
        if bucket == BucketSize.HOUR:
            return base

        midnight = base.replace(hour=0)
        if bucket == BucketSize.DAY:
            return midnight
        if bucket == BucketSize.WEEK:
            return midnight - timedelta(days=midnight.weekday())

        raise ValueError(f"Unknown bucket {bucket}")

    @staticmethod
    def ceiling(start: datetime, bucket: BucketSize) -> datetime:
        if bucket == BucketSize.HOUR:
            return start + timedelta(hours=1)
        if bucket == BucketSize.DAY:
            return start + timedelta(days=1)
        if bucket == BucketSize.WEEK:
            return start + timedelta(days=7)

        raise ValueError(f"Unknown bucket {bucket}")

    @staticmethod
    def coarsen(bucket: BucketSize) -> BucketSize | None:
        sizes = list(BucketSize)
        index = sizes.index(BucketSize(bucket))
        return sizes[index + 1] if index + 1 < len(sizes) else None

    def build(
            self,
            records: list[Record],
            bucket: BucketSize,
            window: DriftWindowMode = DriftWindowMode.CURRENT
    ) -> list[Bucket]:
        grouped = {}
        for record in records:
            start = self.floor(record.created_at, bucket)
            if start not in grouped:
                grouped[start] = Bucket(start_date=start, end_date=self.ceiling(start, bucket), window=window)
            grouped[start].records.append(record)

        return [grouped[key] for key in sorted(grouped)]

    def choose(self, records: list[Record], days: float, requested: str = AUTO_BUCKET) -> BucketSize:
        if requested != AUTO_BUCKET:
            bucket = BucketSize(requested)
        else:
            bucket = BucketSize.DAY
            if days <= 2:
                bucket = BucketSize.HOUR
            elif days > 60:
                bucket = BucketSize.WEEK

            while True:
                buckets = self.build(records, bucket)
                if not buckets:
                    break
                small = sum(1 for item in buckets if item.record_count < self.min_records)
                next_bucket = self.coarsen(bucket)
                if small / len(buckets) > self.config.small_bucket_share and next_bucket:
                    bucket = next_bucket
                    continue
                break

        while len(self.build(records, bucket)) > self.config.max_buckets:
            next_bucket = self.coarsen(bucket)
            if not next_bucket:
                break
            bucket = next_bucket

        return bucket

    def merge_small(self, buckets: list[Bucket]) -> list[Bucket]:
        items = [item.model_copy(update={"records": list(item.records)}) for item in buckets]

        index = 0
        while index < len(items) and len(items) > 1:
            current = items[index]
            if current.record_count >= self.min_records:
                index += 1
                continue

            self.merged_start_dates.append(current.start_date)
            if index + 1 < len(items):
                target = items[index + 1]
                target.records = current.records + target.records
                target.start_date = current.start_date
                target.merged_from += current.merged_from
                items.pop(index)
            else:
                target = items[index - 1]
                target.records = target.records + current.records
                target.end_date = current.end_date
                target.merged_from += current.merged_from
                items.pop(index)
                index -= 1

        return items
