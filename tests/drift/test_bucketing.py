import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mongodb_mcp.drift.config import DriftConfig
from mongodb_mcp.drift.extract import RecordExtractor
from mongodb_mcp.drift.bucketing import (
    bucket_floor, bucket_ceiling, build_buckets, choose_bucket, merge_small_buckets, coarsen
)
from tests.drift.synthetic import generate_rows, cls_prediction

SEOUL = ZoneInfo("Asia/Seoul")


def records_for(days: int, per_day: int, task: str = "cls"):
    rows = generate_rows(task, days, per_day, lambda rng, day: cls_prediction(rng, 0.9, 0.02))
    extractor = RecordExtractor()
    for row in rows:
        extractor.add(row)
    return extractor.records


class TestBoundaries:
    def test_hour_and_day(self):
        moment = datetime(2026, 9, 3, 14, 37, tzinfo=SEOUL)
        assert bucket_floor(moment, "1h") == datetime(2026, 9, 3, 14, tzinfo=SEOUL)
        assert bucket_ceiling(bucket_floor(moment, "1h"), "1h") == datetime(2026, 9, 3, 15, tzinfo=SEOUL)
        assert bucket_floor(moment, "1d") == datetime(2026, 9, 3, tzinfo=SEOUL)
        assert bucket_ceiling(bucket_floor(moment, "1d"), "1d") == datetime(2026, 9, 4, tzinfo=SEOUL)

    def test_week_starts_monday(self):
        thursday = datetime(2026, 9, 3, 14, tzinfo=SEOUL)
        start = bucket_floor(thursday, "1w")
        assert start == datetime(2026, 8, 31, tzinfo=SEOUL) and start.weekday() == 0
        assert bucket_ceiling(start, "1w") == datetime(2026, 9, 7, tzinfo=SEOUL)

    def test_shift_boundaries(self):
        assert bucket_floor(datetime(2026, 9, 3, 14, 30, tzinfo=SEOUL), "shift") == \
            datetime(2026, 9, 3, 14, tzinfo=SEOUL)
        # Before the first shift belongs to the previous day's night shift
        assert bucket_floor(datetime(2026, 9, 3, 3, 0, tzinfo=SEOUL), "shift") == \
            datetime(2026, 9, 2, 22, tzinfo=SEOUL)
        assert bucket_ceiling(datetime(2026, 9, 2, 22, tzinfo=SEOUL), "shift") == \
            datetime(2026, 9, 3, 6, tzinfo=SEOUL)
        assert bucket_ceiling(datetime(2026, 9, 3, 6, tzinfo=SEOUL), "shift") == \
            datetime(2026, 9, 3, 14, tzinfo=SEOUL)

    def test_coarsen_chain(self):
        assert coarsen("1h") == "shift" and coarsen("shift") == "1d" and coarsen("1d") == "1w"
        assert coarsen("1w") is None


class TestBuildAndChoose:
    def test_build_buckets_uses_local_time(self):
        records = records_for(days=2, per_day=24)
        buckets = build_buckets(records, "1d")
        # UTC midnight is 09:00 in Seoul, so two UTC days spill into three local days
        assert len(buckets) == 3
        assert buckets[0].start.tzinfo == SEOUL
        assert sum(bucket.n for bucket in buckets) == len(records)
        assert [bucket.start for bucket in buckets] == sorted(bucket.start for bucket in buckets)

    def test_auto_short_range_is_hourly(self):
        records = records_for(days=1, per_day=24 * 250)
        bucket, warnings = choose_bucket(records, 1.0, "cls")
        assert bucket == "1h" and warnings == []

    def test_auto_two_weeks_is_daily(self):
        records = records_for(days=14, per_day=300)
        assert choose_bucket(records, 14.0, "cls")[0] == "1d"

    def test_auto_coarsens_when_buckets_are_small(self):
        # 450 a day: hourly and shift buckets stay below 200, only one of four local day buckets does
        records = records_for(days=3, per_day=450)
        assert choose_bucket(records, 3.0, "cls")[0] == "1d"

        # Too thin for daily buckets as well, so the rule keeps coarsening
        records = records_for(days=2, per_day=100)
        assert choose_bucket(records, 2.0, "cls")[0] == "1w"

    def test_explicit_bucket_is_capped(self):
        records = records_for(days=14, per_day=300)
        config = DriftConfig(max_buckets=10)
        bucket, warnings = choose_bucket(records, 14.0, "cls", requested="1h", config=config)
        assert bucket == "1w"
        assert len(warnings) == 3


class TestMerge:
    def test_small_buckets_merge_into_neighbours(self):
        records = records_for(days=3, per_day=100)
        raw = build_buckets(records, "1d")
        counts = [bucket.n for bucket in raw]
        assert counts[0] < 100 and counts[-1] < 100

        merged, merged_starts = merge_small_buckets(raw, min_n=100)
        assert sum(bucket.n for bucket in merged) == len(records)
        assert all(bucket.n >= 100 for bucket in merged)
        assert len(merged_starts) == 2
        assert merged[0].start == raw[0].start and merged[-1].end == raw[-1].end
        assert sum(bucket.merged_from for bucket in merged) == len(raw)

    def test_single_bucket_untouched(self):
        records = records_for(days=1, per_day=10)
        merged, merged_starts = merge_small_buckets(build_buckets(records, "1w"), min_n=100)
        assert len(merged) == 1 and merged_starts == []
