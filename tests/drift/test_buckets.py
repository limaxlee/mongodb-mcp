from datetime import datetime, timedelta, timezone

from common.config import SETTINGS
from common.constants import BucketSize, DriftWindowMode
from mongodb_mcp.drift import RecordExtractor
from mongodb_mcp.drift.buckets import BucketBuilder
from tests.drift.synthetic import generate_rows, cls_prediction, det_prediction, START


def records_for(days: int, per_day: int, task: str = "cls", first_day: int = 0):
    start, first_index = START + timedelta(days=first_day), first_day * per_day
    if task == "det":
        rows = generate_rows(task, days, per_day, lambda rng, day: det_prediction(rng, 0.9, 0.02), start=start, first_index=first_index)
    else:
        rows = generate_rows(task, days, per_day, lambda rng, day: cls_prediction(rng, 0.9, 0.02), start=start, first_index=first_index)
    extractor = RecordExtractor()
    for row in rows:
        extractor.add_record(row)
    return extractor.records


class TestBoundaries:
    def test_hour_and_day(self):
        moment = datetime(2026, 9, 3, 14, 37, tzinfo=timezone.utc)
        assert BucketBuilder.floor(moment, BucketSize.HOUR) == datetime(2026, 9, 3, 14, tzinfo=timezone.utc)
        assert BucketBuilder.ceiling(BucketBuilder.floor(moment, BucketSize.HOUR), BucketSize.HOUR) == \
            datetime(2026, 9, 3, 15, tzinfo=timezone.utc)
        assert BucketBuilder.floor(moment, BucketSize.DAY) == datetime(2026, 9, 3, tzinfo=timezone.utc)
        assert BucketBuilder.ceiling(BucketBuilder.floor(moment, BucketSize.DAY), BucketSize.DAY) == \
            datetime(2026, 9, 4, tzinfo=timezone.utc)

    def test_week_starts_monday(self):
        thursday = datetime(2026, 9, 3, 14, tzinfo=timezone.utc)
        start = BucketBuilder.floor(thursday, BucketSize.WEEK)
        assert start == datetime(2026, 8, 31, tzinfo=timezone.utc) and start.weekday() == 0
        assert BucketBuilder.ceiling(start, BucketSize.WEEK) == datetime(2026, 9, 7, tzinfo=timezone.utc)

    def test_coarsen_chain(self):
        assert BucketBuilder.coarsen(BucketSize.HOUR) == BucketSize.DAY
        assert BucketBuilder.coarsen(BucketSize.DAY) == BucketSize.WEEK
        assert BucketBuilder.coarsen(BucketSize.WEEK) is None
        assert BucketBuilder.coarsen("1h") == BucketSize.DAY

    def test_minimum_records_follow_the_task(self):
        assert BucketBuilder("cls").min_records == SETTINGS.data_drift.min_bucket_records_cls
        assert BucketBuilder("det").min_records == SETTINGS.data_drift.min_bucket_records_det


class TestBuildAndChoose:
    def test_build_buckets_by_utc_day(self):
        records = records_for(days=2, per_day=24)
        buckets = BucketBuilder("cls").build(records, BucketSize.DAY, DriftWindowMode.REFERENCE)

        assert len(buckets) == 2
        assert sum(bucket.record_count for bucket in buckets) == len(records)
        assert [bucket.start_date for bucket in buckets] == sorted(bucket.start_date for bucket in buckets)
        assert all(bucket.window == DriftWindowMode.REFERENCE and bucket.merged_from == 1 for bucket in buckets)
        assert buckets[0].end_date == buckets[1].start_date

    def test_build_empty(self):
        assert BucketBuilder("cls").build([], BucketSize.DAY) == []

    def test_auto_short_range_is_hourly(self):
        builder = BucketBuilder("cls")
        assert builder.choose(records_for(days=1, per_day=24 * 250), 1.0) == BucketSize.HOUR

    def test_auto_two_weeks_is_daily(self):
        assert BucketBuilder("cls").choose(records_for(days=14, per_day=300), 14.0) == BucketSize.DAY

    def test_auto_coarsens_when_buckets_are_small(self):
        # 450 a day: hourly buckets stay below 200, the two day buckets do not
        assert BucketBuilder("cls").choose(records_for(days=2, per_day=450), 2.0) == BucketSize.DAY

        # Too thin for daily buckets as well, so the rule keeps coarsening
        assert BucketBuilder("cls").choose(records_for(days=2, per_day=100), 2.0) == BucketSize.WEEK

    def test_explicit_bucket_is_capped(self, mocker):
        builder = BucketBuilder("cls")
        mocker.patch.object(builder, "config", SETTINGS.data_drift.model_copy(update={"max_buckets": 10}))

        assert builder.choose(records_for(days=14, per_day=300), 14.0, requested="1h") == BucketSize.WEEK

    def test_explicit_bucket_is_kept(self):
        builder = BucketBuilder("cls")
        assert builder.choose(records_for(days=3, per_day=10), 3.0, requested="1d") == BucketSize.DAY


class TestMerge:
    def test_small_buckets_merge_into_neighbours(self):
        builder = BucketBuilder("det")
        records = records_for(days=3, per_day=60, task="det")
        raw = builder.build(records, BucketSize.DAY)
        assert all(bucket.record_count < builder.min_records for bucket in raw)

        merged = builder.merge_small(raw)
        assert sum(bucket.record_count for bucket in merged) == len(records)
        assert all(bucket.record_count >= builder.min_records for bucket in merged)
        assert len(builder.merged_start_dates) == 2
        assert merged[0].start_date == raw[0].start_date and merged[-1].end_date == raw[-1].end_date
        assert sum(bucket.merged_from for bucket in merged) == len(raw)

    def test_last_small_bucket_merges_backwards(self):
        builder = BucketBuilder("det")
        records = records_for(days=3, per_day=120, task="det")
        raw = builder.build(records, BucketSize.DAY)
        raw[-1].records = raw[-1].records[:30]

        merged = builder.merge_small(raw)
        assert len(merged) == 2
        assert merged[-1].record_count == 150 and merged[-1].merged_from == 2
        assert merged[-1].end_date == raw[-1].end_date
        assert builder.merged_start_dates == [raw[-1].start_date]

    def test_merge_leaves_input_untouched(self):
        builder = BucketBuilder("det")
        raw = builder.build(records_for(days=3, per_day=60, task="det"), BucketSize.DAY)
        builder.merge_small(raw)
        assert [bucket.record_count for bucket in raw] == [60, 60, 60]

    def test_single_bucket_untouched(self):
        builder = BucketBuilder("cls")
        merged = builder.merge_small(builder.build(records_for(days=1, per_day=10), BucketSize.WEEK))
        assert len(merged) == 1 and builder.merged_start_dates == []


class TestBudget:
    def test_thin_is_even_and_deterministic(self):
        records = records_for(days=1, per_day=10)
        kept = BucketBuilder.thin(records, 3)

        assert kept == [records[0], records[3], records[6]]
        assert BucketBuilder.thin(records, 3) == kept
        assert BucketBuilder.thin(records, 10) == records
        assert BucketBuilder.thin(records, 25) == records
        assert BucketBuilder.thin(records, 0) == []

    def test_zero_budget_disables_thinning(self):
        builder = BucketBuilder("cls")
        buckets = builder.build(records_for(days=3, per_day=300), BucketSize.DAY)

        assert builder.apply_budget(buckets, 0) is buckets

    def test_budget_above_total_leaves_buckets_alone(self):
        builder = BucketBuilder("cls")
        buckets = builder.build(records_for(days=3, per_day=300), BucketSize.DAY)

        assert builder.apply_budget(buckets, 900) is buckets
        assert builder.apply_budget(buckets, 10_000) is buckets

    def test_budget_is_shared_in_proportion(self):
        builder = BucketBuilder("cls")
        records = records_for(days=2, per_day=600) + records_for(days=1, per_day=1200, first_day=2)
        buckets = builder.build(records, BucketSize.DAY)
        thinned = builder.apply_budget(buckets, 1200)

        assert [item.record_count for item in thinned] == [300, 300, 600]
        assert sum(item.record_count for item in thinned) <= 1200
        assert [item.start_date for item in thinned] == [item.start_date for item in buckets]
        assert [item.record_count for item in buckets] == [600, 600, 1200]
        for before, after in zip(buckets, thinned):
            assert after.records == BucketBuilder.thin(before.records, after.record_count)
            assert [record.created_at for record in after.records] == sorted(record.created_at for record in after.records)

    def test_bucket_never_drops_below_minimum(self):
        builder = BucketBuilder("cls")
        minimum = builder.min_records
        buckets = builder.build(records_for(days=4, per_day=minimum * 2), BucketSize.DAY)
        thinned = builder.apply_budget(buckets, minimum)

        assert all(item.record_count == minimum for item in thinned)

    def test_small_bucket_keeps_everything(self):
        builder = BucketBuilder("cls")
        records = records_for(days=1, per_day=50) + records_for(days=3, per_day=2000, first_day=1)
        buckets = builder.build(records, BucketSize.DAY)
        thinned = builder.apply_budget(buckets, 1000)

        assert thinned[0].record_count == buckets[0].record_count < builder.min_records
