import pytest
from datetime import datetime, timedelta, timezone

from common.config import SETTINGS
from common.constants import Granularity
from mongodb_mcp.drift import DriftWindow
from tests.drift.synthetic import START


class TestDriftWindow:
    def test_naive_dates_become_utc(self):
        window = DriftWindow(datetime(2026, 9, 1), datetime(2026, 9, 8))
        assert window.start_date == START and window.start_date.tzinfo == timezone.utc
        assert window.current.days == 7.0

    def test_validate_order(self):
        with pytest.raises(ValueError, match="before end date"):
            DriftWindow(START, START).validate()
        with pytest.raises(ValueError, match="given together"):
            DriftWindow(START, START + timedelta(days=1), reference_start_date=START - timedelta(days=1)).validate()
        with pytest.raises(ValueError, match="end before current"):
            DriftWindow(
                START, START + timedelta(days=1),
                reference_start_date=START - timedelta(days=1), reference_end_date=START + timedelta(hours=1)
            ).validate()

    def test_validate_total_span(self):
        too_long = START + timedelta(days=SETTINGS.data_drift.max_total_days + 1)
        with pytest.raises(ValueError, match="max time window"):
            DriftWindow(START, too_long).validate()

    def test_comparison_and_total_days(self):
        window = DriftWindow(
            START, START + timedelta(days=7), reference_start_date=START - timedelta(days=3), reference_end_date=START
        )
        window.validate()
        assert window.comparison and window.total_days == 10.0
        assert window.reference.days == 3.0

    @pytest.mark.parametrize("days, expected", [
        (1, Granularity.HOURLY),
        (2.5, Granularity.HOURLY),
        (3, Granularity.SHIFT),
        (30, Granularity.SHIFT),
        (31, Granularity.DAILY),
        (60, Granularity.DAILY),
        (61, Granularity.WEEKLY),
        (400, Granularity.WEEKLY)
    ])
    def test_auto_granularity_keeps_periods_within_max(self, days, expected):
        window = DriftWindow(START, START + timedelta(days=days))
        assert window.resolve_granularity("auto") == expected

    def test_auto_granularity_counts_both_windows(self):
        window = DriftWindow(
            START, START + timedelta(days=2), reference_start_date=START - timedelta(days=2), reference_end_date=START
        )
        assert window.resolve_granularity("auto") == Granularity.SHIFT

    def test_explicit_granularity(self):
        window = DriftWindow(START, START + timedelta(days=100))
        assert window.resolve_granularity("hourly") == Granularity.HOURLY
        with pytest.raises(ValueError, match="Unsupported granularity"):
            window.resolve_granularity("1d")
