from datetime import datetime

from common.config import SETTINGS
from common.constants import AUTO_GRANULARITY, Granularity, GRANULARITY_HOURS
from mongodb_mcp.utils import get_elapsed_days, convert_to_utc_datetime
from mongodb_mcp.schemas import DateRange


class DriftWindow:
    def __init__(
            self,
            start_date: datetime,
            end_date: datetime,
            reference_start_date: datetime | None = None,
            reference_end_date: datetime | None = None
    ):
        self.start_date = convert_to_utc_datetime(start_date)
        self.end_date = convert_to_utc_datetime(end_date)
        self.reference_start_date = convert_to_utc_datetime(reference_start_date) if reference_start_date else None
        self.reference_end_date = convert_to_utc_datetime(reference_end_date) if reference_end_date else None

    @property
    def comparison(self) -> bool:
        return self.reference_start_date is not None

    @property
    def current(self) -> DateRange:
        return DateRange(
            start_date=self.start_date,
            end_date=self.end_date,
            days=get_elapsed_days(self.start_date, self.end_date)
        )

    @property
    def reference(self) -> DateRange | None:
        if not self.comparison:
            return None

        return DateRange(
            start_date=self.reference_start_date,
            end_date=self.reference_end_date,
            days=get_elapsed_days(self.reference_start_date, self.reference_end_date)
        )

    @property
    def total_days(self) -> float:
        reference_days = get_elapsed_days(self.reference_start_date, self.reference_end_date)
        current_days = get_elapsed_days(self.start_date, self.end_date)

        return current_days + reference_days

    def validate(self):
        if self.start_date >= self.end_date:
            raise ValueError(f"Start date {self.start_date} for drift analysis must be before end date {self.end_date}")
        if (self.reference_start_date is None) != (self.reference_end_date is None):
            raise ValueError("Reference start date and end date for drift analysis must be given together")
        if self.reference_start_date is not None:
            if self.reference_start_date >= self.reference_end_date:
                raise ValueError("Reference start date for drift analysis must be before reference end date")
            if self.reference_end_date > self.start_date:
                raise ValueError("The reference window for drift analysis must end before current window starts")

        total_days = self.total_days
        if total_days > SETTINGS.data_drift.max_total_days:
            raise ValueError(f"Drift covers {total_days} days: max time window is {SETTINGS.data_drift.max_total_days}")

    def resolve_granularity(self, requested: str) -> Granularity:
        """The requested granularity, or for auto the finest one that keeps both windows within max_periods periods"""
        if requested != AUTO_GRANULARITY:
            if requested not in list(Granularity):
                raise ValueError(
                    f"Unsupported granularity {requested} for drift analysis: "
                    f"supported values are {AUTO_GRANULARITY}, {', '.join(Granularity)}"
                )
            return Granularity(requested)

        total_hours = self.total_days * 24
        for granularity in Granularity:
            if total_hours / GRANULARITY_HOURS[granularity] <= SETTINGS.data_drift.max_periods:
                return granularity

        return Granularity.WEEKLY
