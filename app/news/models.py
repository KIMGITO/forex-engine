"""Typed models for the news & economic-events layer.

Look-ahead discipline: every event carries explicit ``scheduled_at`` and
``released_at`` timestamps. ``available_from`` is the earliest a consumer may
legally use the event's ``actual`` value. A simulated strategy must never see
``actual`` before ``released_at``.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EventImportance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class EventStatus(str, Enum):
    PAST = "past"
    UPCOMING = "upcoming"
    LIVE = "live"
    RELEASED = "released"


class EventCategory(str, Enum):
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    INTEREST_RATE = "interest_rate"
    GDP = "gdp"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    CONSUMER = "consumer"
    HOUSING = "housing"
    TRADE = "trade"
    CENTRAL_BANK = "central_bank"
    SPEECH = "speech"
    OTHER = "other"


class EconomicEvent(BaseModel):
    """A provider-independent scheduled or released economic event.

    Only ``event_id``, ``scheduled_at``, ``country``, ``event_name``, and
    ``importance`` are required. All other fields are optional because not
    every provider supplies every value.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    scheduled_at: datetime = Field(..., description="Scheduled time of the event (tz-aware)")
    timezone: str = Field(default="UTC", description="Original timezone of the event")
    country: str = Field(..., description="Country code, e.g. US, DE, JP")
    currency: str = Field(..., description="Primary currency, e.g. USD")
    affected_currencies: list[str] = Field(
        default_factory=list, description="All currencies potentially affected"
    )
    event_name: str
    category: EventCategory = Field(default=EventCategory.OTHER)
    importance: EventImportance
    actual: float | None = Field(default=None)
    forecast: float | None = Field(default=None)
    previous: float | None = Field(default=None)
    unit: str | None = Field(default=None, description="e.g. %, B, K, M")
    source: str | None = Field(default=None)
    url: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    provider_importance: str | None = Field(
        default=None, description="Raw provider importance label preserved as metadata"
    )
    received_at: datetime | None = Field(
        default=None, description="When the record was received by the system (tz-aware)"
    )
    released_at: datetime | None = Field(
        default=None,
        description="When the actual value was released. If None, actual is not yet available.",
    )
    available_from: datetime | None = Field(
        default=None,
        description="Earliest a consumer may legally use this event. Defaults to "
        "released_at (if released) else scheduled_at.",
    )

    def status(self, now: datetime) -> EventStatus:
        """Classify the event status purely from timestamps, never the name."""
        if self.released_at is not None and self.released_at <= now:
            return EventStatus.RELEASED
        if self.scheduled_at <= now:
            return EventStatus.PAST
        return EventStatus.UPCOMING

    def effective_available_from(self) -> datetime:
        """Resolve the availability timestamp (never None)."""
        if self.available_from is not None:
            return self.available_from
        return self.released_at if self.released_at is not None else self.scheduled_at


class SurpriseResult(BaseModel):
    """Objective actual-vs-forecast surprise. No directional market claim."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    actual: float
    forecast: float | None
    surprise: float | None = Field(default=None, description="actual - forecast")
    surprise_pct: float | None = Field(
        default=None, description="(actual - forecast) / |forecast| * 100 when forecast != 0"
    )


class RiskWindow(BaseModel):
    """A configurable time window around an event."""

    model_config = ConfigDict(frozen=True)

    event: EconomicEvent
    pre_window_minutes: int
    post_window_minutes: int
    start: datetime = Field(..., description="Window start (event - pre)")
    end: datetime = Field(..., description="Window end (event + post)")


class RiskWindowStatus(str, Enum):
    BEFORE = "before"
    INSIDE = "inside"
    AFTER = "after"
    OUTSIDE = "outside"


class PairRiskContext(BaseModel):
    """Structured risk context for a currency pair at a given timestamp."""

    model_config = ConfigDict(frozen=True)

    pair: str
    timestamp: datetime
    active_events: list[RiskWindow] = Field(default_factory=list)
    upcoming_events: list[EconomicEvent] = Field(default_factory=list)
    time_until_next: float | None = Field(
        default=None, description="Seconds until the next event (None if none)"
    )
    time_since_last: float | None = Field(
        default=None, description="Seconds since the last event (None if none)"
    )
    highest_active_importance: EventImportance | None = Field(default=None)
    message: str = Field(default="")
