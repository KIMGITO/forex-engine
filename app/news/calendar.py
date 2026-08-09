"""Calendar queries and event-status classification.

Status is determined purely from timestamps — never from the event name.
"""

from datetime import datetime, timedelta

from app.news.errors import CalendarError
from app.news.models import EconomicEvent, EventImportance, EventStatus

__all__ = [
    "classify_status",
    "get_events_between",
    "get_events_for_currency",
    "get_high_impact_events",
    "get_upcoming_events",
]


def classify_status(event: EconomicEvent, now: datetime) -> EventStatus:
    """Classify an event's status at ``now`` from timestamps only."""
    if event.released_at is not None and event.released_at <= now:
        return EventStatus.RELEASED
    if event.scheduled_at <= now:
        return EventStatus.PAST
    return EventStatus.UPCOMING


def get_events_between(
    events: list[EconomicEvent],
    start: datetime,
    end: datetime,
) -> list[EconomicEvent]:
    """Return events scheduled within [start, end]."""
    if start > end:
        raise CalendarError("start must be <= end.")
    return [e for e in events if start <= e.scheduled_at <= end]


def get_upcoming_events(
    events: list[EconomicEvent],
    now: datetime,
    horizon: timedelta,
) -> list[EconomicEvent]:
    """Return events scheduled within (now, now + horizon]."""
    return get_events_between(events, now, now + horizon)


def get_events_for_currency(
    events: list[EconomicEvent],
    currency: str,
) -> list[EconomicEvent]:
    """Return events affecting the given currency (primary or affected list)."""
    code = currency.upper()
    return [
        e for e in events
        if e.currency.upper() == code or code in [c.upper() for c in e.affected_currencies]
    ]


def get_high_impact_events(
    events: list[EconomicEvent],
    currency: str,
) -> list[EconomicEvent]:
    """Return HIGH-impact events for a currency."""
    return [
        e for e in get_events_for_currency(events, currency)
        if e.importance == EventImportance.HIGH
    ]
