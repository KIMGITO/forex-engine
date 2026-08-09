"""Tests for calendar queries and status classification."""

from datetime import datetime, timedelta, timezone

import pytest

from app.news.calendar import (
    classify_status,
    get_events_between,
    get_events_for_currency,
    get_high_impact_events,
    get_upcoming_events,
)
from app.news.errors import CalendarError
from app.news.models import EconomicEvent, EventCategory, EventImportance, EventStatus


def _make_event(event_id: str, ts: datetime, currency: str = "USD", importance=EventImportance.HIGH) -> EconomicEvent:
    return EconomicEvent(
        event_id=event_id,
        scheduled_at=ts,
        country="US" if currency == "USD" else "EU",
        currency=currency,
        affected_currencies=[currency],
        event_name=f"Event {event_id}",
        category=EventCategory.OTHER,
        importance=importance,
    )


class TestClassifyStatus:
    def test_upcoming(self) -> None:
        now = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        ev = _make_event("e1", datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc))
        assert classify_status(ev, now) == EventStatus.UPCOMING

    def test_past(self) -> None:
        now = datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc)
        ev = _make_event("e1", datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc))
        assert classify_status(ev, now) == EventStatus.PAST

    def test_released(self) -> None:
        now = datetime(2024, 6, 10, 12, 31, tzinfo=timezone.utc)
        ev = _make_event("e1", datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc))
        ev = ev.model_copy(
            update={"released_at": datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)}
        )
        assert classify_status(ev, now) == EventStatus.RELEASED


class TestQueries:
    def _events(self):
        t1 = datetime(2024, 6, 10, 8, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 6, 10, 16, 0, tzinfo=timezone.utc)
        return [
            _make_event("a", t1),
            _make_event("b", t2),
            _make_event("c", t3),
            _make_event("d", t2, currency="EUR", importance=EventImportance.MEDIUM),
        ]

    def test_get_events_between(self) -> None:
        events = self._events()
        start = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 10, 15, 0, tzinfo=timezone.utc)
        result = get_events_between(events, start, end)
        assert [e.event_id for e in result] == ["b", "d"]

    def test_get_events_between_invalid_range(self) -> None:
        events = self._events()
        start = datetime(2024, 6, 10, 15, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        with pytest.raises(CalendarError):
            get_events_between(events, start, end)

    def test_get_upcoming_events(self) -> None:
        events = self._events()
        now = datetime(2024, 6, 10, 11, 0, tzinfo=timezone.utc)
        horizon = timedelta(hours=3)
        result = get_upcoming_events(events, now, horizon)
        # Events within (11:00, 14:00] = b, d (12:00); c (16:00) is outside.
        assert [e.event_id for e in result] == ["b", "d"]

    def test_get_events_for_currency(self) -> None:
        events = self._events()
        result = get_events_for_currency(events, "EUR")
        assert [e.event_id for e in result] == ["d"]

    def test_get_high_impact_events(self) -> None:
        events = self._events()
        result = get_high_impact_events(events, "USD")
        assert [e.event_id for e in result] == ["a", "b", "c"]
        assert all(e.importance == EventImportance.HIGH for e in result)
