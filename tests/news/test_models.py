"""Tests for the economic-event model."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.news.models import EconomicEvent, EventCategory, EventImportance, EventStatus


def _ts(hour=12, day=10):
    return datetime(2024, 6, 10, hour, 0, tzinfo=timezone.utc) + timedelta(days=day)


def _make_event(**overrides) -> EconomicEvent:
    base = {
        "event_id": "test-1",
        "scheduled_at": datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
        "country": "US",
        "currency": "USD",
        "event_name": "CPI m/m",
        "category": EventCategory.INFLATION,
        "importance": EventImportance.HIGH,
    }
    base.update(overrides)
    return EconomicEvent(**base)


class TestEconomicEvent:
    def test_valid_event(self) -> None:
        event = _make_event()
        assert event.event_id == "test-1"
        assert event.importance == EventImportance.HIGH
        assert event.actual is None
        assert event.available_from is None

    def test_optional_fields(self) -> None:
        event = _make_event(
            actual=0.3,
            forecast=0.2,
            previous=0.1,
            unit="%",
            url="https://example.com/cpi",
        )
        assert event.actual == 0.3
        assert event.forecast == 0.2
        assert event.unit == "%"

    def test_invalid_importance_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(importance="crash")  # type: ignore[arg-type]

    def test_status_upcoming(self) -> None:
        event = _make_event()
        now = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        assert event.status(now) == EventStatus.UPCOMING

    def test_status_past(self) -> None:
        event = _make_event()
        now = datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc)
        assert event.status(now) == EventStatus.PAST

    def test_status_released(self) -> None:
        event = _make_event(
            released_at=datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
            actual=0.3,
        )
        now = datetime(2024, 6, 10, 12, 31, tzinfo=timezone.utc)
        assert event.status(now) == EventStatus.RELEASED

    def test_effective_available_from_defaults_to_scheduled(self) -> None:
        event = _make_event()
        assert event.effective_available_from() == event.scheduled_at

    def test_effective_available_from_uses_released(self) -> None:
        released = datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc)
        event = _make_event(released_at=released)
        assert event.effective_available_from() == released

    def test_effective_available_from_explicit(self) -> None:
        explicit = datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc)
        event = _make_event(
            released_at=datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc),
            available_from=explicit,
        )
        assert event.effective_available_from() == explicit

    def test_released_at_before_scheduled_raises(self) -> None:
        # The model itself doesn't enforce ordering; the validator does.
        event = _make_event(
            released_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        )
        from app.news.errors import ValidationError as NewsValidationError
        from app.news.validator import validate_event

        with pytest.raises(NewsValidationError):
            validate_event(event)
