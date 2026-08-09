"""Tests for event validation."""

from datetime import datetime, timezone

import pytest

from app.news.errors import ValidationError
from app.news.models import EconomicEvent, EventImportance
from app.news.validator import detect_duplicates, validate_event, validate_events


def _make_event(**overrides) -> EconomicEvent:
    base = {
        "event_id": "v-1",
        "scheduled_at": datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
        "country": "US",
        "currency": "USD",
        "event_name": "CPI m/m",
        "importance": EventImportance.HIGH,
    }
    base.update(overrides)
    return EconomicEvent(**base)


class TestValidateEvent:
    def test_valid_event_passes(self) -> None:
        validate_event(_make_event())

    def test_empty_event_id(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(_make_event(event_id=""))

    def test_empty_event_name(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(_make_event(event_name="  "))

    def test_empty_country(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(_make_event(country=""))

    def test_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(
                _make_event(scheduled_at=datetime(2024, 6, 10, 12, 30))  # noqa: DTZ001 - intentional naive to test rejection
            )

    def test_impossible_year(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(
                _make_event(scheduled_at=datetime(1800, 1, 1, tzinfo=timezone.utc))
            )

    def test_released_before_scheduled(self) -> None:
        with pytest.raises(ValidationError):
            validate_event(
                _make_event(
                    released_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
                )
            )


class TestDetectDuplicates:
    def test_no_duplicates(self) -> None:
        events = [
            _make_event(event_id="a", provider="p1"),
            _make_event(event_id="b", provider="p1"),
        ]
        assert detect_duplicates(events) == []

    def test_duplicates_detected(self) -> None:
        events = [
            _make_event(event_id="a", provider="p1"),
            _make_event(event_id="a", provider="p1"),
        ]
        dupes = detect_duplicates(events)
        assert len(dupes) == 1
        assert len(dupes[0]) == 2

    def test_same_id_different_provider_not_duplicate(self) -> None:
        events = [
            _make_event(event_id="a", provider="p1"),
            _make_event(event_id="a", provider="p2"),
        ]
        assert detect_duplicates(events) == []

    def test_validate_events_raises_on_duplicates(self) -> None:
        events = [
            _make_event(event_id="a", provider="p1"),
            _make_event(event_id="a", provider="p1"),
        ]
        with pytest.raises(ValidationError):
            validate_events(events)
