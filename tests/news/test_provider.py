"""Tests for the mock economic-calendar provider."""

from datetime import datetime, timezone

import pytest

from app.news.errors import ProviderError
from app.news.provider import BaseEconomicCalendarProvider, MockEconomicCalendarProvider


class TestMockProvider:
    def test_base_provider_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseEconomicCalendarProvider()  # type: ignore[abstract]

    def test_fetch_event(self) -> None:
        provider = MockEconomicCalendarProvider()
        event = provider.fetch_event("mock-us-cpi")
        assert event.event_id == "mock-us-cpi"
        assert event.currency == "USD"

    def test_fetch_event_missing_raises(self) -> None:
        provider = MockEconomicCalendarProvider()
        with pytest.raises(ProviderError):
            provider.fetch_event("does-not-exist")

    def test_fetch_events_between(self) -> None:
        provider = MockEconomicCalendarProvider()
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)
        events = provider.fetch_events_between(start, end)
        assert len(events) == 8

    def test_fetch_events_between_empty(self) -> None:
        provider = MockEconomicCalendarProvider()
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        events = provider.fetch_events_between(start, end)
        assert events == []

    def test_deterministic(self) -> None:
        p1 = MockEconomicCalendarProvider()
        p2 = MockEconomicCalendarProvider()
        s = datetime(2024, 6, 1, tzinfo=timezone.utc)
        e = datetime(2024, 6, 30, tzinfo=timezone.utc)
        e1 = p1.fetch_events_between(s, e)
        e2 = p2.fetch_events_between(s, e)
        assert [ev.event_id for ev in e1] == [ev.event_id for ev in e2]

    def test_events_are_tz_aware(self) -> None:
        provider = MockEconomicCalendarProvider()
        for ev in provider.fetch_events_between(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
        ):
            assert ev.scheduled_at.tzinfo is not None
