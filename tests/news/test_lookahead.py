"""Look-ahead regression tests: actual event data must never be usable before
its release time."""

from datetime import datetime, timezone

import pytest

from app.news.models import EconomicEvent, EventCategory, EventImportance


def _released_event():
    """An event with actual released at 12:35, scheduled 12:30."""
    return EconomicEvent(
        event_id="cpi-release",
        scheduled_at=datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
        released_at=datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc),
        country="US",
        currency="USD",
        affected_currencies=["USD"],
        event_name="CPI m/m",
        category=EventCategory.INFLATION,
        importance=EventImportance.HIGH,
        actual=0.3,
        forecast=0.2,
    )


class TestLookAhead:
    def test_actual_not_available_before_release(self) -> None:
        event = _released_event()
        before_release = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        available = event.effective_available_from()
        # The event must not be usable at its scheduled (pre-release) time.
        assert available > before_release
        assert available == datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc)

    def test_scheduled_event_available_from_scheduled(self) -> None:
        event = EconomicEvent(
            event_id="no-release-yet",
            scheduled_at=datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc),
            country="US",
            currency="USD",
            event_name="Upcoming",
            importance=EventImportance.MEDIUM,
        )
        assert event.effective_available_from() == event.scheduled_at

    def test_engine_uses_available_from_for_upcoming(self) -> None:
        from app.news.risk_windows import RiskWindowConfig, pair_risk_context

        event = _released_event()
        # At 12:32 the actual is already released, but the scheduled window is
        # from 12:00-13:00. pair_context must only list it as "upcoming" if its
        # available_from is in the future.
        now_inside_pre = datetime(2024, 6, 10, 12, 33, tzinfo=timezone.utc)
        ctx = pair_risk_context([event], "EUR/USD", now_inside_pre, RiskWindowConfig())
        # Event is released -> not upcoming.
        assert len(ctx.upcoming_events) == 0

    def test_actual_value_stays_none_before_release(self) -> None:
        # An event scheduled in the future has no actual.
        event = EconomicEvent(
            event_id="future",
            scheduled_at=datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc),
            country="US",
            currency="USD",
            event_name="Future",
            importance=EventImportance.HIGH,
        )
        assert event.actual is None
        assert event.released_at is None

    def test_surprise_recquires_released_actual(self) -> None:
        from app.news.impact import calculate_surprise

        event = _released_event()
        result = calculate_surprise(event)
        assert result.surprise is not None

        # A future (unreleased) event must not be able to produce a surprise.
        future = EconomicEvent(
            event_id="future",
            scheduled_at=datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc),
            country="US",
            currency="USD",
            event_name="Future",
            importance=EventImportance.HIGH,
            actual=None,
        )
        with pytest.raises(ValueError):
            calculate_surprise(future)
