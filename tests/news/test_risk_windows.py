"""Tests for risk-window detection."""

from datetime import datetime, timedelta, timezone

import pytest

from app.news.models import (
    EconomicEvent,
    EventCategory,
    EventImportance,
    RiskWindowStatus,
)
from app.news.risk_windows import (
    RiskWindowConfig,
    build_risk_windows,
    pair_risk_context,
    window_status_at,
)


def _make_event(event_id, ts, currency="USD", importance=EventImportance.MEDIUM) -> EconomicEvent:
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


class TestRiskWindowConfig:
    def test_default_windows(self) -> None:
        cfg = RiskWindowConfig()
        assert cfg.window_for(EventImportance.HIGH) == (30, 30)
        assert cfg.window_for(EventImportance.MEDIUM) == (15, 15)
        assert cfg.window_for(EventImportance.LOW) == (5, 5)

    def test_custom_windows(self) -> None:
        cfg = RiskWindowConfig(high_pre=60, high_post=120, medium_pre=10, medium_post=10)
        assert cfg.window_for(EventImportance.HIGH) == (60, 120)
        assert cfg.window_for(EventImportance.MEDIUM) == (10, 10)


class TestBuildRiskWindows:
    def test_window_bounds(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("e1", ts, importance=EventImportance.HIGH)
        windows = build_risk_windows([event], RiskWindowConfig())
        assert len(windows) == 1
        assert windows[0].start == ts - timedelta(minutes=30)
        assert windows[0].end == ts + timedelta(minutes=30)


class TestWindowStatusAt:
    def test_before(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("e1", ts, importance=EventImportance.HIGH)
        window = build_risk_windows([event], RiskWindowConfig())[0]
        assert window_status_at(window, ts - timedelta(minutes=31)) == RiskWindowStatus.BEFORE

    def test_inside(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("e1", ts, importance=EventImportance.HIGH)
        window = build_risk_windows([event], RiskWindowConfig())[0]
        assert window_status_at(window, ts) == RiskWindowStatus.INSIDE
        assert window_status_at(window, ts - timedelta(minutes=15)) == RiskWindowStatus.INSIDE
        assert window_status_at(window, ts + timedelta(minutes=15)) == RiskWindowStatus.INSIDE

    def test_after(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("e1", ts, importance=EventImportance.HIGH)
        window = build_risk_windows([event], RiskWindowConfig())[0]
        assert window_status_at(window, ts + timedelta(minutes=31)) == RiskWindowStatus.AFTER


class TestPairRiskContext:
    def test_inside_high_impact(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("ecb", ts, currency="EUR", importance=EventImportance.HIGH)
        ctx = pair_risk_context([event], "EUR/USD", ts, RiskWindowConfig())
        assert len(ctx.active_events) == 1
        assert ctx.highest_active_importance == EventImportance.HIGH

    def test_outside_any_window(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        event = _make_event("ecb", ts, currency="EUR", importance=EventImportance.HIGH)
        ctx = pair_risk_context(
            [event], "EUR/USD", ts + timedelta(hours=3), RiskWindowConfig()
        )
        assert len(ctx.active_events) == 0
        assert ctx.time_until_next is None

    def test_upcoming_event_has_time_until(self) -> None:
        now = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        event = _make_event("ecb", now + timedelta(hours=1), currency="EUR", importance=EventImportance.HIGH)
        ctx = pair_risk_context([event], "EUR/USD", now, RiskWindowConfig())
        assert ctx.time_until_next == pytest.approx(3600.0)
        assert len(ctx.upcoming_events) == 1
