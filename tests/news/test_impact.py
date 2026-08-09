"""Tests for the surprise calculation."""

from datetime import datetime, timezone

import pytest

from app.news.impact import calculate_surprise, calculate_surprises
from app.news.models import EconomicEvent, EventCategory, EventImportance


def _make_event(actual=None, forecast=None, **overrides) -> EconomicEvent:
    base = {
        "event_id": "s-1",
        "scheduled_at": datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
        "country": "US",
        "currency": "USD",
        "event_name": "CPI m/m",
        "category": EventCategory.INFLATION,
        "importance": EventImportance.HIGH,
        "actual": actual,
        "forecast": forecast,
    }
    base.update(overrides)
    return EconomicEvent(**base)


class TestSurprise:
    def test_above_forecast(self) -> None:
        result = calculate_surprise(_make_event(actual=0.3, forecast=0.2))
        assert result.surprise == pytest.approx(0.1)
        assert result.surprise_pct == pytest.approx(50.0)

    def test_below_forecast(self) -> None:
        result = calculate_surprise(_make_event(actual=0.1, forecast=0.2))
        assert result.surprise == pytest.approx(-0.1)
        assert result.surprise_pct == pytest.approx(-50.0)

    def test_matching_forecast(self) -> None:
        result = calculate_surprise(_make_event(actual=0.2, forecast=0.2))
        assert result.surprise == pytest.approx(0.0)
        assert result.surprise_pct == pytest.approx(0.0)

    def test_missing_forecast(self) -> None:
        result = calculate_surprise(_make_event(actual=0.3, forecast=None))
        assert result.surprise is None
        assert result.surprise_pct is None

    def test_zero_forecast_no_pct(self) -> None:
        result = calculate_surprise(_make_event(actual=0.3, forecast=0.0))
        assert result.surprise == pytest.approx(0.3)
        assert result.surprise_pct is None

    def test_missing_actual_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_surprise(_make_event(actual=None, forecast=0.2))

    def test_calculate_surprises_skips_missing_actual(self) -> None:
        events = [
            _make_event(event_id="a", actual=0.3, forecast=0.2),
            _make_event(event_id="b", actual=None, forecast=0.2),
            _make_event(event_id="c", actual=0.5, forecast=0.4),
        ]
        results = calculate_surprises(events)
        assert [r.event_id for r in results] == ["a", "c"]
