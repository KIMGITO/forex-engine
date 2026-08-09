"""Tests for currency-pair risk context (EUR/USD considers EUR+USD only)."""

from datetime import datetime, timezone

from app.news.models import EconomicEvent, EventCategory, EventImportance
from app.news.risk_windows import RiskWindowConfig, pair_risk_context


def _make_event(event_id, ts, currency, importance=EventImportance.HIGH) -> EconomicEvent:
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


class TestCurrencyPair:
    def test_eurusd_considers_both(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        usd_ev = _make_event("us-cpi", ts, "USD")
        eur_ev = _make_event("ecb", ts, "EUR")
        ctx = pair_risk_context([usd_ev, eur_ev], "EUR/USD", ts, RiskWindowConfig())
        assert len(ctx.active_events) == 2

    def test_unrelated_currency_ignored(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        jpy_ev = _make_event("boj", ts, "JPY")
        ctx = pair_risk_context([jpy_ev], "EUR/USD", ts, RiskWindowConfig())
        assert len(ctx.active_events) == 0
        assert len(ctx.upcoming_events) == 0

    def test_pair_string_normalized_case(self) -> None:
        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        usd_ev = _make_event("us-cpi", ts, "USD")
        ctx = pair_risk_context([usd_ev], "eur/usd", ts, RiskWindowConfig())
        assert len(ctx.active_events) == 1

    def test_high_impact_flag(self) -> None:
        from app.news.engine import NewsEngine

        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        high = _make_event("us-cpi", ts, "USD", importance=EventImportance.HIGH)
        low = _make_event("us-low", ts, "USD", importance=EventImportance.LOW)
        engine = NewsEngine()
        assert engine.is_pair_in_high_impact([high, low], "EUR/USD", ts) is True

    def test_no_high_impact_flag(self) -> None:
        from app.news.engine import NewsEngine

        ts = datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc)
        low = _make_event("us-low", ts, "USD", importance=EventImportance.LOW)
        engine = NewsEngine()
        assert engine.is_pair_in_high_impact([low], "EUR/USD", ts) is False
