"""Tests for the NewsEngine end-to-end orchestration."""

from datetime import datetime, timedelta, timezone

import pytest

from app.news.engine import NewsEngine
from app.news.errors import NewsError
from app.news.models import EventCategory, EventImportance
from app.news.provider import MockEconomicCalendarProvider
from app.news.repository import ParquetEconomicEventRepository


class TestNewsEngine:
    def test_ingest_fetch_persist(self, tmp_path) -> None:
        repo = ParquetEconomicEventRepository(base_storage_path=str(tmp_path / "processed"))
        engine = NewsEngine(provider=MockEconomicCalendarProvider(), repository=repo)
        events = engine.ingest(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        assert len(events) == 8
        # Persisted and reloadable.
        loaded = repo.load_events()
        assert len(loaded) == 8

    def test_ingest_full(self, tmp_path) -> None:
        engine = NewsEngine(provider=MockEconomicCalendarProvider())
        events = engine.ingest()
        assert len(events) == 8

    def test_ingest_no_provider_raises(self) -> None:
        engine = NewsEngine()
        with pytest.raises(NewsError):
            engine.ingest()

    def test_ingest_partial_range_raises(self) -> None:
        engine = NewsEngine(provider=MockEconomicCalendarProvider())
        with pytest.raises(NewsError):
            engine.ingest(start=datetime(2024, 6, 1, tzinfo=timezone.utc))

    def test_query_methods(self) -> None:
        engine = NewsEngine(provider=MockEconomicCalendarProvider())
        events = engine.ingest(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        # for_currency
        usd_events = engine.for_currency(events, "USD")
        assert len(usd_events) == 2  # CPI + NFP
        # high_impact
        high = engine.high_impact(events, "USD")
        assert all(e.importance == EventImportance.HIGH for e in high)
        # upcoming
        upcoming = engine.upcoming(
            events,
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            timedelta(days=30),
        )
        assert len(upcoming) == 8

    def test_surprise_through_engine(self) -> None:
        engine = NewsEngine()
        event = engine.for_currency(
            [
                __import__("app.news.models", fromlist=["EconomicEvent"]).EconomicEvent(
                    event_id="x",
                    scheduled_at=datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
                    country="US",
                    currency="USD",
                    event_name="CPI",
                    category=EventCategory.INFLATION,
                    importance=EventImportance.HIGH,
                    actual=0.3,
                    forecast=0.2,
                )
            ],
            "USD",
        )[0]
        result = engine.surprise(event)
        assert result.surprise == pytest.approx(0.1)

    def test_risk_window_methods(self) -> None:
        engine = NewsEngine(provider=MockEconomicCalendarProvider())
        events = engine.ingest(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        windows = engine.risk_windows(events)
        assert len(windows) > 0
        ctx = engine.pair_context(
            events,
            "EUR/USD",
            datetime(2024, 6, 3, 13, 45, tzinfo=timezone.utc),
        )
        assert ctx.pair == "EUR/USD"
