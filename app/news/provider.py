"""Economic-calendar provider abstraction.

The rest of the news layer depends on :class:`BaseEconomicCalendarProvider`,
never on a specific vendor. ``MockEconomicCalendarProvider`` provides
deterministic synthetic data for development/testing only — it is explicitly
not connected to any real economic-calendar service.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

from app.news.errors import ProviderError
from app.news.models import EconomicEvent, EventCategory, EventImportance

__all__ = ["BaseEconomicCalendarProvider", "MockEconomicCalendarProvider"]


class BaseEconomicCalendarProvider(ABC):
    """Abstract interface for economic-calendar providers."""

    @abstractmethod
    def fetch_event(self, event_id: str) -> EconomicEvent:
        """Fetch a single event by its provider event id."""

    @abstractmethod
    def fetch_events_between(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        """Fetch all events scheduled within [start, end]."""


class MockEconomicCalendarProvider(BaseEconomicCalendarProvider):
    """Deterministic in-memory provider for development/testing.

    NOT connected to any real economic-calendar service. Events are generated
    from a fixed specification so tests are reproducible.
    """

    def __init__(self) -> None:
        self._events: dict[str, EconomicEvent] = {}
        self._build_events()

    def _build_events(self) -> None:
        """Seed a deterministic set of events across major currencies."""
        base = datetime(2024, 6, 3, 8, 30, tzinfo=pd.Timestamp.utcnow().tz)

        def _ts(hour: int, minute: int = 0, day_offset: int = 0) -> datetime:
            return (base + timedelta(days=day_offset)).replace(hour=hour, minute=minute)

        specs = [
            # (event_id, country, currency, name, category, importance, ts)
            ("mock-us-cpi", "US", "USD", "CPI m/m", EventCategory.INFLATION, EventImportance.HIGH,
             _ts(13, 30)),
            ("mock-us-nfp", "US", "USD", "Nonfarm Payrolls", EventCategory.EMPLOYMENT,
             EventImportance.HIGH, _ts(13, 30)),
            ("mock-ez-cpi", "EU", "EUR", "CPI YoY", EventCategory.INFLATION,
             EventImportance.MEDIUM, _ts(10, 0)),
            ("mock-ez-ecb", "EU", "EUR", "ECB Interest Rate Decision", EventCategory.INTEREST_RATE,
             EventImportance.HIGH, _ts(13, 45)),
            ("mock-gb-gdp", "GB", "GBP", "GDP m/m", EventCategory.GDP,
             EventImportance.MEDIUM, _ts(7, 0)),
            ("mock-jp-boj", "JP", "JPY", "BOJ Policy Rate", EventCategory.INTEREST_RATE,
             EventImportance.HIGH, _ts(3, 0)),
            ("mock-ca-jobs", "CA", "CAD", "Employment Change", EventCategory.EMPLOYMENT,
             EventImportance.MEDIUM, _ts(13, 30)),
            ("mock-au-gdp", "AU", "AUD", "GDP q/q", EventCategory.GDP,
             EventImportance.MEDIUM, _ts(2, 30)),
        ]
        for spec in specs:
            event_id, country, currency, name, category, importance, ts = spec
            self._events[event_id] = EconomicEvent(
                event_id=event_id,
                scheduled_at=ts,
                timezone="UTC",
                country=country,
                currency=currency,
                affected_currencies=[currency],
                event_name=name,
                category=category,
                importance=importance,
                provider="mock",
            )

    def fetch_event(self, event_id: str) -> EconomicEvent:
        if event_id not in self._events:
            raise ProviderError(f"Event '{event_id}' not found in mock provider.")
        return self._events[event_id]

    def fetch_events_between(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        return [
            e for e in self._events.values()
            if start <= e.scheduled_at <= end
        ]
