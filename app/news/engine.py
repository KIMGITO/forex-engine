"""NewsEngine: orchestrates the news & economic-events pipeline.

Pipeline:
  Provider -> Validator -> Repository -> Calendar queries
  -> Impact classification -> Risk-window detection

The engine is provider-agnostic (depends on :class:`BaseEconomicCalendarProvider`)
and repository-agnostic (depends on :class:`ParquetEconomicEventRepository`
for local development).
"""

from datetime import datetime, timedelta, timezone

from app.news.calendar import (
    get_events_between,
    get_events_for_currency,
    get_high_impact_events,
    get_upcoming_events,
)
from app.news.errors import NewsError
from app.news.impact import calculate_surprise, calculate_surprises
from app.news.models import (
    EconomicEvent,
    EventImportance,
    PairRiskContext,
    SurpriseResult,
)
from app.news.provider import BaseEconomicCalendarProvider
from app.news.repository import ParquetEconomicEventRepository
from app.news.risk_windows import (
    RiskWindowConfig,
    build_risk_windows,
    pair_risk_context,
)
from app.news.validator import validate_events

__all__ = ["NewsEngine"]

# Bounded window for full-calendar fetches (mock providers are synthetic and
# finite; real providers will supply their own range).
_EARLIEST = datetime(2000, 1, 1, tzinfo=timezone.utc)
_LATEST = datetime(2100, 1, 1, tzinfo=timezone.utc)


class NewsEngine:
    """Central orchestrator for economic-events data."""

    def __init__(
        self,
        provider: BaseEconomicCalendarProvider | None = None,
        repository: ParquetEconomicEventRepository | None = None,
        risk_config: RiskWindowConfig | None = None,
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.risk_config = risk_config or RiskWindowConfig()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(
        self,
        provider: BaseEconomicCalendarProvider | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EconomicEvent]:
        """Fetch events from a provider, validate them, and store via the repository.

        Returns the validated event list that was persisted (or would be
        persisted, when no repository is configured).
        """
        src = provider or self.provider
        if src is None:
            raise NewsError("No economic-calendar provider configured for ingestion.")

        if start is not None and end is not None:
            events = src.fetch_events_between(start, end)
        elif start is not None or end is not None:
            raise NewsError("Both start and end must be provided for a range fetch.")
        else:
            events = src.fetch_events_between(_EARLIEST, _LATEST)

        validate_events(events)

        if self.repository is not None:
            self.repository.save_events(events)
        return events

    # ── Queries ──────────────────────────────────────────────────────────────

    def events_between(
        self, events: list[EconomicEvent], start: datetime, end: datetime
    ) -> list[EconomicEvent]:
        """Return events scheduled within [start, end]."""
        return get_events_between(events, start, end)

    def upcoming(
        self,
        events: list[EconomicEvent],
        now: datetime,
        horizon: timedelta,
    ) -> list[EconomicEvent]:
        """Return upcoming events within the horizon."""
        return get_upcoming_events(events, now, horizon)

    def for_currency(
        self, events: list[EconomicEvent], currency: str
    ) -> list[EconomicEvent]:
        """Return events affecting a currency."""
        return get_events_for_currency(events, currency)

    def high_impact(
        self, events: list[EconomicEvent], currency: str
    ) -> list[EconomicEvent]:
        """Return high-impact events for a currency."""
        return get_high_impact_events(events, currency)

    # ── Impact ───────────────────────────────────────────────────────────────

    def surprise(self, event: EconomicEvent) -> SurpriseResult:
        """Objectively compute actual-vs-forecast surprise."""
        return calculate_surprise(event)

    def surprises(self, events: list[EconomicEvent]) -> list[SurpriseResult]:
        """Compute surprises for released events only (skips missing actuals)."""
        return calculate_surprises(events)

    # ── Risk windows ─────────────────────────────────────────────────────────

    def risk_windows(self, events: list[EconomicEvent]) -> list:
        """Build risk windows for all events."""
        return build_risk_windows(events, self.risk_config)

    def pair_context(
        self,
        events: list[EconomicEvent],
        pair: str,
        timestamp: datetime,
    ) -> PairRiskContext:
        """Structured risk context for a currency pair (EUR/USD, GBP/JPY, ...)."""
        return pair_risk_context(events, pair, timestamp, self.risk_config)

    def is_pair_in_high_impact(
        self,
        events: list[EconomicEvent],
        pair: str,
        timestamp: datetime,
    ) -> bool:
        """True if either component currency is inside a HIGH-impact window."""
        ctx = self.pair_context(events, pair, timestamp)
        return any(
            w.event.importance == EventImportance.HIGH for w in ctx.active_events
        )
