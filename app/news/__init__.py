"""News & economic-events intelligence layer.

Collects, normalizes, classifies, stores, and analyzes economic events that may
affect Forex markets. This module provides MARKET CONTEXT. It does not generate
trading signals or place trades.
"""

from app.news.calendar import (
    classify_status,
    get_events_between,
    get_events_for_currency,
    get_high_impact_events,
    get_upcoming_events,
)
from app.news.engine import NewsEngine
from app.news.errors import (
    CalendarError,
    NewsError,
    NormalizationError,
    ProviderError,
    StorageError,
    ValidationError,
)
from app.news.impact import calculate_surprise, calculate_surprises
from app.news.models import (
    EconomicEvent,
    EventCategory,
    EventImportance,
    EventStatus,
    PairRiskContext,
    RiskWindow,
    RiskWindowStatus,
    SurpriseResult,
)
from app.news.normalizer import (
    map_category,
    map_country_currency,
    map_importance,
    normalize_provider_event,
)
from app.news.provider import (
    BaseEconomicCalendarProvider,
    MockEconomicCalendarProvider,
)
from app.news.repository import ParquetEconomicEventRepository
from app.news.risk_windows import (
    RiskWindowConfig,
    build_risk_windows,
    pair_risk_context,
)
from app.news.validator import validate_event, validate_events

__all__ = [
    "BaseEconomicCalendarProvider",
    "CalendarError",
    "EconomicEvent",
    "EventCategory",
    "EventImportance",
    "EventStatus",
    "MockEconomicCalendarProvider",
    "NewsEngine",
    "NewsError",
    "NormalizationError",
    "PairRiskContext",
    "ParquetEconomicEventRepository",
    "ProviderError",
    "RiskWindow",
    "RiskWindowConfig",
    "RiskWindowStatus",
    "StorageError",
    "SurpriseResult",
    "ValidationError",
    "build_risk_windows",
    "calculate_surprise",
    "calculate_surprises",
    "classify_status",
    "get_events_between",
    "get_events_for_currency",
    "get_high_impact_events",
    "get_upcoming_events",
    "map_category",
    "map_country_currency",
    "map_importance",
    "normalize_provider_event",
    "pair_risk_context",
    "validate_event",
    "validate_events",
]
