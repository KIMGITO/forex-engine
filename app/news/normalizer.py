"""Normalization of raw provider data into internal :class:`EconomicEvent` models.

Mappings are intentionally best-effort and provider-agnostic:
* importance: provider labels such as ``red``/``3``/``high`` -> ``HIGH``
* category: keyword matching against a canonical set
* currency: country-code mapping, supporting multiple affected currencies
* numerics: malformed values become ``None`` (never guessed)
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from app.news.errors import NormalizationError
from app.news.models import EconomicEvent, EventCategory, EventImportance

logger = logging.getLogger(__name__)

__all__ = ["map_category", "map_country_currency", "map_importance", "normalize_provider_event"]

# ── Importance mapping ────────────────────────────────────────────────────────

_IMPORTANCE_MAP: dict[str, EventImportance] = {
    # Explicit enum values
    "low": EventImportance.LOW,
    "medium": EventImportance.MEDIUM,
    "high": EventImportance.HIGH,
    "unknown": EventImportance.UNKNOWN,
    # Common provider conventions
    "red": EventImportance.HIGH,
    "red3": EventImportance.HIGH,
    "3": EventImportance.HIGH,
    "2": EventImportance.MEDIUM,
    "1": EventImportance.LOW,
    "orange": EventImportance.MEDIUM,
    "yellow": EventImportance.MEDIUM,
    "amber": EventImportance.MEDIUM,
    "green": EventImportance.LOW,
    "gray": EventImportance.UNKNOWN,
    "grey": EventImportance.UNKNOWN,
    "": EventImportance.UNKNOWN,
}


def map_importance(raw: str | None) -> EventImportance:
    """Map a provider importance label to the canonical enum."""
    if raw is None:
        return EventImportance.UNKNOWN
    key = str(raw).strip().lower()
    if key in _IMPORTANCE_MAP:
        return _IMPORTANCE_MAP[key]
    # Star ratings: e.g. "★★★" or "3 stars"
    if "star" in key or "★" in key:
        stars = sum(1 for ch in key if ch == "★") or int("".join(x for x in key if x.isdigit()) or 0)
        if stars >= 3:
            return EventImportance.HIGH
        if stars == 2:
            return EventImportance.MEDIUM
        if stars == 1:
            return EventImportance.LOW
    logger.warning("Unmapped provider importance label %r -> UNKNOWN", raw)
    return EventImportance.UNKNOWN


# ── Category mapping ──────────────────────────────────────────────────────────

_CATEGORY_KEYWORDS: list[tuple] = [
    (EventCategory.INFLATION, ("cpi", "inflation", "ppi", "core cpi")),
    (EventCategory.EMPLOYMENT, ("nonfarm", "payroll", "unemployment", "jobless", "employment",
                                "nfp", "average earnings", "labor")),
    (EventCategory.INTEREST_RATE, ("rate decision", "interest rate", "policy rate", "overnight rate",
                                   "rate statement", "refinance rate")),
    (EventCategory.GDP, ("gdp", "gross domestic product")),
    (EventCategory.MANUFACTURING, ("manufacturing", "ism", "industrial production",
                                   "factory orders", "durable goods")),
    (EventCategory.SERVICES, ("services pmi", "ism services", "non-manufacturing")),
    (EventCategory.CONSUMER, ("consumer", "retail sales", "consumer confidence", "consumer sentiment",
                              "household", "personal spending")),
    (EventCategory.HOUSING, ("housing", "building permits", "new home", "existing home",
                             "home sales", "house price")),
    (EventCategory.TRADE, ("trade balance", "trade deficit", "current account", "exports", "imports")),
    (EventCategory.CENTRAL_BANK, ("central bank", "fed", "ecb", "boj", "boe", "fomc",
                                  "minutes", "monetary policy")),
    (EventCategory.SPEECH, ("speech", "press conference", "testimony")),
]


def map_category(raw: str) -> EventCategory:
    """Map a provider event-name/category string to a canonical category."""
    text = str(raw).lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return category
    return EventCategory.OTHER


# ── Country -> currency mapping ───────────────────────────────────────────────

_COUNTRY_CURRENCIES: dict[str, list[str]] = {
    "US": ["USD"],
    "CA": ["CAD"],
    "EU": ["EUR"], "DE": ["EUR"], "FR": ["EUR"], "IT": ["EUR"], "ES": ["EUR"],
    "NL": ["EUR"], "BE": ["EUR"], "PT": ["EUR"], "IE": ["EUR"], "FI": ["EUR"],
    "GB": ["GBP"], "UK": ["GBP"],
    "JP": ["JPY"],
    "AU": ["AUD"], "NZ": ["NZD"], "CH": ["CHF"], "CN": ["CNY"],
    "SE": ["SEK"], "NO": ["NOK"], "DK": ["DKK"],
    "BR": ["BRL"], "MX": ["MXN"], "IN": ["INR"], "ZA": ["ZAR"], "TR": ["TRY"],
    "SG": ["SGD"], "HK": ["HKD"], "KR": ["KRW"], "TW": ["TWD"],
}


def map_country_currency(country: str, extra: list[str] | None = None) -> list[str]:
    """Return the currencies affected by an event for a country code.

    Unknown countries map to an empty list (never invented). ``extra`` allows
    multi-currency events to add currencies explicitly.
    """
    currencies = list(_COUNTRY_CURRENCIES.get(country.upper(), []))
    if extra:
        for c in extra:
            code = c.upper()
            if code not in currencies:
                currencies.append(code)
    return currencies


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _to_tz_aware(value: Any, timezone_name: str = "UTC") -> datetime | None:
    """Convert a raw timestamp to a tz-aware datetime, or None if malformed."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize(timezone_name)
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> float | None:
    """Convert a raw numeric to float, or None if malformed."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).strip().replace(",", "").replace("%", "")
        if cleaned in ("", "-", "n/a", "N/A", "nan"):
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def normalize_provider_event(raw: dict[str, Any], provider: str) -> EconomicEvent:
    """Normalize a raw provider event record into an internal EconomicEvent.

    Fields that are missing or malformed are left ``None`` — the normalizer
    never invents values. Any parse error raises :class:`NormalizationError`.
    """
    try:
        country = str(raw.get("country") or raw.get("countryCode") or "").upper()
        if not country:
            raise NormalizationError("Missing required field 'country'.")

        event_name = str(raw.get("event_name") or raw.get("name") or raw.get("event") or "").strip()
        if not event_name:
            raise NormalizationError("Missing required field 'event_name'.")

        scheduled = _to_tz_aware(
            raw.get("timestamp") or raw.get("scheduled_at") or raw.get("time"),
            raw.get("timezone", "UTC") or "UTC",
        )
        if scheduled is None:
            raise NormalizationError(f"Malformed timestamp in record for '{event_name}'.")

        released = _to_tz_aware(
            raw.get("released_at") or raw.get("actual_time"),
            raw.get("timezone", "UTC") or "UTC",
        )
        received = _to_tz_aware(
            raw.get("received_at"),
            raw.get("timezone", "UTC") or "UTC",
        )

        raw_importance = raw.get("importance") or raw.get("impact")
        importance = map_importance(None if raw_importance is None else str(raw_importance))
        raw_category = raw.get("category") or raw.get("event_type") or event_name
        category = map_category(str(raw_category))

        currencies = map_country_currency(country, raw.get("affected_currencies"))

        event_id = str(
            raw.get("event_id")
            or raw.get("id")
            or f"{provider}:{country}:{scheduled.isoformat()}:{event_name}"
        )

        return EconomicEvent(
            event_id=event_id,
            scheduled_at=scheduled,
            timezone=str(raw.get("timezone", "UTC")),
            country=country,
            currency=currencies[0] if currencies else "",
            affected_currencies=currencies,
            event_name=event_name,
            category=category,
            importance=importance,
            actual=_to_float(raw.get("actual")),
            forecast=_to_float(raw.get("forecast")),
            previous=_to_float(raw.get("previous")),
            unit=raw.get("unit"),
            source=raw.get("source"),
            url=raw.get("url"),
            provider=provider,
            provider_importance=None if raw_importance is None else str(raw_importance),
            received_at=received,
            released_at=released,
            available_from=raw.get("available_from") and _to_tz_aware(
                raw.get("available_from"), raw.get("timezone", "UTC") or "UTC"
            ),
        )
    except KeyError as exc:
        raise NormalizationError(f"Missing key {exc} in provider record.") from exc
