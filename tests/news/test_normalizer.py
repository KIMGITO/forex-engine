"""Tests for the normalizer (importance/category/currency/timestamp mapping)."""

from app.news.errors import NormalizationError
from app.news.models import EventCategory, EventImportance
from app.news.normalizer import (
    map_category,
    map_country_currency,
    map_importance,
    normalize_provider_event,
)


class TestImportanceMapping:
    def test_high_labels(self) -> None:
        for label in ("high", "red", "3", "★★★", "3 stars"):
            assert map_importance(label) == EventImportance.HIGH

    def test_medium_labels(self) -> None:
        for label in ("medium", "2", "orange", "amber", "★★"):
            assert map_importance(label) == EventImportance.MEDIUM

    def test_low_labels(self) -> None:
        for label in ("low", "1", "green", "★"):
            assert map_importance(label) == EventImportance.LOW

    def test_unknown_labels(self) -> None:
        assert map_importance(None) == EventImportance.UNKNOWN
        assert map_importance("weird-label") == EventImportance.UNKNOWN


class TestCategoryMapping:
    def test_inflation(self) -> None:
        assert map_category("CPI m/m") == EventCategory.INFLATION
        assert map_category("Inflation Rate YoY") == EventCategory.INFLATION

    def test_employment(self) -> None:
        assert map_category("Nonfarm Payrolls") == EventCategory.EMPLOYMENT
        assert map_category("NFP") == EventCategory.EMPLOYMENT
        assert map_category("Unemployment Rate") == EventCategory.EMPLOYMENT

    def test_interest_rate(self) -> None:
        assert map_category("ECB Interest Rate Decision") == EventCategory.INTEREST_RATE
        assert map_category("policy rate") == EventCategory.INTEREST_RATE

    def test_gdp(self) -> None:
        assert map_category("GDP m/m") == EventCategory.GDP

    def test_other(self) -> None:
        assert map_category("Something Completely Different") == EventCategory.OTHER


class TestCurrencyMapping:
    def test_us(self) -> None:
        assert map_country_currency("US") == ["USD"]

    def test_eu(self) -> None:
        assert map_country_currency("EU") == ["EUR"]
        assert map_country_currency("DE") == ["EUR"]

    def test_gb(self) -> None:
        assert map_country_currency("UK") == ["GBP"]

    def test_jp(self) -> None:
        assert map_country_currency("JP") == ["JPY"]

    def test_unknown_country_empty(self) -> None:
        assert map_country_currency("ZZ") == []

    def test_extra_currencies(self) -> None:
        assert map_country_currency("US", extra=["EUR"]) == ["USD", "EUR"]


class TestNormalizeProviderEvent:
    def test_minimal_record(self) -> None:
        event = normalize_provider_event(
            {
                "event_id": "x-1",
                "timestamp": "2024-06-10T12:30:00Z",
                "country": "US",
                "event_name": "CPI m/m",
                "importance": "red",
            },
            provider="mock",
        )
        assert event.event_id == "x-1"
        assert event.currency == "USD"
        assert event.affected_currencies == ["USD"]
        assert event.importance == EventImportance.HIGH
        assert event.category == EventCategory.INFLATION
        assert event.provider == "mock"
        assert event.provider_importance == "red"
        assert event.scheduled_at.tzinfo is not None

    def test_numeric_parsing(self) -> None:
        event = normalize_provider_event(
            {
                "timestamp": "2024-06-10T12:30:00Z",
                "country": "US",
                "event_name": "CPI",
                "importance": "3",
                "actual": "0.3%",
                "forecast": "0.2",
                "previous": "1,000",
            },
            provider="mock",
        )
        assert event.actual == 0.3
        assert event.forecast == 0.2
        assert event.previous == 1000.0

    def test_malformed_numeric_becomes_none(self) -> None:
        event = normalize_provider_event(
            {
                "timestamp": "2024-06-10T12:30:00Z",
                "country": "US",
                "event_name": "CPI",
                "importance": "3",
                "actual": "n/a",
            },
            provider="mock",
        )
        assert event.actual is None

    def test_naive_timestamp_localized(self) -> None:
        event = normalize_provider_event(
            {
                "timestamp": "2024-06-10 12:30",
                "timezone": "UTC",
                "country": "US",
                "event_name": "CPI",
                "importance": "high",
            },
            provider="mock",
        )
        assert event.scheduled_at.tzinfo is not None

    def test_missing_country_raises(self) -> None:
        import pytest

        with pytest.raises(NormalizationError):
            normalize_provider_event(
                {"timestamp": "2024-06-10T12:30:00Z", "event_name": "CPI"},
                provider="mock",
            )

    def test_missing_timestamp_raises(self) -> None:
        import pytest

        with pytest.raises(NormalizationError):
            normalize_provider_event(
                {"country": "US", "event_name": "CPI", "importance": "high"},
                provider="mock",
            )

    def test_missing_event_name_raises(self) -> None:
        import pytest

        with pytest.raises(NormalizationError):
            normalize_provider_event(
                {"timestamp": "2024-06-10T12:30:00Z", "country": "US"},
                provider="mock",
            )

    def test_released_at_preserved(self) -> None:
        event = normalize_provider_event(
            {
                "timestamp": "2024-06-10T12:30:00Z",
                "released_at": "2024-06-10T12:35:00Z",
                "actual": 0.3,
                "country": "US",
                "event_name": "CPI",
                "importance": "high",
            },
            provider="mock",
        )
        assert event.released_at is not None
        assert event.effective_available_from() == event.released_at
