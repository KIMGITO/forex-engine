"""Deterministic tests for the Twelve Data adapter (mocked HTTP).

These tests never touch the real Twelve Data API. Network interactions are
replaced by pre-canned responses injected into the shared ``HttpClient``.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest
from requests import Session

from app.data.exceptions import MalformedResponseError, UnavailableTimeframeError
from app.data.providers.twelvedata import (
    TwelveDataMarketDataProvider,
    _to_twelve_symbol,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _ts(hours=0) -> datetime:
    return datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc) + pd.Timedelta(hours=hours)


def _values_payload(n=5, base=None):
    base = base or _ts()
    out = []
    for i in range(n):
        t = base + pd.Timedelta(hours=i)
        out.append(
            {
                "datetime": t.strftime("%Y-%m-%d %H:%M:%S"),
                "open": "1.0850",
                "high": "1.0860",
                "low": "1.0840",
                "close": "1.0855",
            }
        )
    return out


def _ok_payload(n=5):
    return {"meta": {"symbol": "EUR/USD"}, "values": _values_payload(n), "status": "ok"}


def _patch_http(monkeypatch, payload, status=200):
    class FakeResp:
        def __init__(self, payload, status):
            self._payload = payload
            self.status_code = status
            self.headers = {}
            self.text = ""

        def json(self):
            return self._payload

    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeResp(payload, status)

    monkeypatch.setattr(Session, "get", fake_get)


# ── tests ────────────────────────────────────────────────────────────────────

class TestTwelveSymbolMapping:
    def test_eurusd_underscore(self):
        assert _to_twelve_symbol("EURUSD") == "EUR/USD"

    def test_with_underscore(self):
        assert _to_twelve_symbol("EUR_USD") == "EUR/USD"

    def test_with_dash(self):
        assert _to_twelve_symbol("EUR-USD") == "EUR/USD"


class TestTwelveDataProvider:
    def test_fetch_candles_valid(self, monkeypatch):
        _patch_http(monkeypatch, _ok_payload(10), 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        candles = provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(20))
        assert len(candles) == 10
        assert candles[0].symbol == "EURUSD"
        assert candles[0].high > candles[0].low

    def test_fetch_candles_empty(self, monkeypatch):
        _patch_http(monkeypatch, {"meta": {}, "values": [], "status": "ok"}, 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        assert provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(10)) == []

    def test_fetch_candles_error_status(self, monkeypatch):
        _patch_http(
            monkeypatch,
            {"status": "error", "message": "invalid symbol"},
            200,
        )
        provider = TwelveDataMarketDataProvider(api_key="fake")
        with pytest.raises(MalformedResponseError, match="status='error'"):
            provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(10))

    def test_fetch_candles_missing_values(self, monkeypatch):
        _patch_http(monkeypatch, {"meta": {}, "status": "ok"}, 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        with pytest.raises(MalformedResponseError, match="missing 'values'"):
            provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(10))

    def test_fetch_candles_deduplicates_on_timestamp(self, monkeypatch):
        payload = _ok_payload(5)
        payload["values"] = _values_payload(5) + _values_payload(5)[:2]
        _patch_http(monkeypatch, payload, 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        candles = provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(10))
        assert len(candles) == 5

    def test_fetch_candles_sorted(self, monkeypatch):
        # Twelve Data returns newest-first; ensure we sort chronologically.
        payload = _ok_payload(8)
        payload["values"] = list(reversed(_values_payload(8)))
        _patch_http(monkeypatch, payload, 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        candles = provider.fetch_candles("EURUSD", "1h", _ts(0), _ts(20))
        times = [c.timestamp for c in candles]
        assert times == sorted(times)

    def test_unsupported_timeframe(self, monkeypatch):
        _patch_http(monkeypatch, _ok_payload(1), 200)
        provider = TwelveDataMarketDataProvider(api_key="fake")
        with pytest.raises(UnavailableTimeframeError):
            provider.fetch_candles("EURUSD", "3h", _ts(0), _ts(10))

    def test_supports_bid_ask_false(self):
        provider = TwelveDataMarketDataProvider(api_key="fake")
        assert provider.supports_bid_ask() is False