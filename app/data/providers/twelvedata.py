"""Twelve Data market-data adapter.

Implements the provider-independent :class:`BaseMarketDataProvider` interface.
Handles authentication, request construction, pagination, response parsing,
and conversion to our internal :class:`Candle` model. No Twelve Data response
objects are exposed to the rest of the application.

Data received: Twelve Data ``time_series`` returns OHLC candles with a UTC
``datetime`` string and ``open/high/low/close`` numeric strings. Values are
returned newest-first; we sort chronologically and deduplicate on timestamp.

LIMITATIONS (documented, not hidden):
- Twelve Data is POLLING-only. There is no true streaming/tick feed; the
  ``LiveCandlePoller`` here is a polling approximation, NOT a tick stream.
- Twelve Data supplies OHLC only — no bid/ask/spread. ``close`` is therefore
  NOT an executable price. Realistic execution modeling (spread, commissions,
  swaps, slippage, latency) belongs to the future execution stage.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.data.client import HttpClient
from app.data.exceptions import MalformedResponseError, UnavailableTimeframeError
from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider

__all__ = ["TWELVE_DATA_INTERVALS", "LiveCandlePoller", "TwelveDataMarketDataProvider"]

TWELVE_DATA_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "45m": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1day",
    "1w": "1week",
    "1M": "1month",
}

# Twelve Data free-tier per-request outputsize cap (documented).
MAX_PER_REQUEST = 800


def _to_twelve_symbol(symbol: str) -> str:
    """Normalize EURUSD / EUR_USD -> EUR/USD (Twelve Data format)."""
    s = symbol.upper().replace("_", "/").replace("-", "/").strip("/")
    if "/" not in s and len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    return s


def _canonical_symbol(symbol: str) -> str:
    """Normalize any input symbol to the internal canonical form (e.g. EURUSD)."""
    return symbol.upper().replace("/", "").replace("_", "").replace("-", "")


class TwelveDataMarketDataProvider(BaseMarketDataProvider):
    """Real Twelve Data historical OHLC provider (polling)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.twelvedata.com",
        max_retries: int = 3,
    ) -> None:
        self.client = HttpClient(
            base_url=base_url,
            api_key=None,  # Twelve Data uses apikey query param, not Bearer
            max_retries=max_retries,
        )
        self.api_key = api_key
        self.name = "twelvedata"

    def supports_bid_ask(self) -> bool:
        """Twelve Data supplies OHLC only — no bid/ask."""
        return False

    def fetch_candles(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Candle]:
        """Fetch OHLC candles over [start, end] with pagination.

        Twelve Data's free tier caps ``outputsize`` per request, so we paginate
        by sliding date windows. Responses are newest-first; we sort
        chronologically and deduplicate on timestamp. Missing candles are never
        fabricated.
        """
        interval = _resolve_interval(timeframe)
        twelve_symbol = _to_twelve_symbol(symbol)

        cursor = start
        candles: list[Candle] = []
        while cursor < end:
            window_end = min(cursor + timedelta(days=30), end)
            params = {
                "symbol": twelve_symbol,
                "interval": interval,
                "start_date": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": window_end.strftime("%Y-%m-%d %H:%M:%S"),
                "outputsize": MAX_PER_REQUEST,
                "timezone": "UTC",
                "apikey": self.api_key,
            }
            data = self.client.get_json("/time_series", params=params)
            candles.extend(self._parse_candles(data, symbol, timeframe))
            cursor = window_end

        return self._dedupe_sort(candles)

    def _parse_candles(self, data: dict, symbol: str, timeframe: str) -> list[Candle]:
        if not isinstance(data, dict):
            raise MalformedResponseError("Twelve Data response is not a JSON object.")
        if data.get("status") != "ok":
            raise MalformedResponseError(
                f"Twelve Data returned status={data.get('status')!r}: {data.get('message', '')}"
            )
        values = data.get("values")
        if not isinstance(values, list):
            raise MalformedResponseError("Twelve Data response missing 'values' list.")

        out: list[Candle] = []
        for entry in values:
            if not isinstance(entry, dict):
                raise MalformedResponseError("Twelve Data candle is not an object.")
            try:
                ts = pd.Timestamp(entry["datetime"]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts = ts.astimezone(timezone.utc)

                def _f(key: str, entry: dict = entry) -> float:
                    val = entry.get(key)
                    if val is None:
                        raise MalformedResponseError(f"Twelve Data candle missing '{key}'.")
                    return float(val)

                out.append(
                    Candle(
                        symbol=_canonical_symbol(symbol),
                        timeframe=timeframe,
                        timestamp=ts,
                        open=_f("open"),
                        high=_f("high"),
                        low=_f("low"),
                        close=_f("close"),
                        volume=None,
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise MalformedResponseError(f"Malformed Twelve Data candle: {exc}") from exc

        return out

    @staticmethod
    def _dedupe_sort(candles: list[Candle]) -> list[Candle]:
        seen = set()
        out: list[Candle] = []
        for c in sorted(candles, key=lambda c: c.timestamp):
            if c.timestamp in seen:
                continue
            seen.add(c.timestamp)
            out.append(c)
        return out


class LiveCandlePoller:
    """Polling-based live-data interface (NOT a true tick stream).

    Twelve Data has no streaming API; this poller repeatedly fetches recent
    candles on a configurable interval. It is explicitly a polling
    approximation, never a tick stream.
    """

    def __init__(
        self,
        provider: TwelveDataMarketDataProvider,
        symbol: str,
        timeframe: str,
        poll_interval_seconds: int = 60,
    ) -> None:
        self.provider = provider
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_interval = poll_interval_seconds

    def poll_once(self, now: datetime | None = None) -> list[Candle]:
        """Fetch candles for the most recent intervals (last two frames)."""
        now = now or datetime.now(timezone.utc)
        frame = timedelta(minutes=_interval_minutes_for_timeframe(self.timeframe))
        start = now - 2 * frame
        return self.provider.fetch_candles(self.symbol, self.timeframe, start, now)


def _normalize_timeframe(timeframe: str) -> str:
    """Normalize a timeframe string to the adapter's canonical native key.

    Accepts BOTH the lowercase native conventions used by the market-structure /
    MTF / backtest layers (``15m``, ``1h``, ``4h``, ``1d``) AND the uppercase
    research-layer conventions (``M15``, ``H1``, ``H4``, ``D1``). This is purely
    additive: existing lowercase usage (tests, adapters, MTF engine) is unchanged.
    """
    s = timeframe.strip().lower().replace(" ", "")
    # Uppercase research forms: M5/M15/H1/H2/H4/D1/W1/MN1.
    mapping = {
        "m1": "1m", "m5": "5m", "m15": "15m", "m30": "30m", "m45": "45m",
        "h1": "1h", "h2": "2h", "h4": "4h",
        "d1": "1d", "w1": "1w", "mn1": "1M",
    }
    mapped = mapping.get(s, s)
    return mapped


def _resolve_interval(timeframe: str) -> str:
    interval = TWELVE_DATA_INTERVALS.get(_normalize_timeframe(timeframe))
    if interval is None:
        raise UnavailableTimeframeError(
            f"Timeframe {timeframe!r} not supported by Twelve Data adapter. "
            f"Supported: {sorted(TWELVE_DATA_INTERVALS)}"
        )
    return interval


def _interval_minutes(interval: str) -> int:
    return {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "45min": 45,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1day": 1440,
        "1week": 10080,
        "1month": 43200,
    }[interval]


def _interval_minutes_for_timeframe(timeframe: str) -> int:
    return _interval_minutes(_resolve_interval(timeframe))