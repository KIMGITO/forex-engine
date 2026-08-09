"""Deterministic tests for incremental sync and gap detection."""

from datetime import datetime, timedelta, timezone

import pytest

from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.data.repository import ParquetMarketDataRepository
from app.data.sync import detect_gaps, sync_candles

# ── helpers ──────────────────────────────────────────────────────────────────

class FakeProvider(BaseMarketDataProvider):
    def __init__(self, candles):
        self._candles = candles

    def fetch_candles(self, symbol, timeframe, start, end):
        # Return only candles within [start, end]
        return [c for c in self._candles if start <= c.timestamp <= end]


def _make_candle(ts, symbol="EURUSD", timeframe="1h"):
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        volume=100.0,
    )


def _ts(hours):
    base = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(hours=hours)


# ── tests ────────────────────────────────────────────────────────────────────

class TestSyncCandles:
    def test_first_download(self, tmp_path):
        fetched = [_make_candle(_ts(0)), _make_candle(_ts(1))]
        provider = FakeProvider(fetched)
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))

        result = sync_candles(provider, repo, "EURUSD", "1h", _ts(10), start=_ts(-5))
        assert len(result) == 2

    def test_incremental_appends_from_latest_local(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        # Seed local data
        repo.save_candles([_make_candle(_ts(0)), _make_candle(_ts(1))])

        fetched = [_make_candle(_ts(2)), _make_candle(_ts(3))]
        provider = FakeProvider(fetched)

        result = sync_candles(provider, repo, "EURUSD", "1h", _ts(10))
        assert len(result) == 4
        loaded = repo.load_candles_df("EURUSD", "1h")
        assert len(loaded) == 4

    def test_overlap_dedupe_keeps_last(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        repo.save_candles([_make_candle(_ts(1))])

        fetched = [_make_candle(_ts(1)), _make_candle(_ts(2))]
        provider = FakeProvider(fetched)
        result = sync_candles(provider, repo, "EURUSD", "1h", _ts(10))
        assert len(result) == 2
        loaded = repo.load_candles_df("EURUSD", "1h")
        assert len(loaded) == 2

    def test_no_start_and_no_local_raises(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        provider = FakeProvider([])
        with pytest.raises(ValueError, match="No local data and no explicit start"):
            sync_candles(provider, repo, "EURUSD", "1h", _ts(10))


class TestDetectGaps:
    def test_no_gaps(self):
        candles = [_make_candle(_ts(i)) for i in range(10)]
        gaps = detect_gaps(candles, expected_interval_minutes=60)
        assert gaps == []

    def test_single_gap(self):
        candles = [_make_candle(_ts(0)), _make_candle(_ts(2))]
        gaps = detect_gaps(candles, expected_interval_minutes=60)
        assert len(gaps) == 1
        assert gaps[0][0] == _ts(0)
        assert gaps[0][1] == _ts(2)

    def test_fewer_than_two_candles(self):
        assert detect_gaps([_make_candle(_ts(0))]) == []