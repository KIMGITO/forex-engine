"""Tests for partitioned multi-timeframe dataset repository + sync idempotency."""

from datetime import datetime, timezone

from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.research.dataset import PartitionedResearchRepository, sync_partition


class _FakeProvider(BaseMarketDataProvider):
    def __init__(self, candles):
        self._candles = candles

    def fetch_candles(self, symbol, timeframe, start, end):
        return [c for c in self._candles if start <= c.timestamp <= end]


def _candle(symbol, timeframe, ts):
    return Candle(
        symbol=symbol, timeframe=timeframe, timestamp=ts,
        open=1.0, high=1.1, low=0.9, close=1.05, volume=100.0,
    )


def _ts(day=10, hour=12):
    return datetime(2026, 8, day, hour, 0, tzinfo=timezone.utc)


class TestPartitionedRepository:
    def test_partition_layout_per_symbol_timeframe(self, tmp_path):
        repo = PartitionedResearchRepository(str(tmp_path))
        assert repo.candles_path("EURUSD", "H1").parent.name == "H1"
        assert repo.candles_path("EURUSD", "H1").parent.parent.name == "EURUSD"
        assert not repo.exists("EURUSD", "H1")

    def test_save_and_load_roundtrip(self, tmp_path):
        repo = PartitionedResearchRepository(str(tmp_path))
        candles = [_candle("EURUSD", "H1", _ts(10)), _candle("EURUSD", "H1", _ts(11))]
        repo.merge_candles(candles)
        df = repo.load_df("EURUSD", "H1")
        assert df is not None and len(df) == 2

    def test_merge_is_idempotent(self, tmp_path):
        repo = PartitionedResearchRepository(str(tmp_path))
        candles = [_candle("EURUSD", "H1", _ts(10))]
        repo.merge_candles(candles)
        repo.merge_candles(candles)  # same data again
        df = repo.load_df("EURUSD", "H1")
        assert len(df) == 1  # no duplicates

    def test_sync_partition_first_then_incremental(self, tmp_path):
        repo = PartitionedResearchRepository(str(tmp_path))
        provider = _FakeProvider([_candle("EURUSD", "H1", _ts(10)), _candle("EURUSD", "H1", _ts(11))])
        # First download (explicit start and end covering the provider candles).
        _, final = sync_partition(
            provider, repo, "EURUSD", "H1", start=_ts(9), end=_ts(12)
        )
        assert final == 2
        # Incremental re-run with overlapping candle → idempotent.
        _, final2 = sync_partition(
            provider, repo, "EURUSD", "H1", end=_ts(12)
        )
        assert final2 == 2

    def test_describe_reports_gaps_and_provenance(self, tmp_path):
        repo = PartitionedResearchRepository(str(tmp_path))
        repo.merge_candles([
            _candle("EURUSD", "H1", _ts(10)),
            _candle("EURUSD", "H1", _ts(14)),  # 4h gap in 1h data
        ])
        description = repo.describe("EURUSD", "H1", expected_minutes=60)
        assert description is not None
        assert description.row_count == 2
        assert description.gaps == 1  # 12:00→14:00 gap detected, never filled
