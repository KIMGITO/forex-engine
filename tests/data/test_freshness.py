"""Deterministic tests for data-freshness classification."""

from datetime import datetime, timedelta, timezone

from app.data.freshness import classify_freshness
from app.data.repository import ParquetMarketDataRepository


def _ts(hours=0):
    return datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc) + timedelta(hours=hours)


class TestFreshness:
    def test_fresh_when_recent(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        # Latest local = now (within 1h of now).
        repo.save_candles(
            [
                Candle(
                    symbol="EURUSD",
                    timeframe="1h",
                    timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                )
            ]
        )
        report = classify_freshness(repo, "EURUSD", "1h", interval_minutes=60)
        assert report.status == "fresh"

    def test_stale_when_old(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        repo.save_candles(
            [
                Candle(
                    symbol="EURUSD",
                    timeframe="1h",
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=100),
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                )
            ]
        )
        report = classify_freshness(repo, "EURUSD", "1h", interval_minutes=60)
        assert report.status == "stale"

    def test_unknown_when_no_local(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        report = classify_freshness(repo, "EURUSD", "1h", interval_minutes=60)
        assert report.status == "unknown"


from app.data.models import Candle