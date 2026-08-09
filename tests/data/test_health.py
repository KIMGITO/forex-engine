"""Deterministic tests for market-data health reporting."""

from datetime import datetime, timezone

from app.data.health import check_health
from app.data.repository import ParquetMarketDataRepository


def _make_candle(ts):
    return {
        "symbol": "EURUSD",
        "timeframe": "1h",
        "timestamp": ts,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 100.0,
    }


class TestHealth:
    def test_missing_local_reports_issue(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        health = check_health(None, repo, "EURUSD", "1h")
        assert health.latest_local_timestamp is None
        assert any("no local data" in issue for issue in health.issues)

    def test_duplicates_counted(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        ts = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        repo.save_candles(
            [
                Candle(
                    symbol="EURUSD",
                    timeframe="1h",
                    timestamp=ts,
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                ),
                Candle(
                    symbol="EURUSD",
                    timeframe="1h",
                    timestamp=ts,
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                ),
            ]
        )
        health = check_health(None, repo, "EURUSD", "1h")
        assert health.duplicate_count >= 1

    def test_staleness_detected(self, tmp_path):
        repo = ParquetMarketDataRepository(base_storage_path=str(tmp_path))
        remote = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)
        local = datetime(2024, 6, 9, 0, 0, tzinfo=timezone.utc)
        repo.save_candles(
            [
                Candle(
                    symbol="EURUSD",
                    timeframe="1h",
                    timestamp=local,
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=100.0,
                )
            ]
        )
        health = check_health(None, repo, "EURUSD", "1h", remote_latest=remote)
        assert health.is_stale is True


from app.data.models import Candle