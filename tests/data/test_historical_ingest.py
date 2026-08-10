"""Tests for provider-independent historical dataset ingestion (app/data/)."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.data.historical import (
    HistoricalIngestionConfig,
    infer_timeframe_from_filename,
    ingest_csv_file,
)
from app.data.models import Candle
from app.research.dataset import PartitionedResearchRepository


def _write_csv(path, text):
    path.write_text(text)
    return path


class TestSymbolAndTimeframeNormalization:
    def test_infer_timeframe_from_filename(self):
        assert infer_timeframe_from_filename("EAUD_15.csv") == "M15"
        assert infer_timeframe_from_filename("eurusd_1h.parquet") == "H1"
        assert infer_timeframe_from_filename("EURUSD_min1.csv") == "M1"
        assert infer_timeframe_from_filename("no_hint.csv") is None

    def test_header_aliases_accepted(self):
        p = _write_csv(Path("/tmp/alias_test.csv"), (
            "Datetime,Open,High,Low,Close,TickVolume\n"
            "2022-01-03 00:00:00,1.05000,1.05050,1.04980,1.05020,100\n"
            "2022-01-03 01:00:00,1.05020,1.05060,1.04990,1.05040,100\n"
        ))
        res = ingest_csv_file(p, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="kaggle"))
        assert res.rows_parsed == 2
        assert res.rows_rejected == 0


class TestParsingAndValidity:
    def test_parses_and_rejects_malformed(self):
        p = _write_csv(Path("/tmp/parse_test.csv"), (
            "timestamp,open,high,low,close,volume\n"
            "2022-01-03 00:00:00,1.05,1.06,1.04,1.055,100\n"
            "BAD_ROW,1.05,1.06,1.04,1.055,100\n"
            "2022-01-03 02:00:00,1.05,1.02,1.04,1.055,100\n"  # high < open
        ))
        res = ingest_csv_file(p, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="kaggle"))
        # First row valid, second (bad ts) rejected, third (invalid OHLC) rejected.
        assert res.rows_parsed == 1
        assert res.rows_rejected == 2

    def test_timestamp_timezone_aware_utc(self):
        p = _write_csv(Path("/tmp/tz_test.csv"), (
            "timestamp,open,high,low,close\n"
            "2022-01-03 00:00:00,1.05,1.06,1.04,1.055\n"
        ))
        res = ingest_csv_file(p, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="x"))
        c = res.candles[0]
        assert c.timestamp.tzinfo is not None

    def test_deduplicates_and_sorts(self):
        p = _write_csv(Path("/tmp/dedup_test.csv"), (
            "timestamp,open,high,low,close\n"
            "2022-01-03 02:00:00,1.06,1.07,1.05,1.065\n"
            "2022-01-03 00:00:00,1.05,1.06,1.04,1.055\n"
            "2022-01-03 02:00:00,9.90,9.91,9.89,9.90\n"  # duplicate ts (2nd wins)
        ))
        res = ingest_csv_file(p, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="x"))
        times = [c.timestamp for c in res.candles]
        assert times == sorted(times)
        assert len(res.candles) == 2  # 02:00 appears once
        c2 = next(c for c in res.candles if c.timestamp.hour == 2)
        assert abs(c2.close - 9.90) < 1e-9  # later duplicate wins


class TestAggregation:
    def test_m1_aggregates_to_h1(self):
        # 4 M1 bars -> 1 H1 bar (same hour).
        p = _write_csv(Path("/tmp/agg_test.csv"), (
            "timestamp,open,high,low,close,volume\n"
            "2022-01-03 00:00:00,1.0500,1.0502,1.0498,1.0501,10\n"
            "2022-01-03 00:01:00,1.0501,1.0505,1.0497,1.0503,20\n"
            "2022-01-03 00:02:00,1.0503,1.0506,1.0499,1.0502,30\n"
            "2022-01-03 00:03:00,1.0502,1.0504,1.0496,1.0500,40\n"
        ))
        res = ingest_csv_file(
            p,
            HistoricalIngestionConfig(symbol="EURUSD", timeframe="M1", source="x", aggregate_to="H1"),
        )
        assert res.timeframe == "H1"
        assert len(res.candles) == 1
        c = res.candles[0]
        assert abs(c.open - 1.0500) < 1e-9  # open-first
        assert abs(c.high - 1.0506) < 1e-9  # high-max
        assert abs(c.low - 1.0496) < 1e-9   # low-min
        assert abs(c.close - 1.0500) < 1e-9  # close-last


class TestPersistenceAndProvenance:
    def test_idempotent_partition_merge(self):
        repo = PartitionedResearchRepository(tempfile.mkdtemp())
        candles = [
            Candle(symbol="EURUSD", timeframe="H1",
                   timestamp=datetime(2022, 1, 3, tzinfo=timezone.utc),
                   open=1.05, high=1.06, low=1.04, close=1.055, volume=100),
            Candle(symbol="EURUSD", timeframe="H1",
                   timestamp=datetime(2022, 1, 3, 1, tzinfo=timezone.utc),
                   open=1.055, high=1.065, low=1.045, close=1.06, volume=100),
        ]
        _, f1 = repo.merge_candles(candles, symbol="EURUSD", timeframe="H1")
        _, f2 = repo.merge_candles(candles, symbol="EURUSD", timeframe="H1")
        assert f1 == f2 == 2  # idempotent: no duplicates after re-merge
        df = repo.load_df("EURUSD", "H1")
        assert len(df) == 2
        assert not df.index.duplicated().any()

    def test_deterministic_reingest_same_result(self):
        text = (
            "timestamp,open,high,low,close\n"
            "2022-01-03 00:00:00,1.05,1.06,1.04,1.055\n"
            "2022-01-03 01:00:00,1.055,1.065,1.045,1.06\n"
        )
        p1 = _write_csv(Path("/tmp/det1.csv"), text)
        p2 = _write_csv(Path("/tmp/det2.csv"), text)
        r1 = ingest_csv_file(p1, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="x"))
        r2 = ingest_csv_file(p2, HistoricalIngestionConfig(symbol="EURUSD", timeframe="H1", source="x"))
        assert [c.model_dump() for c in r1.candles] == [c.model_dump() for c in r2.candles]
