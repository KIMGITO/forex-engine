"""Unit test suite for the app.data market-data foundation layer."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from app.data.exceptions import StorageError
from app.data.exceptions import ValidationError as DataValidationError
from app.data.models import Candle
from app.data.normalizer import DataNormalizer
from app.data.provider import BaseMarketDataProvider, MockMarketDataProvider
from app.data.repository import ParquetMarketDataRepository
from app.data.validator import DataValidator


@pytest.fixture
def valid_candle() -> Candle:
    return Candle(
        symbol="EURUSD",
        timeframe="1h",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=1.0850,
        high=1.0858,
        low=1.0842,
        close=1.0855,
        volume=100.0,
    )


@pytest.fixture
def candle_list() -> list[Candle]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(
            symbol="EURUSD",
            timeframe="1h",
            timestamp=base,
            open=1.0850,
            high=1.0858,
            low=1.0842,
            close=1.0855,
            volume=100.0,
        ),
        Candle(
            symbol="EURUSD",
            timeframe="1h",
            timestamp=pd.Timestamp("2024-01-01 01:00:00", tz="UTC").to_pydatetime(),
            open=1.0855,
            high=1.0862,
            low=1.0848,
            close=1.0860,
            volume=150.0,
        ),
    ]


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "open": [1.0850] * 5,
            "high": [1.0858] * 5,
            "low": [1.0842] * 5,
            "close": [1.0855] * 5,
            "volume": [100.0] * 5,
        }
    )


@pytest.fixture
def repo(tmp_path) -> ParquetMarketDataRepository:
    return ParquetMarketDataRepository(base_storage_path=str(tmp_path / "processed"))


class TestCandleModel:
    """Pydantic Candle model boundary and relationship validation."""

    def test_valid_candle_passes(self, valid_candle: Candle) -> None:
        assert valid_candle.symbol == "EURUSD"
        assert valid_candle.volume == 100.0

    def test_invalid_high_below_open_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0849,
                low=1.0842,
                close=1.0855,
            )

    def test_invalid_high_below_close_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0854,
                low=1.0842,
                close=1.0855,
            )

    def test_invalid_high_below_low_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0838,
                low=1.0842,
                close=1.0855,
            )

    def test_invalid_low_above_open_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0858,
                low=1.0852,
                close=1.0855,
            )

    def test_invalid_low_above_close_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0858,
                low=1.0856,
                close=1.0855,
            )

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=0.0,
                high=1.0858,
                low=1.0842,
                close=1.0855,
            )

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Candle(
                symbol="EURUSD",
                timeframe="1h",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=1.0850,
                high=1.0858,
                low=1.0842,
                close=1.0855,
                volume=-1.0,
            )

    def test_model_is_frozen(self, valid_candle: Candle) -> None:
        with pytest.raises((TypeError, PydanticValidationError)):
            valid_candle.open = 1.1


class TestDataValidator:
    """DataFrame-level structural, temporal, and financial integrity checks."""

    def test_valid_dataframe_passes(self, valid_df: pd.DataFrame) -> None:
        DataValidator.validate_dataframe(valid_df)

    def test_missing_column_raises(self, valid_df: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(valid_df.drop(columns=["high"]))

    def test_nan_price_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[2, "open"] = np.nan
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(bad)

    def test_infinite_price_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[2, "close"] = np.inf
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(bad)

    def test_unsorted_timestamps_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.iloc[::-1].copy()
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(bad)

    def test_duplicate_timestamps_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[4, "timestamp"] = bad.loc[0, "timestamp"]
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(bad)

    def test_invalid_ohlc_raises(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[2, "high"] = bad.loc[2, "low"] - 0.01
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(bad)


class TestDataNormalizer:
    """Vendor dictionary normalization and Candle -> DataFrame conversion."""

    def test_short_key_mapping(self) -> None:
        candle = DataNormalizer.normalize_vendor_dict(
            {
                "ts": "2024-01-01T00:00:00Z",
                "o": 1.0850,
                "h": 1.0858,
                "l": 1.0842,
                "c": 1.0855,
                "v": 100.0,
            },
            "eurusd",
            "1H",
        )
        assert candle.symbol == "EURUSD"
        assert candle.timeframe == "1h"
        assert candle.open == 1.0850
        assert candle.high == 1.0858
        assert candle.low == 1.0842
        assert candle.close == 1.0855
        assert candle.volume == 100.0

    def test_long_key_mapping(self) -> None:
        candle = DataNormalizer.normalize_vendor_dict(
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 1.0850,
                "high": 1.0858,
                "low": 1.0842,
                "close": 1.0855,
                "volume": 42.0,
            },
            "EURUSD",
            "1h",
        )
        assert candle.symbol == "EURUSD"
        assert candle.volume == 42.0

    def test_volume_absent_defaults_to_none(self) -> None:
        candle = DataNormalizer.normalize_vendor_dict(
            {"ts": "2024-01-01T00:00:00Z", "o": 1.0850, "h": 1.0858, "l": 1.0842, "c": 1.0855},
            "EURUSD",
            "1h",
        )
        assert candle.volume is None

    def test_volume_zero_becomes_none(self) -> None:
        candle = DataNormalizer.normalize_vendor_dict(
            {"ts": "2024-01-01T00:00:00Z", "o": 1.0850, "h": 1.0858, "l": 1.0842, "c": 1.0855, "v": 0.0},
            "EURUSD",
            "1h",
        )
        assert candle.volume is None

    def test_missing_timestamp_key_raises(self) -> None:
        with pytest.raises(KeyError):
            DataNormalizer.normalize_vendor_dict(
                {"o": 1.0850, "h": 1.0858, "l": 1.0842, "c": 1.0855},
                "EURUSD",
                "1h",
            )

    def test_candles_to_df(self, candle_list: list[Candle]) -> None:
        df = DataNormalizer.candles_to_df(candle_list)
        assert list(df.columns) == [
            "symbol",
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 2
        assert df.loc[0, "symbol"] == "EURUSD"
        assert df.loc[1, "volume"] == 150.0


class TestMockMarketDataProvider:
    """Synthetic provider contract: volume, OHLC integrity, determinism."""

    START = datetime(2024, 1, 1, tzinfo=timezone.utc)
    END = datetime(2024, 1, 2, tzinfo=timezone.utc)

    def test_returns_non_empty_candles(self) -> None:
        provider = MockMarketDataProvider()
        candles = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        assert len(candles) > 0

    def test_symbol_prefixed_with_synthetic(self) -> None:
        provider = MockMarketDataProvider()
        candles = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        assert all(c.symbol == "SYNTHETIC_EURUSD" for c in candles)

    def test_ohlc_integrity(self) -> None:
        provider = MockMarketDataProvider()
        candles = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        for candle in candles:
            assert candle.high >= max(candle.open, candle.close, candle.low)
            assert candle.low <= min(candle.open, candle.close, candle.high)

    def test_monotonic_timestamps(self) -> None:
        provider = MockMarketDataProvider()
        candles = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        timestamps = [c.timestamp for c in candles]
        assert timestamps == sorted(timestamps)

    def test_deterministic_output(self) -> None:
        provider = MockMarketDataProvider()
        first = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        second = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        assert [(c.open, c.high, c.low, c.close) for c in first] == [
            (c.open, c.high, c.low, c.close) for c in second
        ]

    def test_fixed_volume(self) -> None:
        provider = MockMarketDataProvider()
        candles = provider.fetch_candles("EURUSD", "1h", self.START, self.END)
        assert all(c.volume == 100.0 for c in candles)

    def test_base_provider_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseMarketDataProvider()  # type: ignore[abstract]


class TestParquetMarketDataRepository:
    """Parquet persistence: round-trips, file layout, and error paths."""

    def test_save_and_load_roundtrip(
        self, repo: ParquetMarketDataRepository, candle_list: list[Candle]
    ) -> None:
        repo.save_candles(candle_list)
        loaded = repo.load_candles_df("EURUSD", "1h")
        assert len(loaded) == 2
        for col in ["open", "high", "low", "close"]:
            assert loaded[col].tolist() == [getattr(c, col) for c in candle_list]
        assert loaded["volume"].tolist() == [100.0, 150.0]

    def test_save_creates_partition_file(
        self, repo: ParquetMarketDataRepository, candle_list: list[Candle], tmp_path
    ) -> None:
        repo.save_candles(candle_list)
        assert (tmp_path / "processed" / "eurusd_1h.parquet").exists()

    def test_load_missing_file_raises(self, repo: ParquetMarketDataRepository) -> None:
        with pytest.raises(StorageError):
            repo.load_candles_df("EURUSD", "1h")

    def test_save_empty_list_is_noop(self, repo: ParquetMarketDataRepository, tmp_path) -> None:
        repo.save_candles([])
        assert len(list((tmp_path / "processed").glob("*.parquet"))) == 0

    def test_repository_creates_base_path(self, tmp_path) -> None:
        target = tmp_path / "nested" / "processed"
        ParquetMarketDataRepository(base_storage_path=str(target))
        assert target.is_dir()