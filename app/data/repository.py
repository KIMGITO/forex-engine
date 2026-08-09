"""Local research storage handling Parquet file persistency."""

from pathlib import Path

import pandas as pd

from app.data.exceptions import StorageError
from app.data.models import Candle
from app.data.normalizer import DataNormalizer


class ParquetMarketDataRepository:
    """Independent Parquet file repository for local historical research."""

    def __init__(self, base_storage_path: str = "data/processed"):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, symbol: str, timeframe: str) -> Path:
        return self.base_path / f"{symbol.lower()}_{timeframe.lower()}.parquet"

    def save_candles(self, candles: list[Candle]) -> None:
        """Saves candles to a local Parquet partition."""
        if not candles:
            return

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        df = DataNormalizer.candles_to_df(candles)
        filepath = self._get_filepath(symbol, timeframe)

        try:
            df.to_parquet(filepath, index=False)
        except Exception as e:
            raise StorageError(f"Failed to write Parquet storage file at {filepath}: {e!s}") from e

    def load_candles_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Loads market data as a pandas DataFrame."""
        filepath = self._get_filepath(symbol, timeframe)
        if not filepath.exists():
            raise StorageError(f"Market data file not found at {filepath}")

        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            raise StorageError(f"Failed to load Parquet storage file at {filepath}: {e!s}") from e