"""Local research storage handling Parquet file persistency.

Extends the original repository with additive methods for:
- latest-timestamp lookups
- append-with-dedup (for incremental sync)
- provenance sidecar read/write

Existing ``save_candles``/``load_candles_df`` behavior is preserved unchanged.
"""

from pathlib import Path

import pandas as pd

from app.data.exceptions import StorageError
from app.data.models import Candle
from app.data.normalizer import DataNormalizer
from app.data.provenance import (
    ProviderMetadata,
    read_metadata,
    write_metadata,
)

__all__ = ["ParquetMarketDataRepository"]


class ParquetMarketDataRepository:
    """Independent Parquet file repository for local historical research."""

    def __init__(self, base_storage_path: str = "data/processed"):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, symbol: str, timeframe: str) -> Path:
        # Sanitize the symbol for filesystem safety (e.g. EUR/USD -> eur_usd).
        safe_symbol = symbol.lower().replace("/", "_").replace("\\", "_").strip("_")
        return self.base_path / f"{safe_symbol}_{timeframe.lower()}.parquet"

    def save_candles(self, candles: list[Candle]) -> None:
        """Saves candles to a local Parquet partition.

        Existing behavior: overwrite the partition with the supplied candles.
        """
        if not candles:
            return

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        df = DataNormalizer.candles_to_df(candles)
        filepath = self._get_filepath(symbol, timeframe)

        try:
            df.to_parquet(filepath, index=False)
        except Exception as e:
            raise StorageError(f"Failed to write Parquet storage file at {filepath}: {e}") from e

    def append_candles(self, candles: list[Candle]) -> None:
        """Append candles, deduplicating on timestamp.

        New candles overwrite existing rows with identical timestamps. This is
        intended for incremental sync where an overlap window is intentionally
        re-downloaded.
        """
        if not candles:
            return

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        filepath = self._get_filepath(symbol, timeframe)

        incoming = DataNormalizer.candles_to_df(candles)
        if filepath.exists():
            existing = pd.read_parquet(filepath)
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged["timestamp"] = pd.to_datetime(merged["timestamp"])
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
            merged = merged.sort_values("timestamp").reset_index(drop=True)
        else:
            merged = incoming.sort_values("timestamp").reset_index(drop=True)

        try:
            merged.to_parquet(filepath, index=False)
        except Exception as e:
            raise StorageError(f"Failed to append Parquet storage file at {filepath}: {e}") from e

    def load_candles_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Loads market data as a pandas DataFrame."""
        filepath = self._get_filepath(symbol, timeframe)
        if not filepath.exists():
            raise StorageError(f"Market data file not found at {filepath}")

        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            raise StorageError(f"Failed to load Parquet storage file at {filepath}: {e}") from e

    def latest_timestamp(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        """Return the latest candle timestamp for a dataset, or None if missing."""
        filepath = self._get_filepath(symbol, timeframe)
        if not filepath.exists():
            return None
        try:
            df = pd.read_parquet(filepath, columns=["timestamp"])
            if df.empty:
                return None
            ts = pd.to_datetime(df["timestamp"]).dropna()
            if ts.empty:
                return None
            return ts.max()
        except Exception:  # noqa: BLE001 - unreadable/missing treated as no data
            return None

    def filepath(self, symbol: str, timeframe: str) -> Path:
        """Public accessor for the dataset filepath (for sidecar metadata)."""
        return self._get_filepath(symbol, timeframe)

    def save_candles_with_meta(
        self,
        candles: list[Candle],
        metadata: ProviderMetadata,
    ) -> None:
        """Persist candles + provenance sidecar."""
        self.save_candles(candles)
        if candles:
            fp = self.filepath(candles[0].symbol, candles[0].timeframe)
            write_metadata(fp, metadata)

    def load_metadata(self, symbol: str, timeframe: str) -> ProviderMetadata | None:
        """Load provenance sidecar, if present."""
        fp = self.filepath(symbol, timeframe)
        return read_metadata(fp)