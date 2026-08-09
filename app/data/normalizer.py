"""Transforms external vendor formats into standardized domain representations."""

from typing import Any

import pandas as pd

from app.data.models import Candle


class DataNormalizer:
    """Standardizes external raw data records into internal models."""

    @staticmethod
    def normalize_vendor_dict(raw_data: dict[str, Any], symbol: str, timeframe: str) -> Candle:
        """Converts vendor raw dictionary record into an internal Candle model."""
        return Candle(
            symbol=symbol.upper(),
            timeframe=timeframe.lower(),
            timestamp=pd.to_datetime(raw_data["ts"] if "ts" in raw_data else raw_data["timestamp"]),
            open=float(raw_data["o"] if "o" in raw_data else raw_data["open"]),
            high=float(raw_data["h"] if "h" in raw_data else raw_data["high"]),
            low=float(raw_data["l"] if "l" in raw_data else raw_data["low"]),
            close=float(raw_data["c"] if "c" in raw_data else raw_data["close"]),
            volume=float(raw_data.get("v", raw_data.get("volume", 0.0))) or None,
        )

    @staticmethod
    def candles_to_df(candles: list[Candle]) -> pd.DataFrame:
        """Converts a sequence of internal Candles into a standard Pandas DataFrame."""
        return pd.DataFrame([c.model_dump() for c in candles])