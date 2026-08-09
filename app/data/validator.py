"""Data quality and integrity checks for DataFrames and Candle sequences."""

import numpy as np
import pandas as pd

from app.data.exceptions import ValidationError


class DataValidator:
    """Validates structural and temporal integrity of market data pandas DataFrames."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """Runs strict checks on a raw/normalized market dataframe."""
        required_cols = {"timestamp", "open", "high", "low", "close"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Missing required market data columns: {missing_cols}")

        # Missing or infinite values check
        price_cols = ["open", "high", "low", "close"]
        if df[price_cols].isna().any().any():
            raise ValidationError("Dataset contains missing (NaN) price values.")
        if np.isinf(df[price_cols].to_numpy()).any():
            raise ValidationError("Dataset contains infinite price values.")

        # Temporal integrity
        timestamps = pd.to_datetime(df["timestamp"])
        if not timestamps.is_monotonic_increasing:
            raise ValidationError("Timestamps are not strictly sorted in chronological order.")
        if timestamps.duplicated().any():
            raise ValidationError("Dataset contains duplicate timestamps.")

        # Financial OHLC integrity vector check
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])

        if invalid_high.any() or invalid_low.any():
            raise ValidationError("Dataset contains mathematically invalid OHLC relationship records.")