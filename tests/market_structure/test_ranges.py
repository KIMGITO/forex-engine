"""Tests for range/consolidation detection."""

import numpy as np
import pandas as pd
import pytest

from app.market_structure.errors import RangeError
from app.market_structure.ranges import detect_ranges


def _make_ohlc(highs, lows, closes):
    n = len(highs)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": closes.copy(), "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestRanges:
    def test_insufficient_data_raises(self) -> None:
        n = 20
        data = _make_ohlc([100.0] * n, [99.0] * n, [99.5] * n)
        with pytest.raises(RangeError):
            detect_ranges(
                data, "EURUSD", "1h",
                atr_window=14, range_window=30, min_range_bars=5,
            )

    def test_compressed_range_detected(self) -> None:
        """A long flat/low-volatility segment should be flagged as a range."""
        n = 120
        # Volatile first segment
        highs = [100.0 + np.sin(i) * 2.0 for i in range(30)]
        lows = [98.0 + np.sin(i) * 2.0 for i in range(30)]
        closes = [99.0 + np.sin(i) * 1.0 for i in range(30)]
        # Compressed second segment (very tight range)
        for i in range(30, n):
            highs.append(100.05)
            lows.append(99.95)
            closes.append(100.0)
        data = _make_ohlc(highs, lows, closes)
        ranges = detect_ranges(
            data, "EURUSD", "1h",
            atr_window=5, range_window=20, min_range_bars=15,
        )
        # Expect at least one range covering the compressed segment
        assert len(ranges) >= 1
        # The range should cover the compressed portion
        assert ranges[0].lower >= 99.9
        assert ranges[0].upper <= 100.1

    def test_expanding_market_no_range(self) -> None:
        """An expanding/trending market should produce no range events."""
        n = 100
        highs = [100.0 + i * 0.5 for i in range(n)]
        lows = [99.0 + i * 0.5 for i in range(n)]
        closes = [99.5 + i * 0.5 for i in range(n)]
        data = _make_ohlc(highs, lows, closes)
        ranges = detect_ranges(
            data, "EURUSD", "1h",
            atr_window=5, range_window=20, min_range_bars=15,
            compression_threshold=0.85,
        )
        # In a steadily expanding market, ATR grows with price, so the
        # compression ratio stays near 1.0 -> no compressed runs.
        assert len(ranges) == 0

    def test_configurable_threshold(self) -> None:
        """A very loose threshold should detect more ranges than a strict one."""
        n = 80
        highs = [100.0 + (i % 4) * 0.1 for i in range(n)]
        lows = [99.9 + (i % 4) * 0.1 for i in range(n)]
        closes = [100.0 for _ in range(n)]
        data = _make_ohlc(highs, lows, closes)
        loose = detect_ranges(
            data, "EURUSD", "1h",
            atr_window=5, range_window=20, min_range_bars=10,
            compression_threshold=0.99,
        )
        strict = detect_ranges(
            data, "EURUSD", "1h",
            atr_window=5, range_window=20, min_range_bars=10,
            compression_threshold=0.5,
        )
        assert len(loose) >= len(strict)