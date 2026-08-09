"""Tests for the displacement metric."""

import numpy as np
import pandas as pd
import pytest

from app.market_structure.displacement import compute_displacement
from app.market_structure.errors import DisplacementError
from app.market_structure.models import DisplacementClass


def _make_ohlc(highs, lows, opens, closes):
    n = len(highs)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestDisplacement:
    def test_normal_candle(self) -> None:
        # 40 bars with varied-but-small ranges -> mix of normal classifications
        n = 40
        highs = [100.0 + i * 0.1 + (i % 3) * 0.05 for i in range(n)]
        lows = [99.9 + i * 0.1 - (i % 3) * 0.05 for i in range(n)]
        opens = [100.0 + i * 0.1 for i in range(n)]
        closes = [100.05 + i * 0.1 for i in range(n)]
        data = _make_ohlc(highs, lows, opens, closes)
        events = compute_displacement(data, "EURUSD", "1h", atr_window=5)
        # First bars have NaN range_ratio (ATR not yet defined)
        assert np.isnan(events[0].range_ratio)
        # Most bars should be normal
        classes = {e.classification for e in events[10:]}
        assert DisplacementClass.NORMAL in classes

    def test_unusually_large_candle(self) -> None:
        # Many small bars then one huge bar -> large/extreme classification
        n = 60
        highs = [100.0 + i * 0.1 for i in range(n)]
        lows = [99.9 + i * 0.1 for i in range(n)]
        opens = [100.0 + i * 0.1 for i in range(n)]
        closes = [100.05 + i * 0.1 for i in range(n)]
        # Make the last bar huge
        highs[-1] = 115.0
        lows[-1] = 99.0
        opens[-1] = 100.0
        closes[-1] = 114.0
        data = _make_ohlc(highs, lows, opens, closes)
        events = compute_displacement(data, "EURUSD", "1h", atr_window=5)
        last = events[-1]
        assert last.classification in (DisplacementClass.LARGE, DisplacementClass.EXTREME)
        assert last.range_ratio > 1.0

    def test_direction_detection(self) -> None:
        n = 40
        highs = [100.0 + i * 0.1 for i in range(n)]
        lows = [99.9 + i * 0.1 for i in range(n)]
        opens = [100.0 + i * 0.1 for i in range(n)]
        closes = [100.5 + i * 0.1 for i in range(n)]  # always above open
        data = _make_ohlc(highs, lows, opens, closes)
        events = compute_displacement(data, "EURUSD", "1h", atr_window=5)
        # Bars beyond ATR seed should be 'up'
        assert all(e.direction == "up" for e in events[5:])

    def test_insufficient_data_raises(self) -> None:
        n = 10
        highs = [100.0] * n
        lows = [99.0] * n
        opens = [99.5] * n
        closes = [99.5] * n
        data = _make_ohlc(highs, lows, opens, closes)
        with pytest.raises(DisplacementError):
            compute_displacement(data, "EURUSD", "1h", atr_window=20)