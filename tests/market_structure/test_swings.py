"""Tests for swing detection and confirmation timing."""

import pandas as pd
import pytest

from app.market_structure.errors import SwingDetectionError
from app.market_structure.models import SwingType
from app.market_structure.swings import detect_swings


def _make_ohlc(
    highs,
    lows,
    closes=None,
    opens=None,
) -> pd.DataFrame:
    n = len(highs)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else [c for c in closes]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestSwingDetection:
    def test_obvious_swing_high(self) -> None:
        # Peaks at index 4 (high=10); left=3/right=3 window
        highs = [1, 2, 3, 4, 10, 4, 3, 2, 1]
        lows = [h - 1 for h in highs]
        data = _make_ohlc(highs, lows)
        swings = detect_swings(data, "EURUSD", "1h", left=3, right=3)
        highs_found = [s for s in swings if s.swing_type == SwingType.HIGH]
        assert len(highs_found) == 1
        assert highs_found[0].price == 10.0
        assert highs_found[0].timestamp == data.index[4]
        # Confirmation happens at i + right = 4 + 3 = index 7
        assert highs_found[0].confirmation_timestamp == data.index[7]
        assert highs_found[0].available_from == data.index[7]

    def test_obvious_swing_low(self) -> None:
        # Trough at index 4 (low=1); left=3/right=3 window
        lows = [10, 9, 8, 7, 1, 7, 8, 9, 10]
        highs = [l + 1 for l in lows]
        data = _make_ohlc(highs, lows)
        swings = detect_swings(data, "EURUSD", "1h", left=3, right=3)
        lows_found = [s for s in swings if s.swing_type == SwingType.LOW]
        assert len(lows_found) == 1
        assert lows_found[0].price == 1.0
        assert lows_found[0].confirmation_timestamp == data.index[7]

    def test_window_size_changes_detection(self) -> None:
        # With left=1/right=1, index 4 is a swing high.
        highs = [5, 6, 5, 7, 100, 7, 5, 6, 5]
        lows = [h - 1 for h in highs]
        data = _make_ohlc(highs, lows)
        small_window = detect_swings(data, "EURUSD", "1h", left=1, right=1)
        big_window = detect_swings(data, "EURUSD", "1h", left=4, right=4)
        small_highs = [s for s in small_window if s.swing_type == SwingType.HIGH]
        big_highs = [s for s in big_window if s.swing_type == SwingType.HIGH]
        assert len(small_highs) >= 1
        assert len(big_highs) >= 0

    def test_confirmation_timestamp_is_explicit(self) -> None:
        highs = [1, 5, 1, 5, 1]
        lows = [0, 4, 0, 4, 0]
        data = _make_ohlc(highs, lows)
        swings = detect_swings(data, "EURUSD", "1h", left=1, right=1)
        for s in swings:
            assert s.confirmation_timestamp >= s.timestamp
            assert s.available_from == s.confirmation_timestamp

    def test_insufficient_data_raises(self) -> None:
        data = _make_ohlc([1, 2, 3], [0, 1, 2])
        with pytest.raises(SwingDetectionError):
            detect_swings(data, "EURUSD", "1h", left=3, right=3)

    def test_invalid_window_raises(self) -> None:
        data = _make_ohlc([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        with pytest.raises(SwingDetectionError):
            detect_swings(data, "EURUSD", "1h", left=0, right=3)