"""Tests for market structure (HH/HL/LH/LL) and break-of-structure events."""

import pandas as pd

from app.market_structure.models import (
    BreakType,
    StructureType,
    Swing,
    SwingType,
)
from app.market_structure.structure import build_structure, detect_breaks


def _make_swing(
    timestamp,
    swing_type: SwingType,
    price: float,
    confirmation=None,
    symbol: str = "EURUSD",
    timeframe: str = "1h",
) -> Swing:
    conf = confirmation if confirmation is not None else timestamp
    return Swing(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        swing_type=swing_type,
        price=price,
        confirmation_timestamp=conf,
        available_from=conf,
        left=3,
        right=3,
    )


def _make_ohlc(highs, lows, closes):
    n = len(highs)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": closes.copy(), "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestBuildStructure:
    def test_higher_high_sequence(self) -> None:
        idx = pd.date_range("2024-01-01", periods=7, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.HIGH, 10.0, confirmation=idx[4]),
            _make_swing(idx[2], SwingType.LOW, 5.0, confirmation=idx[5]),
            _make_swing(idx[3], SwingType.HIGH, 12.0, confirmation=idx[6]),
        ]
        points = build_structure(swings, "EURUSD", "1h")
        high_points = [
            p for p in points
            if p.structure_type in (StructureType.HIGHER_HIGH, StructureType.LOWER_HIGH)
        ]
        assert len(high_points) == 1
        assert high_points[0].structure_type == StructureType.HIGHER_HIGH
        assert high_points[0].price == 12.0
        assert high_points[0].prior_price == 10.0

    def test_lower_low_sequence(self) -> None:
        idx = pd.date_range("2024-01-01", periods=7, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.LOW, 10.0, confirmation=idx[4]),
            _make_swing(idx[2], SwingType.HIGH, 15.0, confirmation=idx[5]),
            _make_swing(idx[3], SwingType.LOW, 8.0, confirmation=idx[6]),
        ]
        points = build_structure(swings, "EURUSD", "1h")
        low_points = [
            p for p in points
            if p.structure_type in (StructureType.HIGHER_LOW, StructureType.LOWER_LOW)
        ]
        assert len(low_points) == 1
        assert low_points[0].structure_type == StructureType.LOWER_LOW
        assert low_points[0].price == 8.0
        assert low_points[0].prior_price == 10.0

    def test_bullish_structure_sequence(self) -> None:
        """Low -> Higher High -> Higher Low -> Higher High: bullish sequence."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.LOW, 5.0, confirmation=idx[4]),
            _make_swing(idx[2], SwingType.HIGH, 10.0, confirmation=idx[5]),
            _make_swing(idx[3], SwingType.LOW, 6.0, confirmation=idx[6]),
            _make_swing(idx[4], SwingType.HIGH, 12.0, confirmation=idx[7]),
        ]
        points = build_structure(swings, "EURUSD", "1h")
        types = [p.structure_type for p in points]
        assert StructureType.HIGHER_HIGH in types
        assert StructureType.HIGHER_LOW in types

    def test_empty_swings_returns_empty(self) -> None:
        assert build_structure([], "EURUSD", "1h") == []


class TestDetectBreaks:
    def test_close_break_above(self) -> None:
        idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
        swing = _make_swing(idx[2], SwingType.HIGH, 10.0, confirmation=idx[4])
        highs = [5.0] * 12
        lows = [4.0] * 12
        closes = [4.5] * 12
        highs[7] = 11.0
        closes[7] = 10.5
        data = _make_ohlc(highs, lows, closes)
        # min_move_pct=6.0: the 5% move beyond 10.0 (close 10.5) is below the
        # 6% threshold, so this classifies as a plain CLOSE_BREAK, not a
        # confirmed break.
        events = detect_breaks(data, [swing], "EURUSD", "1h", confirm_bars=0, min_move_pct=6.0)
        up_breaks = [e for e in events if e.direction == "up"]
        assert len(up_breaks) >= 1
        assert up_breaks[0].break_type == BreakType.CLOSE_BREAK
        assert up_breaks[0].level == 10.0

    def test_wick_only_breach(self) -> None:
        idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
        swing = _make_swing(idx[2], SwingType.HIGH, 10.0, confirmation=idx[4])
        highs = [5.0] * 12
        lows = [4.0] * 12
        closes = [4.5] * 12
        highs[6] = 10.5
        closes[6] = 9.5
        data = _make_ohlc(highs, lows, closes)
        events = detect_breaks(data, [swing], "EURUSD", "1h", confirm_bars=0)
        up_wick = [
            e for e in events
            if e.direction == "up" and e.break_type == BreakType.WICK_BREACH
        ]
        assert len(up_wick) == 1
        assert up_wick[0].level == 10.0
        assert up_wick[0].available_from == data.index[6]

    def test_confirmed_break_timing(self) -> None:
        idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
        swing = _make_swing(idx[2], SwingType.HIGH, 10.0, confirmation=idx[4])
        highs = [5.0] * 12
        lows = [4.0] * 12
        closes = [4.5] * 12
        highs[6] = 11.0
        closes[6] = 10.5
        closes[7] = 10.6
        closes[8] = 10.7
        data = _make_ohlc(highs, lows, closes)
        events = detect_breaks(data, [swing], "EURUSD", "1h", confirm_bars=2)
        confirmed = [
            e for e in events
            if e.break_type == BreakType.CONFIRMED_BREAK and e.direction == "up"
        ]
        assert len(confirmed) == 1
        assert confirmed[0].confirmation_timestamp == data.index[8]
        assert confirmed[0].available_from == data.index[8]