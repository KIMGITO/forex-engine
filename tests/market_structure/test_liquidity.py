"""Tests for liquidity zones (equal highs/lows) and sweep detection."""

import pandas as pd
import pytest

from app.market_structure.errors import LiquidityError
from app.market_structure.liquidity import detect_liquidity_zones, detect_sweeps
from app.market_structure.models import (
    SweepType,
    Swing,
    SwingType,
)


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


class TestLiquidityZones:
    def test_equal_highs_grouped(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.00, confirmation=idx[4]),
            _make_swing(idx[3], SwingType.HIGH, 100.02, confirmation=idx[5]),
        ]
        zones = detect_liquidity_zones(
            swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        assert len(zones) == 1
        assert zones[0].zone_type == "equal_highs"
        assert zones[0].swing_count == 2
        assert zones[0].upper == 100.02
        assert zones[0].lower == 100.00

    def test_equal_lows_grouped(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.LOW, 99.90, confirmation=idx[4]),
            _make_swing(idx[3], SwingType.LOW, 99.92, confirmation=idx[5]),
        ]
        zones = detect_liquidity_zones(
            swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        assert len(zones) == 1
        assert zones[0].zone_type == "equal_lows"
        assert zones[0].mid == pytest.approx(99.91)

    def test_tolerance_prevents_grouping(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.0, confirmation=idx[4]),
            _make_swing(idx[3], SwingType.HIGH, 105.0, confirmation=idx[5]),
        ]
        zones = detect_liquidity_zones(
            swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        assert len(zones) == 0

    def test_unrelated_highs_not_grouped(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.0, confirmation=idx[4]),
            _make_swing(idx[3], SwingType.HIGH, 100.0, confirmation=idx[5]),
            _make_swing(idx[2], SwingType.HIGH, 90.0, confirmation=idx[4]),
        ]
        zones = detect_liquidity_zones(
            swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        # The two 100.0s group; the 90.0 stays alone (not enough swings).
        assert len(zones) == 1

    def test_min_swings_too_high(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.0, confirmation=idx[4]),
            _make_swing(idx[3], SwingType.HIGH, 100.0, confirmation=idx[5]),
        ]
        zones = detect_liquidity_zones(
            swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=3
        )
        assert len(zones) == 0

    def test_invalid_tolerance_raises(self) -> None:
        with pytest.raises(LiquidityError):
            detect_liquidity_zones([], "EURUSD", "1h", tolerance_pct=-1.0)


class TestSweeps:
    def test_valid_high_sweep(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        zones_swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.0, confirmation=idx[3]),
            _make_swing(idx[2], SwingType.HIGH, 100.0, confirmation=idx[4]),
        ]
        zones = detect_liquidity_zones(
            zones_swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        assert len(zones) == 1

        highs = [99.0] * 10
        lows = [98.0] * 10
        closes = [98.5] * 10
        # Sweep above the zone at index 5, return below at index 6
        highs[5] = 101.0
        closes[6] = 99.5
        data = _make_ohlc(highs, lows, closes)
        events = detect_sweeps(data, zones, "EURUSD", "1h", sweep_bars=3)
        high_sweeps = [e for e in events if e.sweep_type == SweepType.HIGH_SWEEP]
        assert len(high_sweeps) == 1
        assert high_sweeps[0].level == 100.0
        assert high_sweeps[0].extreme_price == 101.0
        assert high_sweeps[0].close_price == 99.5
        assert high_sweeps[0].available_from == data.index[6]

    def test_valid_low_sweep(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        zones_swings = [
            _make_swing(idx[1], SwingType.LOW, 100.0, confirmation=idx[3]),
            _make_swing(idx[2], SwingType.LOW, 100.0, confirmation=idx[4]),
        ]
        zones = detect_liquidity_zones(
            zones_swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        assert len(zones) == 1

        highs = [101.0] * 10
        lows = [100.5] * 10
        closes = [100.8] * 10
        # Sweep below the zone at index 5, return above at index 6
        lows[5] = 99.0
        closes[6] = 100.5
        data = _make_ohlc(highs, lows, closes)
        events = detect_sweeps(data, zones, "EURUSD", "1h", sweep_bars=3)
        low_sweeps = [e for e in events if e.sweep_type == SweepType.LOW_SWEEP]
        assert len(low_sweeps) == 1
        assert low_sweeps[0].level == 100.0
        assert low_sweeps[0].extreme_price == 99.0

    def test_wick_without_return_is_not_sweep(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        zones_swings = [
            _make_swing(idx[1], SwingType.HIGH, 100.0, confirmation=idx[3]),
            _make_swing(idx[2], SwingType.HIGH, 100.0, confirmation=idx[4]),
        ]
        zones = detect_liquidity_zones(
            zones_swings, "EURUSD", "1h", tolerance_pct=0.1, min_swings=2
        )
        highs = [99.0] * 10
        lows = [98.0] * 10
        closes = [100.5] * 10  # closes stay ABOVE the zone
        # Wick above the zone, but no return close back below -> no sweep
        highs[5] = 101.0
        data = _make_ohlc(highs, lows, closes)
        events = detect_sweeps(data, zones, "EURUSD", "1h", sweep_bars=3)
        assert len(events) == 0

    def test_return_without_prior_level_is_not_sweep(self) -> None:
        highs = [99.0, 101.0, 99.0] + [98.0] * 7
        lows = [98.0] * 10
        closes = [98.5] * 10
        data = _make_ohlc(highs, lows, closes)
        # No zones -> no sweeps even though price spiked and returned.
        events = detect_sweeps(data, [], "EURUSD", "1h", sweep_bars=3)
        assert len(events) == 0