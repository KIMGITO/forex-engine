"""Look-ahead regression tests for the market-structure engine.

The invariant: changing future candles must never cause an event to become
available *earlier* than in the full analysis. Any event whose
``available_from`` lies strictly before the first modified bar must exist in
both analyses with the identical ``available_from``.
"""

import numpy as np
import pandas as pd

from app.market_structure.engine import MarketStructureEngine


def _make_varied_ohlc(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
        },
        index=idx,
    )


def _swing_key(event) -> tuple:
    return (event.timestamp, event.swing_type.value, round(event.price, 6), event.available_from)


def _break_key(event) -> tuple:
    return (
        event.timestamp,
        event.break_type.value,
        event.direction,
        round(event.level, 6),
        event.available_from,
    )


def _zone_key(event) -> tuple:
    return (
        event.zone_type,
        round(event.mid, 6),
        event.swing_count,
        event.available_from,
    )


class TestLookAheadProtection:
    """Future data must never advance an event's availability."""

    def test_swing_availability_never_advances(self) -> None:
        data = _make_varied_ohlc()
        engine = MarketStructureEngine()
        full = engine.analyze(data, symbol="EURUSD", timeframe="1h")

        # Modify the last 30 bars drastically (future-only perturbation).
        modified = data.copy()
        modified.iloc[-30:] *= 2.0
        recomputed = engine.analyze(modified, symbol="EURUSD", timeframe="1h")

        cutoff = data.index[-30]

        full_early = {_swing_key(s) for s in full.swings if s.available_from < cutoff}
        mod_early = {_swing_key(s) for s in recomputed.swings if s.available_from < cutoff}

        # Every swing available before the cutoff in the modified analysis must
        # also be available (with the same availability) in the full analysis.
        assert mod_early <= full_early

    def test_break_availability_never_advances(self) -> None:
        data = _make_varied_ohlc()
        engine = MarketStructureEngine()
        full = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        modified = data.copy()
        modified.iloc[-30:] *= 2.0
        recomputed = engine.analyze(modified, symbol="EURUSD", timeframe="1h")

        cutoff = data.index[-30]

        full_early = {_break_key(b) for b in full.breaks if b.available_from < cutoff}
        mod_early = {_break_key(b) for b in recomputed.breaks if b.available_from < cutoff}

        assert mod_early <= full_early

    def test_zones_availability_never_advances(self) -> None:
        data = _make_varied_ohlc()
        engine = MarketStructureEngine()
        full = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        modified = data.copy()
        modified.iloc[-30:] *= 2.0
        recomputed = engine.analyze(modified, symbol="EURUSD", timeframe="1h")

        cutoff = data.index[-30]

        full_early = {_zone_key(z) for z in full.liquidity_zones if z.available_from < cutoff}
        mod_early = {_zone_key(z) for z in recomputed.liquidity_zones if z.available_from < cutoff}

        assert mod_early <= full_early

    def test_displacement_is_causal(self) -> None:
        """Per-bar displacement at T must only depend on information <= T."""
        data = _make_varied_ohlc()
        engine = MarketStructureEngine()
        full = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        modified = data.copy()
        modified.iloc[-30:] *= 5.0
        recomputed = engine.analyze(modified, symbol="EURUSD", timeframe="1h")

        cutoff = data.index[-30]

        def _key(d) -> tuple:
            # Normalize NaN into a string sentinel so equality is well-defined
            # in the comparison set (NaN != NaN otherwise).
            ratio = "nan" if np.isnan(d.range_ratio) else round(d.range_ratio, 6)
            return (d.timestamp, ratio, d.classification.value, d.direction)

        full_early = {_key(d) for d in full.displacement if d.timestamp < cutoff}
        mod_early = {_key(d) for d in recomputed.displacement if d.timestamp < cutoff}

        # Displacement before the cutoff must be identical even though future
        # bars changed.
        assert mod_early == full_early

    def test_no_event_has_available_from_after_end(self) -> None:
        """Sanity: no event's availability can exceed the data end."""
        # 60 bars >= range_window(30) + atr_window(14) required by detect_ranges.
        data = _make_varied_ohlc(60)
        engine = MarketStructureEngine()
        result = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        end = data.index[-1]
        for event_list in (
            result.swings,
            result.breaks,
            result.liquidity_zones,
            result.sweeps,
            result.ranges,
            result.displacement,
        ):
            for event in event_list:
                assert event.available_from <= end
