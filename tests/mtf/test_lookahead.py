"""MTF look-ahead regression tests — 8 mandatory scenarios.

A lower-timeframe observation must never see information from an unfinished
higher-timeframe candle, future events, or future candles.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.market_structure.models import (
    LiquidityZone,
    MarketStructureResult,
)
from app.mtf import MtfConfig, MtfContextBuilder, MtfEngine


def _frame(start, periods, freq, seed=1):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = np.random.default_rng(seed).normal(1.08, 0.01, periods)
    return pd.DataFrame(
        {"open": close, "high": close + 0.002, "low": close - 0.002, "close": close},
        index=idx,
    )


def _valid_structure(symbol="EURUSD", tf="1h"):
    """A minimal valid MarketStructureResult (no future events)."""
    return MarketStructureResult(
        symbol=symbol,
        timeframe=tf,
        swings=[],
        structure=[],
        breaks=[],
        liquidity_zones=[],
        sweeps=[],
        displacement=[],
        ranges=[],
    )


def _ctx_key(ctx) -> tuple:
    return tuple(
        (
            t.timeframe,
            t.present,
            t.trend_state,
            t.volatility_state,
            t.market_state,
            t.structural_bias,
            len(t.liquidity_zones),
            len(t.sweeps),
        )
        for t in ctx.hierarchy
    ) + (ctx.alignment.value, tuple(ctx.alignment_reasons))


class TestIncompleteCandle:
    def test_scenario1_incomplete_h1_cannot_affect_m15(self):
        """M15 observation at 09:45 must see H1 08:00-09:00, never 09:00+."""
        h1 = _frame("2026-08-03", 2000, "1h", seed=3)
        # Observation just before the 10:00 M15 bar → H1 09:00 is incomplete.
        obs_ts = datetime(2026, 8, 3, 9, 45, tzinfo=timezone.utc)
        # Build a single context via the builder for the H1 tier.
        builder = MtfContextBuilder(MtfConfig(), "EURUSD")
        tc = builder.build("1h", obs_ts, h1, None, _valid_structure(), [], None)
        assert tc.present is True
        assert tc.candle_open == datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)
        assert tc.candle_open != datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)

    def test_scenario2_incomplete_h4_cannot_affect_m15(self):
        """M15 observation at 09:45 must see H4 04:00-08:00, never 08:00+."""
        obs_ts = datetime(2026, 8, 3, 9, 45, tzinfo=timezone.utc)
        h4 = _frame("2026-08-03", 2000, "4h", seed=4)
        builder = MtfContextBuilder(MtfConfig(), "EURUSD")
        tc = builder.build("4h", obs_ts, h4, None, _valid_structure(), [], None)
        assert tc.present is True
        assert tc.candle_open == datetime(2026, 8, 3, 4, 0, tzinfo=timezone.utc)
        assert tc.candle_open != datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)


class TestFutureData:
    def test_scenario3_future_d1_cannot_change_earlier_m15(self):
        base = _frame("2026-08-01", 2000, "15min", seed=5)
        d1_full = _frame("2026-08-01", 60, "1d", seed=6)
        d1_modified = d1_full.copy()
        d1_modified.iloc[-10:] *= 2.0  # mutate future D1 candles

        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1d",)), "EURUSD"
        )
        ctx_full = mtf.analyze(
            {"15m": base, "1d": d1_full}, base_timeframe="15m"
        )
        ctx_mod = mtf.analyze(
            {"15m": base, "1d": d1_modified}, base_timeframe="15m"
        )
        cutoff = len(base) - 100
        assert _ctx_key(ctx_full[cutoff - 1]) == _ctx_key(ctx_mod[cutoff - 1])

    def test_scenario4_future_liquidity_not_in_historical_ctx(self):
        """A future liquidity zone must not appear in earlier MTF context."""
        now_zone = LiquidityZone(
            symbol="EURUSD", timeframe="1h", zone_type="equal_lows",
            upper=1.081, lower=1.079, mid=1.080, swing_count=2,
            first_timestamp=datetime(2026, 8, 1, tzinfo=timezone.utc),
            last_timestamp=datetime(2026, 8, 1, tzinfo=timezone.utc),
            available_from=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        future_zone = now_zone.model_copy(
            update={"available_from": datetime(2026, 8, 10, tzinfo=timezone.utc)}
        )
        structure = _valid_structure()
        structure = structure.model_copy(
            update={"liquidity_zones": [now_zone, future_zone]}
        )
        obs = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
        builder = MtfContextBuilder(MtfConfig(), "EURUSD")
        tc = builder.build("1h", obs, _frame("2026-08-01", 100, "1h"), None, structure, [], None)
        # Only the available (08-01) zone appears; the future one is excluded.
        assert len(tc.liquidity_zones) == 1

    def test_scenario5_future_news_not_before_available_from(self):
        from app.news.models import EconomicEvent, EventImportance

        future = EconomicEvent(
            event_id="f", scheduled_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
            country="US", currency="USD", event_name="X", importance=EventImportance.HIGH,
            available_from=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )
        obs = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        builder = MtfContextBuilder(MtfConfig(), "EURUSD")
        tc = builder.build(
            "1h", obs, _frame("2026-08-01", 100, "1h"),
            None, _valid_structure(), [], [future],
        )
        assert tc.news_risk_max is None  # future event excluded

    def test_scenario6_mutating_future_candles_does_not_change_past_ctx(self):
        base = _frame("2026-08-01", 2000, "15min", seed=7)
        h1_full = _frame("2026-08-01", 2000, "1h", seed=8)
        h1_mod = h1_full.copy()
        h1_mod.iloc[-200:] *= 2.0
        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)), "EURUSD"
        )
        ctx_full = mtf.analyze({"15m": base, "1h": h1_full}, "15m")
        ctx_mod = mtf.analyze({"15m": base, "1h": h1_mod}, "15m")
        cutoff = len(base) - 200
        assert _ctx_key(ctx_full[cutoff - 1]) == _ctx_key(ctx_mod[cutoff - 1])

    def test_scenario7_rerun_identical(self):
        base = _frame("2026-08-01", 2000, "15min", seed=20)
        h1 = _frame("2026-08-01", 2000, "1h", seed=21)
        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)), "EURUSD"
        )
        a = mtf.analyze({"15m": base, "1h": h1}, "15m")
        b = mtf.analyze({"15m": base, "1h": h1}, "15m")
        assert [_ctx_key(x) for x in a] == [_ctx_key(x) for x in b]

    def test_scenario8_missing_tf_not_fabricated(self):
        base = _frame("2026-08-01", 2000, "15min", seed=30)
        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("4h",)), "EURUSD"
        )
        # No 4h data provided → the 4h tier must be present=False, never
        # fabricated with fabricated values.
        ctxs = mtf.analyze({"15m": base}, "15m")
        for ctx in ctxs:
            htf_tier = [t for t in ctx.hierarchy if t.timeframe == "4h"]
            assert htf_tier, "expected a 4h tier entry"
            assert htf_tier[0].present is False
            assert htf_tier[0].trend_state is None