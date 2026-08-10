"""Real Twelve Data EUR/USD MTF integration test.

Uses the locally stored Twelve Data H1 dataset. Since the store only keeps H1,
we derive synthetic M15 (resampled down) and aggregate H4/D1 from H1 to build a
multi-timeframe hierarchy. This validates the MTF engine's causal alignment on
real market data — NOT profitability.
"""

import numpy as np
import pandas as pd
import pytest

from app.mtf import MtfConfig, MtfEngine

DATA_PATH = "data/processed/eurusd_1h.parquet"


def _load_h1() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close"]]


def _make_m15(h1: pd.DataFrame) -> pd.DataFrame:
    """Synthetic M15 frame derived from H1 (for MTF alignment testing only)."""
    idx = pd.date_range(h1.index[0], h1.index[-1], freq="15min", tz="UTC")
    # Slightly varied values around each H1 close so M15 candles exist.
    rng = np.random.default_rng(42)
    baseline = h1["close"].reindex(idx, method="ffill")
    close = baseline + rng.normal(0, 0.0001, len(idx))
    return pd.DataFrame(
        {"open": close, "high": close + 0.0003, "low": close - 0.0003, "close": close},
        index=idx,
    )


def _aggregate_h4(h1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate H1 into H4 (open-first, high-max, low-min, close-last)."""
    g = h1.resample("4h")
    return pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).dropna()


def _aggregate_d1(h1: pd.DataFrame) -> pd.DataFrame:
    g = h1.resample("1D")
    return pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).dropna()


class TestRealMtfIntegration:
    def test_m15_h1_h4_d1_alignment(self):
        try:
            h1 = _load_h1()
        except FileNotFoundError:
            pytest.skip("Real Twelve Data dataset not present.")
        assert len(h1) > 0

        m15 = _make_m15(h1)
        h4 = _aggregate_h4(h1)
        d1 = _aggregate_d1(h1)

        mtf = MtfEngine(
            MtfConfig(
                base_timeframe="15m",
                higher_timeframes=("1h", "4h", "1d"),
            ),
            "EURUSD",
        )
        contexts = mtf.analyze(
            {"15m": m15, "1h": h1, "4h": h4, "1d": d1},
            "15m",
        )
        assert len(contexts) > 0
        # Every context is causal (available_from == observation timestamp).
        for c in contexts:
            assert c.available_from == c.timestamp
            assert c.base_timeframe == "15m"
            # Higher-timeframe tiers are aligned to completed candles only:
            # a tier's available_from must be <= its observation timestamp.
            for t in c.hierarchy[1:]:
                if t.present:
                    assert t.available_from <= c.timestamp

    def test_h1_only_present_tiers_marked(self):
        try:
            h1 = _load_h1()
        except FileNotFoundError:
            pytest.skip("Real Twelve Data dataset not present.")
        m15 = _make_m15(h1)

        # Only M15 + H1 given; H4/D1 absent → present=False, never fabricated.
        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h", "4h", "1d")),
            "EURUSD",
        )
        contexts = mtf.analyze({"15m": m15, "1h": h1}, "15m")
        for c in contexts:
            tiers = {t.timeframe: t for t in c.hierarchy}
            assert tiers["4h"].present is False
            assert tiers["1d"].present is False
            assert tiers["4h"].trend_state is None