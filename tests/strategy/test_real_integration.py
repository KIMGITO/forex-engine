"""Real Twelve Data EUR/USD strategy integration (no profitability claims)."""

import pandas as pd
import pytest

from app.strategy import (
    HistoricalSignalScanner,
    LiquidityReversalStrategy,
    TrendStructureStrategy,
)

DATA_PATH = "data/processed/eurusd_1h.parquet"


def _load_real_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close"]]


class TestRealIntegration:
    def test_both_strategies_scan_real_data(self):
        try:
            frame = _load_real_frame()
        except FileNotFoundError:
            pytest.skip("Real Twelve Data dataset not present.")
        assert len(frame) > 0

        for strat in (TrendStructureStrategy(), LiquidityReversalStrategy()):
            result = HistoricalSignalScanner().scan(
                frame, strat, "EURUSD", "1h"
            )
            assert result.bars_processed == len(frame)
            # Signals (if any) must satisfy the signal model invariants.
            for s in result.signals:
                assert s.symbol == "EURUSD"
                assert s.available_from == s.timestamp
                if s.direction.value == "long":
                    assert s.stop_loss < s.entry < s.take_profit
                else:
                    assert s.take_profit < s.entry < s.stop_loss
                assert s.risk_reward_ratio >= 1.0
            # Documented: very few signals is acceptable; no rule relaxation.
