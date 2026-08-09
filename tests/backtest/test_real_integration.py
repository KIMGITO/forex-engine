"""Real-data integration test for the backtest engine.

Runs on the locally stored Twelve Data EUR/USD dataset when present.
Skips gracefully when the dataset is absent (CI/dev machines without data).
"""

import pandas as pd
import pytest

from app.backtest import BacktestConfig, EventBacktester, NoOpStrategy

DATA_PATH = "data/processed/eurusd_1h.parquet"


def _load_real_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close"]]


def test_real_data_backtest_noop():
    try:
        frame = _load_real_frame()
    except FileNotFoundError:
        pytest.skip("Real Twelve Data dataset not present.")
    assert len(frame) > 0

    config = BacktestConfig(symbol="EURUSD", timeframe="1h")
    result = EventBacktester(config).run(
        frame, NoOpStrategy(), provider="twelvedata", source_type="historical"
    )
    # No-op strategy: no trades, equity unchanged.
    assert result.metrics.trade_count == 0
    assert abs(result.equity_curve[-1].equity - config.initial_balance) < 1e-9
    assert result.metadata.provider == "twelvedata"
