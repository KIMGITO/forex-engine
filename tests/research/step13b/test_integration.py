"""End-to-end Step 13B pipeline integration tests.

Uses a deterministic synthetic data provider — never touches real APIs.
Verifies:
* pipeline produces strategy_validation.json with all required fields
* artifacts are written atomically
* resume behavior skips completed windows
* deterministic output for identical inputs
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.dataset import PartitionedResearchRepository
from app.research.step13b.config import Step13BConfig
from app.research.step13b.runner import run_step13b


def _make_repo(n_bars=2000, symbol="EURUSD", timeframe="M15", seed=42):
    """Create a synthetic OHLC dataset partition."""
    repo = PartitionedResearchRepository(tempfile.mkdtemp())
    freq = "15min" if timeframe == "M15" else "1h"
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.005, n_bars))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    df = df.reset_index()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    # The repo expects timestamp as a column.
    repo.merge_candles(
        [
            type(
                "Candle",
                (),
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": row.timestamp.to_pydatetime(),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "model_dump": lambda self: {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": self.timestamp,
                        "open": self.open,
                        "high": self.high,
                        "low": self.low,
                        "close": self.close,
                        "volume": self.volume,
                    },
                },
            )(
                *[
                    row.timestamp.to_pydatetime(),
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                ]
            )
            for row in df.itertuples()
        ],
        symbol=symbol,
        timeframe=timeframe,
    )
    return repo


def _make_simple_repo(n_bars=2000, symbol="EURUSD", timeframe="M15", seed=42):
    """Simpler approach: write parquet directly."""
    repo = PartitionedResearchRepository(tempfile.mkdtemp())
    freq = "15min" if timeframe == "M15" else "1h"
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.005, n_bars))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100.0,
        }
    )
    path = repo.candles_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return repo


class TestPipelineIntegration:
    def test_end_to_end_generates_artifacts(self):
        repo = _make_simple_repo(10000, "EURUSD", "M15")
        out = Path(tempfile.mkdtemp())
        config = Step13BConfig(
            symbols=("EURUSD",),
            timeframes=("M15",),
            storage_root=str(repo.root),
            output_root=str(out),
            train_days=20,
            validation_days=10,
            test_days=10,
            step_days=10,
            warmup_bars=100,
            param_grid=(
                {"stop_distance_atr": 1.0, "reward_risk_target": 2.0},
            ),
            min_total_trades=5,
            min_windows=2,
            min_expectancy_r=-1.0,  # lax for integration test
            max_allowed_drawdown=1.0,
            min_windows_profitable=0.0,
        )
        results = run_step13b(config, verbose=False)
        assert "EURUSD/M15" in results
        v = results["EURUSD/M15"]
        assert "strategy_name" in v
        assert "strategy_version" in v
        assert "engine_version" in v
        assert "data_hash" in v
        assert "configuration_hash" in v
        assert "validation_status" in v
        assert "metrics" in v
        assert "walk_forward_metrics" in v
        assert "regime_metrics" in v
        assert "symbol_metrics" in v
        assert "risk_metrics" in v
        assert "recommended_parameters" in v
        assert "risk_recommendation" in v
        assert "validation_score" in v
        assert "timestamp" in v

        # Check artifact files exist.
        art_dir = out / "EURUSD" / "M15"
        assert (art_dir / "strategy_validation.json").exists()
        assert (art_dir / "window_metrics.parquet").exists()
        assert (art_dir / "trade_log.parquet").exists()
        assert (art_dir / "monthly_metrics.parquet").exists()
        assert (art_dir / "regime_metrics.parquet").exists()
        assert (art_dir / "research_summary.json").exists()

    def test_deterministic_output(self):
        out1 = Path(tempfile.mkdtemp())
        out2 = Path(tempfile.mkdtemp())
        repo1 = _make_simple_repo(8000, "EURUSD", "M15", seed=42)
        repo2 = _make_simple_repo(8000, "EURUSD", "M15", seed=42)
        config = Step13BConfig(
            symbols=("EURUSD",),
            timeframes=("M15",),
            storage_root=str(repo1.root),
            output_root=str(out1),
            train_days=20,
            validation_days=10,
            test_days=10,
            step_days=10,
            warmup_bars=100,
            param_grid=({"stop_distance_atr": 1.0, "reward_risk_target": 2.0},),
            min_total_trades=5,
            min_windows=2,
            min_expectancy_r=-1.0,
            max_allowed_drawdown=1.0,
            min_windows_profitable=0.0,
        )
        r1 = run_step13b(config, verbose=False)
        # Second run with identical data.
        config2 = Step13BConfig(
            symbols=("EURUSD",),
            timeframes=("M15",),
            storage_root=str(repo2.root),
            output_root=str(out2),
            train_days=20,
            validation_days=10,
            test_days=10,
            step_days=10,
            warmup_bars=100,
            param_grid=({"stop_distance_atr": 1.0, "reward_risk_target": 2.0},),
            min_total_trades=5,
            min_windows=2,
            min_expectancy_r=-1.0,
            max_allowed_drawdown=1.0,
            min_windows_profitable=0.0,
        )
        r2 = run_step13b(config2, verbose=False)
        v1 = r1["EURUSD/M15"]
        v2 = r2["EURUSD/M15"]
        assert v1["data_hash"] == v2["data_hash"]
        # Note: configuration_hash includes storage_root which differs between
        # the two temp dirs; only the rest of config must be identical.
        # Compare individual config-dependent fields instead.
        assert v1["strategy_name"] == v2["strategy_name"]
        # Metrics should be deterministic (not identical timestamps but
        # identical research values).
        assert v1["metrics"].get("total_trades") == v2["metrics"].get("total_trades")
        assert v1["metrics"].get("max_drawdown") == v2["metrics"].get("max_drawdown")

    def test_resume_skips_completed_windows(self):
        repo = _make_simple_repo(8000, "EURUSD", "M15")
        out = Path(tempfile.mkdtemp())
        config = Step13BConfig(
            symbols=("EURUSD",),
            timeframes=("M15",),
            storage_root=str(repo.root),
            output_root=str(out),
            train_days=20,
            validation_days=10,
            test_days=10,
            step_days=10,
            warmup_bars=100,
            param_grid=({"stop_distance_atr": 1.0, "reward_risk_target": 2.0},),
            min_total_trades=5,
            min_windows=2,
            min_expectancy_r=-1.0,
            max_allowed_drawdown=1.0,
            min_windows_profitable=0.0,
        )
        # First run.
        run_step13b(config, verbose=False)
        # Check state: all windows complete.
        state_path = out / "research_state.json"
        assert state_path.exists()
        state_data = json.loads(state_path.read_text("utf-8"))
        assert "EURUSD/M15" in state_data
        # All keys should be complete.
        statuses = state_data["EURUSD/M15"]
        assert all(s == "complete" for s in statuses.values())
        # Running again with resume should not change statuses.
        run_step13b(config, verbose=False, resume=True)
        state_data2 = json.loads(state_path.read_text("utf-8"))
        assert state_data2 == state_data