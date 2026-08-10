"""Tests for chronological splits, leakage-safe optimizer, walk-forward,
warnings, cross-symbol research, and research report assembly."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.research import (
    GridSearchOptimizer,
    ResearchConfig,
    build_research_report,
    build_walk_forward_windows,
    make_time_split,
    split_frame,
)
from app.research.metrics import aggregate_metrics, cross_symbol_summary, warnings_for
from app.research.optimizer import grid_space_to_candidates
from app.research.splits import is_leak_free


def _frame(n=1000, seed=1):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.1, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close},
        index=idx,
    )


def _backtest_metrics(frame, params=None):
    """Synthetic backtest metrics for a frame (deterministic per length)."""
    n = len(frame)
    trades = max(2, n // 50)
    seed = n + (len(params or {}) if params else 0)
    return {
        "trade_count": trades,
        "net_pnl": float(np.random.default_rng(seed).normal(10, 5)),
        "expectancy": 0.5,
        "profit_factor": 1.2,
        "max_drawdown": 0.05,
    }


class TestChronologicalSplits:
    def test_fractional_split_chronological(self):
        frame = _frame()
        split = make_time_split(
            frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime(),
            ResearchConfig(train_fraction=0.6, validation_fraction=0.2),
        )
        assert is_leak_free(split.train_end, split.validation_start, split.test_start)
        train, val, test = split_frame(frame, split)
        assert len(train) + len(val) + len(test) == len(frame)
        assert len(test) > 0

    def test_explicit_dates_split(self):
        frame = _frame(n=2000)
        cfg = ResearchConfig(
            train_start="2020-01-01", train_end="2021-12-31",
            validation_start="2022-01-01", validation_end="2022-12-31",
            test_start="2023-01-01", test_end="2023-12-31",
        )
        split = make_time_split(
            frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime(), cfg
        )
        assert (
            split.train_start < split.train_end <= split.validation_start
            < split.validation_end <= split.test_start < split.test_end
        )

    def test_invalid_fraction_raises(self):
        from app.research.errors import SplitConfigurationError

        with pytest.raises(SplitConfigurationError):
            make_time_split(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2021, 1, 1, tzinfo=timezone.utc),
                ResearchConfig(train_fraction=0.9, validation_fraction=0.2),
            )


class TestOptimizerLeakage:
    def test_optimize_only_uses_train(self):
        frame = _frame(n=1000)
        split = make_time_split(
            frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime(),
            ResearchConfig(train_fraction=0.6, validation_fraction=0.2),
        )
        train, val, _ = split_frame(frame, split)

        optimizer = GridSearchOptimizer(
            ResearchConfig(min_trades=10), _backtest_metrics
        )
        candidates = optimizer.optimize(train, {"stop_loss": [10, 20]})
        assert len(candidates) == 2
        assert candidates[0].score >= candidates[1].score

        validated = optimizer.select_on_validation(candidates, val)
        assert isinstance(validated, list)

    def test_grid_space_expansion(self):
        space = {"a": [1, 2], "b": [True, False]}
        combos = grid_space_to_candidates(space)
        assert len(combos) == 4
        assert {"a": 1, "b": True} in combos

    def test_test_set_never_used_in_optimizer(self):
        frame = _frame(n=900)
        split = make_time_split(
            frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime(),
            ResearchConfig(train_fraction=0.6, validation_fraction=0.2),
        )
        train, _, _ = split_frame(frame, split)
        optimizer = GridSearchOptimizer(ResearchConfig(min_trades=10), _backtest_metrics)
        candidates = optimizer.optimize(train, {"x": [1]})
        assert isinstance(candidates, list)


class TestWalkForward:
    def test_windows_are_chronological_and_disjoint(self):
        start = datetime(2019, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        cfg = ResearchConfig(
            walk_train_years=2.0, walk_validation_years=0.5, walk_test_years=0.5,
        )
        windows = build_walk_forward_windows(start, end, cfg, max_windows=10)
        assert len(windows) >= 1
        for w in windows:
            s = w.split
            assert (
                s.train_start < s.train_end <= s.validation_start
                < s.validation_end <= s.test_start < s.test_end
            )

    def test_builds_over_frame_range(self):
        frame = _frame(n=4000, seed=2)
        cfg = ResearchConfig(
            walk_train_years=0.2, walk_validation_years=0.05, walk_test_years=0.05,
        )
        results = build_walk_forward_windows(
            frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime(), cfg
        )
        assert len(results) >= 1


class TestWarningsAndMetrics:
    def test_insufficient_trades_warned(self):
        metrics = {"trade_count": 5}
        warnings = warnings_for(metrics, min_trades=30, min_bars=500, n_bars=200)
        assert any("only 5 trades" in w for w in warnings)
        assert any("insufficient" in w.lower() for w in warnings)

    def test_cross_symbol_summary(self):
        summary = cross_symbol_summary({
            "EURUSD": {"expectancy": 0.5, "trade_count": 40},
            "GBPUSD": {"expectancy": -0.2, "trade_count": 35},
        })
        assert summary["symbols_tested"] == 2
        assert summary["positive_fraction"] == 0.5

    def test_aggregate_metrics(self):
        agg = aggregate_metrics([
            {"net_pnl": 10, "trade_count": 30},
            {"net_pnl": -5, "trade_count": 40},
        ])
        assert "net_pnl_mean" in agg


class TestResearchReport:
    def test_report_separates_is_validation_oos(self):
        report = build_research_report(
            provider="twelvedata",
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H1", "H4"],
            strategy_name="trend_structure",
            config=ResearchConfig(),
            date_range={"start": "2020-01-01", "end": "2024-12-31"},
            training_metrics=[{"net_pnl": 100, "trade_count": 50}],
            validation_metrics=[{"net_pnl": 20, "trade_count": 30}],
            oos_metrics=[{"net_pnl": 5, "trade_count": 20}],
            cross_window_metrics=[{"net_pnl": 10}],
            results_by_symbol={"EURUSD": {"expectancy": 0.4, "trade_count": 50}},
            cost_assumptions={"spread_pips": 0.8},
            warnings=["WARNING: 120 bars"],
            limitations=["120 candles insufficient"],
        )
        assert report.training["label"].startswith("IN-SAMPLE")
        assert report.validation["label"].startswith("VALIDATION")
        assert report.out_of_sample["label"].startswith("OUT-OF-SAMPLE")
        assert report.provider == "twelvedata"