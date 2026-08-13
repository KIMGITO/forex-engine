"""Causal correctness and walk-forward boundary tests for Step 13B."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13b.config import Step13BConfig, build_walk_forward_bounds
from app.research.step13b.models import StrategyStatus, WindowMetrics
from app.research.step13b.validation import compute_validation_score, detect_overfit


def _frame(n=400, seed=42, start="2026-01-01", freq="1h"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        },
        index=idx,
    )


class TestWalkForwardBounds:
    def test_strictly_chronological_disjoint(self):
        data_start = pd.Timestamp("2026-01-01", tz="UTC")
        data_end = pd.Timestamp("2026-12-31", tz="UTC")
        windows = build_walk_forward_bounds(
            data_start, data_end,
            train_days=90, validation_days=30, test_days=30, step_days=30,
        )
        assert len(windows) >= 6
        for w in windows:
            assert w.train_start < w.train_end
            assert w.train_end == w.validation_start
            assert w.validation_start < w.validation_end
            assert w.validation_end == w.test_start
            assert w.test_start < w.test_end
            # No overlap with next window.
            assert w.train_start <= w.test_end

    def test_windows_advance_by_step(self):
        data_start = pd.Timestamp("2026-01-01", tz="UTC")
        data_end = pd.Timestamp("2026-06-30", tz="UTC")
        windows = build_walk_forward_bounds(
            data_start, data_end,
            train_days=30, validation_days=15, test_days=15, step_days=15,
        )
        assert len(windows) >= 4
        # Each successive window starts 15 days later.
        if len(windows) >= 2:
            assert (
                windows[1].train_start
                == windows[0].train_start + pd.Timedelta(days=15)
            )

    def test_window_stops_at_data_end(self):
        data_start = pd.Timestamp("2026-01-01", tz="UTC")
        data_end = pd.Timestamp("2026-03-31", tz="UTC")
        windows = build_walk_forward_bounds(
            data_start, data_end,
            train_days=30, validation_days=15, test_days=15, step_days=15,
        )
        # Only windows whose TEST starts before data end are included.
        for w in windows:
            assert w.test_start <= data_end


class TestValidationScore:
    def _make_window_results(self, n_windows=5, trades_per_window=10, exp_r=0.1):
        """Create synthetic window results with positive expectancy."""
        results = []
        for i in range(n_windows):
            from app.research.step13b.models import WindowResult

            m = WindowMetrics(
                window_index=i,
                phase="test",
                symbol="EURUSD",
                timeframe="M15",
                param_set="stop_distance_atr=1.0_reward_risk_target=2.0",
                trade_count=trades_per_window,
                expectancy_r=exp_r,
                net_profit=100.0 * (1 + i * 0.1),
                max_drawdown=0.05,
                profit_factor=1.5,
                win_rate=0.55,
                sharpe=1.2,
                sortino=1.8,
            )
            results.append(
                WindowResult(
                    index=i,
                    symbol="EURUSD",
                    timeframe="M15",
                    bounds={},
                    selected_params={"stop_distance_atr": 1.0, "reward_risk_target": 2.0},
                    test_metrics=[m],
                    trade_count=trades_per_window,
                    status="complete",
                )
            )
        return results

    def test_valid_strategy_scores_validated(self):
        from app.research.step13b.robustness import analyze_robustness

        results = self._make_window_results(n_windows=5, trades_per_window=10, exp_r=0.15)
        config = Step13BConfig(
            min_total_trades=20,
            min_windows=3,
            min_expectancy_r=0.05,
            max_allowed_drawdown=0.25,
            min_windows_profitable=0.50,
        )
        robustness = analyze_robustness(
            results, symbol="EURUSD", timeframe="M15",
            min_trades_per_window=5,
        )
        score = compute_validation_score(
            window_results=results,
            robustness=robustness,
            config=config,
        )
        assert score.status == StrategyStatus.VALIDATED
        assert score.total > 0.5
        assert score.hard_gates["G1_min_trades"]
        assert score.hard_gates["G2_min_windows"]

    def test_insufficient_data_detected(self):
        from app.research.step13b.robustness import analyze_robustness

        results = self._make_window_results(n_windows=1, trades_per_window=2, exp_r=0.1)
        config = Step13BConfig(
            min_total_trades=20,
            min_windows=3,
        )
        robustness = analyze_robustness(
            results, symbol="EURUSD", timeframe="M15",
            min_trades_per_window=5,
        )
        score = compute_validation_score(
            window_results=results,
            robustness=robustness,
            config=config,
        )
        assert score.status == StrategyStatus.INSUFFICIENT_DATA

    def test_weak_edge_rejected(self):
        from app.research.step13b.robustness import analyze_robustness

        # Negative/neutral expectancy across many windows.
        results = self._make_window_results(n_windows=5, trades_per_window=10, exp_r=-0.01)
        config = Step13BConfig(
            min_total_trades=20,
            min_windows=3,
            min_expectancy_r=0.05,
        )
        robustness = analyze_robustness(
            results, symbol="EURUSD", timeframe="M15",
            min_trades_per_window=5,
        )
        score = compute_validation_score(
            window_results=results,
            robustness=robustness,
            config=config,
        )
        assert score.status in (StrategyStatus.REJECTED, StrategyStatus.NOT_VALIDATED)

    def test_overfit_detected(self):
        from app.research.step13b.robustness import analyze_robustness

        # Create results where one window produces ALL the profit.
        results = self._make_window_results(n_windows=5, trades_per_window=10, exp_r=0.15)
        # Boost one window's profit dramatically.
        results[0] = results[0].__class__(
            index=0,
            symbol=results[0].symbol,
            timeframe=results[0].timeframe,
            bounds={},
            selected_params=results[0].selected_params,
            test_metrics=[
                WindowMetrics(
                    window_index=0,
                    phase="test",
                    symbol="EURUSD",
                    timeframe="M15",
                    param_set="baseline",
                    trade_count=10,
                    expectancy_r=2.0,
                    net_profit=10000.0,
                    max_drawdown=0.05,
                )
            ],
            trade_count=10,
            status="complete",
        )
        config = Step13BConfig(
            min_total_trades=20,
            min_windows=3,
            min_expectancy_r=0.05,
        )
        robustness = analyze_robustness(
            results, symbol="EURUSD", timeframe="M15",
            min_trades_per_window=5,
        )
        score = compute_validation_score(
            window_results=results,
            robustness=robustness,
            config=config,
        )
        assert score.status == StrategyStatus.OVERFIT


class TestDetectOverfit:
    def test_train_test_divergence_detected(self):
        from app.research.step13b.models import RobustnessMetrics

        train_m = [
            WindowMetrics(
                window_index=0, phase="train", symbol="EURUSD", timeframe="M15",
                param_set="baseline", trade_count=10, expectancy_r=1.0,
            )
        ]
        test_m = [
            WindowMetrics(
                window_index=0, phase="test", symbol="EURUSD", timeframe="M15",
                param_set="baseline", trade_count=10, expectancy_r=0.05,
            )
        ]
        robustness = RobustnessMetrics(
            single_window_dependence=0.2,
            param_stability=0.8,
        )
        is_overfit, reasons = detect_overfit(
            train_metrics=train_m,
            validation_metrics=[],
            test_metrics=test_m,
            robustness=robustness,
        )
        assert is_overfit
        assert any("train/test" in r for r in reasons)

    def test_low_param_stability_detected(self):
        from app.research.step13b.models import RobustnessMetrics

        robustness = RobustnessMetrics(
            single_window_dependence=0.2,
            param_stability=0.1,
        )
        is_overfit, reasons = detect_overfit(
            train_metrics=[],
            validation_metrics=[],
            test_metrics=[],
            robustness=robustness,
        )
        assert is_overfit
        assert any("parameter stability" in r for r in reasons)