"""Cross-window robustness analysis for Step 13B.

A strategy must NOT be marked validated simply because total profit is
positive. Robustness requires consistency across windows, symbols, regimes,
and parameter variations.
"""

from __future__ import annotations

from typing import Any

from app.research.step13b.models import RobustnessMetrics, WindowMetrics, WindowResult


def analyze_robustness(
    window_results: list[WindowResult],
    *,
    symbol: str,
    timeframe: str,
    min_trades_per_window: int = 5,
) -> RobustnessMetrics:
    """Analyze cross-window robustness from completed window results.

    Metrics include:
    * profitable / losing / insufficient window counts
    * profit-factor consistency (mean + std)
    * expectancy consistency (mean + std in R)
    * drawdown consistency (mean + std)
    * parameter stability (fraction of windows selecting the same param set)
    * single-window dependence (fraction of total profit from best window)
    * regime coverage (fraction of windows spanning each regime type)
    """
    completed = [w for w in window_results if w.status == "complete"]
    if not completed:
        return RobustnessMetrics(total_windows=0)

    profitable = 0
    losing = 0
    insufficient = 0
    window_profits: list[float] = []
    window_pf: list[float] = []
    window_exp: list[float] = []
    window_dd: list[float] = []
    param_selections: dict[str, int] = {}
    regime_seen: set[str] = set()

    for w in completed:
        test_m = w.test_metrics[0] if w.test_metrics else None
        if test_m is None:
            insufficient += 1
            continue

        trades = test_m.trade_count
        if trades < min_trades_per_window:
            insufficient += 1
        else:
            net = test_m.net_profit
            window_profits.append(net)
            if net > 0:
                profitable += 1
            elif net < 0:
                losing += 1
            else:
                losing += 1  # breakeven window counts as non-profitable

        if test_m.profit_factor is not None:
            window_pf.append(test_m.profit_factor)
        if test_m.expectancy_r is not None:
            window_exp.append(test_m.expectancy_r)
        window_dd.append(test_m.max_drawdown)

        # Parameter stability: count selections.
        pname = _param_name(w.selected_params)
        param_selections[pname] = param_selections.get(pname, 0) + 1

        # Regime coverage: use the window's train regime count heuristic.
        if w.test_metrics:
            tm = w.test_metrics[0]
            # Regime type is derived from the trade log, not stored in metrics.
            # As a fallback we use the trade log parquet from the artifact.
            # For in-memory aggregation, we infer from trade snapshots via
            # whatever regimes are present in the window's trades parquet.
            # (The runner writes regime_metrics.parquet separately.)

    total_windows_logged = len(completed)
    pf_mean = _mean(window_pf) or None
    pf_std = _std(window_pf) or None
    exp_mean = _mean(window_exp) or None
    exp_std = _std(window_exp) or None
    dd_mean = _mean(window_dd) or None
    dd_std = _std(window_dd) or None

    # Parameter stability: fraction of windows selecting the most common param.
    max_param_count = max(param_selections.values()) if param_selections else 0
    param_stability = (max_param_count / total_windows_logged) if total_windows_logged else 1.0

    # Single-window dependence: fraction of profit from the best window.
    total_profit = sum(window_profits) if window_profits else 0.0
    best_profit = max(window_profits) if window_profits else 0.0
    single_dep = (best_profit / total_profit) if total_profit > 0 else 0.0

    return RobustnessMetrics(
        total_windows=total_windows_logged,
        profitable_windows=profitable,
        losing_windows=losing,
        insufficient_windows=insufficient,
        profitable_window_fraction=(profitable / total_windows_logged) if total_windows_logged else 0.0,
        profit_factor_mean=pf_mean,
        profit_factor_std=pf_std,
        expectancy_mean=exp_mean,
        expectancy_std=exp_std,
        drawdown_mean=dd_mean,
        drawdown_std=dd_std,
        param_stability=param_stability,
        single_window_dependence=single_dep,
        positive_symbol_fraction=0.0,  # filled by caller when multi-symbol known
        regime_coverage={},  # filled by caller from regime_metrics.parquet
        per_window_summary=[
            {
                "index": w.index,
                "status": w.status,
                "selected_params": w.selected_params,
                "test_net_pnl": (
                    w.test_metrics[0].net_profit if w.test_metrics else 0.0
                ),
                "test_expectancy_r": (
                    w.test_metrics[0].expectancy_r if w.test_metrics else None
                ),
                "test_max_drawdown": (
                    w.test_metrics[0].max_drawdown if w.test_metrics else 0.0
                ),
            }
            for w in completed
        ],
    )


def _param_name(params: dict[str, Any]) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted((params or {}).items()))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values) or 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5