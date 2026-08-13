"""Strategy validation scoring and overfit detection for Step 13B.

=============================================================================
VALIDATION SCORING FORMULA (documented)
=============================================================================

The validation score is an explicitly weighted combination of components,
each normalized to 0..1. The strategy is NEVER selected purely because its
backtest made money.

Components (each 0..1):

1. EXPECTANCY_COMPONENT (weight 0.25)
   = min(max(median_expectancy_R / 0.20, 0), 1)
   Rewards positive median expectancy in R across validation windows.
   A strategy with median expectancy of +0.20R or more gets full credit.

2. DRAWDOWN_COMPONENT (weight 0.20)
   = 1 - min(max_drawdown / 0.50, 1)
   Rewards low maximum drawdown. 0% drawdown = 1.0, 50%+ = 0.0.

3. CONSISTENCY_COMPONENT (weight 0.20)
   = fraction_of_test_windows_with_positive_expectancy
   Rewards the fraction of windows that are profitable.

4. TRADE_COUNT_COMPONENT (weight 0.15)
   = min(total_trades / 100, 1)
   Rewards sufficient sample size. 100+ trades = full credit.

5. PARAM_STABILITY_COMPONENT (weight 0.10)
   = param_stability (fraction of windows selecting the same best params)
   Rewards parameter stability across windows.

6. SYMBOL_CONSISTENCY_COMPONENT (weight 0.10)
   = positive_symbol_fraction
   Rewards consistency across symbols (filled when multi-symbol data exists;
   defaults to 1.0 for single-symbol research runs for simplicity but is
   explicitly documented as a limitation).

The TOTAL score is the weighted sum. It ranges 0..1.

-----------------------------------------------------------------------------
HARD GATES (any failure => status below VALIDATED)
-----------------------------------------------------------------------------

The following are HARD gates; a strategy CANNOT be VALIDATED if any fails:

  * G1: total_trades >= config.min_total_trades
  * G2: completed_windows >= config.min_windows
  * G3: max_drawdown <= config.max_allowed_drawdown
  * G4: median_expectancy_R > config.min_expectancy_r (strictly positive edge)
  * G5: windows_profitable_fraction >= config.min_windows_profitable

-----------------------------------------------------------------------------
STATUS ASSIGNMENT
-----------------------------------------------------------------------------

  * INSUFFICIENT_DATA  if G1 or G2 fails (too few trades / windows)
  * OVERFIT            if G1-G5 pass BUT single_window_dependence > 0.60
                       OR param_stability < 0.35
                       OR train/validation performance diverges wildly from
                       test performance (train edge not repeated in test)
  * REJECTED           if G2-G5 all pass but edge is too weak/negative
                       (median_expectancy_R <= min_expectancy_r)
  * PROMISING          if G3-G5 pass but score < 0.50 (some edge but
                       insufficient robustness to validate)
  * VALIDATED          if ALL hard gates pass AND score >= 0.50
  * NOT_VALIDATED      otherwise (insufficient evidence)
"""

from __future__ import annotations

import math
from typing import Any

from app.research.step13b.config import Step13BConfig
from app.research.step13b.models import (
    RobustnessMetrics,
    StrategyStatus,
    ValidationScore,
    WindowResult,
)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _min_max(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_validation_score(
    *,
    window_results: list[WindowResult],
    robustness: RobustnessMetrics,
    config: Step13BConfig,
    positive_symbol_fraction: float = 1.0,
) -> ValidationScore:
    """Compute the documented validation score for a research run.

    Uses TEST-phase metrics exclusively (the final TEST set must never be used
    to optimize parameters; it IS used here only for final evaluation).
    """
    # Collect test-phase metrics across all completed windows.
    test_metric_list = []
    for w in window_results:
        if w.status == "complete" and w.test_metrics:
            test_metric_list.extend(w.test_metrics)

    total_trades = sum(m.trade_count for m in test_metric_list) if test_metric_list else 0
    completed_windows = robustness.total_windows

    # Expectancy in R values across test windows (use per-window median).
    exp_r_values = [
        m.expectancy_r
        for m in test_metric_list
        if m.expectancy_r is not None
    ]
    median_exp_r = _median(exp_r_values) if exp_r_values else 0.0

    # Max drawdown across test windows.
    max_dd = max((m.max_drawdown for m in test_metric_list), default=0.0)

    # Profitable window fraction.
    profitable_windows = robustness.profitable_windows
    win_frac = (
        profitable_windows / completed_windows
        if completed_windows > 0
        else 0.0
    )

    # Components (0..1 each).
    exp_component = _min_max(median_exp_r / 0.20)
    dd_component = _min_max(1.0 - (max_dd / 0.50))
    consistency_component = win_frac
    trade_count_component = _min_max(total_trades / 100.0)
    param_stability_component = _min_max(robustness.param_stability)
    symbol_component = _min_max(positive_symbol_fraction)

    weighted = {
        "expectancy": exp_component * 0.25,
        "drawdown": dd_component * 0.20,
        "consistency": consistency_component * 0.20,
        "trade_count": trade_count_component * 0.15,
        "param_stability": param_stability_component * 0.10,
        "symbol_consistency": symbol_component * 0.10,
    }
    total = sum(weighted.values())

    # ── Hard gates ───────────────────────────────────────────────────────────
    gates = {
        "G1_min_trades": bool(
            total_trades >= config.min_total_trades
        ),
        "G2_min_windows": bool(
            completed_windows >= config.min_windows
        ),
        "G3_max_drawdown": bool(
            max_dd <= config.max_allowed_drawdown
        ),
        "G4_positive_expectancy": bool(
            median_exp_r > config.min_expectancy_r
        ),
        "G5_window_consistency": bool(
            win_frac >= config.min_windows_profitable
        ),
    }
    all_gates_pass = all(gates.values())

    reasons: list[str] = []
    if not gates["G1_min_trades"]:
        reasons.append(
            f"insufficient total trades ({total_trades} < {config.min_total_trades})"
        )
    if not gates["G2_min_windows"]:
        reasons.append(
            f"insufficient windows ({completed_windows} < {config.min_windows})"
        )
    if not gates["G3_max_drawdown"]:
        reasons.append(f"max drawdown {max_dd:.2%} exceeds {config.max_allowed_drawdown:.2%}")
    if not gates["G4_positive_expectancy"]:
        reasons.append(
            f"median expectancy {median_exp_r:.4f}R <= {config.min_expectancy_r}R"
        )
    if not gates["G5_window_consistency"]:
        reasons.append(
            f"only {win_frac:.2%} of windows profitable "
            f"(min {config.min_windows_profitable:.2%})"
        )

    # ── Status assignment ────────────────────────────────────────────────────
    status = StrategyStatus.NOT_VALIDATED

    if not gates["G1_min_trades"] or not gates["G2_min_windows"]:
        status = StrategyStatus.INSUFFICIENT_DATA
    elif all_gates_pass:
        # Check overfit indicators.
        single_dep = robustness.single_window_dependence
        param_stab = robustness.param_stability

        if single_dep > 0.60:
            status = StrategyStatus.OVERFIT
            reasons.append(
                f"single-window dependence {single_dep:.2%} exceeds 60%"
            )
        elif param_stab < 0.35:
            status = StrategyStatus.OVERFIT
            reasons.append(
                f"parameter stability {param_stab:.2%} below 35%"
            )
        elif total >= 0.50:
            status = StrategyStatus.VALIDATED
        else:
            status = StrategyStatus.PROMISING
    elif not gates["G4_positive_expectancy"] and all(
        gates[k] for k in ("G1_min_trades", "G2_min_windows")
    ):
        # Gates 1&2 pass but edge is weak/negative.
        status = StrategyStatus.REJECTED
    else:
        status = StrategyStatus.NOT_VALIDATED

    return ValidationScore(
        total=round(total, 4),
        components=weighted,
        status=status,
        reasons=reasons,
        hard_gates=gates,
    )


def detect_overfit(
    *,
    train_metrics: list[Any],
    validation_metrics: list[Any],
    test_metrics: list[Any],
    robustness: RobustnessMetrics,
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Detect overfitting using train/validation/test separation.

    Overfitting is indicated when:
    1. Train performance is much better than test performance (edge doesn't
       generalize to unseen data).
    2. Single-window dependence is excessive.
    3. Parameter stability is very low (parameter chosen by luck).

    Returns (is_overfit, reasons).
    """
    t = thresholds or {
        "train_test_expectancy_ratio": 3.0,  # train expectancy > 3x test
        "max_single_window_dependence": 0.60,
        "min_param_stability": 0.35,
    }
    reasons: list[str] = []

    train_exp = [
        m.expectancy_r for m in train_metrics if m.expectancy_r is not None
    ]
    test_exp = [
        m.expectancy_r for m in test_metrics if m.expectancy_r is not None
    ]
    train_median = _median(train_exp) if train_exp else 0.0
    test_median = _median(test_exp) if test_exp else 0.0

    if train_median > 0 and test_median > 0:
        ratio = train_median / test_median
        if ratio > t["train_test_expectancy_ratio"]:
            reasons.append(
                f"train/test expectancy ratio {ratio:.2f} exceeds "
                f"{t['train_test_expectancy_ratio']:.2f}"
            )

    if robustness.single_window_dependence > t["max_single_window_dependence"]:
        reasons.append(
            f"single-window dependence {robustness.single_window_dependence:.2%} "
            f"exceeds {t['max_single_window_dependence']:.2%}"
        )

    if robustness.param_stability < t["min_param_stability"]:
        reasons.append(
            f"parameter stability {robustness.param_stability:.2%} "
            f"below {t['min_param_stability']:.2%}"
        )

    return bool(reasons), reasons