"""Discovery scoring and candidate ranking for Step 13 Alpha Discovery.

The discovery score is a transparent, documented combination of factors.
It is NOT a profitability ranking.

Components (each 0..1, weighted):

1. EXPECTANCY_COMPONENT (weight 0.30)
   = min(max(mean_R / 0.10, 0), 1)

2. STATISTICAL_CONFIDENCE (weight 0.20)
   = min(n / 100, 1) * min(max(1 - ci_width / 0.40, 0), 1)

3. STABILITY_COMPONENT (weight 0.20)
   = positive_group_fraction (across sessions/regimes/symbols)

4. EFFECT_SIZE_COMPONENT (weight 0.15)
   = min(max(cohens_d / 0.30, 0), 1)

5. DRAWDOWN_COMPONENT (weight 0.15)
   = 1 - min(max_drawdown_R / 10.0, 1)

Total = weighted sum (0..1), deterministic.

Overfit warnings fire when:
- sample < min_sample
- stability by halves degrades
- CI spans zero (not significant)
- positive group fraction < 0.5
- candidate depends on one symbol (>60% of samples)
"""

from __future__ import annotations

from typing import Any

from app.research.step13.evaluator import ResearchEvaluatorResult
from app.research.step13.statistics import (
    bootstrap_ci,
    cohens_d,
    expectancy_stats,
    stability_by_group,
    stability_by_halves,
)


class DiscoveryScore:
    """Documented discovery score for one candidate."""

    def __init__(
        self,
        total: float,
        components: dict[str, float],
        overfit_warnings: list[str],
        stats: dict[str, Any],
    ) -> None:
        self.total = total
        self.components = components
        self.overfit_warnings = overfit_warnings
        self.stats = stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "components": self.components,
            "overfit_warnings": self.overfit_warnings,
            "stats": self.stats,
        }


def compute_discovery_score(
    evaluator_result: ResearchEvaluatorResult,
    *,
    min_sample: int = 30,
    bootstrap_seed: int = 42,
) -> DiscoveryScore:
    """Compute the documented discovery score for an evaluated hypothesis.

    Parameters
    ----------
    evaluator_result : output of the FastResearchEvaluator.
    min_sample : minimum sample threshold for overfit warnings.
    """
    r = evaluator_result.r_values
    warnings: list[str] = []

    if not r:
        return DiscoveryScore(
            total=0.0,
            components={
                "expectancy": 0.0,
                "statistical_confidence": 0.0,
                "stability": 0.0,
                "effect_size": 0.0,
                "drawdown": 0.0,
            },
            overfit_warnings=["no samples: hypothesis rejected"],
            stats={"n": 0},
        )

    n = len(r)
    if n < min_sample:
        warnings.append(f"sample size {n} < minimum {min_sample}")

    exp = expectancy_stats(r)
    boot = bootstrap_ci(r, seed=bootstrap_seed)
    d = cohens_d(r)
    halves = stability_by_halves(r)
    groups = stability_by_group(dict(evaluator_result.groups))

    mean_r = exp["mean_r"] or 0.0
    ci_lo = boot["ci_lower"]
    ci_hi = boot["ci_upper"]

    # CI spans zero → not significant.
    if ci_lo is not None and ci_hi is not None and ci_lo < 0.0 < ci_hi:
        warnings.append(
            "confidence interval spans zero: edge not statistically significant"
        )

    # Half degradation.
    if not halves.get("degradation_ok", True):
        warnings.append("second half expectancy is non-positive: edge may not persist")

    # Group consistency.
    pos_frac = groups.get("positive_group_fraction", 0.0)
    if pos_frac < 0.5:
        warnings.append(f"only {pos_frac:.1%} of groups have positive mean R")

    # Single-symbol dependence.
    per_group = groups.get("per_group", {})
    symbol_counts = [
        v.get("n", 0) for k, v in per_group.items() if k.startswith("symbol=")
    ]
    if symbol_counts and max(symbol_counts) > 0.6 * n:
        warnings.append("performance depends heavily on a single symbol")

    # Components (0..1).
    exp_component = min(max(mean_r / 0.10, 0.0), 1.0)
    ci_width = (ci_hi - ci_lo) if ci_lo is not None and ci_hi is not None else 1.0
    sample_component = min(n / 100.0, 1.0)
    confidence_component = sample_component * min(max(1.0 - ci_width / 0.40, 0.0), 1.0)
    stability_component = float(pos_frac)
    effect_component = min(max((d or 0.0) / 0.30, 0.0), 1.0)
    max_dd_r = _max_drawdown_r(r)
    dd_component = min(max(1.0 - max_dd_r / 10.0, 0.0), 1.0)

    components = {
        "expectancy": round(exp_component * 0.30, 4),
        "statistical_confidence": round(confidence_component * 0.20, 4),
        "stability": round(stability_component * 0.20, 4),
        "effect_size": round(effect_component * 0.15, 4),
        "drawdown": round(dd_component * 0.15, 4),
    }
    total = sum(components.values())

    stats = {
        "n": n,
        "expectancy": exp,
        "bootstrap": boot,
        "effect_size_d": d,
        "stability_halves": halves,
        "stability_groups": groups,
        "max_drawdown_r": round(max_dd_r, 4),
        "win_rate": round(evaluator_result.win_count / n, 4) if n else 0.0,
        "avg_holding_bars": (
            round(sum(evaluator_result.holding_bars) / n, 2)
            if evaluator_result.holding_bars
            else None
        ),
    }

    return DiscoveryScore(
        total=total,
        components=components,
        overfit_warnings=warnings,
        stats=stats,
    )


def _max_drawdown_r(r_values: list[float]) -> float:
    """Max drawdown of the cumulative R curve (peak-to-trough)."""
    if not r_values:
        return 0.0
    peak = 0.0
    max_dd = 0.0
    cum = 0.0
    for r in r_values:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def rank_candidates(
    scores: dict[str, DiscoveryScore],
) -> list[tuple[str, float]]:
    """Deterministic ranking by discovery score (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1].total, reverse=True)