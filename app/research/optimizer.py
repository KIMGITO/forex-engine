"""Leakage-safe parameter optimizer (grid search)."""

import itertools
from collections.abc import Callable

import pandas as pd

from app.research.config import ResearchConfig
from app.research.models import OptimizerCandidate

__all__ = ["GridSearchOptimizer", "grid_space_to_candidates"]


def grid_space_to_candidates(grid_space: dict[str, list]) -> list[dict]:
    """Expand a grid space dict into a list of parameter dicts."""
    if not grid_space:
        return [{}]
    keys = list(grid_space.keys())
    values = list(grid_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _normalized_score(metrics: dict, config: ResearchConfig) -> float:
    """Multi-metric selection score (higher is better), never a probability."""
    net = float(metrics.get("net_pnl", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    profit_factor = float(metrics.get("profit_factor") or 0.0)
    expectancy = float(metrics.get("expectancy") or 0.0)
    trades = int(metrics.get("trade_count", 0))

    if trades < config.min_trades:
        return float("-inf")

    # Penalize catastrophic drawdown hard.
    dd_penalty = 1.0 - min(max_dd * 5.0, 0.9)
    score = net * 0.5 + expectancy * 2.0 + profit_factor * 0.5
    return score * dd_penalty


class GridSearchOptimizer:
    """Grid-search over candidate configs on TRAIN data only."""

    def __init__(
        self,
        config: ResearchConfig,
        backtest_callable: Callable[[pd.DataFrame, dict], dict],
    ) -> None:
        self.config = config
        self.backtest_callable = backtest_callable

    def optimize(
        self,
        train_frame: pd.DataFrame,
        grid_space: dict[str, list],
    ) -> list[OptimizerCandidate]:
        """Backtest each candidate on TRAIN only; return ranked by score.

        Never touches validation or test. The caller applies the VALIDATION
        selection gate separately.
        """
        candidates: list[OptimizerCandidate] = []
        for params in grid_space_to_candidates(grid_space):
            result = self.backtest_callable(train_frame, params)
            metrics = result if isinstance(result, dict) else {}
            score = _normalized_score(metrics, self.config)
            candidates.append(
                OptimizerCandidate(
                    params=params,
                    score=score,
                    metrics=metrics,
                    trade_count=int(metrics.get("trade_count", 0)),
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def select_on_validation(
        self,
        candidates: list[OptimizerCandidate],
        validation_frame: pd.DataFrame,
        top_n: int = 5,
    ) -> list[OptimizerCandidate]:
        """Re-evaluate top-N train candidates on VALIDATION; return ranked.

        The selection gate. TEST data is never used here.
        """
        if not candidates:
            return []
        top = candidates[:top_n]
        validated: list[OptimizerCandidate] = []
        for c in top:
            metrics = self.backtest_callable(validation_frame, c.params)
            if not isinstance(metrics, dict):
                metrics = {}
            score = _normalized_score(metrics, self.config)
            if score == float("-inf"):
                continue
            validated.append(
                OptimizerCandidate(
                    params=c.params,
                    score=score,
                    metrics=metrics,
                    trade_count=int(metrics.get("trade_count", 0)),
                )
            )
        validated.sort(key=lambda c: c.score, reverse=True)
        return validated
