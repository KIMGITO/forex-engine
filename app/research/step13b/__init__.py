"""Step 13B — Strategy Research & Walk-Forward Validation Engine.

An ALTERNATIVE research path to the giant MTF pipeline. Purpose: determine
whether a strategy has a robust, repeatable statistical edge and produce a
machine-readable strategy configuration for the live trading application.

Key design principles:
* bounded memory (one symbol / timeframe / window at a time)
* incremental processing with resumable state
* strict train/validation/test separation (TEST never used for optimization)
* compact research artifacts (no giant Pydantic object graphs)
* reuse of existing engines (features, structure, regime, strategy, risk,
  backtest, causal-index, cache)
"""

from app.research.step13b.config import Step13BConfig, WalkForwardBounds
from app.research.step13b.models import (
    RobustnessMetrics,
    StrategyStatus,
    StrategyValidation,
    ValidationScore,
    WindowMetrics,
    WindowResult,
)
from app.research.step13b.runner import run_step13b

__all__ = [
    "RobustnessMetrics",
    "Step13BConfig",
    "StrategyStatus",
    "StrategyValidation",
    "ValidationScore",
    "WalkForwardBounds",
    "WindowMetrics",
    "WindowResult",
    "run_step13b",
]