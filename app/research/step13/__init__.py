"""Step 13 — Market Event Research & Candidate Generation Engine.

Step 13's responsibility: DATA → MARKET EVENTS → CAUSAL FEATURES → CANDIDATES.
Step 13B's responsibility: CANDIDATES → STRATEGY RESEARCH → WALK-FORWARD
VALIDATION → VALIDATED STRATEGY.

This module produces compact columnar event datasets (features, structure,
liquidity zones, sweeps, displacement, regime, MTF context) plus candidate
setups with strict feature/label separation. It NEVER claims profitability
and NEVER duplicates the authoritative domain engines.
"""

from __future__ import annotations

from typing import Any

from app.research.step13.config import Step13Config

__all__ = ["Step13Config", "run_step13"]


def __getattr__(name: str) -> Any:
    """Lazy export of ``run_step13``.

    IMPORTANT: ``runner`` is deliberately NOT imported here eagerly. If it were,
    ``python3 -m app.research.step13.runner`` would pre-load the runner module
    (with __name__ = 'app.research.step13.runner', not '__main__') when Python
    first imports the parent package, so the ``if __name__ == '__main__'`` guard
    would never fire and the CLI would silently exit. Lazy ``__getattr__`` keeps
    the ``from app.research.step13 import run_step13`` API while allowing the
    module entrypoint to execute as __main__.
    """
    if name == "run_step13":
        from app.research.step13.runner import run_step13 as fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
