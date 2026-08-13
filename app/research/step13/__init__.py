"""Step 13 — Market Event Research & Candidate Generation Engine.

Step 13's responsibility: DATA → MARKET EVENTS → CAUSAL FEATURES → CANDIDATES.

Step 13B's responsibility: CANDIDATES → STRATEGY RESEARCH → WALK-FORWARD
VALIDATION → VALIDATED STRATEGY.

This module produces compact columnar event datasets (features, structure,
liquidity zones, sweeps, displacement, regime, MTF context) plus candidate
setups with strict feature/label separation. It NEVER claims profitability
and NEVER duplicates the authoritative domain engines.
"""

from app.research.step13.config import Step13Config
from app.research.step13.runner import run_step13

__all__ = ["Step13Config", "run_step13"]