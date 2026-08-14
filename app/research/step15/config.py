"""Step 15 — Walk-Forward Validation configuration.

Defines the canonical temporal architecture (TRAIN -> VALIDATION -> TEST),
walk-forward window durations, purge/embargo policy, cost model, and
determinism controls used by the validation engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step15Config:
    """Configuration for the Step 15 walk-forward validation engine."""

    # ── Fold architecture (calendar days) ───────────────────────────────────
    # The canonical single split for the FINAL report:
    train_days: int = 240      # ~8 months
    validation_days: int = 60  # ~2 months
    test_days: int = 60        # ~2 months

    # Rolling walk-forward windows:
    wf_train_days: int = 120   # ~4 months
    wf_validation_days: int = 30
    wf_test_days: int = 30
    wf_step_days: int = 30

    # ── Temporal contamination policy ────────────────────────────────────────
    # Maximum number of bars a candidate can extend into the future (the
    # holding/exit horizon + entry-confirmation search). This engine's label
    # computation uses:
    #   * entry-confirmation search lookback : Step13Config.label_lookback_bars
    #   * outcome holding window             : Step13Config.label_lookback_bars
    # Worst case a candidate timestamp can touch: lookback search to find the
    # entry bar, then a full holding window after that. For M15 with
    # label_lookback_bars=100 that is 200 bars (~50h).
    # A train candidate whose PURGED horizon (timestamp +
    # purge_horizon_bars * bar_minutes) crosses the train/validation boundary
    # is EXCLUDED from training discovery and evaluation.
    purge_horizon_bars: int = 200
    purge_enabled: bool = True

    # ── Determinism / reproducibility ────────────────────────────────────────
    random_seed: int = 42
    bootstrap_seed: int = 42

    # ── Cost model (identical on TRAIN and TEST) ─────────────────────────────
    spread_pips: float = 0.8
    slippage_pips: float = 0.0
    commission_per_lot: float = 0.0  # account currency per 100k lot

    # ── Sample gates ─────────────────────────────────────────────────────────
    min_train_sample: int = 30       # mins samples to even consider a hypothesis
    min_oos_trades: int = 5          # below this an OOS fold is "insufficient"
    min_windows: int = 3

    # ── Baselines ────────────────────────────────────────────────────────────
    baseline_random_seed: int = 1234

    # ── Max windows cap (safety) ─────────────────────────────────────────────
    max_windows: int = 20

    # ── Output ───────────────────────────────────────────────────────────────
    output_root: str = "research/results/step15"

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "test_days": self.test_days,
            "wf_train_days": self.wf_train_days,
            "wf_validation_days": self.wf_validation_days,
            "wf_test_days": self.wf_test_days,
            "wf_step_days": self.wf_step_days,
            "purge_horizon_bars": self.purge_horizon_bars,
            "purge_enabled": self.purge_enabled,
            "random_seed": self.random_seed,
            "bootstrap_seed": self.bootstrap_seed,
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "commission_per_lot": self.commission_per_lot,
            "min_train_sample": self.min_train_sample,
            "min_oos_trades": self.min_oos_trades,
            "min_windows": self.min_windows,
            "baseline_random_seed": self.baseline_random_seed,
            "max_windows": self.max_windows,
            "output_root": self.output_root,
        }

    def config_hash(self) -> str:
        """Deterministic hash of this configuration."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()
