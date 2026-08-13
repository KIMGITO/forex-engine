"""Step 13B — Strategy Research & Walk-Forward Validation Engine configuration.

This is an ALTERNATIVE research path to the giant MTF pipeline. Its purpose is
to determine whether a strategy has a robust, repeatable statistical edge using
bounded-memory, incremental, resumable research.

The configuration is deliberately conservative:
* one symbol and one timeframe processed at a time
* one walk-forward window at a time
* a small, explicitly bounded parameter grid (no exhaustive search)
* configurable memory limit with safe abort
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step13BConfig:
    """Configuration for the Step 13B research engine."""

    # ── Data scope ──────────────────────────────────────────────────────────
    symbols: tuple = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
    timeframes: tuple = ("M15", "H1")
    storage_root: str = "data/processed"

    # ── Walk-forward windows (calendar days) ─────────────────────────────────
    train_days: int = 90
    validation_days: int = 30
    test_days: int = 30
    step_days: int = 30
    warmup_bars: int = 200  # extra causal warmup bars before train_start

    # ── Parameter grid (explicitly bounded, no exhaustive search) ───────────
    # Each dict is a StrategyConfig override. First entry is the baseline.
    param_grid: tuple[dict[str, Any], ...] = (
        {"stop_distance_atr": 1.0, "reward_risk_target": 2.0},  # baseline
        {"stop_distance_atr": 1.5, "reward_risk_target": 2.0},
        {"stop_distance_atr": 1.0, "reward_risk_target": 2.5},
        {"stop_distance_atr": 1.5, "reward_risk_target": 2.5},
    )
    strategy_name: str = "trend_structure"  # trend_structure | liquidity_reversal

    # ── Backtest / cost assumptions ─────────────────────────────────────────
    spread_pips: float = 0.8
    slippage_pips: float = 0.0
    commission_percent: float = 0.0
    commission_per_trade: float = 0.0
    initial_balance: float = 10_000.0

    # ── Risk engine (Step 15) ───────────────────────────────────────────────
    risk_percent: float = 0.01  # 1% per trade
    max_daily_loss_pct: float | None = 0.03
    max_drawdown_pct: float | None = 0.10
    max_open_positions: int = 5

    # ── Validation thresholds ────────────────────────────────────────────────
    min_trades_per_window: int = 5
    min_windows: int = 3
    min_total_trades: int = 20
    max_allowed_drawdown: float = 0.25  # 25% max acceptable drawdown
    min_expectancy_r: float = 0.05  # minimum median expectancy in R for validation
    min_windows_profitable: float = 0.50  # fraction for consistency

    # ── Memory guard ─────────────────────────────────────────────────────────
    max_rss_mb: int = 3000  # abort before OOM on 8GB dev machine
    gc_after_window: bool = True

    # ── Output ───────────────────────────────────────────────────────────────
    output_root: str = "research/results/step13b"

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "storage_root": self.storage_root,
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
            "warmup_bars": self.warmup_bars,
            "param_grid": [dict(p) for p in self.param_grid],
            "strategy_name": self.strategy_name,
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "commission_percent": self.commission_percent,
            "commission_per_trade": self.commission_per_trade,
            "initial_balance": self.initial_balance,
            "risk_percent": self.risk_percent,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_open_positions": self.max_open_positions,
            "min_trades_per_window": self.min_trades_per_window,
            "min_windows": self.min_windows,
            "min_total_trades": self.min_total_trades,
            "max_allowed_drawdown": self.max_allowed_drawdown,
            "min_expectancy_r": self.min_expectancy_r,
            "min_windows_profitable": self.min_windows_profitable,
            "max_rss_mb": self.max_rss_mb,
            "gc_after_window": self.gc_after_window,
            "output_root": self.output_root,
        }

    def config_hash(self) -> str:
        """Deterministic hash of this configuration for reproducibility."""
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class WalkForwardBounds:
    """One walk-forward window's chronological boundaries (tz-aware datetimes)."""

    index: int
    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    test_start: Any
    test_end: Any

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "validation_start": str(self.validation_start),
            "validation_end": str(self.validation_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def build_walk_forward_bounds(
    data_start,
    data_end,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    max_windows: int = 50,
) -> list[WalkForwardBounds]:
    """Build rolling walk-forward window bounds in calendar days.

    Each window is strictly chronological and disjoint:
        TRAIN -> VALIDATION -> TEST
    The window advances by ``step_days`` after each TEST block.

    A window is only included when its TEST period falls within the data.
    """
    import pandas as pd

    windows: list[WalkForwardBounds] = []
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)

    for idx in range(max_windows):
        offset = pd.Timedelta(days=step_days * idx)
        w_train_start = start + offset
        w_train_end = w_train_start + pd.Timedelta(days=train_days)
        w_val_start = w_train_end
        w_val_end = w_val_start + pd.Timedelta(days=validation_days)
        w_test_start = w_val_end
        w_test_end = w_test_start + pd.Timedelta(days=test_days)

        # Stop when this window's TEST starts after available data.
        if w_test_start > end:
            break

        windows.append(
            WalkForwardBounds(
                index=idx,
                train_start=w_train_start,
                train_end=w_train_end,
                validation_start=w_val_start,
                validation_end=w_val_end,
                test_start=w_test_start,
                test_end=w_test_end,
            )
        )
    return windows