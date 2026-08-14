"""Typed models for Step 15 walk-forward validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemporalSplit:
    """A strictly chronological TRAIN / VALIDATION / TEST split."""

    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    test_start: Any
    test_end: Any

    def to_dict(self) -> dict:
        return {
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "validation_start": str(self.validation_start),
            "validation_end": str(self.validation_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }

    def is_chronological(self) -> bool:
        return bool(
            self.train_start < self.train_end
            <= self.validation_start < self.validation_end
            <= self.test_start < self.test_end
        )


@dataclass(frozen=True)
class Step15Fold:
    """One walk-forward fold with its canonical split and results."""

    index: int
    split: TemporalSplit
    selected_hypothesis: str = ""          # hypothesis_id (frozen)
    selection_metrics: dict = field(default_factory=dict)
    train_sample: int = 0
    train_bars: int = 0
    validation_results: dict = field(default_factory=dict)
    test_results: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "split": self.split.to_dict(),
            "selected_hypothesis": self.selected_hypothesis,
            "selection_metrics": self.selection_metrics,
            "train_sample": self.train_sample,
            "train_bars": self.train_bars,
            "validation_results": self.validation_results,
            "test_results": self.test_results,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class OosMetrics:
    """Comprehensive out-of-sample metrics for one frozen hypothesis on one fold."""

    trades: int = 0
    win_rate: float | None = None
    loss_rate: float | None = None
    average_r: float | None = None
    median_r: float | None = None
    total_r: float = 0.0
    profit_factor: float | None = None
    expectancy: float | None = None
    max_drawdown_r: float = 0.0
    sharpe: float | None = None
    mfe: float = 0.0
    mae: float = 0.0
    average_holding_bars: float | None = None
    tp_count: int = 0
    sl_count: int = 0
    time_exit_count: int = 0
    ambiguous_exit_count: int = 0
    gross_r: float = 0.0
    net_r: float = 0.0
    r_values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "average_r": self.average_r,
            "median_r": self.median_r,
            "total_r": self.total_r,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown_r": self.max_drawdown_r,
            "sharpe": self.sharpe,
            "mfe": self.mfe,
            "mae": self.mae,
            "average_holding_bars": self.average_holding_bars,
            "tp_count": self.tp_count,
            "sl_count": self.sl_count,
            "time_exit_count": self.time_exit_count,
            "ambiguous_exit_count": self.ambiguous_exit_count,
            "gross_r": self.gross_r,
            "net_r": self.net_r,
            "r_sample_count": len(self.r_values),
        }