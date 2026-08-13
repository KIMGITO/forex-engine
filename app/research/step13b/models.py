"""Typed models for Step 13B research artifacts.

These models are deliberately COMPACT — they represent research snapshots, not
giant analytical object graphs. Full Pydantic objects (signals, structure,
regime) are never retained for the entire historical dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StrategyStatus(str, Enum):
    """Explicit validation status for a strategy candidate.

    A profitable backtest does NOT make a strategy valid. Status is determined
    by the full validation scoring formula (see ``validation.py``).
    """

    NOT_VALIDATED = "not_validated"
    PROMISING = "promising"
    VALIDATED = "validated"
    REJECTED = "rejected"
    OVERFIT = "overfit"
    INSUFFICIENT_DATA = "insufficient_data"


class WindowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SignalSnapshot:
    """Compact research representation of one evaluated signal.

    Fields represent the causal context at signal time plus the trade result.
    Stored as DataFrame rows / parquet — never as giant Pydantic graphs.
    """

    timestamp: datetime
    symbol: str
    timeframe: str

    # Causal context at signal time
    trend: str = "unknown"
    regime: str = "unknown"
    volatility: str = "unknown"
    structure_bias: str = "neutral"
    liquidity_state: str = "none"
    sweep_detected: bool = False
    break_of_structure: bool = False
    displacement: str = "none"
    mtf_bias: str | None = None
    session: str = "utc"

    # Trade geometry (from the signal)
    direction: str = ""
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_distance: float = 0.0
    reward_distance: float = 0.0
    risk_reward_ratio: float = 0.0

    # Risk integration
    risk_rejected: bool = False
    risk_rejection_reason: str | None = None
    risk_percent: float = 0.0
    position_size: float = 0.0

    # Trade outcome
    result: str = "pending"  # win | loss | breakeven | pending
    r_multiple: float = 0.0
    exit_reason: str = ""

    # Strategy metadata
    param_set: str = "baseline"
    score: float = 0.0
    strength: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "trend": self.trend,
            "regime": self.regime,
            "volatility": self.volatility,
            "structure_bias": self.structure_bias,
            "liquidity_state": self.liquidity_state,
            "sweep_detected": self.sweep_detected,
            "break_of_structure": self.break_of_structure,
            "displacement": self.displacement,
            "mtf_bias": self.mtf_bias,
            "session": self.session,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_distance": self.risk_distance,
            "reward_distance": self.reward_distance,
            "risk_reward_ratio": self.risk_reward_ratio,
            "risk_rejected": self.risk_rejected,
            "risk_rejection_reason": self.risk_rejection_reason,
            "risk_percent": self.risk_percent,
            "position_size": self.position_size,
            "result": self.result,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason,
            "param_set": self.param_set,
            "score": self.score,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class WindowMetrics:
    """Aggregated metrics for one walk-forward window phase."""

    window_index: int
    phase: str  # train | validation | test
    symbol: str
    timeframe: str
    param_set: str

    # Core metrics
    net_return: float = 0.0
    net_profit: float = 0.0
    profit_factor: float | None = None
    expectancy: float | None = None
    expectancy_r: float | None = None
    win_rate: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    max_drawdown: float = 0.0
    average_drawdown: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    trade_count: int = 0
    average_trades_per_month: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    bars_in_phase: int = 0

    # Risk integration
    risk_rejected_count: int = 0
    risk_approved_count: int = 0
    daily_loss_breaches: int = 0
    drawdown_limit_breaches: int = 0
    position_limit_breaches: int = 0
    exposure_breaches: int = 0

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "phase": self.phase,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "param_set": self.param_set,
            "net_return": self.net_return,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "expectancy_r": self.expectancy_r,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "max_drawdown": self.max_drawdown,
            "average_drawdown": self.average_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "trade_count": self.trade_count,
            "average_trades_per_month": self.average_trades_per_month,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_consecutive_wins": self.max_consecutive_wins,
            "bars_in_phase": self.bars_in_phase,
            "risk_rejected_count": self.risk_rejected_count,
            "risk_approved_count": self.risk_approved_count,
            "daily_loss_breaches": self.daily_loss_breaches,
            "drawdown_limit_breaches": self.drawdown_limit_breaches,
            "position_limit_breaches": self.position_limit_breaches,
            "exposure_breaches": self.exposure_breaches,
        }


@dataclass(frozen=True)
class WindowResult:
    """Complete result of evaluating one walk-forward window."""

    index: int
    symbol: str
    timeframe: str
    bounds: dict
    selected_params: dict = field(default_factory=dict)
    candidate_metrics: list[WindowMetrics] = field(default_factory=list)
    train_metrics: list[WindowMetrics] = field(default_factory=list)
    validation_metrics: list[WindowMetrics] = field(default_factory=list)
    test_metrics: list[WindowMetrics] = field(default_factory=list)
    trade_count: int = 0
    status: str = "complete"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bounds": self.bounds,
            "selected_params": self.selected_params,
            "candidate_metrics": [m.to_dict() for m in self.candidate_metrics],
            "train_metrics": [m.to_dict() for m in self.train_metrics],
            "validation_metrics": [m.to_dict() for m in self.validation_metrics],
            "test_metrics": [m.to_dict() for m in self.test_metrics],
            "trade_count": self.trade_count,
            "status": self.status,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RobustnessMetrics:
    """Cross-window robustness assessment."""

    total_windows: int = 0
    profitable_windows: int = 0
    losing_windows: int = 0
    insufficient_windows: int = 0
    profitable_window_fraction: float = 0.0
    profit_factor_mean: float | None = None
    profit_factor_std: float | None = None
    expectancy_mean: float | None = None
    expectancy_std: float | None = None
    drawdown_mean: float | None = None
    drawdown_std: float | None = None
    param_stability: float = 1.0  # 1.0 = perfectly stable, 0.0 = unstable
    single_window_dependence: float = 0.0  # fraction of profit from best window
    positive_symbol_fraction: float = 0.0
    regime_coverage: dict[str, float] = field(default_factory=dict)
    per_window_summary: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_windows": self.total_windows,
            "profitable_windows": self.profitable_windows,
            "losing_windows": self.losing_windows,
            "insufficient_windows": self.insufficient_windows,
            "profitable_window_fraction": self.profitable_window_fraction,
            "profit_factor_mean": self.profit_factor_mean,
            "profit_factor_std": self.profit_factor_std,
            "expectancy_mean": self.expectancy_mean,
            "expectancy_std": self.expectancy_std,
            "drawdown_mean": self.drawdown_mean,
            "drawdown_std": self.drawdown_std,
            "param_stability": self.param_stability,
            "single_window_dependence": self.single_window_dependence,
            "positive_symbol_fraction": self.positive_symbol_fraction,
            "regime_coverage": self.regime_coverage,
            "per_window_summary": self.per_window_summary,
        }


@dataclass(frozen=True)
class ValidationScore:
    """Documented validation scoring result.

    The score is an explicitly weighted sum of components (0..1 each).
    See ``validation.py`` for the formula documentation.
    """

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    status: StrategyStatus = StrategyStatus.NOT_VALIDATED
    reasons: list[str] = field(default_factory=list)
    hard_gates: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "components": self.components,
            "status": self.status.value,
            "reasons": self.reasons,
            "hard_gates": self.hard_gates,
        }


@dataclass(frozen=True)
class StrategyValidation:
    """Complete machine-readable strategy validation artifact."""

    strategy_name: str
    strategy_version: str
    engine_version: str
    data_hash: str
    configuration_hash: str
    validation_status: StrategyStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward_metrics: dict[str, Any] = field(default_factory=dict)
    regime_metrics: dict[str, Any] = field(default_factory=dict)
    symbol_metrics: dict[str, Any] = field(default_factory=dict)
    risk_metrics: dict[str, Any] = field(default_factory=dict)
    recommended_parameters: dict[str, Any] = field(default_factory=dict)
    risk_recommendation: dict[str, Any] = field(default_factory=dict)
    validation_score: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "engine_version": self.engine_version,
            "data_hash": self.data_hash,
            "configuration_hash": self.configuration_hash,
            "validation_status": self.validation_status.value,
            "metrics": self.metrics,
            "walk_forward_metrics": self.walk_forward_metrics,
            "regime_metrics": self.regime_metrics,
            "symbol_metrics": self.symbol_metrics,
            "risk_metrics": self.risk_metrics,
            "recommended_parameters": self.recommended_parameters,
            "risk_recommendation": self.risk_recommendation,
            "validation_score": self.validation_score,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ResearchSummary:
    """Compact research summary across all symbols/timeframes."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "created_at": self.created_at,
        }