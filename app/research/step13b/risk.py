"""Step 15 RiskEngine integration for Step 13B research.

Research must evaluate trades through the SAME risk rules the future live
system will use. This module provides a research-facing wrapper that:
* builds AccountState from backtest context (portfolio values)
* evaluates each proposed signal through RiskEngine
* records approved/rejected decisions with structured reasons
* reports risk counters (daily loss, drawdown, position limits, exposure)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.backtest.engine import BacktestContext
from app.risk import (
    AccountState,
    PositionSide,
    ProposedTrade,
    RejectionReason,
    RiskConfig,
    RiskDecision,
    RiskEngine,
)
from app.strategy.models import Signal, SignalDirection


class RiskResearchTracker:
    """Tracks risk engine evaluations and cumulative counters for a phase."""

    def __init__(self, risk_engine: RiskEngine) -> None:
        self.risk_engine = risk_engine
        self.approved: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.counters = {
            "approved": 0,
            "rejected": 0,
            "daily_loss_limit": 0,
            "drawdown_limit": 0,
            "max_open_positions": 0,
            "position_size_too_small": 0,
            "position_size_too_large": 0,
            "per_trade_risk_exceeded": 0,
            "symbol_exposure": 0,
            "total_exposure": 0,
            "exposure_group": 0,
            "duplicate_position": 0,
            "invalid_trade": 0,
            "invalid_instrument": 0,
            "stop_on_wrong_side": 0,
            "insufficient_margin": 0,
            "emergency_stop": 0,
            "other": 0,
        }

    def _account_state(self, context: BacktestContext) -> AccountState:
        """Build an AccountState from the backtest context's portfolio."""
        portfolio = context._portfolio  # private but stable across the engine
        if portfolio is None:
            return AccountState(
                balance=10_000.0,
                equity=10_000.0,
                peak_equity=10_000.0,
                daily_pnl=0.0,
                open_positions=[],
            )

        mid = float(context.current_candle()["close"])
        equity = portfolio.equity(mid)
        peak = max(
            equity,
            portfolio.balance,
            getattr(portfolio, "_peak_equity", equity),
        )
        setattr(portfolio, "_peak_equity", peak)

        # Daily P&L approximator: track equity at start of UTC day.
        now = context.now.to_pydatetime()
        day_key = now.date().isoformat()
        if not hasattr(self, "_day_equity"):
            self._day_equity: dict[str, float] = {}
        if day_key not in self._day_equity:
            self._day_equity[day_key] = equity

        open_positions = []
        for sym, pos in (portfolio.positions or {}).items():
            open_positions.append(
                {
                    "symbol": sym,
                    "side": "buy" if pos.side.value == "buy" else "sell",
                    "quantity": pos.quantity,
                    "entry_price": pos.average_entry,
                }
            )

        return AccountState(
            balance=portfolio.balance,
            equity=equity,
            peak_equity=peak,
            daily_pnl=equity - self._day_equity.get(day_key, equity),
            open_positions=open_positions,
            drawdown_pct=(
                ((peak - equity) / peak) if peak > 0 else 0.0
            ),
        )

    def evaluate_signal(
        self,
        signal: Signal,
        context: BacktestContext,
        *,
        param_set: str = "baseline",
    ) -> RiskDecision:
        """Evaluate a signal through RiskEngine and record the result."""
        account = self._account_state(context)
        side = (
            PositionSide.BUY
            if signal.direction == SignalDirection.LONG
            else PositionSide.SELL
        )
        trade = ProposedTrade(
            symbol=signal.symbol,
            side=side,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            timeframe=signal.timeframe,
            signal_id=signal.signal_id,
        )
        decision = self.risk_engine.evaluate(trade, account)

        self._record(decision, signal, param_set)
        return decision

    # Maps RejectionReason enum values to research counter keys.
    _REASON_COUNTER_MAP = {
        "daily_loss_limit_exceeded": "daily_loss_limit",
        "drawdown_limit_exceeded": "drawdown_limit",
        "max_open_positions_reached": "max_open_positions",
        "position_size_too_small": "position_size_too_small",
        "position_size_too_large": "position_size_too_large",
        "per_trade_risk_exceeded": "per_trade_risk_exceeded",
        "symbol_exposure_exceeded": "symbol_exposure",
        "total_exposure_exceeded": "total_exposure",
        "exposure_group_exceeded": "exposure_group",
        "duplicate_position": "duplicate_position",
        "invalid_trade": "invalid_trade",
        "invalid_instrument": "invalid_instrument",
        "stop_on_wrong_side": "stop_on_wrong_side",
        "insufficient_margin": "insufficient_margin",
        "emergency_stop": "emergency_stop",
    }

    def _record(self, decision: RiskDecision, signal: Signal, param_set: str) -> None:
        entry = {
            "timestamp": signal.timestamp.isoformat(),
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "direction": signal.direction.value,
            "param_set": param_set,
            "decision": decision.type.value,
            "reason": (
                decision.reason.value if decision.reason is not None else None
            ),
            "message": decision.message,
            "position_size": decision.position_size,
            "monetary_risk": decision.monetary_risk,
        }

        if decision.approved:
            self.approved.append(entry)
            self.counters["approved"] += 1
        else:
            self.rejected.append(entry)
            self.counters["rejected"] += 1
            reason = (
                decision.reason.value if decision.reason is not None else "other"
            )
            counter_key = self._REASON_COUNTER_MAP.get(reason, "other")
            self.counters[counter_key] += 1

    def summary(self) -> dict[str, Any]:
        """Return risk research counters for the current phase."""
        return dict(self.counters)


def build_research_risk_engine(
    risk_percent: float = 0.01,
    max_daily_loss_pct: float | None = 0.03,
    max_drawdown_pct: float | None = 0.10,
    max_open_positions: int = 5,
) -> RiskEngine:
    """Build a RiskEngine configured for research.

    Instruments use default_specs() with USDJPY/USDCHF excluded by default
    (no constant quote->account conversion). For research, we approximate
    quote_to_account=1.0 for all configured pairs (documented assumption).
    """
    from app.risk.instrument import InstrumentSpec

    config = RiskConfig(
        risk_percent=risk_percent,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_open_positions=max_open_positions,
        prevent_duplicate_position=True,
    )
    instruments = {}
    for sym in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"):
        # Research approximation: quote_to_account=1.0 for all pairs.
        instruments[sym] = InstrumentSpec(symbol=sym, quote_to_account=1.0)
    return RiskEngine(config=config, instruments=instruments)


def risk_counters_to_metrics(tracker: RiskResearchTracker) -> dict[str, int]:
    """Convert a risk tracker to a metrics-ready dict subset."""
    c = tracker.counters
    return {
        "risk_rejected_count": c["rejected"],
        "risk_approved_count": c["approved"],
        "daily_loss_breaches": c["daily_loss_limit"],
        "drawdown_limit_breaches": c["drawdown_limit"],
        "position_limit_breaches": c["max_open_positions"],
        "exposure_breaches": c["symbol_exposure"] + c["total_exposure"] + c["exposure_group"],
    }