"""Backtest configuration.

All values are DOCUMENTED DEVELOPMENT DEFAULTS. They are not claimed optimal
for any market or strategy, and simulated cost assumptions are explicit.
"""

from dataclasses import dataclass
from datetime import datetime

from app.backtest.models import FillPolicy


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run."""

    # ── Data scope ───────────────────────────────────────────────────────────
    symbol: str = "EURUSD"
    timeframe: str = "1h"
    start: datetime | None = None
    end: datetime | None = None

    # ── Account ──────────────────────────────────────────────────────────────
    initial_balance: float = 10_000.0
    account_currency: str = "USD"
    leverage: int = 30
    max_position_size: float = 10_000.0  # notional in account currency

    # ── Cost models (identifiers resolved by the engine) ──────────────────────
    spread_model: str = "fixed"  # "fixed" | (future: historical/tod/volatility)
    spread_pips: float = 0.8
    slippage_model: str = "fixed"  # "fixed" (deterministic)
    slippage_pips: float = 0.0
    commission_model: str = "zero"  # "zero" | "fixed" | "percentage"
    commission_per_trade: float = 0.0  # account currency, per side
    commission_percent: float = 0.0  # fraction of notional, per side
    swap_model: str = "none"  # "none" | (future: broker swap schedule)

    # ── Fill policy ──────────────────────────────────────────────────────────
    fill_policy: FillPolicy = FillPolicy.CONSERVATIVE_SL_FIRST

    # ── Pip conventions ──────────────────────────────────────────────────────
    # Auto-inferred from pair format unless overridden here.
    pip_size: float | None = None

    # ── Reporting ────────────────────────────────────────────────────────────
    benchmark: str = "buy_and_hold"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "initial_balance": self.initial_balance,
            "account_currency": self.account_currency,
            "leverage": self.leverage,
            "max_position_size": self.max_position_size,
            "spread_model": self.spread_model,
            "spread_pips": self.spread_pips,
            "slippage_model": self.slippage_model,
            "slippage_pips": self.slippage_pips,
            "commission_model": self.commission_model,
            "commission_per_trade": self.commission_per_trade,
            "commission_percent": self.commission_percent,
            "swap_model": self.swap_model,
            "fill_policy": self.fill_policy.value,
            "pip_size": self.pip_size,
            "benchmark": self.benchmark,
        }