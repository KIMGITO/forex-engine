"""Strongly typed models for the research-grade backtesting engine.

All models are immutable where possible; classification and accounting values
use strict numeric validators so invalid P&L/equity states are caught at
construction time rather than silently propagated.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FILLED = "filled"
    CANCELLED = "cancelled"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class FillPolicy(str, Enum):
    """Policy for resolving intrabar ambiguity when both SL and TP are hit.

    OHLC data alone cannot reveal the exact intrabar sequence. We therefore
    apply a documented conservative policy by default (SL first), which is the
    less favorable outcome.
    """

    CONSERVATIVE_SL_FIRST = "conservative_sl_first"
    FULL_OHLC_WHIPSAW = "full_ohlc_whipsaw"


class EquityPoint(BaseModel):
    """A single timestamped equity observation."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    equity: float = Field(..., ge=0.0)
    balance: float
    unrealized_pnl: float


class OrderIntent(BaseModel):
    """Strategy-facing order request. Kept separate from the executed Order."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    order_type: OrderType = OrderType.MARKET
    requested_price: float | None = Field(default=None, gt=0.0)
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    timestamp: datetime


class Order(BaseModel):
    """A submitted order with its lifecycle state."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    order_type: OrderType
    requested_price: float | None = Field(default=None, gt=0.0)
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = Field(default=None, gt=0.0)
    filled_at: datetime | None = None
    reject_reason: str | None = None


class Fill(BaseModel):
    """An execution fill."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    timestamp: datetime
    slippage_applied: float = 0.0
    gross_value: float


class Position(BaseModel):
    """An open position (long or short)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    average_entry: float = Field(..., gt=0.0)
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    opened_at: datetime
    realized_pnl: float = 0.0
    fees: float = 0.0
    financing: float = 0.0
    holding_bars: int = 0


class Trade(BaseModel):
    """A completed entry→exit round trip."""

    model_config = ConfigDict(frozen=True)

    trade_id: str
    symbol: str
    side: OrderSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float = Field(..., gt=0.0)
    exit_price: float = Field(..., gt=0.0)
    quantity: float = Field(..., gt=0.0)
    net_pnl: float
    fees: float
    financing: float
    holding_bars: int


class PortfolioState(BaseModel):
    """Snapshot of account state at a timestamp."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    balance: float
    equity: float = Field(..., ge=0.0)
    margin: float
    free_margin: float
    used_margin: float
    open_positions: list[Position] = Field(default_factory=list)
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    financing: float


class BacktestConfigMeta(BaseModel):
    """Reproducibility metadata for a backtest result."""

    model_config = ConfigDict(frozen=True)

    provider: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    source_type: str
    strategy: str
    retrieved_at: datetime | None = None
    notes: str | None = None


class PerformanceMetrics(BaseModel):
    """Research metrics. None + ``insufficient_data`` flags where appropriate."""

    model_config = ConfigDict(frozen=True)

    total_return: float | None = None
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    win_rate: float | None = None
    loss_rate: float | None = None
    profit_factor: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    expectancy: float | None = None
    max_drawdown: float = 0.0
    drawdown_duration_bars: int = 0
    sharpe: float | None = None
    sortino: float | None = None
    trade_count: int = 0
    exposure_fraction: float = 0.0
    average_holding_bars: float | None = None
    insufficient_data: list[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    """Complete backtest output with provenance."""

    model_config = ConfigDict(frozen=True)

    metadata: BacktestConfigMeta
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    portfolio_states: list[PortfolioState] = Field(default_factory=list)
    metrics: PerformanceMetrics
    config_dump: dict = Field(default_factory=dict)