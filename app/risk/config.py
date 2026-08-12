"""Risk configuration.

All values are DOCUMENTED DEVELOPMENT DEFAULTS, not claimed optimal for any
market or strategy. They are intentionally simple and conservative; a live
deployment must supply broker- and account-specific values through this
dataclass rather than modifying risk rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    """Configurable limits for the risk engine."""

    # ── Per-trade risk ────────────────────────────────────────────────────────
    # Fraction of account equity risked per trade (0 < risk_percent <= 1).
    risk_percent: float = 0.01  # 1% per trade
    # Absolute cap on monetary risk per trade in account currency (None = no cap).
    max_risk_per_trade: float | None = None

    # ── Position-size bounds (in base units) ──────────────────────────────────
    min_position_units: float = 0.0
    max_position_units: float | None = None

    # ── Daily loss limit ─────────────────────────────────────────────────────
    # Fraction of equity that may be lost in a single day before new trades are
    # rejected (None = disabled).
    max_daily_loss_pct: float | None = 0.03  # 3% daily stop

    # ── Maximum drawdown ─────────────────────────────────────────────────────
    # Fraction of peak equity by which the account may draw down before new
    # trades are rejected (None = disabled).
    max_drawdown_pct: float | None = 0.10  # 10% max drawdown

    # ── Maximum open positions ────────────────────────────────────────────────
    max_open_positions: int = 5

    # ── Exposure caps (notional in account currency) ──────────────────────────
    # Maximum notional exposure to a single symbol (None = disabled). These are
    # ABSOLUTE notional caps in account currency; equity-multiple semantics are
    # the caller's responsibility when constructing the value.
    max_symbol_exposure: float | None = None
    # Maximum total notional exposure across all symbols (None = disabled).
    max_total_exposure: float | None = None
    # Maximum notional exposure per exposure group (None = disabled).
    max_exposure_per_group: float | None = None

    # ── Duplicate position protection ─────────────────────────────────────────
    # Reject a new trade when a position already exists in the same symbol and
    # direction.
    prevent_duplicate_position: bool = False

    # ── Emergency stop ────────────────────────────────────────────────────────
    # Account-level trading-disabled state.
    emergency_stop: bool = False

    # ── Instrument / symbol safety ────────────────────────────────────────────
    # Reject symbols without a configured instrument spec (Unknown group).
    allow_unknown_symbols: bool = False

    # ── Related exposure groups ───────────────────────────────────────────────
    # Symbol -> exposure group override. When empty, ExposureGroup.from_symbol
    # is used (conservative USD-quote/USD-base-JPY/USD-base-CHF groups).
    exposure_groups: dict[str, str] = field(default_factory=dict)
