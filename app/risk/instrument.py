"""Instrument specification and position sizing.

Separation of concerns
----------------------
* **Instrument specification** — the pip size, lot/unit conventions, and
  per-unit value of a price move for a given symbol. This is broker-adjacent
  knowledge and must never be hardcoded into the core risk rules.
* **Position sizing** — deterministic sizing that converts a monetary risk
  budget into a quantity using the instrument's per-unit risk value. The
  calculation itself stays broker-independent; only the instrument spec
  carries execution-level precision.

``pip_size_for_symbol`` from ``app.backtest.costs`` is reused (JPY-quoted
pairs use 0.01, everything else 0.0001) rather than duplicating pip logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.backtest.costs import pip_size_for_symbol
from app.risk.errors import RiskError


@dataclass(frozen=True)
class InstrumentSpec:
    """Execution-level instrument specification.

    Parameters
    ----------
    symbol : str
        Normalised 6-letter FX symbol (e.g. "EURUSD").
    pip_size : float | None
        Price distance of one pip. When ``None``, inferred via
        :func:`pip_size_for_symbol` (JPY→0.01, else 0.0001). Explicit override
        is supported for exotic quote conventions.
    quote_to_account : float
        Conversion factor from one quote-currency unit to one account-currency
        unit. Defaults to 1.0, which is exact for USD-quoted accounts and
        USD-quote pairs (EURUSD, GBPUSD, AUDUSD, USDCAD). For USDJPY/USDCHF the
        caller must supply the inverse of the current price (a live conversion
        the engine must never guess).
    lot_size : float
        Standard lot size in *units of base currency* (default 100_000).
    min_lot : float
        Broker minimum lot size in lots (default 0.01).
    lot_step : float
        Broker lot step in lots (default 0.01).
    """

    symbol: str
    pip_size: float | None = None
    quote_to_account: float = 1.0
    lot_size: float = 100_000.0
    min_lot: float = 0.01
    lot_step: float = 0.01

    def __post_init__(self) -> None:
        sym = self.symbol.upper().replace("/", "").replace("_", "")
        if len(sym) != 6:
            raise RiskError(f"Invalid FX symbol: {self.symbol!r}")
        object.__setattr__(self, "symbol", sym)
        object.__setattr__(
            self,
            "pip_size",
            self.pip_size if self.pip_size is not None else pip_size_for_symbol(sym),
        )
        if self.pip_size <= 0:
            raise RiskError("pip_size must be > 0")
        if self.quote_to_account <= 0:
            raise RiskError("quote_to_account must be > 0")
        if self.lot_size <= 0:
            raise RiskError("lot_size must be > 0")
        if self.min_lot < 0:
            raise RiskError("min_lot cannot be negative")
        if self.lot_step <= 0:
            raise RiskError("lot_step must be > 0")

    @property
    def normalized_symbol(self) -> str:
        return self.symbol

    def pip_distance(self, entry: float, stop: float) -> float:
        """Number of pips between two prices (absolute)."""
        return abs(entry - stop) / self.pip_size

    def pip_value_per_unit(self) -> float:
        """Account-currency value of one pip per one unit of base currency.

        ``pip size in price terms x quote_to_account``. For a standard lot this
        equals ``pip_value_per_unit() * lot_size`` account currency per pip.
        """
        return self.pip_size * self.quote_to_account

    def units_per_lot(self, lots: float) -> float:
        """Base units represented by ``lots`` standard lots."""
        return lots * self.lot_size

    def quantize_lots(self, lots: float) -> float:
        """Round a lot size down to the broker lot step, respecting the minimum.

        Rounding down (never up) ensures the broker minimum is honoured without
        oversizing a position (conservative safety rule).
        """
        if lots <= 0:
            return 0.0
        stepped = (lots / self.lot_step) // 1.0 * self.lot_step
        # Floating-noise guard so e.g. 0.3000000004 snaps to 0.3.
        stepped = round(stepped, 10)
        if stepped < self.min_lot:
            return 0.0  # below broker minimum -> not tradable by this instrument
        return stepped


def default_specs() -> dict[str, InstrumentSpec]:
    """Instrument specs for the six configured FX pairs.

    quote_to_account defaults to 1.0 (exact for USD-account with USD-quote
    pairs). USDJPY/USDCHF require a live conversion at runtime and are NOT
    pre-populated with a fake constant — the engine must reject them until the
    caller supplies the correct conversion factor.
    """
    specs: dict[str, InstrumentSpec] = {}
    for sym in ("EURUSD", "GBPUSD", "AUDUSD", "USDCAD"):
        specs[sym] = InstrumentSpec(symbol=sym, quote_to_account=1.0)
    # USDJPY / USDCHF intentionally absent: no constant quote->account factor
    # may be assumed; see InstrumentSpec.quote_to_account.
    return specs


def position_size_for_risk(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    spec: InstrumentSpec,
) -> tuple[float, float]:
    """Compute the position size (in units) and monetary risk for a trade.

    Parameters
    ----------
    account_equity : float
        Account equity in account currency (must be > 0).
    risk_percent : float
        Fraction of equity to risk (0 < risk_percent <= 1).
    entry_price, stop_loss : float
        Trade levels in price terms.
    spec : InstrumentSpec
        Instrument specification.

    Returns
    -------
    (units, monetary_risk) where ``units`` is in base units and
    ``monetary_risk`` is in account currency.

    Raises
    ------
    RiskError
        For invalid inputs (zero/negative equity, risk_percent, or pip size).
    """
    if account_equity <= 0:
        raise RiskError("account_equity must be > 0")
    if not 0 < risk_percent <= 1:
        raise RiskError("risk_percent must be in (0, 1]")
    if entry_price <= 0 or stop_loss <= 0:
        raise RiskError("entry_price and stop_loss must be > 0")
    if entry_price == stop_loss:
        raise RiskError("zero stop distance: entry price equals stop loss")
    if spec.pip_size is None or spec.pip_size <= 0:
        raise RiskError("invalid instrument spec: pip_size must be > 0")
    if spec.quote_to_account <= 0:
        raise RiskError("invalid instrument spec: quote_to_account must be > 0")

    stop_distance = abs(entry_price - stop_loss)
    risk_budget = account_equity * risk_percent
    # Per-unit risk in account currency.
    per_unit_risk = stop_distance * spec.quote_to_account
    units = risk_budget / per_unit_risk
    monetary_risk = units * per_unit_risk
    return units, monetary_risk