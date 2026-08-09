"""Cost models and currency/pip utilities for the backtest engine."""

from abc import ABC, abstractmethod

__all__ = [
    "CommissionModel",
    "FixedPerTradeCommissionModel",
    "FixedSlippageModel",
    "FixedSpreadModel",
    "NoSwapModel",
    "PercentageCommissionModel",
    "SlippageModel",
    "SpreadModel",
    "SwapModel",
    "ZeroCommissionModel",
    "pip_distance",
    "pip_size_for_symbol",
]


def pip_size_for_symbol(symbol: str) -> float:
    """Return the pip size for a common Forex pair format.

    JPY-quoted pairs use 0.01; most others use 0.0001. Pairs with exotic quote
    currencies (e.g. USDHUF = 0.01-ish quote convention) are documented
    assumptions: we support the common 0.0001/0.01 conventions and allow an
    explicit override via BacktestConfig.pip_size.
    """
    uppercase = symbol.upper().replace("/", "").replace("_", "")
    if len(uppercase) != 6:
        # Unrecognized format: sensible default (most FX pairs).
        return 0.0001
    quote = uppercase[3:]
    if quote in ("JPY", "HUF", "VND"):
        return 0.01
    return 0.0001


def pip_distance(price_a: float, price_b: float, pip_size: float) -> float:
    """Number of pips between two prices."""
    if pip_size <= 0:
        raise ValueError("pip_size must be > 0")
    return abs(price_a - price_b) / pip_size


class SpreadModel(ABC):
    """Abstract spread model. Baseline: FixedSpreadModel (simulated)."""

    @abstractmethod
    def bid_ask(self, mid: float) -> tuple:
        """Return (bid, ask) for a given mid price."""
        raise NotImplementedError


class FixedSpreadModel(SpreadModel):
    """Fixed spread in pips. Explicitly SIMULATED: the underlying Twelve Data
    OHLC data has no historical bid/ask. This is an assumption, documented."""

    def __init__(self, spread_pips: float, pip_size: float) -> None:
        if spread_pips < 0:
            raise ValueError("spread_pips must be >= 0")
        self.spread = spread_pips
        self.pip = pip_size

    def bid_ask(self, mid: float) -> tuple:
        half = (self.spread * self.pip) / 2.0
        return mid - half, mid + half


class SlippageModel(ABC):
    """Abstract slippage model (deterministic baseline)."""

    @abstractmethod
    def slippage_price(self, requested: float, side) -> float:
        raise NotImplementedError


class FixedSlippageModel(SlippageModel):
    """Deterministic slippage in pips applied against the trader.

    BUY fills at requested + slippage; SELL fills at requested - slippage.
    No random component in the baseline model.
    """

    def __init__(self, slippage_pips: float, pip_size: float) -> None:
        if slippage_pips < 0:
            raise ValueError("slippage_pips must be >= 0")
        self.slippage = slippage_pips
        self.pip = pip_size

    def slippage_price(self, requested: float, side) -> float:
        delta = self.slippage * self.pip
        if side == "buy":
            return requested + delta
        return requested - delta


class CommissionModel(ABC):
    @abstractmethod
    def commission(self, notional: float, quantity: float) -> float:
        raise NotImplementedError


class ZeroCommissionModel(CommissionModel):
    def commission(self, notional: float, quantity: float) -> float:
        return 0.0


class FixedPerTradeCommissionModel(CommissionModel):
    def __init__(self, per_trade: float) -> None:
        self.per_trade = per_trade

    def commission(self, notional: float, quantity: float) -> float:
        return self.per_trade


class PercentageCommissionModel(CommissionModel):
    def __init__(self, percent: float) -> None:
        if percent < 0:
            raise ValueError("percent must be >= 0")
        self.percent = percent

    def commission(self, notional: float, quantity: float) -> float:
        return notional * self.percent


class SwapModel(ABC):
    """Abstract swap/financing extension point."""

    @abstractmethod
    def financing(self, position_notional: float, holding_bars: int) -> float:
        raise NotImplementedError


class NoSwapModel(SwapModel):
    """Development baseline: zero financing.

    Real broker swap schedules must be supplied as a future SwapModel
    implementation; this model invents nothing.
    """

    def financing(self, position_notional: float, holding_bars: int) -> float:
        return 0.0