"""Research-layer configuration (documented development defaults)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchConfig:
    """Configuration for research dataset building, splits, and reporting."""

    # Dataset
    provider: str = "twelvedata"
    symbols: tuple = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
    timeframes: tuple = ("M5", "M15", "H1", "H4", "D1")
    storage_root: str = "data/processed"

    # Chronological split (explicit dates preferred when provided)
    train_start: str | None = None
    train_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    # Fractional fallback when dates are None
    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    # Walk-forward durations
    walk_train_years: float = 2.0
    walk_validation_years: float = 0.5
    walk_test_years: float = 0.5

    # Optimizer
    grid_space: dict = field(default_factory=dict)
    selection_metric: str = "expectancy"
    min_trades: int = 10

    # Cost assumptions (explicit; NOT historical bid/ask)
    spread_pips: float = 0.8
    slippage_pips: float = 0.0
    commission_per_trade: float = 0.0
    commission_percent: float = 0.0

    # Warnings
    warn_min_trades: int = 30
    warn_min_bars: int = 500

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "storage_root": self.storage_root,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "walk_train_years": self.walk_train_years,
            "walk_validation_years": self.walk_validation_years,
            "walk_test_years": self.walk_test_years,
            "selection_metric": self.selection_metric,
            "min_trades": self.min_trades,
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "commission_per_trade": self.commission_per_trade,
            "commission_percent": self.commission_percent,
            "warn_min_trades": self.warn_min_trades,
            "warn_min_bars": self.warn_min_bars,
        }
