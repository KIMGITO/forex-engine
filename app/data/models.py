"""Internal provider-agnostic domain models for market data."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Candle(BaseModel):
    """Internal OHLCV candle representation.

    All prices are strictly positive floats, timestamps are UTC datetimes, and
    the OHLC relationships are mathematically enforced at validation time.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Standardized currency pair, e.g. EURUSD")
    timeframe: str = Field(..., description="Timeframe string, e.g. 1m, 1h, 1d")
    timestamp: datetime = Field(..., description="UTC timestamp of candle open")
    open: float = Field(..., gt=0.0, description="Open price")
    high: float = Field(..., gt=0.0, description="High price")
    low: float = Field(..., gt=0.0, description="Low price")
    close: float = Field(..., gt=0.0, description="Close price")
    volume: float | None = Field(default=None, ge=0.0, description="Volume if available")

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "Candle":
        """Enforce mathematical OHLC sanity rules."""
        if self.high < self.open or self.high < self.close or self.high < self.low:
            raise ValueError(f"High price {self.high} violates OHLC boundary constraints.")
        if self.low > self.open or self.low > self.close or self.low > self.high:
            raise ValueError(f"Low price {self.low} violates OHLC boundary constraints.")
        return self