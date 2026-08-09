"""Provider abstraction for external market data sources."""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

from app.data.models import Candle


class BaseMarketDataProvider(ABC):
    """Abstract interface for all data vendors (OANDA, Interactive Brokers, Synthetic)."""

    @abstractmethod
    def fetch_candles(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Candle]:
        """Fetch normalized candle list from provider."""


class MockMarketDataProvider(BaseMarketDataProvider):
    """Synthetic Data Provider explicitly labeled for local testing/development."""

    def fetch_candles(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Candle]:
        """Generates deterministic synthetic candle data (Clearly marked as synthetic)."""
        dt_range = pd.date_range(start=start, end=end, freq="1h")
        candles = []
        base_price = 1.0850

        for i, ts in enumerate(dt_range):
            # Synthetic price Walk
            open_p = base_price + np.sin(i / 10.0) * 0.0050
            close_p = open_p + (0.0005 if i % 2 == 0 else -0.0005)
            high_p = max(open_p, close_p) + 0.0008
            low_p = min(open_p, close_p) - 0.0008

            candle = Candle(
                symbol=f"SYNTHETIC_{symbol.upper()}",
                timeframe=timeframe,
                timestamp=ts.to_pydatetime(),
                open=round(open_p, 5),
                high=round(high_p, 5),
                low=round(low_p, 5),
                close=round(close_p, 5),
                volume=100.0,
            )
            candles.append(candle)

        return candles