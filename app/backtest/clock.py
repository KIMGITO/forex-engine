"""Deterministic historical backtest clock.

Walks a pre-sorted, tz-aware DataFrame one timestamp at a time. The clock
surface is deliberately minimal: the backtest loop and the strategy can only
ever see the current bar index and the set of timestamps that have already
passed. There is intentionally no method to enumerate future timestamps.
"""

from collections.abc import Iterator

import pandas as pd

__all__ = ["BacktestClock"]


class BacktestClock:
    """Iterates a DataFrame index sequentially; never exposes future rows."""

    def __init__(self, timestamps: list[pd.Timestamp]) -> None:
        if not timestamps:
            raise ValueError("BacktestClock requires at least one timestamp.")
        self._timestamps = list(timestamps)

    def __iter__(self) -> Iterator[tuple]:
        yield from enumerate(self._timestamps)

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def start(self) -> pd.Timestamp:
        """First timestamp (chronologically earliest)."""
        return self._timestamps[0]

    @property
    def end(self) -> pd.Timestamp:
        """Last timestamp (chronologically latest)."""
        return self._timestamps[-1]

    def timestamps_up_to(self, i: int) -> list[pd.Timestamp]:
        """Timestamps [0..i] inclusive (current and history, never future)."""
        return self._timestamps[: i + 1]