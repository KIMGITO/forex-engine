"""Incremental market-data synchronization.

Given a local Parquet dataset, computes the next start boundary and fetches
only the missing tail from the provider. Overlapping candles are deduplicated
on timestamp; gaps are *reported*, never fabricated.
"""

from datetime import datetime

import pandas as pd

from app.data.exceptions import StorageError
from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.data.repository import ParquetMarketDataRepository

__all__ = ["detect_gaps", "sync_candles"]


def sync_candles(
    provider: BaseMarketDataProvider,
    repository: ParquetMarketDataRepository,
    symbol: str,
    timeframe: str,
    end: datetime,
    start: datetime | None = None,
) -> list[Candle]:
    """Synchronize local data with the provider up to ``end``.

    Strategy:
    1. Determine the latest local timestamp (or use ``start`` for a first
       download). The incremental fetch begins at the latest local candle so
       the overlap window is intentionally re-downloaded and deduplicated.
    2. Fetch [fetch_start, end] from the provider.
    3. Merge: existing local candles + fetched candles, deduplicated on
       timestamp, sorted chronologically.
    4. Persist the merged dataset (existing data is never destroyed).

    Returns the merged candle list.
    """
    latest_local = repository.latest_timestamp(symbol, timeframe)

    if start is not None:
        fetch_start = start
    elif latest_local is not None:
        fetch_start = latest_local
    else:
        raise ValueError(
            "No local data and no explicit start; cannot determine fetch window."
        )

    fetched = provider.fetch_candles(symbol, timeframe, fetch_start, end)

    target_path = repository.filepath(symbol, timeframe)
    if not target_path.exists():
        # First download: no local data yet.
        existing = []
    else:
        try:
            existing_df = repository.load_candles_df(symbol, timeframe)
        except Exception as exc:
            # A real storage failure is surfaced, never silently discarded.
            raise StorageError(f"Failed to load existing candles during sync: {exc}") from exc
        existing = _df_to_candles(existing_df)

    merged = _merge_dedupe(existing + fetched)
    # Persist under the canonical (uppercase) symbol used by the candles.
    candles = merged
    if candles:
        repository.save_candles(candles)
    return candles


def detect_gaps(
    candles: list[Candle],
    expected_interval_minutes: int | None = None,
    tolerance_pct: float = 0.02,
) -> list[tuple]:
    """Return a list of (gap_start, gap_end) tuples where candles are missing.

    ``expected_interval_minutes`` is inferred from the most common timestamp
    delta when not supplied. Weekends/holidays naturally produce gaps around
    Friday-close/Monday-open in many FX pairs — these are reported, never
    filled.

    Returns
    -------
    List[(datetime, datetime)]
        Chronological list of gaps, each a (previous_timestamp, next_timestamp)
        pair whose delta exceeds the expected interval.
    """
    if len(candles) < 2:
        return []

    ordered = sorted(candles, key=lambda c: c.timestamp)
    deltas = [
        (ordered[i + 1].timestamp - ordered[i].timestamp).total_seconds() / 60.0
        for i in range(len(ordered) - 1)
    ]
    if expected_interval_minutes is None:
        expected_interval_minutes = int(_mode(deltas))

    threshold = expected_interval_minutes * (1.0 + tolerance_pct)
    gaps: list[tuple] = []
    for i in range(len(ordered) - 1):
        delta = (ordered[i + 1].timestamp - ordered[i].timestamp).total_seconds() / 60.0
        if delta > threshold:
            gaps.append((ordered[i].timestamp, ordered[i + 1].timestamp))
    return gaps


def _merge_dedupe(candles: list[Candle]) -> list[Candle]:
    """Merge and deduplicate candles on timestamp, chronologically sorted."""
    by_ts: dict = {}
    for c in candles:
        by_ts[c.timestamp] = c  # later occurrence wins for identical timestamps
    return [by_ts[ts] for ts in sorted(by_ts)]


def _df_to_candles(df: pd.DataFrame) -> list[Candle]:
    from app.data.models import Candle

    out: list[Candle] = []
    for _, row in df.iterrows():
        out.append(
            Candle(
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=(
                    None
                    if row.get("volume") is None or pd.isna(row.get("volume"))
                    else float(row["volume"])
                ),
            )
        )
    return out


def _mode(values: list[float]) -> float:
    """Return the most common value (simple frequency mode)."""
    if not values:
        return 0.0
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.keys(), key=lambda k: counts[k])
