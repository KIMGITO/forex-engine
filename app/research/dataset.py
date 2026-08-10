"""Research dataset repository and incremental sync.

Uses the existing Parquet repository (additively) for persistence, but stores
each symbol/timeframe under a partitioned path so M5/M15/H1/H4/D1 datasets can
coexist without collisions. Every dataset retains full provenance.

The sync is incremental and idempotent: running it twice must not duplicate
candles. Missing candles are never fabricated — gaps are surfaced in provenance.
"""

from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import pandas as pd

from app.data.models import Candle
from app.data.provenance import ProviderMetadata, read_metadata, write_metadata
from app.data.provider import BaseMarketDataProvider
from app.research.models import TimeframeDataset

__all__ = ["PartitionedResearchRepository", "sync_partition"]


def _safe_symbol(symbol: str) -> str:
    return symbol.upper().replace("/", "_").replace("-", "")


class PartitionedResearchRepository:
    """Stores one Parquet partition per symbol/timeframe, with provenance.

    Layout: ``storage_root / <SYMBOL> / <TIMEFRAME> / data.parquet``.
    """

    def __init__(self, storage_root: str = "data/processed"):
        self.root = Path(storage_root)

    def _partition_path(self, symbol: str, timeframe: str) -> Path:
        return (
            self.root
            / _safe_symbol(symbol)
            / timeframe.upper()
        )

    def candles_path(self, symbol: str, timeframe: str) -> Path:
        return self._partition_path(symbol, timeframe) / "data.parquet"

    def meta_path(self, symbol: str, timeframe: str) -> Path:
        return self._partition_path(symbol, timeframe) / "meta.json"

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self.candles_path(symbol, timeframe).exists()

    def save_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: list[Candle],
        metadata: ProviderMetadata | None = None,
    ) -> None:
        """Overwrite partition with candles (existing data destroyed is NOT
        intended for sync; use ``merge_candles`` for incremental append)."""
        if not candles:
            return
        path = self.candles_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([c.model_dump() for c in candles])
        df.to_parquet(path, index=False)
        if metadata is not None:
            write_metadata(path, metadata)

    def merge_candles(
        self,
        candles: list[Candle],
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> tuple[int, int]:
        """Append-and-dedupe on timestamp. Returns (existing_count, final_count).

        Idempotent: running twice with the same candles yields no duplicate rows.

        ``symbol``/``timeframe`` may override the values inferred from the first
        candle. This is necessary because some providers normalize the symbol
        (e.g. mock prefixes ``SYNTHETIC_``); the partition must be stored under
        the symbol the caller requested, not the provider's internal label.
        """
        if not candles:
            return 0, 0
        symbol = symbol or candles[0].symbol
        timeframe = timeframe or candles[0].timeframe
        path = self.candles_path(symbol, timeframe)
        incoming = pd.DataFrame([c.model_dump() for c in candles])
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"])

        if path.exists():
            existing = pd.read_parquet(path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            existing_count = len(existing)
            merged = pd.concat([existing, incoming], ignore_index=True)
        else:
            existing_count = 0
            merged = incoming

        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        return existing_count, len(merged)

    def load_df(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load a partition as a DataFrame (or None if absent)."""
        path = self.candles_path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").set_index("timestamp")

    def latest_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        df = self.load_df(symbol, timeframe)
        if df is None or df.empty:
            return None
        return df.index[-1].to_pydatetime()

    def detect_gaps(self, df: pd.DataFrame, expected_minutes: int) -> list[tuple]:
        """Return list of (prev_ts, next_ts) gaps exceeding expected interval."""
        if df is None or len(df) < 2:
            return []
        gaps: list[tuple] = []
        idx = df.index
        for a, b in pairwise(idx):
            delta_min = (b - a).total_seconds() / 60.0
            if delta_min > expected_minutes * 1.02:
                gaps.append((a.to_pydatetime(), b.to_pydatetime()))
        return gaps

    def provenance(self, symbol: str, timeframe: str) -> ProviderMetadata | None:
        path = self.candles_path(symbol, timeframe)
        if not path.exists():
            return None
        return read_metadata(path)

    def describe(self, symbol: str, timeframe: str, expected_minutes: int) -> TimeframeDataset | None:
        """Structured description of a partition including gaps/provenance."""
        df = self.load_df(symbol, timeframe)
        if df is None:
            return None
        meta = self.provenance(symbol, timeframe)
        gaps = self.detect_gaps(df, expected_minutes)
        return TimeframeDataset(
            symbol=symbol,
            timeframe=timeframe,
            provider=meta.provider if meta else "unknown",
            start=df.index[0].to_pydatetime(),
            end=df.index[-1].to_pydatetime(),
            row_count=len(df),
            timezone=str(getattr(df.index, "tz", None) or "UTC"),
            gaps=len(gaps),
            source_metadata={
                "retrieved_at": (
                    meta.retrieved_at.isoformat() if meta and meta.retrieved_at else None
                ),
                "timezone": meta.timezone if meta else "UTC",
                "notes": meta.notes if meta else None,
            },
        )


# Expected minutes per timeframe for gap detection.
_TIMEFRAME_MINUTES = {
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def _expected_minutes(timeframe: str) -> int:
    return _TIMEFRAME_MINUTES.get(timeframe.upper(), 60)


def sync_partition(
    provider: BaseMarketDataProvider,
    repo: PartitionedResearchRepository,
    symbol: str,
    timeframe: str,
    end: datetime | None = None,
    start: datetime | None = None,
) -> tuple[int, int]:
    """Incrementally sync one symbol/timeframe partition (idempotent).

    Fetches only the missing tail (from the latest local timestamp, with a
    small overlap for dedup), then merges. Returns (existing_count, new_count).

    No infinite retry loop: provider rate-limit/retry behavior is delegated to
    the shared HttpClient (Step 7) which bounds retries.
    """
    end = end or datetime.now(timezone.utc)
    latest = repo.latest_timestamp(symbol, timeframe)

    if start is not None:
        fetch_start = start
    elif latest is not None:
        # Re-download the final candle period to bridge boundary (deduped).
        fetch_start = latest
    else:
        raise ValueError(
            "No local data and no explicit start; cannot determine fetch window."
        )

    fetched = provider.fetch_candles(symbol, timeframe, fetch_start, end)
    # Store under the caller-requested symbol/timeframe (providers may
    # normalize the symbol internally, e.g. mock prefixes SYNTHETIC_).
    existing_count, final_count = repo.merge_candles(
        fetched, symbol=symbol, timeframe=timeframe
    )
    return existing_count, final_count