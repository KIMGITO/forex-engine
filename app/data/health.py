"""Market-data health reporting.

Produces a deterministic, structured health assessment for a local dataset
and its provider. Health checks are lightweight and never download large
datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.data.provider import BaseMarketDataProvider
from app.data.repository import ParquetMarketDataRepository
from app.data.sync import detect_gaps

__all__ = ["MarketDataHealth", "check_health"]


@dataclass
class MarketDataHealth:
    """Structured health report for a symbol/timeframe dataset."""

    symbol: str
    timeframe: str
    provider_reachable: bool | None
    auth_valid: bool | None
    latest_remote_timestamp: datetime | None
    latest_local_timestamp: datetime | None
    is_stale: bool | None
    gap_count: int
    duplicate_count: int
    last_request_error: str | None
    issues: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MarketDataHealth(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"reachable={self.provider_reachable}, stale={self.is_stale}, "
            f"gaps={self.gap_count}, dupes={self.duplicate_count}, "
            f"last_error={self.last_request_error!r})"
        )


def check_health(
    provider: BaseMarketDataProvider | None,
    repository: ParquetMarketDataRepository,
    symbol: str,
    timeframe: str,
    remote_latest: datetime | None = None,
    provider_reachable: bool | None = None,
    auth_valid: bool | None = None,
) -> MarketDataHealth:
    """Return a deterministic health report for a dataset.

    Parameters
    ----------
    provider / repository: backing data layer objects. ``provider`` may be
      None when the caller has already probed reachability/auth externally.
    symbol / timeframe: dataset identifier.
    remote_latest: caller may pass the latest candle timestamp from a recent
      lightweight provider probe (or None if not probed).
    provider_reachable / auth_valid: caller may pre-fill these probe results.

    The report does not perform network I/O itself; it consumes whatever the
    caller has already probed. This keeps the health check deterministic and
    fast.
    """
    issues: list[str] = []

    # Local latest timestamp (None if dataset missing/empty).
    try:
        local_latest = repository.latest_timestamp(symbol, timeframe)
    except Exception:  # noqa: BLE001 - missing file treated as no data
        local_latest = None
        issues.append("local dataset missing or unreadable")
    if local_latest is None:
        issues.append("no local data available")

    # Duplicates + gaps from local data if present.
    duplicate_count = 0
    gap_count = 0
    if local_latest is not None:
        try:
            df = repository.load_candles_df(symbol, timeframe)
            duplicate_count = int(df["timestamp"].duplicated().sum())
            from app.data.models import Candle

            candles = [
                Candle(
                    symbol=str(r["symbol"]),
                    timeframe=str(r["timeframe"]),
                    timestamp=pd.Timestamp(r["timestamp"]).to_pydatetime(),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=(
                        None
                        if r.get("volume") is None or pd.isna(r.get("volume"))
                        else float(r["volume"])
                    ),
                )
                for _, r in df.iterrows()
            ]
            gaps = detect_gaps(candles)
            gap_count = len(gaps)
            if gap_count:
                issues.append(f"{gap_count} gap(s) detected")
        except Exception as exc:  # noqa: BLE001 - report, never crash health check
            issues.append(f"failed to inspect local data: {exc}")

    # Staleness: compare local latest against remote latest if both present.
    is_stale = None
    if local_latest is not None and remote_latest is not None:
        is_stale = local_latest < remote_latest

    return MarketDataHealth(
        symbol=symbol,
        timeframe=timeframe,
        provider_reachable=provider_reachable,
        auth_valid=auth_valid,
        latest_remote_timestamp=remote_latest,
        latest_local_timestamp=local_latest,
        is_stale=is_stale,
        gap_count=gap_count,
        duplicate_count=duplicate_count,
        last_request_error=None,
        issues=issues,
    )