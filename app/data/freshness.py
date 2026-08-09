"""Data-freshness classification.

Reports whether local market data is fresh, delayed, or stale relative to the
provider and the symbol's timeframe interval. Thresholds are configurable;
the defaults are documented development values, not claimed optimal.
"""

from datetime import datetime, timezone

from app.data.repository import ParquetMarketDataRepository

__all__ = ["FreshnessReport", "classify_freshness"]


class FreshnessReport:
    """Structured freshness assessment for a symbol/timeframe."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        local_latest: datetime | None,
        remote_latest: datetime | None,
        local_age_minutes: float,
        status: str,
        interval_minutes: int,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.local_latest = local_latest
        self.remote_latest = remote_latest
        self.local_age_minutes = local_age_minutes
        self.status = status  # "fresh" | "delayed" | "stale" | "unknown"
        self.interval_minutes = interval_minutes

    def __repr__(self) -> str:
        return (
            f"FreshnessReport(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"status={self.status!r}, local_age_minutes={self.local_age_minutes:.2f}, "
            f"local_latest={self.local_latest}, remote_latest={self.remote_latest})"
        )


def classify_freshness(
    repository: ParquetMarketDataRepository,
    symbol: str,
    timeframe: str,
    remote_latest: datetime | None = None,
    *,
    interval_minutes: int | None = None,
    fresh_multiplier: float = 1.0,
    delayed_multiplier: float = 5.0,
) -> FreshnessReport:
    """Classify local data freshness.

    - ``fresh``: local latest is within ``interval_minutes * fresh_multiplier``
    - ``delayed``: within ``interval_minutes * delayed_multiplier``
    - ``stale``: older than delayed
    - ``unknown``: insufficient data (no local rows / interval unknown)

    ``interval_minutes`` defaults to a few common timeframe mappings when not
    supplied.
    """
    try:
        local_latest = repository.latest_timestamp(symbol, timeframe)
    except Exception:  # noqa: BLE001 - missing file treated as no data
        local_latest = None

    if interval_minutes is None:
        interval_minutes = _guess_interval_minutes(timeframe)

    now = datetime.now(timezone.utc)
    if local_latest is None:
        return FreshnessReport(
            symbol=symbol,
            timeframe=timeframe,
            local_latest=None,
            remote_latest=remote_latest,
            local_age_minutes=float("inf"),
            status="unknown",
            interval_minutes=interval_minutes,
        )

    age = (now - local_latest).total_seconds() / 60.0
    if age <= interval_minutes * fresh_multiplier:
        status = "fresh"
    elif age <= interval_minutes * delayed_multiplier:
        status = "delayed"
    else:
        status = "stale"

    return FreshnessReport(
        symbol=symbol,
        timeframe=timeframe,
        local_latest=local_latest,
        remote_latest=remote_latest,
        local_age_minutes=age,
        status=status,
        interval_minutes=interval_minutes,
    )


def _guess_interval_minutes(timeframe: str) -> int:
    tf = timeframe.lower()
    return {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
        "1M": 43200,
    }.get(tf, 60)