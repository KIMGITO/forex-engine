"""Local development repository for economic events.

Stores :class:`EconomicEvent` objects in a Parquet file. The repository is
independent from any provider and can later be replaced with Supabase,
PostgreSQL, or another database.
"""

from pathlib import Path

import pandas as pd

from app.news.errors import StorageError
from app.news.models import EconomicEvent

__all__ = ["ParquetEconomicEventRepository"]


def _event_to_dict(event: EconomicEvent) -> dict:
    """Flatten an EconomicEvent into a JSON-safe dict for storage."""
    return {
        "event_id": event.event_id,
        "scheduled_at": event.scheduled_at.isoformat(),
        "timezone": event.timezone,
        "country": event.country,
        "currency": event.currency,
        "affected_currencies": ",".join(event.affected_currencies),
        "event_name": event.event_name,
        "category": event.category.value,
        "importance": event.importance.value,
        "actual": event.actual,
        "forecast": event.forecast,
        "previous": event.previous,
        "unit": event.unit,
        "source": event.source,
        "url": event.url,
        "provider": event.provider,
        "provider_importance": event.provider_importance,
        "received_at": event.received_at.isoformat() if event.received_at else None,
        "released_at": event.released_at.isoformat() if event.released_at else None,
        "available_from": event.effective_available_from().isoformat(),
    }


class ParquetEconomicEventRepository:
    """Parquet file repository for structured economic events."""

    def __init__(self, base_storage_path: str = "data/processed"):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.filepath = self.base_path / "economic_events.parquet"

    def save_events(self, events: list[EconomicEvent]) -> None:
        """Save events to Parquet, overwriting the previous snapshot."""
        if not events:
            self._clear()
            return
        try:
            df = pd.DataFrame([_event_to_dict(e) for e in events])
            df.to_parquet(self.filepath, index=False)
        except Exception as e:
            raise StorageError(f"Failed to save economic events: {e}") from e

    def _clear(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()

    def load_events(self) -> list[EconomicEvent]:
        """Load all stored events as EconomicEvent objects."""
        df = self._load_df()
        events: list[EconomicEvent] = []
        for _, row in df.iterrows():
            events.append(_row_to_event(row))
        return events

    def _load_df(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise StorageError(
                f"Economic event file not found at {self.filepath}"
            )
        try:
            return pd.read_parquet(self.filepath)
        except Exception as e:
            raise StorageError(f"Failed to load economic events: {e}") from e


def _row_to_event(row: pd.Series) -> EconomicEvent:
    from datetime import datetime

    from app.news.models import EventCategory, EventImportance

    scheduled = datetime.fromisoformat(row["scheduled_at"])
    aff = [c for c in str(row.get("affected_currencies") or "").split(",") if c]

    return EconomicEvent(
        event_id=row["event_id"],
        scheduled_at=scheduled,
        timezone=row["timezone"],
        country=row["country"],
        currency=row["currency"],
        affected_currencies=aff,
        event_name=row["event_name"],
        category=EventCategory(row["category"]),
        importance=EventImportance(row["importance"]),
        actual=_opt_float(row.get("actual")),
        forecast=_opt_float(row.get("forecast")),
        previous=_opt_float(row.get("previous")),
        unit=row.get("unit"),
        source=row.get("source"),
        url=row.get("url"),
        provider=row.get("provider"),
        provider_importance=row.get("provider_importance"),
        received_at=_opt_dt(row.get("received_at")),
        released_at=_opt_dt(row.get("released_at")),
        available_from=_opt_dt(row.get("available_from")),
    )


def _opt_float(value):
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _opt_dt(value):
    from datetime import datetime

    if value is None:
        return None
    return datetime.fromisoformat(value)
