"""Dataset provenance/sidecar metadata.

Every persisted dataset can have an accompanying metadata file that records
where the data came from, when it was retrieved, and how it was produced.
This is essential for reproducible backtests and for understanding the
limitations of a dataset before using it in production.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = ["ProviderMetadata", "read_metadata", "write_metadata"]


@dataclass
class ProviderMetadata:
    """Provenance record for a stored dataset."""

    symbol: str
    timeframe: str
    source_type: str  # "historical", "live", "mock"
    provider: str  # "oanda", "mock", ...
    data_type: str  # "OHLC", "bid_ask", "mid"
    retrieved_at: datetime | None
    timezone: str
    api_version: str | None
    notes: str | None = None

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source_type": self.source_type,
            "provider": self.provider,
            "data_type": self.data_type,
            "timezone": self.timezone,
            "notes": self.notes,
        }
        if self.retrieved_at is not None:
            d["retrieved_at"] = self.retrieved_at.isoformat()
        else:
            d["retrieved_at"] = None
        if self.api_version is not None:
            d["api_version"] = self.api_version
        return d


def _meta_path(data_path: Path) -> Path:
    """Companion metadata path for a parquet file."""
    return data_path.with_suffix(".meta.json")


def write_metadata(
    data_path: Path,
    metadata: ProviderMetadata,
) -> None:
    """Write a metadata sidecar next to the parquet file."""
    path = _meta_path(data_path)
    path.write_text(json.dumps(metadata.to_dict(), indent=2, default=str))


def read_metadata(data_path: Path) -> ProviderMetadata | None:
    """Read a metadata sidecar, if it exists."""
    path = _meta_path(data_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - malformed sidecar treated as absent
        return None
    return ProviderMetadata(
        symbol=raw["symbol"],
        timeframe=raw["timeframe"],
        source_type=raw["source_type"],
        provider=raw["provider"],
        data_type=raw["data_type"],
        retrieved_at=(
            datetime.fromisoformat(raw["retrieved_at"]) if raw.get("retrieved_at") else None
        ),
        timezone=raw.get("timezone", "UTC"),
        api_version=raw.get("api_version"),
        notes=raw.get("notes"),
    )