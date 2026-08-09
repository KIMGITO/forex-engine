"""Tests for the Parquet economic-event repository."""

from datetime import datetime, timezone

import pytest

from app.news.errors import StorageError
from app.news.models import EconomicEvent, EventCategory, EventImportance
from app.news.repository import ParquetEconomicEventRepository


def _make_event(event_id, ts=None, **overrides) -> EconomicEvent:
    base = {
        "event_id": event_id,
        "scheduled_at": ts if ts is not None else datetime(2024, 6, 10, 12, 30, tzinfo=timezone.utc),
        "country": "US",
        "currency": "USD",
        "affected_currencies": ["USD"],
        "event_name": f"Event {event_id}",
        "category": EventCategory.INFLATION,
        "importance": EventImportance.HIGH,
        "provider": "mock",
    }
    base.update(overrides)
    return EconomicEvent(**base)


class TestParquetRepository:
    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        repo = ParquetEconomicEventRepository(base_storage_path=str(tmp_path / "processed"))
        events = [
            _make_event("a", actual=0.3),
            _make_event("b"),
            _make_event("c", released_at=datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc)),
        ]
        repo.save_events(events)
        loaded = repo.load_events()
        assert len(loaded) == 3
        ids = {e.event_id for e in loaded}
        assert ids == {"a", "b", "c"}
        # Round-tripped event preserves importance + tz-aware scheduled_at.
        a = next(e for e in loaded if e.event_id == "a")
        assert a.importance == EventImportance.HIGH
        assert a.scheduled_at.tzinfo is not None

    def test_save_empty_clears(self, tmp_path) -> None:
        repo = ParquetEconomicEventRepository(base_storage_path=str(tmp_path / "processed"))
        repo.save_events([_make_event("a")])
        assert len(repo.load_events()) == 1
        repo.save_events([])
        assert repo.filepath.exists() is False

    def test_load_missing_file_raises(self, tmp_path) -> None:
        repo = ParquetEconomicEventRepository(base_storage_path=str(tmp_path / "processed"))
        with pytest.raises(StorageError):
            repo.load_events()

    def test_repository_creates_path(self, tmp_path) -> None:
        target = tmp_path / "nested" / "processed"
        ParquetEconomicEventRepository(base_storage_path=str(target))
        assert target.is_dir()

    def test_released_at_roundtrip(self, tmp_path) -> None:
        repo = ParquetEconomicEventRepository(base_storage_path=str(tmp_path / "processed"))
        released = datetime(2024, 6, 10, 12, 35, tzinfo=timezone.utc)
        repo.save_events([_make_event("a", released_at=released, actual=0.3)])
        loaded = repo.load_events()
        assert loaded[0].released_at == released
        assert loaded[0].actual == 0.3
