"""Resume behavior, atomic artifact writing, and state management tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.research.step13b.artifacts import (
    ArtifactManager,
    atomic_write_json,
    atomic_write_parquet,
    read_json_if_valid,
    read_parquet_if_valid,
)
from app.research.step13b.state import ResearchState


class TestAtomicArtifacts:
    def test_atomic_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            atomic_write_json(path, {"key": "value", "n": 42})
            assert path.exists()
            data = read_json_if_valid(path)
            assert data == {"key": "value", "n": 42}

    def test_atomic_write_parquet_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            atomic_write_parquet(path, df)
            loaded = read_parquet_if_valid(path)
            assert loaded is not None
            assert len(loaded) == 3
            assert list(loaded["a"]) == [1, 2, 3]

    def test_corrupt_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not valid json", "utf-8")
            assert read_json_if_valid(path) is None

    def test_corrupt_parquet_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.parquet"
            path.write_bytes(b"not a parquet file")
            assert read_parquet_if_valid(path) is None

    def test_artifact_manager_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            am = ArtifactManager(tmp, "EURUSD", "M15")
            assert am.dir == Path(tmp) / "EURUSD" / "M15"
            assert am.windows_dir == Path(tmp) / "EURUSD" / "M15" / "windows"
            # Write a window result + trades parquet (both required for validity).
            am.write_window_result(0, {"index": 0, "status": "complete"})
            # JSON alone is not enough — trades parquet must also exist.
            assert am.window_json_path(0).exists()
            assert not am.window_artifacts_valid(0)
            am.write_window_trades(
                0, pd.DataFrame({"timestamp": [], "result": []})
            )
            assert am.window_artifacts_valid(0)

    def test_window_artifacts_not_valid_without_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            am = ArtifactManager(tmp, "EURUSD", "M15")
            am.write_window_result(0, {"index": 0, "status": "complete"})
            # Trades parquet doesn't exist, so not fully valid.
            assert not am.window_artifacts_valid(0)


class TestResearchState:
    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchState(Path(tmp) / "state.json")
            state.set_status("EURUSD", "M15", 0, "complete")
            state.set_status("EURUSD", "M15", 1, "running")
            assert state.status("EURUSD", "M15", 0) == "complete"
            assert state.status("EURUSD", "M15", 1) == "running"
            assert state.status("EURUSD", "M15", 2) == "pending"
            assert state.is_complete("EURUSD", "M15", 0)
            assert not state.is_complete("EURUSD", "M15", 1)

    def test_state_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = ResearchState(path)
            state.set_status("EURUSD", "M15", 2, "complete")
            # New instance reads from disk.
            state2 = ResearchState(path)
            assert state2.status("EURUSD", "M15", 2) == "complete"

    def test_next_incomplete_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchState(Path(tmp) / "state.json")
            state.set_status("EURUSD", "M15", 0, "complete")
            state.set_status("EURUSD", "M15", 1, "complete")
            state.set_status("EURUSD", "M15", 2, "running")
            assert state.next_incomplete_window("EURUSD", "M15", 5) == 2
            # All complete.
            state.set_status("EURUSD", "M15", 2, "complete")
            state.set_status("EURUSD", "M15", 3, "complete")
            state.set_status("EURUSD", "M15", 4, "complete")
            assert state.next_incomplete_window("EURUSD", "M15", 5) == 5
            assert state.all_complete("EURUSD", "M15", 5)

    def test_corrupt_state_falls_back_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{not valid", "utf-8")
            state = ResearchState(path)
            assert state.status("EURUSD", "M15", 0) == "pending"

    def test_state_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = ResearchState(path)
            for i in range(10):
                state.set_status("EURUSD", "M15", i, "complete")
            # No temp files left behind.
            tmps = list(Path(tmp).glob("*.tmp"))
            assert len(tmps) == 0
            data = json.loads(path.read_text("utf-8"))
            assert len(data["EURUSD/M15"]) == 10