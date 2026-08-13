"""Resumable state for Step 13 chunked event extraction.

State layout::

    {
      "EURUSD/M15": {
        "chunk_0": "complete",
        "chunk_1": "running",
        "chunk_2": "pending"
      }
    }

A chunk is only marked complete AFTER its artifact + meta are committed and
validated. On ``--resume``, complete chunks are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class Step13State:
    """Persistent per-chunk progress tracker (atomic updates)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self.path.read_text("utf-8"))
            if isinstance(raw, dict):
                self._data = raw
        except Exception:  # noqa: BLE001 - corrupt state treated as empty
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, default=str),
            "utf-8",
        )
        os.replace(str(tmp), str(self.path))

    @staticmethod
    def _key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}/{timeframe.upper()}"

    @staticmethod
    def _chunk_key(chunk_index: int) -> str:
        return f"chunk_{chunk_index}"

    def status(self, symbol: str, timeframe: str, chunk_index: int) -> str:
        return self._data.get(
            self._key(symbol, timeframe), {}
        ).get(self._chunk_key(chunk_index), "pending")

    def set_status(
        self, symbol: str, timeframe: str, chunk_index: int, status: str
    ) -> None:
        key = self._key(symbol, timeframe)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][self._chunk_key(chunk_index)] = status
        self._save()

    def next_incomplete_chunk(
        self, symbol: str, timeframe: str, total_chunks: int
    ) -> int:
        """First chunk index not marked complete; ``total_chunks`` when all done."""
        statuses = self._data.get(self._key(symbol, timeframe), {})
        for idx in range(total_chunks):
            if statuses.get(f"chunk_{idx}", "pending") != "complete":
                return idx
        return total_chunks

    def to_dict(self) -> dict:
        return dict(self._data)