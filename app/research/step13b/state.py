"""Research state file management for Step 13B (resumability).

State layout::

    {
      "EURUSD/M15": {
        "window_1": "complete",
        "window_2": "complete",
        "window_3": "running"
      }
    }

A window is only marked ``complete`` AFTER its artifacts are atomically
committed and validated. On ``--resume``, completed windows are skipped.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class ResearchState:
    """Persistent progress tracker for Step 13B, written atomically."""

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
        """Atomically persist state via temp file + os.replace."""
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
    def _window_key(window_index: int) -> str:
        return f"window_{window_index}"

    def status(
        self, symbol: str, timeframe: str, window_index: int
    ) -> str:
        """Return 'pending', 'running', 'complete', or 'failed'."""
        return self._data.get(
            self._key(symbol, timeframe), {}
        ).get(self._window_key(window_index), "pending")

    def set_status(
        self,
        symbol: str,
        timeframe: str,
        window_index: int,
        status: str,
    ) -> None:
        """Set window status and persist atomically."""
        key = self._key(symbol, timeframe)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][self._window_key(window_index)] = status
        self._save()

    def window_statuses(self, symbol: str, timeframe: str) -> dict[str, str]:
        """Return all window statuses for a symbol/timeframe."""
        return dict(self._data.get(self._key(symbol, timeframe), {}))

    def is_complete(self, symbol: str, timeframe: str, window_index: int) -> bool:
        return self.status(symbol, timeframe, window_index) == "complete"

    def next_incomplete_window(
        self, symbol: str, timeframe: str, total_windows: int
    ) -> int:
        """Return the first window index that is not complete.

        Returns ``total_windows`` when all windows are complete.
        """
        statuses = self.window_statuses(symbol, timeframe)
        for idx in range(total_windows):
            if statuses.get(f"window_{idx}", "pending") != "complete":
                return idx
        return total_windows

    def all_complete(self, symbol: str, timeframe: str, total_windows: int) -> bool:
        return self.next_incomplete_window(symbol, timeframe, total_windows) >= total_windows

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)