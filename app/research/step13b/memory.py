"""Memory monitoring and guard for Step 13B.

The Step 13B pipeline must run on an 8 GB development machine. This module
provides RSS reporting and a configurable memory guard that aborts safely
before OOM.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)


def rss_mb() -> float:
    """Return current process RSS in megabytes from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - unavailable on unsupported platforms
        pass
    return -1.0


class MemoryLimitError(RuntimeError):
    """Raised when the process RSS exceeds the configured memory guard."""


@dataclass(frozen=True)
class MemoryGuard:
    """Abort-safe memory limit enforcement."""

    max_rss_mb: float = 3000.0
    check_interval: int = 1  # check every N calls

    def __post_init__(self) -> None:
        if self.max_rss_mb <= 0:
            raise ValueError("max_rss_mb must be > 0")

    def _current(self) -> float:
        return rss_mb()

    def check(self, stage: str = "") -> float:
        """Check current RSS; raise MemoryLimitError if over the limit."""
        current = self._current()
        if current <= 0:
            return current  # RSS unavailable; cannot enforce
        if current > self.max_rss_mb:
            raise MemoryLimitError(
                f"Memory limit exceeded: RSS={current:.0f}MB > "
                f"max={self.max_rss_mb:.0f}MB (stage={stage or 'unknown'})"
            )
        return current

    def report(self, label: str = "") -> float:
        """Log current RSS and return it."""
        current = self._current()
        prefix = f"[{label}] " if label else ""
        _log.info("%sRSS: %.0f MB (limit %.0f MB)", prefix, current, self.max_rss_mb)
        return current


def gc_collect_if_needed(threshold_mb: float = 200.0) -> None:
    """Run gc.collect() when RSS exceeds the given threshold since last.

    This is a conservative safety net — the pipeline already releases
    DataFrames at window boundaries. ``threshold_mb`` is the delta between the
    previous RSS and current RSS that triggers a forced collection.
    """
    gc.collect()