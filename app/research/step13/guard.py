"""Pre-computation RSS guard for Step 13.

The guard runs BEFORE heavy analytical objects are constructed (features,
market structure, regime, MTF). On Linux it reads MemAvailable and fails
early with an actionable message rather than letting the host OOM.

This mirrors the pattern already proven in ``app/mtf/engine.py``
(``_require_rss_headroom``) and is intentionally testable without exhausting
system memory.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def rss_mb() -> float:
    """Current process RSS in MB (Linux /proc/self/status) or -1."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - non-Linux / unreadable
        return -1.0
    return -1.0


def mem_available_mb() -> float:
    """System MemAvailable in MB (Linux /proc/meminfo) or -1."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - non-Linux
        return -1.0
    return -1.0


class RssGuardError(RuntimeError):
    """Raised when the pre-computation RSS guard blocks the run."""


def require_rss_headroom(
    *,
    rss_limit_mb: float,
    min_mem_available_mb: float = 256.0,
    stage: str = "analysis",
    extra_hint: str = "",
) -> float:
    """Fail BEFORE heavy computation if RSS/MemAvailable are unsafe.

    Returns current RSS (MB) when safe.

    Raises
    ------
    RssGuardError
        With an actionable message when the guard trips.
    """
    rss = rss_mb()
    avail = mem_available_mb()

    if rss_limit_mb > 0.0 and 0.0 < rss > rss_limit_mb:
        raise RssGuardError(
            f"RSS limit exceeded before {stage} precompute.\n"
            f"  Current RSS: {rss:.0f} MB\n"
            f"  Available memory: {avail:.0f} MB\n"
            f"  Configured limit: {rss_limit_mb:.0f} MB\n"
            f"Suggested actions:\n"
            f"  --max-bars 20000\n"
            f"  --chunk-size 2500\n"
            f"  close memory-heavy applications"
            + (f"\n  {extra_hint}" if extra_hint else "")
        )

    if 0.0 < avail < min_mem_available_mb:
        raise RssGuardError(
            f"System memory too low before {stage} precompute.\n"
            f"  Current RSS: {rss:.0f} MB\n"
            f"  Available memory: {avail:.0f} MB (min {min_mem_available_mb:.0f})\n"
            f"Suggested actions:\n"
            f"  close memory-heavy applications\n"
            f"  reduce --max-bars"
            + (f"\n  {extra_hint}" if extra_hint else "")
        )

    return rss