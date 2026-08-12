"""Shared causal-index helper for strategy/backtest contexts.

Contexts filter lists by ``available_from <= now`` per bar. This module
builds a sorted index once and shares it across per-bar contexts, turning
O(n) per-bar scans into O(log n) boundary + O(k) prefix.
"""

from __future__ import annotations

from typing import Any, Sequence


def build_causal_index(
    items: Sequence[Any] | None,
) -> tuple[list[Any], list[Any]]:
    """Return (sorted_items, keys) with None availability first."""
    if not items:
        return [], []
    pairs = sorted(
        items,
        key=lambda it: (it.available_from is None, it.available_from),
    )
    return pairs, [it.available_from for it in pairs]


def available_prefix(items: list[Any], keys: list[Any], now: Any) -> list[Any]:
    """Prefix of items whose availability <= now (None always usable)."""
    if not items:
        return []
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] is None or keys[mid] <= now:
            lo = mid + 1
        else:
            hi = mid
    return items[:lo]