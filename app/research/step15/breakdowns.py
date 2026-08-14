"""Regime / session / market-context breakdown of OOS performance.

These breakdowns are computed ONLY on out-of-sample data and are NEVER used
to select the hypothesis. They describe where the frozen hypothesis made or
lost money after validation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _group_r(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    group_field: str,
) -> dict[str, dict[str, float | int | str]]:
    """Group per-trade net R by a causal feature field.

    ``group_field`` is a column on the candidate events frame (e.g.
    ``regime``, ``feature_session``, ``feature_structure_bias``,
    ``feature_htf_trend``, ``direction``).
    """
    if labels is None or labels.empty or events is None or events.empty:
        return {}
    if group_field not in events.columns:
        return {}

    by_cand: dict[str, str] = {}
    for _, e in events.iterrows():
        by_cand[str(e.get("candidate_id", ""))] = str(e.get(group_field, "unknown"))

    grouped: dict[str, list[float]] = {}
    for _, row in labels.iterrows():
        rv = row.get("label_r")
        if rv is None:
            continue
        g = by_cand.get(str(row.get("candidate_id", "")), "unknown")
        grouped.setdefault(g, []).append(float(rv))

    out: dict[str, Any] = {}
    for g, vals in grouped.items():
        if not vals:
            continue
        out[g] = {
            "trades": len(vals),
            "net_r": round(sum(vals), 4),
            "mean_r": round(sum(vals) / len(vals), 4),
            "win_rate": round(
                sum(1 for v in vals if v > 0) / len(vals), 4
            ),
        }
    return out


def regime_breakdown(events: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """OOS performance by market regime (causal regime at candidate time)."""
    return _group_r(events, labels, "regime")


def structure_bias_breakdown(
    events: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, Any]:
    """OOS performance by bullish/bearish structure bias."""
    return _group_r(events, labels, "feature_structure_bias")


def session_breakdown(events: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """OOS performance by session (Europe / New York / Asia / Late)."""
    return _group_r(events, labels, "feature_session")


def direction_breakdown(events: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """OOS performance by long/short direction."""
    return _group_r(events, labels, "direction")


def htf_alignment_breakdown(
    events: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, Any]:
    """OOS performance by HTF alignment (bullish/bearish/neutral/none)."""
    field = "feature_htf_trend"
    if field not in events.columns:
        field = "feature_htf_alignment"
    return _group_r(events, labels, field)


def full_breakdown(
    events: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Compute ALL documented OOS context breakdowns."""
    return {
        "regime": regime_breakdown(events, labels),
        "structure_bias": structure_bias_breakdown(events, labels),
        "session": session_breakdown(events, labels),
        "direction": direction_breakdown(events, labels),
        "htf_alignment": htf_alignment_breakdown(events, labels),
    }