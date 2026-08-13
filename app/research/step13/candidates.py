"""Candidate generation for Step 13.

A candidate is a potentially meaningful trading setup identified by a
STRICTLY CAUSAL join of market events:

    liquidity sweep
    + displacement after sweep (bounded lookback)
    + market structure context (bounded)
    + HTF alignment (ALWAYS a hypothesis condition, never a global hard filter)
    + regime
    + session

This module emits CANDIDATE rows with `feature_*` columns only (information
known at the candidate timestamp). Labels are computed separately and stored
in a distinct dataset — feature/label separation is enforced by the schema.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from app.research.step13.schema import (
    CANDIDATE_EVENTS_COLUMNS,
    CANDIDATE_ID_COLUMNS,
    CANDIDATE_LABELS_COLUMNS,
)


def _candidate_id(
    symbol: str, timeframe: str, ts, sweep_ref: str, displacement_ref: str
) -> str:
    """Deterministic candidate identifier."""
    raw = f"{symbol}|{timeframe}|{ts.isoformat()}|{sweep_ref}|{displacement_ref}"
    return f"cand_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


class CandidateGenerator:
    """Joins sweep/displacement/structure/HTF/regime events causally.

    HTF alignment is deliberately NOT a hard filter: it is recorded as
    ``feature_htf_trend`` on every candidate so hypotheses can compare
    WITH vs WITHOUT HTF alignment on the same event population.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        sweep_displacement_lookback: int = 5,
        min_displacement_class: tuple = ("large", "extreme"),
        require_htf_alignment: bool = False,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.sweep_lookback = sweep_displacement_lookback
        self.min_displacement_classes = set(min_displacement_class)
        self.require_htf_alignment = require_htf_alignment

    def generate(
        self,
        sweeps: list[dict[str, Any]],
        displacements: list[dict[str, Any]],
        regimes: list[dict[str, Any]],
        features: list[dict[str, Any]],
        mtf_rows: list[dict[str, Any]] | None = None,
        structure_rows: list[dict[str, Any]] | None = None,
        bar_index_map: dict[Any, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate candidate rows (feature_* only, strictly causal).

        ``bar_index_map`` maps timestamp key -> integer bar index so the
        displacement lookback is measured in ACTUAL BARS (not event counts).
        """
        if not sweeps:
            return []

        disp_sorted = sorted(displacements, key=lambda d: _ts(d["timestamp"]))
        regime_sorted = sorted(regimes, key=lambda r: _ts(r["timestamp"]))
        feature_sorted = sorted(features, key=lambda f: _ts(f["timestamp"]))
        mtf_sorted = (
            sorted(mtf_rows, key=lambda m: _ts(m["timestamp"]))
            if mtf_rows
            else []
        )
        struct_sorted = (
            sorted(structure_rows, key=lambda s: _ts(s["timestamp"]))
            if structure_rows
            else []
        )

        candidates: list[dict[str, Any]] = []
        for sweep in sweeps:
            sweep_ts = _ts(sweep["timestamp"])
            direction = sweep.get("direction", "")

            # 1. Find confirming displacement within BAR-based lookback.
            disp = _find_displacement(
                disp_sorted, sweep_ts, direction, self.sweep_lookback,
                self.min_displacement_classes, bar_index_map=bar_index_map,
            )
            if disp is None:
                continue

            # 2. Causal regime at candidate timestamp.
            regime = _latest_before(regime_sorted, sweep_ts)
            if regime is None:
                continue

            # 3. Causal features at candidate timestamp.
            feats = _latest_before(feature_sorted, sweep_ts)
            atr = feats.get("atr", float("nan")) if feats else float("nan")
            rsi = feats.get("rsi", float("nan")) if feats else float("nan")

            # 4. Causal BOUNDED structure bias (last 6 points, immutable).
            bias = _structure_bias_before(
                struct_sorted, sweep_ts, lookback_points=6
            )

            # 5. HTF context recorded (never globally hard-filtered).
            htf_alignment, htf_trend, htf_vol = _htf_at(mtf_sorted, sweep_ts)

            # 6. Candidate fields.
            # entry_ref MUST be a PRICE (the sweep candle's confirmed close) —
            # labels/execution_model do float(entry_ref) as event_close.
            # sweep_ref / displacement_ref are IDENTITY references (ISO
            # timestamps), used for the deterministic candidate_id.
            ref_disp = disp.get("available_from") or disp.get("timestamp")
            ref_sweep = sweep.get("available_from") or sweep.get("timestamp")
            entry_price = float(sweep.get("close_price") or 0.0) or None
            session = feats.get("session", "") if feats else ""

            candidates.append(
                {
                    "candidate_id": _candidate_id(
                        self.symbol, self.timeframe, sweep_ts,
                        str(ref_sweep), str(ref_disp),
                    ),
                    "timestamp": sweep_ts.isoformat(),
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "direction": direction,
                    "entry_ref": entry_price,
                    "sweep_ref": (
                        ref_sweep.isoformat()
                        if hasattr(ref_sweep, "isoformat")
                        else str(ref_sweep)
                    ),
                    "displacement_ref": (
                        ref_disp.isoformat()
                        if hasattr(ref_disp, "isoformat")
                        else str(ref_disp)
                    ),
                    "structure_ref": bias,
                    "htf_ref": htf_trend,
                    "regime": regime.get("market_state", "unknown"),
                    "session": session,
                    "available_from": (
                        ref_disp.isoformat()
                        if hasattr(ref_disp, "isoformat")
                        else str(ref_disp)
                    ),
                    "feature_atr": float(atr) if not _is_nan(atr) else None,
                    "feature_rsi": float(rsi) if not _is_nan(rsi) else None,
                    "feature_volatility": regime.get("volatility_state", "unknown"),
                    "feature_structure_bias": bias,
                    "feature_sweep_penetration": float(sweep.get("penetration", 0.0) or 0.0),
                    "feature_sweep_excursion": float(sweep.get("excursion", 0.0) or 0.0),
                    "feature_displacement_ratio": (
                        float(disp.get("range_ratio") or 0.0)
                        if disp.get("range_ratio") is not None
                        else None
                    ),
                    "feature_htf_alignment": htf_alignment,
                    "feature_htf_trend": htf_trend,
                    "feature_htf_volatility": htf_vol,
                    "feature_session": session,
                }
            )
        return candidates


def candidates_to_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=CANDIDATE_EVENTS_COLUMNS)
    df = pd.DataFrame(candidates)
    for c in CANDIDATE_EVENTS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_EVENTS_COLUMNS]


def labels_to_frame(labels: list[dict[str, Any]]) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(columns=CANDIDATE_LABELS_COLUMNS)
    df = pd.DataFrame(labels)
    for c in CANDIDATE_LABELS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_LABELS_COLUMNS]


# ── Helpers (strictly causal lookups) ────────────────────────────────────────


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _ts_key(value) -> str:
    return str(value)


def _is_nan(v) -> bool:
    import math

    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _latest_before(rows: list[dict[str, Any]], ts) -> dict[str, Any] | None:
    key = _ts(ts)
    latest = None
    latest_ts = None
    for r in rows:
        r_ts = _ts(r["timestamp"])
        if r_ts <= key and (latest_ts is None or r_ts > latest_ts):
            latest = r
            latest_ts = r_ts
    return latest


def _direction_key(direction: str) -> str:
    """Normalize a direction string to the long/short trading axis.

    Sweep rows use ``long`` (low swept, buy) / ``short`` (high swept, sell).
    Displacement rows use ``up`` / ``down`` (bar movement). Comparing these
    in their raw forms NEVER matches, which silently kills every candidate.

    Mapping: up -> long, down -> short (a long sweep confirms on an UP
    displacement; a short sweep confirms on a DOWN displacement).
    """
    d = str(direction).lower()
    if d in ("up", "long"):
        return "long"
    if d in ("down", "short"):
        return "short"
    return d


def _find_displacement(
    displacements: list[dict[str, Any]],
    sweep_ts,
    direction: str,
    lookback_bars: int,
    min_classes: set[str],
    *,
    bar_index_map: dict[Any, int] | None = None,
):
    """Find first displacement AFTER sweep with matching direction/class.

    ``lookback_bars`` counts ACTUAL BARS between the sweep and the confirming
    displacement when ``bar_index_map`` (timestamp key -> bar index) is
    provided. Without a bar map we count displacement events (documented
    heuristic fallback).

    Direction matching is namespace-normalized via ``_direction_key``: sweep
    direction is ``long``/``short``; displacement direction is ``up``/``down``.
    """
    sweep_ts = _ts(sweep_ts)
    sweep_idx = bar_index_map.get(_ts_key(sweep_ts)) if bar_index_map else None
    want = _direction_key(direction)
    for d in displacements:
        d_ts = _ts(d["timestamp"])
        if d_ts <= sweep_ts:
            continue
        if sweep_idx is not None and bar_index_map is not None:
            d_idx = bar_index_map.get(_ts_key(d_ts))
            if d_idx is None:
                continue
            if d_idx - sweep_idx > lookback_bars:
                break
        if _direction_key(d.get("direction", "")) != want:
            continue
        if d.get("classification", "normal") not in min_classes:
            continue
        return d
    return None


def _structure_bias_before(
    structure_rows: list[dict[str, Any]],
    ts,
    *,
    lookback_points: int = 6,
) -> str:
    """Bounded causal structure bias from the last N structure events
    strictly at-or-before ``ts``. Future structure events are excluded, so a
    historical candidate's bias is immutable (causal regression test).
    """
    key = _ts(ts)
    prior = sorted(
        [r for r in structure_rows if _ts(r["timestamp"]) <= key],
        key=lambda r: _ts(r["timestamp"]),
    )[-lookback_points:]
    bullish = sum(
        1 for r in prior
        if r.get("structure_type", "") in ("higher_high", "higher_low")
    )
    bearish = sum(
        1 for r in prior
        if r.get("structure_type", "") in ("lower_high", "lower_low")
    )
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _htf_at(mtf_rows: list[dict[str, Any]], ts) -> tuple[str, str, str]:
    """Return (alignment, trend, volatility) from newest HTF row <= ts."""
    if not mtf_rows:
        return "unknown", "", ""
    latest = _latest_before(mtf_rows, ts)
    if latest is None:
        return "unknown", "", ""
    return (
        latest.get("htf_trend_state", "unknown") or "unknown",
        latest.get("htf_trend_state", "") or "",
        latest.get("htf_volatility_state", "") or "",
    )