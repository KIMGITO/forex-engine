"""Candidate generation for Step 13.

A candidate is a potentially meaningful trading setup identified by a
STRICTLY CAUSAL join of market events:

    liquidity sweep
    + displacement after sweep (lookback)
    + market structure context
    + HTF alignment (causal)
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
    CANDIDATE_LABEL_COLUMNS,
    CANDIDATE_LABELS_COLUMNS,
)


def _candidate_id(
    symbol: str, timeframe: str, ts, sweep_ref: str, displacement_ref: str
) -> str:
    """Deterministic candidate identifier."""
    raw = f"{symbol}|{timeframe}|{ts.isoformat()}|{sweep_ref}|{displacement_ref}"
    return f"cand_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


class CandidateGenerator:
    """Joins sweep/displacement/structure/HTF/regime events causally."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        sweep_displacement_lookback: int = 5,
        min_displacement_class: tuple = ("large", "extreme"),
        require_htf_alignment: bool = True,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.sweep_lookback = sweep_displacement_lookback
        self.min_displacement_classes = set(min_displacement_class)
        self.require_htf_alignment = require_htf_alignment

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        sweeps: list[dict[str, Any]],
        displacements: list[dict[str, Any]],
        regimes: list[dict[str, Any]],
        features: list[dict[str, Any]],
        mtf_rows: list[dict[str, Any]] | None = None,
        structure_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate candidate rows (feature_* only, strictly causal).

        Parameters
        ----------
        sweeps : list of sweep row dicts (each with timestamp/direction)
        displacements : list of displacement row dicts (timestamp/direction/classification)
        regimes : list of regime row dicts (timestamp + states)
        features : list of feature row dicts (timestamp + atr/rsi)
        mtf_rows : optional list of MTF context rows (timestamp + htf_*)
        structure_rows : optional list of structure event rows (timestamp + structure_type)

        Returns candidate event rows (feature namespace only).
        """
        if not sweeps:
            return []

        # Build causal indices (sorted by available_from/timestamp).
        sweep_by_ts = {_ts_key(s["timestamp"]): i for i, s in enumerate(sweeps)}
        disp_sorted = sorted(displacements, key=lambda d: _ts(d["timestamp"]))
        disp_by_ts = {_ts_key(d["timestamp"]): i for i, d in enumerate(disp_sorted)}
        regime_sorted = sorted(regimes, key=lambda r: _ts(r["timestamp"]))
        feature_sorted = sorted(features, key=lambda f: _ts(f["timestamp"]))
        mtf_sorted = sorted(mtf_rows, key=lambda m: _ts(m["timestamp"])) if mtf_rows else []
        struct_sorted = sorted(structure_rows, key=lambda s: _ts(s["timestamp"])) if structure_rows else []

        candidates: list[dict[str, Any]] = []
        for sweep in sweeps:
            sweep_ts = _ts(sweep["timestamp"])
            sweep_key = _ts_key(sweep["timestamp"])
            direction = sweep.get("direction", "")

            # 1. Find confirming displacement within lookback after sweep.
            disp = _find_displacement(
                disp_sorted, sweep_ts, direction, self.sweep_lookback,
                self.min_displacement_classes,
            )
            if disp is None:
                continue
            disp_ts = _ts(disp["timestamp"])

            # 2. Causal regime at candidate timestamp.
            regime = _latest_before(regime_sorted, sweep_ts)
            if regime is None:
                continue

            # 3. Causal features at candidate timestamp.
            feats = _latest_before(feature_sorted, sweep_ts)
            atr = feats.get("atr", float("nan")) if feats else float("nan")
            rsi = feats.get("rsi", float("nan")) if feats else float("nan")

            # 4. Causal structure bias (bulk of structure rows before sweep).
            bias = _structure_bias_before(struct_sorted, sweep_ts)

            # 5. HTF alignment (causal).
            htf_alignment, htf_trend, htf_vol = _htf_at(mtf_sorted, sweep_ts)
            if self.require_htf_alignment and mtf_sorted:
                # Require at least one aligning HTF tier.
                if direction == "long" and htf_trend != "bullish":
                    continue
                if direction == "short" and htf_trend != "bearish":
                    continue

            # 6. Candidate fields.
            ref_disp = disp.get("available_from") or disp.get("timestamp")
            ref_sweep = sweep.get("available_from") or sweep.get("timestamp")
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
                    "entry_ref": ref_disp,
                    "sweep_ref": ref_sweep,
                    "displacement_ref": ref_disp,
                    "structure_ref": bias,
                    "htf_ref": htf_trend,
                    "regime": regime.get("market_state", "unknown"),
                    "session": session,
                    "available_from": ref_disp,
                    # Feature namespace (causal only).
                    "feature_atr": float(atr) if not _is_nan(atr) else None,
                    "feature_rsi": float(rsi) if not _is_nan(rsi) else None,
                    "feature_volatility": regime.get("volatility_state", "unknown"),
                    "feature_structure_bias": bias,
                    "feature_sweep_penetration": float(sweep.get("penetration", 0.0) or 0.0),
                    "feature_sweep_excursion": float(sweep.get("excursion", 0.0) or 0.0),
                    "feature_displacement_ratio": float(disp.get("range_ratio") or 0.0) if disp.get("range_ratio") is not None else None,
                    "feature_htf_alignment": htf_alignment,
                    "feature_htf_trend": htf_trend,
                    "feature_htf_volatility": htf_vol,
                    "feature_session": session,
                }
            )
        return candidates


def candidates_to_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert candidate rows to a stable feature-only DataFrame."""
    if not candidates:
        return pd.DataFrame(columns=CANDIDATE_EVENTS_COLUMNS)
    df = pd.DataFrame(candidates)
    for c in CANDIDATE_EVENTS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_EVENTS_COLUMNS]


def labels_to_frame(labels: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert label rows to a stable label-only DataFrame."""
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
    """Last row with row.timestamp <= ts (causal prefix semantics)."""
    key = _ts(ts)
    latest = None
    latest_ts = None
    for r in rows:
        r_ts = _ts(r["timestamp"])
        if r_ts <= key and (latest_ts is None or r_ts > latest_ts):
            latest = r
            latest_ts = r_ts
    return latest


def _find_displacement(
    displacements: list[dict[str, Any]],
    sweep_ts,
    direction: str,
    lookback_bars: int,
    min_classes: set[str],
):
    """Find first displacement AFTER sweep with matching direction/class."""
    sweep_ts = _ts(sweep_ts)
    count = 0
    for d in displacements:
        d_ts = _ts(d["timestamp"])
        if d_ts <= sweep_ts:
            continue
        count += 1
        if count > lookback_bars:
            break
        if d.get("direction", "") != direction:
            continue
        if d.get("classification", "normal") not in min_classes:
            continue
        return d
    return None


def _structure_bias_before(structure_rows: list[dict[str, Any]], ts) -> str:
    """Simple bullish/bearish/neutral bias from structure events <= ts."""
    key = _ts(ts)
    bullish = 0
    bearish = 0
    for r in structure_rows:
        r_ts = _ts(r["timestamp"])
        if r_ts > key:
            continue
        t = r.get("structure_type", "")
        if t in ("higher_high", "higher_low"):
            bullish += 1
        elif t in ("lower_high", "lower_low"):
            bearish += 1
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