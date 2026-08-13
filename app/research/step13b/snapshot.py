"""Compact signal snapshot extraction for Step 13B.

The Step 13B research engine records ONLY the information required for
research — never the full Pydantic object graph of every signal. Each signal
is flattened into a single dict row (then stored as parquet via atomic writes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from app.market_structure.models import MarketStructureResult
from app.regime.models import MarketRegime
from app.strategy.models import Signal, SignalDirection


def _session_label(ts: datetime) -> str:
    """Map a UTC hour to a coarse session label (research only)."""
    h = ts.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "europe"
    if 13 <= h < 21:
        return "newyork"
    return "late"


def _structure_bias(structure: MarketStructureResult | None, ts: datetime) -> str:
    """Determine a coarse structural bias at ``ts`` (causal: available <= ts)."""
    if structure is None or not structure.structure:
        return "neutral"
    # Count bullish vs bearish structure points causally available at ts.
    bullish = 0
    bearish = 0
    for p in structure.structure:
        if p.available_from is not None and p.available_from > ts:
            continue
        if p.structure_type.value in ("higher_high", "higher_low"):
            bullish += 1
        elif p.structure_type.value in ("lower_high", "lower_low"):
            bearish += 1
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _liquidity_state(structure: MarketStructureResult | None, ts: datetime) -> str:
    """Coarse liquidity state at ``ts`` (causal)."""
    if structure is None or not structure.liquidity_zones:
        return "none"
    # Count zones available at or before ts.
    count = 0
    for z in structure.liquidity_zones:
        if z.available_from is not None and z.available_from <= ts:
            count += 1
    if count >= 3:
        return "abundant"
    if count >= 1:
        return "present"
    return "none"


def _sweep_detected(structure: MarketStructureResult | None, ts: datetime) -> bool:
    if structure is None or not structure.sweeps:
        return False
    for s in structure.sweeps:
        if s.available_from is not None and s.available_from <= ts:
            return True
    return False


def _break_of_structure(structure: MarketStructureResult | None, ts: datetime) -> bool:
    if structure is None or not structure.breaks:
        return False
    for b in structure.breaks:
        if b.available_from is not None and b.available_from <= ts:
            return True
    return False


def _displacement_label(structure: MarketStructureResult | None, ts: datetime) -> str:
    if structure is None or not structure.displacement:
        return "none"
    for d in reversed(structure.displacement):
        if d.available_from is not None and d.available_from <= ts:
            return f"{d.direction}_{d.classification.value}"
    return "none"


def _regime_snapshot(regimes: list[MarketRegime] | None, ts: datetime) -> dict[str, str]:
    """Extract the latest causal regime snapshot at ``ts``."""
    if not regimes:
        return {
            "trend": "unknown",
            "regime": "unknown",
            "volatility": "unknown",
        }
    latest = None
    for r in regimes:
        if r.available_from is not None and r.available_from <= ts:
            latest = r
    if latest is None:
        return {
            "trend": "unknown",
            "regime": "unknown",
            "volatility": "unknown",
        }
    return {
        "trend": latest.trend_state.value,
        "regime": latest.market_state.value,
        "volatility": latest.volatility_state.value,
    }


def _mtf_bias(signal: Signal) -> str | None:
    """Extract MTF bias from signal metadata if available."""
    md = signal.metadata or {}
    mtf = md.get("mtf") if isinstance(md, dict) else None
    if isinstance(mtf, dict):
        return mtf.get("alignment")
    return None


def signal_to_snapshot(
    signal: Signal,
    *,
    structure: MarketStructureResult | None = None,
    regimes: list[MarketRegime] | None = None,
    param_set: str = "baseline",
) -> dict[str, Any]:
    """Convert a Signal to a compact research snapshot dict.

    The dict contains ONLY research-relevant fields. No giant object graph.
    """
    ts = signal.timestamp
    regime_info = _regime_snapshot(regimes, ts)
    return {
        "timestamp": ts.isoformat(),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "trend": regime_info["trend"],
        "regime": regime_info["regime"],
        "volatility": regime_info["volatility"],
        "structure_bias": _structure_bias(structure, ts),
        "liquidity_state": _liquidity_state(structure, ts),
        "sweep_detected": _sweep_detected(structure, ts),
        "break_of_structure": _break_of_structure(structure, ts),
        "displacement": _displacement_label(structure, ts),
        "mtf_bias": _mtf_bias(signal),
        "session": _session_label(ts),
        "direction": signal.direction.value,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "risk_distance": signal.risk_distance,
        "reward_distance": signal.reward_distance,
        "risk_reward_ratio": signal.risk_reward_ratio,
        "risk_rejected": False,
        "risk_rejection_reason": None,
        "risk_percent": 0.0,
        "position_size": 0.0,
        "result": "pending",
        "r_multiple": 0.0,
        "exit_reason": "",
        "param_set": param_set,
        "score": signal.score,
        "strength": signal.strength.value,
    }


def snapshots_to_frame(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert snapshot dicts to a compact DataFrame."""
    if not snapshots:
        return pd.DataFrame(
            columns=[
                "timestamp", "symbol", "timeframe", "trend", "regime",
                "volatility", "structure_bias", "liquidity_state",
                "sweep_detected", "break_of_structure", "displacement",
                "mtf_bias", "session", "direction", "entry", "stop_loss",
                "take_profit", "risk_distance", "reward_distance",
                "risk_reward_ratio", "risk_rejected", "risk_rejection_reason",
                "risk_percent", "position_size", "result", "r_multiple",
                "exit_reason", "param_set", "score", "strength",
            ]
        )
    df = pd.DataFrame(snapshots)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
    return df