"""Bounded event extraction for Step 13.

For one bounded chunk of base candles this module runs the authoritative
engines ONCE (FeatureEngine, MarketStructureEngine, RegimeEngine) and emits
COMPACT dict rows — never full Pydantic object graphs. All rows carry
explicit ``available_from`` timestamps; only causally available data is used
(for crosses-timeframe lookups, the shared causal index applies).

The extraction is cache-friendly: the runner wraps this computation in the
existing ResearchCache keyed by symbol/timeframe/data-hash/config-hash.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.features import FeatureEngine
from app.market_structure.engine import MarketStructureEngine
from app.market_structure.models import MarketStructureResult
from app.regime import RegimeEngine
from app.regime.models import MarketRegime


def _session_label(ts) -> str:
    """Coarse UTC session label (research only)."""
    h = ts.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "europe"
    if 13 <= h < 21:
        return "newyork"
    return "late"


def _to_utc_dt(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class EventExtractor:
    """Extracts compact event rows from a bounded chunk of base candles."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        feature_names: tuple = ("atr", "rsi"),
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.feature_names = list(feature_names)
        self._feature_engine = FeatureEngine()

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        chunk: pd.DataFrame,
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract all event rows from a bounded chunk.

        Returns a dict keyed by dataset name:
            features, structure_events, liquidity_zones, sweeps,
            displacement, regime
        plus the raw analytical objects for the runner's candidate stage
        (as ``_structure`` and ``_regimes``).
        """
        chunk = chunk.sort_index()
        rows: dict[str, list[dict[str, Any]]] = {
            "features": [],
            "structure_events": [],
            "liquidity_zones": [],
            "sweeps": [],
            "displacement": [],
            "regime": [],
        }

        # Features (vectorized once).
        feats = self._feature_engine.calculate(
            chunk, features=self.feature_names
        )

        # Market structure (authoritative engine, causal by design).
        structure = MarketStructureEngine().analyze(
            chunk, self.symbol, self.timeframe
        )

        # Regime (consumes structure output; causal).
        regimes = RegimeEngine().analyze(
            chunk, self.symbol, self.timeframe, market_structure=structure
        )

        # ── Feature rows ─────────────────────────────────────────────────────
        for ts, row in chunk.iterrows():
            frow = {
                "timestamp": _to_utc_dt(ts),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "session": _session_label(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "atr": float(feats.loc[ts, "atr"]) if "atr" in feats.columns else float("nan"),
                "rsi": float(feats.loc[ts, "rsi"]) if "rsi" in feats.columns else float("nan"),
                "return_1": float(chunk["close"].pct_change().loc[ts])
                if ts != chunk.index[0]
                else float("nan"),
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
            rows["features"].append(frow)

        # ── Structure event rows ──────────────────────────────────────────────
        for i, p in enumerate(structure.structure):
            rows["structure_events"].append(
                {
                    "timestamp": _to_utc_dt(p.timestamp),
                    "symbol": p.symbol,
                    "timeframe": p.timeframe,
                    "structure_type": p.structure_type.value,
                    "price": float(p.price),
                    "prior_price": float(p.prior_price),
                    "available_from": _to_utc_dt(p.available_from),
                }
            )

        # ── Liquidity zone rows ───────────────────────────────────────────────
        for zi, z in enumerate(structure.liquidity_zones):
            rows["liquidity_zones"].append(
                {
                    "zone_id": f"{self.symbol}_{self.timeframe}_{zi}",
                    "symbol": z.symbol,
                    "timeframe": z.timeframe,
                    "zone_type": z.zone_type,
                    "upper": float(z.upper),
                    "lower": float(z.lower),
                    "mid": float(z.mid),
                    "swing_count": int(z.swing_count),
                    "first_timestamp": _to_utc_dt(z.first_timestamp),
                    "last_timestamp": _to_utc_dt(z.last_timestamp),
                    "available_from": _to_utc_dt(z.available_from),
                }
            )

        # ── Sweep rows ────────────────────────────────────────────────────────
        for s in structure.sweeps:
            penetration = float(s.extreme_price - s.level) if s.sweep_type.value == "high_sweep" else float(s.level - s.extreme_price)
            excursion = abs(float(s.close_price - s.level))
            zone_id = ""
            for z in structure.liquidity_zones:
                if z.zone_type == ("equal_highs" if s.sweep_type.value == "high_sweep" else "equal_lows"):
                    if abs(z.upper - s.level) < 1e-9 or abs(z.lower - s.level) < 1e-9:
                        zone_id = f"{self.symbol}_{self.timeframe}_{structure.liquidity_zones.index(z)}"
                        break
            rows["sweeps"].append(
                {
                    "timestamp": _to_utc_dt(s.timestamp),
                    "symbol": s.symbol,
                    "timeframe": s.timeframe,
                    "direction": "short" if s.sweep_type.value == "high_sweep" else "long",
                    "sweep_type": s.sweep_type.value,
                    "level": float(s.level),
                    "extreme_price": float(s.extreme_price),
                    "close_price": float(s.close_price),
                    "zone_id": zone_id,
                    "penetration": float(penetration),
                    "excursion": float(excursion),
                    "session": _session_label(s.timestamp),
                    "regime": _regime_at(regimes, s.timestamp),
                    "htf_bias": "",
                    "available_from": _to_utc_dt(s.available_from),
                }
            )

        # ── Displacement rows ─────────────────────────────────────────────────
        for d in structure.displacement:
            rows["displacement"].append(
                {
                    "timestamp": _to_utc_dt(d.timestamp),
                    "symbol": d.symbol,
                    "timeframe": d.timeframe,
                    "direction": d.direction,
                    "range_ratio": float(d.range_ratio) if not np.isnan(d.range_ratio) else None,
                    "body_ratio": float(d.body_ratio),
                    "classification": d.classification.value,
                    "available_from": _to_utc_dt(d.available_from),
                }
            )

        # ── Regime rows ───────────────────────────────────────────────────────
        for r in regimes:
            rows["regime"].append(
                {
                    "timestamp": _to_utc_dt(r.timestamp),
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "trend_state": r.trend_state.value,
                    "volatility_state": r.volatility_state.value,
                    "market_state": r.market_state.value,
                    "strength": float(r.strength),
                    "available_from": _to_utc_dt(r.available_from),
                }
            )

        # Attach analytical objects for the candidate stage (same chunk).
        rows["_structure"] = [structure]
        rows["_regimes"] = [regimes]
        return rows


def _regime_at(regimes: list[MarketRegime], ts) -> str:
    """Latest causal regime market_state at ``ts``."""
    latest = None
    for r in regimes:
        if r.available_from <= ts:
            latest = r
    return latest.market_state.value if latest else "unknown"


def extract_rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Convert compact row dicts to a normalized DataFrame with stable columns."""
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist (fill missing with None).
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]