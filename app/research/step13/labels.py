"""Hypothesis-aware label computation for Step 13.

Labels are computed by ACTUALLY SIMULATING the candidate through the
hypothesis's entry/stop/exit rules against future OHLC bars. The resulting
R corresponds exactly to the hypothesis recorded in research_candidate.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13.hypotheses import Hypothesis
from app.research.step13.schema import CANDIDATE_LABELS_COLUMNS


def labels_to_frame(labels: list[dict[str, Any]]) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(columns=CANDIDATE_LABELS_COLUMNS)
    df = pd.DataFrame(labels)
    for c in CANDIDATE_LABELS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_LABELS_COLUMNS]


def _pip_size(symbol: str) -> float:
    quote = symbol.upper().replace("/", "").replace("_", "")[3:]
    return 0.01 if quote in ("JPY", "CHF") else 0.0001


def _entry_price(
    entry_rule: str,
    event_close: float,
    displacement_close: float | None,
) -> tuple[float, str]:
    if entry_rule == "displacement_confirmation" and displacement_close is not None:
        return displacement_close, "displacement_confirmation"
    if entry_rule == "retest" and displacement_close is not None:
        return displacement_close, "retest"
    return event_close, "immediate"


def _stop_price(
    stop_rule: str,
    entry: float,
    direction: float,
    atr: float,
    stop_atr_multiple: float,
    event_level: float | None,
) -> tuple[float, float]:
    if stop_rule == "atr" and atr > 0:
        stop = entry - direction * stop_atr_multiple * atr
    elif stop_rule == "liquidity" and event_level is not None:
        stop = event_level
    else:
        stop = entry - direction * stop_atr_multiple * (atr if atr > 0 else 0.001)
    risk_distance = abs(entry - stop)
    if risk_distance <= 0:
        risk_distance = abs(entry) * 0.005
        stop = entry - direction * risk_distance
    return stop, risk_distance


def _target_price(
    exit_rule: str,
    entry: float,
    direction: float,
    risk_distance: float,
    atr: float,
    exit_atr_multiple: float,
) -> float:
    if exit_rule.startswith("fixed_rr_"):
        n = float(exit_rule.split("_")[-1])
        return entry + direction * n * risk_distance
    if exit_rule == "atr" and atr > 0:
        return entry + direction * exit_atr_multiple * atr
    return entry + direction * 2.0 * risk_distance


def _simulate_ohlc_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry: float,
    stop: float,
    target: float,
    direction: float,
    max_bars: int,
) -> tuple[float, int, str]:
    for i in range(len(highs)):
        bar_high = float(highs[i])
        bar_low = float(lows[i])
        sl_hit = bar_low <= stop if direction > 0 else bar_high >= stop
        tp_hit = bar_high >= target if direction > 0 else bar_low <= target
        if sl_hit and tp_hit:
            return stop, i + 1, "conser_sl_first"
        if sl_hit:
            return stop, i + 1, "stop_loss"
        if tp_hit:
            return target, i + 1, "take_profit"
        if i + 1 >= max_bars:
            break
    return float(closes[-1]), min(len(highs), max_bars), "time_exit"


def compute_labels(
    candidates: list[dict[str, Any]],
    candles: pd.DataFrame,
    hypothesis: Hypothesis | None = None,
    *,
    lookback_bars: int = 100,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    commission_per_unit: float = 0.0,
) -> list[dict[str, Any]]:
    if not candidates or candles is None or candles.empty:
        return []

    candles = candles.sort_index()
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)

    out: list[dict[str, Any]] = []
    for cand in candidates:
        ts = pd.Timestamp(cand["timestamp"])
        direction = 1.0 if cand.get("direction") == "long" else -1.0
        atr = float(cand.get("feature_atr", 0.01) or 0.01)
        event_close = float(cand.get("entry_ref") or 0.0)
        pos = _index_at(candles, ts)
        if pos is None or pos + 1 >= len(candles):
            continue
        if event_close <= 0:
            event_close = closes[pos]

        # Hypothesis-aware entry.
        entry_rule = hypothesis.entry_rule if hypothesis else "immediate"
        displacement_close = _find_displacement_close(
            closes, pos, cand, lookback_bars, direction
        )
        entry, _ = _entry_price(entry_rule, event_close, displacement_close)

        # Hypothesis-aware stop/target.
        stop_rule = hypothesis.stop_rule if hypothesis else "atr"
        exit_rule = hypothesis.exit_rule if hypothesis else "fixed_rr_2.0"
        stop, risk_distance = _stop_price(
            stop_rule, entry, direction, atr,
            hypothesis.stop_atr_multiple if hypothesis else 1.0,
            cand.get("level"),
        )
        target = _target_price(
            exit_rule, entry, direction, risk_distance, atr,
            hypothesis.exit_atr_multiple if hypothesis else 2.0,
        )

        future_highs = highs[pos + 1 : pos + 1 + lookback_bars]
        future_lows = lows[pos + 1 : pos + 1 + lookback_bars]
        future_closes = closes[pos + 1 : pos + 1 + lookback_bars]
        if len(future_highs) == 0:
            continue

        exit_price, holding, exit_reason = _simulate_ohlc_outcome(
            future_highs, future_lows, future_closes,
            entry, stop, target, direction, lookback_bars,
        )

        # Costs.
        pip = _pip_size(cand.get("symbol", "EURUSD"))
        cost_in_price = pip * (spread_pips + slippage_pips) + commission_per_unit
        r_after_cost = direction * (exit_price - entry) / risk_distance - (
            cost_in_price / risk_distance
        )

        mfe = float(
            (future_highs - entry).max() if direction > 0
            else (entry - future_lows).max()
        )
        mae = float(
            (entry - future_lows).min() if direction > 0
            else (entry - future_highs).min()
        )

        out.append(
            {
                "candidate_id": cand.get("candidate_id", "unknown"),
                "timestamp": cand.get("timestamp"),
                "label_entry_price": round(entry, 6),
                "label_stop_price": round(stop, 6),
                "label_target_price": round(target, 6),
                "label_risk_distance": round(risk_distance, 6),
                "label_exit_price": round(exit_price, 6),
                "label_r": round(r_after_cost, 4),
                "label_mfe": round(mfe, 6),
                "label_mae": round(mae, 6),
                "label_tp_hit": bool(exit_reason == "take_profit"),
                "label_sl_hit": bool(exit_reason == "stop_loss")
                or bool(exit_reason == "conser_sl_first"),
                "label_exit_reason": exit_reason,
                "label_holding_bars": int(holding),
                "label_entry_reason": (
                    "displacement_confirmation"
                    if displacement_close is not None
                    else "immediate"
                ),
            }
        )
    return out


def _find_displacement_close(
    closes: np.ndarray,
    pos: int,
    cand: dict[str, Any],
    lookback: int,
    direction: float,
) -> float | None:
    ref = cand.get("displacement_ref")
    if ref:
        try:
            return float(ref)
        except (TypeError, ValueError):
            pass
    end = min(pos + 1 + lookback, len(closes))
    for i in range(pos + 1, end):
        move = closes[i] - closes[pos]
        if direction > 0 and move > 0.0:
            return closes[i]
        if direction < 0 and move < 0.0:
            return closes[i]
    return None


def _index_at(candles: pd.DataFrame, ts) -> int | None:
    if ts in candles.index:
        return candles.index.get_loc(ts)
    prior = candles.index[candles.index <= ts]
    if len(prior) == 0:
        return None
    return candles.index.get_loc(prior[-1])