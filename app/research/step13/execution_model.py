"""Hypothesis-aware execution model for Step 13.

Simulates a candidate through the hypothesis's actual entry, stop, and
exit rules against future OHLC bars, applying the configured trading-cost
model. The resulting R corresponds EXACTLY to the hypothesis recorded in
``research_candidate.json``.

Conservative policy: when a bar touches BOTH stop and target, the stop is
assumed to fill first (mirrors EventBacktester's CONSERVATIVE_SL_FIRST).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13.hypotheses import Hypothesis


def _pip_size(symbol: str) -> float:
    quote = symbol.upper().replace("/", "").replace("_", "")[3:]
    return 0.01 if quote in ("JPY", "CHF") else 0.0001


def _index_at(candles: pd.DataFrame, ts) -> int | None:
    ts = pd.Timestamp(ts)
    if ts in candles.index:
        return candles.index.get_loc(ts)
    prior = candles.index[candles.index <= ts]
    if len(prior) == 0:
        return None
    return candles.index.get_loc(prior[-1])


def _entry_price(
    entry_rule: str,
    event_close: float,
    displacement_close: float | None,
) -> float:
    if entry_rule in ("displacement_confirmation", "retest") and displacement_close is not None:
        return displacement_close
    return event_close


def _stop(
    stop_rule: str,
    entry: float,
    direction: float,
    atr: float,
    stop_atr_multiple: float,
    event_level: float | None,
) -> tuple[float, float]:
    if stop_rule == "liquidity" and event_level is not None:
        stop = event_level
    else:
        stop = entry - direction * stop_atr_multiple * (atr if atr > 0 else 0.001)
    risk = abs(entry - stop)
    if risk <= 0:
        risk = abs(entry) * 0.005
        stop = entry - direction * risk
    return stop, risk


def _target(
    exit_rule: str,
    entry: float,
    direction: float,
    risk: float,
    atr: float,
    exit_atr_multiple: float,
) -> float:
    if exit_rule.startswith("fixed_rr_"):
        n = float(exit_rule.split("_")[-1])
        return entry + direction * n * risk
    if exit_rule == "atr" and atr > 0:
        return entry + direction * exit_atr_multiple * atr
    return entry + direction * 2.0 * risk


def simulate_hypothesis_outcome(
    hypothesis: Hypothesis,
    candidate: dict[str, Any],
    candles: pd.DataFrame,
    *,
    lookback_bars: int = 100,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    commission_per_lot: float = 0.0,
) -> dict[str, Any] | None:
    """Simulate the candidate through the hypothesis rules; return outcome.

    Returns None when the candidate cannot be simulated (no future bars).
    The outcome contains entry/stop/target/exit prices, risk distance,
    exit reason, holding bars, and R AFTER costs — consistent with the
    hypothesis recorded in the artifact.
    """
    if candles is None or candles.empty:
        return None

    candles = candles.sort_index()
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)

    pos = _index_at(candles, candidate.get("timestamp"))
    if pos is None or pos + 1 >= len(candles):
        return None

    direction = 1.0 if candidate.get("direction") == "long" else -1.0
    atr = float(candidate.get("feature_atr", 0.01) or 0.01)
    event_close = float(candidate.get("entry_ref") or closes[pos])

    displacement_close = _find_displacement_close(
        closes, pos, candidate, lookback_bars, direction
    )
    entry = _entry_price(hypothesis.entry_rule, event_close, displacement_close)

    stop, risk = _stop(
        hypothesis.stop_rule, entry, direction, atr,
        hypothesis.stop_atr_multiple, candidate.get("level"),
    )
    target = _target(
        hypothesis.exit_rule, entry, direction, risk, atr,
        hypothesis.exit_atr_multiple,
    )

    fut_high = highs[pos + 1 : pos + 1 + lookback_bars]
    fut_low = lows[pos + 1 : pos + 1 + lookback_bars]
    fut_close = closes[pos + 1 : pos + 1 + lookback_bars]
    if len(fut_high) == 0:
        return None

    max_bars = (
        hypothesis.max_holding_bars
        if hypothesis.max_holding_bars > 0
        else lookback_bars
    )
    exit_price, holding, reason = _simulate(
        fut_high, fut_low, fut_close, entry, stop, target, direction, max_bars
    )

    pip = _pip_size(candidate.get("symbol", "EURUSD"))
    lot_units = 100_000.0
    commission_per_unit = commission_per_lot / lot_units
    cost_in_price = pip * (spread_pips + slippage_pips) + commission_per_unit
    r_after_cost = direction * (exit_price - entry) / risk - (cost_in_price / risk)

    return {
        "candidate_id": candidate.get("candidate_id", "unknown"),
        "entry_price": round(entry, 6),
        "stop_price": round(stop, 6),
        "target_price": round(target, 6),
        "risk_distance": round(risk, 6),
        "exit_price": round(exit_price, 6),
        "r": round(r_after_cost, 4),
        "exit_reason": reason,
        "holding_bars": int(holding),
    }


def _simulate(
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
        sl = lows[i] <= stop if direction > 0 else highs[i] >= stop
        tp = highs[i] >= target if direction > 0 else lows[i] <= target
        if sl and tp:
            return stop, i + 1, "conser_sl_first"
        if sl:
            return stop, i + 1, "stop_loss"
        if tp:
            return target, i + 1, "take_profit"
        if i + 1 >= max_bars:
            break
    return float(closes[-1]), min(len(highs), max_bars), "time_exit"


def _find_displacement_close(
    closes: np.ndarray,
    pos: int,
    candidate: dict[str, Any],
    lookback: int,
    direction: float,
) -> float | None:
    ref = candidate.get("displacement_ref")
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