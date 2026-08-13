"""Label computation for Step 13 candidates.

Labels are observed AFTER the candidate timestamp: MFE, MAE, TP/SL hit,
future return, future displacement. Stored in a SEPARATE dataset from
candidate features — the schema validator rejects any mixing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.research.step13.schema import CANDIDATE_LABELS_COLUMNS


def labels_to_frame(labels: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert label rows to a stable label-only DataFrame.

    Labels are stored in a SEPARATE dataset from candidate features to
    enforce structural feature/label separation.
    """
    if not labels:
        return pd.DataFrame(columns=CANDIDATE_LABELS_COLUMNS)
    df = pd.DataFrame(labels)
    for c in CANDIDATE_LABELS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_LABELS_COLUMNS]


def compute_labels(
    candidates: list[dict[str, Any]],
    candles: pd.DataFrame,
    *,
    lookback_bars: int = 100,
    tp_multiple: float = 2.0,
) -> list[dict[str, Any]]:
    """Compute label_* rows for each candidate.

    Parameters
    ----------
    candidates : list of candidate event rows (candidate_id, timestamp, direction, entry_ref)
    candles : base OHLC DataFrame indexed by tz-aware timestamps
    lookback_bars : max bars after candidate to inspect for MFE/MAE/TP/SL
    tp_multiple : take-profit distance as a multiple of the sweep-displacement reference.

    Returns a list of label row dicts (label_* namespace only).
    """
    if not candidates or candles is None or candles.empty:
        return []

    candles = candles.sort_index()
    closes = candles["close"].to_numpy(dtype=float)
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    timestamps = candles.index.to_numpy()

    labels: list[dict[str, Any]] = []
    for cand in candidates:
        ts = pd.Timestamp(cand["timestamp"])
        direction = cand.get("direction", "")
        entry = float(cand.get("entry_ref") or 0.0)
        if entry <= 0:
            # No numerical entry; use close at candidate timestamp.
            idx = _index_at(candles, ts)
            if idx is None:
                continue
            entry = closes[idx]

        idx = _index_at(candles, ts)
        if idx is None or idx + 1 >= len(candles):
            continue

        end = min(idx + lookback_bars + 1, len(candles))
        window_high = highs[idx + 1 : end]
        window_low = lows[idx + 1 : end]
        window_close = closes[idx + 1 : end]

        if len(window_high) == 0:
            continue

        if direction == "long":
            mfe = float(window_high.max() - entry)
            mae = float(entry - window_low.min())
            tp_reached = bool((window_high >= entry + tp_multiple * max(entry * 0.005, mfe * 0.1)).any())
            sl_hit = bool((window_low <= entry - max(entry * 0.005, mae * 0.1)).any())
            future_return = float(window_close[-1] / entry - 1.0) if entry > 0 else 0.0
        elif direction == "short":
            mfe = float(entry - window_low.min())
            mae = float(window_high.max() - entry)
            tp_reached = bool((window_low <= entry - tp_multiple * max(entry * 0.005, mfe * 0.1)).any())
            sl_hit = bool((window_high >= entry + max(entry * 0.005, mae * 0.1)).any())
            future_return = float(entry / window_close[-1] - 1.0) if entry > 0 else 0.0
        else:
            mfe = 0.0
            mae = 0.0
            tp_reached = False
            sl_hit = False
            future_return = 0.0

        # Future displacement (within lookback): max |range| / ATR approximate.
        future_disp = float(
            (highs[idx + 1 : end] - lows[idx + 1 : end]).max()
        ) if end > idx + 1 else 0.0

        labels.append(
            {
                "candidate_id": cand["candidate_id"],
                "timestamp": cand["timestamp"],
                "label_mfe": mfe,
                "label_mae": mae,
                "label_tp_hit": tp_reached,
                "label_sl_hit": sl_hit,
                "label_excursion_after_bars": int(len(window_high)),
                "label_future_return": future_return,
                "label_displacement_after": future_disp,
            }
        )
    return labels


def _index_at(candles: pd.DataFrame, ts) -> int | None:
    """Index of exact timestamp in candles; None when absent."""
    if ts in candles.index:
        return candles.index.get_loc(ts)
    # Fall back to nearest prior bar (strictly <=).
    prior = candles.index[candles.index <= ts]
    if len(prior) == 0:
        return None
    return candles.index.get_loc(prior[-1])