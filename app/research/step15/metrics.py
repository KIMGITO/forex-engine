"""Out-of-sample metric computation for Step 15.

For every frozen hypothesis on every test fold the following are computed:
trades, win rate, loss rate, average R, median R, total R, profit factor,
expectancy, max drawdown (R), Sharpe (only when statistically appropriate),
MFE, MAE, average holding bars, TP/SL/time/ambiguous exit counts.

Gross R excludes costs; net R includes the configured cost model. Both are
reported; low sample sizes are never hidden.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.research.step15.models import OosMetrics


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _max_drawdown_r(r_values: list[float]) -> float:
    """Max drawdown of the cumulative R curve (peak to trough)."""
    if not r_values:
        return 0.0
    peak = 0.0
    max_dd = 0.0
    cum = 0.0
    for r in r_values:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def _sharpe(r_values: list[float]) -> float | None:
    """Annualized Sharpe-like ratio on per-trade R.

    Only reported when there are >= 30 trades (statistically more meaningful).
    Uses per-trade R, not per-bar returns; annotated as a research heuristic.
    """
    if len(r_values) < 30:
        return None
    arr = np.asarray(r_values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return None
    return round(mean / std * math.sqrt(len(arr)), 4)


def compute_oos_metrics(
    r_values: list[float],
    holding_bars: list[int] | None = None,
    exit_reasons: list[str] | None = None,
    mfe_values: list[float] | None = None,
    mae_values: list[float] | None = None,
    gross_r_values: list[float] | None = None,
    *,
    min_trades_for_sharpe: int = 30,
) -> OosMetrics:
    """Compute comprehensive OOS metrics.

    Parameters
    ----------
    r_values : per-trade NET R values (after costs).
    holding_bars : per-trade holding bars.
    exit_reasons : per-trade exit reason strings
        (take_profit | stop_loss | time_exit | conser_sl_first | ambiguous).
    mfe_values / mae_values : per-trade MFE/MAE excursions.
    gross_r_values : per-trade GROSS R values (before costs) when available;
        defaults to ``r_values`` when absent.
    """
    r = _finite(r_values)
    n = len(r)
    if n == 0:
        return OosMetrics(trades=0)

    wins = [v for v in r if v > 0]
    losses = [v for v in r if v < 0]
    breakeven = [v for v in r if abs(v) < 1e-12]

    gross = _finite(gross_r_values) if gross_r_values is not None else list(r)
    gross_total = sum(gross)

    total = sum(r)
    avg_r = total / n if n else None
    med_r = _median(r)
    win_rate = len(wins) / n
    loss_rate = len(losses) / n

    gross_win = sum(v for v in gross if v > 0)
    gross_loss = abs(sum(v for v in gross if v < 0))
    profit_factor = (
        round(gross_win / gross_loss, 4) if gross_loss > 1e-12 else None
    )

    bars = _finite(holding_bars) if holding_bars is not None else []
    reasons = list(exit_reasons or [])
    mfe = _finite(mfe_values) if mfe_values is not None else []
    mae = _finite(mae_values) if mae_values is not None else []

    return OosMetrics(
        trades=n,
        win_rate=round(win_rate, 4) if n else None,
        loss_rate=round(loss_rate, 4) if n else None,
        average_r=round(avg_r, 4) if avg_r is not None else None,
        median_r=round(med_r, 4) if med_r is not None else None,
        total_r=round(total, 4),
        profit_factor=profit_factor,
        expectancy=round(total / n, 4) if n else None,
        max_drawdown_r=round(_max_drawdown_r(r), 4),
        sharpe=_sharpe(r) if n >= min_trades_for_sharpe else None,
        mfe=round(sum(mfe) / len(mfe), 4) if mfe else 0.0,
        mae=round(sum(mae) / len(mae), 4) if mae else 0.0,
        average_holding_bars=round(sum(bars) / len(bars), 2) if bars else None,
        tp_count=sum(1 for x in reasons if x == "take_profit"),
        sl_count=sum(
            1 for x in reasons
            if x in ("stop_loss", "conser_sl_first")
        ),
        time_exit_count=sum(1 for x in reasons if x == "time_exit"),
        ambiguous_exit_count=sum(
            1 for x in reasons if x not in (
                "take_profit", "stop_loss", "conser_sl_first", "time_exit"
            )
        ),
        gross_r=round(gross_total, 4),
        net_r=round(total, 4),
        r_values=r,
    )


def r_from_labels(labels: pd.DataFrame) -> tuple[list[float], list[int], list[str]]:
    """Extract (net_r, holding_bars, exit_reasons) from a candidate_labels frame."""
    if labels is None or labels.empty:
        return [], [], []
    r = []
    bars = []
    reasons = []
    for _, row in labels.iterrows():
        rv = row.get("label_r")
        if rv is not None:
            r.append(float(rv))
        bars.append(int(row.get("label_holding_bars", 0) or 0))
        reasons.append(str(row.get("label_exit_reason", "unknown")))
    return r, bars, reasons


def gross_r_from_labels(
    labels: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> list[float] | None:
    """Reconstruct GROSS R (before costs) from the label frame.

    The label frame stores ``label_r`` AFTER costs as ``direction * (exit - entry) / risk
    - cost_component``. Gross R before costs is ``direction * (exit - entry) / risk``.

    Direction is taken from ``events`` (candidate_events frame) joined by
    candidate_id; when absent the caller falls back to net-only reporting.
    """
    if labels is None or labels.empty:
        return None
    dir_map: dict[Any, float] = {}
    if events is not None and not events.empty and "direction" in events.columns:
        for _, erow in events.iterrows():
            dir_map[erow.get("candidate_id")] = (
                1.0 if str(erow.get("direction", "")) == "long" else -1.0
            )
    out: list[float] = []
    for _, row in labels.iterrows():
        entry = row.get("label_entry_price")
        stop = row.get("label_stop_price")
        exit_p = row.get("label_exit_price")
        risk = row.get("label_risk_distance")
        direction = dir_map.get(row.get("candidate_id"))
        if direction is None:
            return None
        if entry is None or stop is None or exit_p is None or risk is None:
            return None
        if float(risk) <= 0:
            return None
        out.append(round(
            direction * (float(exit_p) - float(entry)) / float(risk), 4
        ))
    return out


def compute_single_fold_metrics(
    labels: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> OosMetrics:
    """Compute OOS metrics from a candidate_labels frame alone."""
    if labels is None or labels.empty:
        return OosMetrics(trades=0)
    r, bars, reasons = r_from_labels(labels)
    gross = gross_r_from_labels(labels, events)
    mfe = [
        float(v) for v in labels["label_mfe"]
        if v is not None and math.isfinite(float(v))
    ]
    mae = [
        float(v) for v in labels["label_mae"]
        if v is not None and math.isfinite(float(v))
    ]
    return compute_oos_metrics(
        r, bars, reasons, mfe, mae, gross
    )