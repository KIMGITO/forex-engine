"""Stability analysis for Step 15 walk-forward validation.

For every fold reports metrics (sample, expectancy, win rate, total R,
drawdown, before/after costs) and answers the critical questions:
* Does performance persist across time?
* Does one fold generate almost all profits?
* Does performance collapse after costs?
* Is the strategy dependent on one regime / session / direction / period?
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def fold_stability_report(folds: list[Any]) -> dict[str, Any]:
    """Build a per-fold stability report from Step15Fold objects.

    Returns dict with per_fold details plus aggregate stability flags.
    """
    rows: list[dict[str, Any]] = []
    for f in folds:
        # Accept either Step15Fold objects or serialized dicts.
        if isinstance(f, dict):
            f_obj = f
            tr = f_obj.get("test_results") or {}
            vr = f_obj.get("validation_results") or {}
            index = f_obj.get("index", 0)
            hypothesis = f_obj.get("selected_hypothesis") or ""
            train_sample = f_obj.get("train_sample", 0)
        else:
            f_obj = f
            tr = f_obj.test_results or {}
            vr = f_obj.validation_results or {}
            index = f_obj.index
            hypothesis = f_obj.selected_hypothesis
            train_sample = f_obj.train_sample
        rows.append(
            {
                "fold": index,
                "hypothesis": hypothesis,
                "train_samples": train_sample,
                "val_trades": int(vr.get("trades", 0)),
                "test_trades": int(tr.get("trades", 0)),
                "expectancy_r": (
                    float(tr["average_r"]) if tr.get("average_r") is not None else None
                ),
                "win_rate": (
                    float(tr["win_rate"]) if tr.get("win_rate") is not None else None
                ),
                "total_r": float(tr.get("total_r", 0.0)),
                "net_r": float(tr.get("net_r", 0.0)),
                "gross_r": float(tr.get("gross_r", 0.0)),
                "max_drawdown_r": float(tr.get("max_drawdown_r", 0.0)),
                "sharpe": tr.get("sharpe"),
            }
        )

    completed = [r for r in rows if r["test_trades"] > 0]
    n = len(completed)
    if n == 0:
        return {
            "per_fold": rows,
            "folds_with_trades": 0,
            "aggregate_flags": {
                "persistence": "INSUFFICIENT_DATA",
                "single_fold_dependence": None,
                "collapses_after_costs": None,
                "regime_dependent": None,
                "session_dependent": None,
                "direction_dependent": None,
            },
        }

    total_net = sum(r["net_r"] for r in completed)
    total_gross = sum(r["gross_r"] for r in completed)
    profitable = sum(1 for r in completed if r["net_r"] > 0)
    noprofit = sum(1 for r in completed if r["net_r"] <= 0)

    # Single-fold dependence: fraction of |total NET R| contributed by the
    # fold with the largest |net R|. Handles negative totals.
    if completed:
        dominant = max(completed, key=lambda r: abs(r["net_r"]))
        denom = abs(total_net) if total_net != 0 else 1e-12
        single_dep = abs(dominant["net_r"]) / denom
        dominant_fold = dominant["fold"]
    else:
        single_dep = 0.0
        dominant_fold = None

    # Persistence: fraction of folds profitable, but the strategy only
    # "persists" when the TOTAL is also positive. A majority of mildly
    # positive folds with a huge losing fold is NOT persistence.
    persistence = profitable / n
    if total_net <= 0:
        persistence_flag = "UNSTABLE (total OOS R is negative)"
    elif persistence >= 0.60 and n >= 3:
        persistence_flag = "PERSISTS"
    elif persistence < 0.50:
        persistence_flag = "UNSTABLE"
    else:
        persistence_flag = "MIXED"

    # Cost collapse: gross positive but net below 50% gross.
    collapse = False
    if total_gross > 0 and total_net < 0.5 * total_gross:
        collapse = True

    return {
        "per_fold": rows,
        "folds_with_trades": n,
        "aggregate_flags": {
            "persistence": persistence_flag,
            "persistent_fraction": round(persistence, 4),
            "single_fold_dependence": round(single_dep, 4),
            "single_fold_dominates": bool(single_dep > 0.60),
            "dominant_fold": dominant_fold,
            "collapses_after_costs": collapse,
            "gross_net_ratio": (
                round(total_net / total_gross, 4) if total_gross != 0 else None
            ),
            "regime_dependent": None,  # filled by regime/session breakdown
            "session_dependent": None,
            "direction_dependent": None,
        },
    }


def direction_breakdown(events: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """Break OOS outcome R by trade direction (long/short)."""
    if labels is None or labels.empty:
        return {}
    out: dict[str, dict[str, float | int]] = {"long": {}, "short": {}}
    dir_map: dict[str, str] = {}
    if events is not None and not events.empty and "direction" in events.columns:
        for _, e in events.iterrows():
            dir_map[str(e.get("candidate_id", ""))] = str(e.get("direction", ""))
    by_dir: dict[str, list[float]] = {"long": [], "short": []}
    for _, row in labels.iterrows():
        rv = row.get("label_r")
        if rv is None:
            continue
        cand_id = str(row.get("candidate_id", ""))
        d = dir_map.get(cand_id, "long")
        if d not in by_dir:
            continue
        by_dir[d].append(float(rv))
    for d, vals in by_dir.items():
        if vals:
            out[d] = {
                "trades": len(vals),
                "net_r": round(sum(vals), 4),
                "mean_r": round(sum(vals) / len(vals), 4),
                "win_rate": round(
                    sum(1 for v in vals if v > 0) / len(vals), 4
                ),
            }
    return out