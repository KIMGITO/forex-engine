"""Baseline comparisons for Step 15.

The purpose of baselines is to determine whether the discovered hypothesis
actually adds predictive value over simple null models. Every baseline is
constructed WITHOUT leaking future outcomes:

A. All candidates      — evaluate the frozen hypothesis on ALL candidates in
                         the fold's test window (no conditions filter).
B. Random candidates   — deterministic random selection using a fixed seed.
C. No HTF filter       — candidates with any HTF alignment (align=unknown/'').
D. HTF-aligned         — candidates with hTF alignment == direction.
E. Opposite/noise      — the hypothesis applied to the OPPOSITE direction
                         (noise baseline; a real edge must beat it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13.evaluator import FastResearchEvaluator
from app.research.step13.hypotheses import Hypothesis
from app.research.step15.metrics import compute_oos_metrics


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def evaluate_baseline(
    hypothesis: Hypothesis,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    name: str,
    filter_mask: np.ndarray | None = None,
    random_sample: int | None = None,
    seed: int = 1234,
    candles: pd.DataFrame | None = None,
    costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a baseline on a candidate partition.

    Parameters
    ----------
    hypothesis : the frozen hypothesis (used for labels/outcome).
    events : candidate events for the fold test window.
    labels : matching candidate labels.
    filter_mask : optional boolean array (same length as events) to select
        a subset. When None, ALL candidates are used.
    random_sample : if set, randomly select this many candidate rows using a
        deterministic seed (B. Random baseline).
    """
    if events is None or events.empty:
        return {"name": name, "trades": 0}

    ev = events
    lab = labels
    if filter_mask is not None:
        ev = ev[filter_mask]
        if lab is not None and not lab.empty and "candidate_id" in ev.columns:
            ids = set(ev["candidate_id"])
            lab = lab[lab["candidate_id"].isin(ids)]

    if ev.empty:
        return {"name": name, "trades": 0}

    if random_sample is not None:
        idx = _rng(seed).choice(len(ev), size=min(random_sample, len(ev)), replace=False)
        ev = ev.iloc[idx]
        if lab is not None and not lab.empty and "candidate_id" in ev.columns:
            ids = set(ev["candidate_id"])
            lab = lab[lab["candidate_id"].isin(ids)]

    evaluator = FastResearchEvaluator()
    result = evaluator.evaluate(
        hypothesis, ev, lab, candles, costs=costs or {}
    )
    if result.sample_count == 0:
        return {"name": name, "trades": 0}

    metrics = compute_oos_metrics(
        r_values=result.r_values,
        holding_bars=result.holding_bars,
    )
    d = metrics.to_dict()
    d["name"] = name
    d["trades"] = d["trades"]
    return d


def compute_baselines(
    hypothesis: Hypothesis,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    candles: pd.DataFrame | None = None,
    costs: dict[str, float] | None = None,
    random_seed: int = 1234,
    random_size: int = 100,
) -> dict[str, Any]:
    """Compute all documented baselines for one test partition.

    Returns dict keyed by baseline name; each value is an OosMetrics dict.
    """
    if events is None or events.empty:
        return {}

    results: dict[str, Any] = {}

    # A. All candidates (no condition filter).
    results["all_candidates"] = evaluate_baseline(
        hypothesis, events, labels,
        name="A_all_candidates",
        candles=candles, costs=costs,
    )

    # B. Random selection (deterministic seed).
    results["random_selected"] = evaluate_baseline(
        hypothesis, events, labels,
        name="B_random_selected",
        random_sample=random_size,
        seed=random_seed,
        candles=candles, costs=costs,
    )

    # C. No HTF filter (only candidates with no/failed HTF alignment).
    if "feature_htf_alignment" in events.columns:
        mask_no_htf = (
            events["feature_htf_alignment"].isna()
            | (events["feature_htf_alignment"].astype(str).isin(
                ["unknown", "none", "neutral", ""]
            ))
        ).to_numpy()
        results["no_htf_filter"] = evaluate_baseline(
            hypothesis, events, labels,
            name="C_no_htf_filter",
            filter_mask=mask_no_htf,
            candles=candles, costs=costs,
        )

        # D. HTF-aligned candidates (alignment matches direction).
        direction_aligned = events["feature_htf_alignment"].astype(str).str.lower()
        want = "bullish" if hypothesis.direction == "long" else "bearish"
        mask_htf = (direction_aligned == want).to_numpy()
        results["htf_aligned"] = evaluate_baseline(
            hypothesis, events, labels,
            name="D_htf_aligned",
            filter_mask=mask_htf,
            candles=candles, costs=costs,
        )

    # E. Opposite/noise baseline: the hypothesis applied to the opposite
    # direction (a real edge must beat the noise it should not have).
    opposite_hyp = Hypothesis(
        symbol=hypothesis.symbol,
        timeframe=hypothesis.timeframe,
        strategy_family=hypothesis.strategy_family,
        event_type=hypothesis.event_type,
        direction="short" if hypothesis.direction == "long" else "long",
        conditions=hypothesis.conditions,
        entry_rule=hypothesis.entry_rule,
        stop_rule=hypothesis.stop_rule,
        exit_rule=hypothesis.exit_rule,
        stop_atr_multiple=hypothesis.stop_atr_multiple,
        exit_atr_multiple=hypothesis.exit_atr_multiple,
        max_holding_bars=hypothesis.max_holding_bars,
    )
    opposite_events = events.copy()
    if "direction" in opposite_events.columns:
        opposite_events["direction"] = (
            "short" if hypothesis.direction == "long" else "long"
        )
    opposite_labels = labels.copy() if labels is not None else None
    if opposite_labels is not None and not opposite_labels.empty:
        # Reverse the long/short label R (the R sign flips when direction flips).
        opposite_labels = opposite_labels.copy()
        if "label_r" in opposite_labels.columns:
            opposite_labels["label_r"] = -opposite_labels["label_r"].astype(float)

    results["opposite_noise"] = evaluate_baseline(
        opposite_hyp,
        opposite_events,
        opposite_labels,
        name="E_opposite_noise",
        candles=candles,
        costs=costs,
    )

    return results