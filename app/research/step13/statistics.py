"""Statistical analysis for Step 13 Alpha Discovery.

Provides statistically responsible analysis for candidate hypotheses:
- expectancy (mean/median R) with standard error
- bootstrap confidence interval
- Cohen's d effect size
- stability across chronological periods / symbols / regimes / sessions
- train-half vs second-half comparison
- Benjamini-Hochberg FDR correction for multiple testing

Assumptions are documented per function. The methods are deliberately
pragmatic — not academic overkill — while remaining statistically honest.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


# ── Expectancy ───────────────────────────────────────────────────────────────

def expectancy_stats(r_values: list[float]) -> dict[str, float | None]:
    """Mean/median R, standard error of the mean.

    Assumption: R values are IID draws (independent event outcomes). This is
    a research approximation; serial correlation is not modelled here.
    """
    arr = _finite(r_values)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean_r": None,
            "median_r": None,
            "std_r": None,
            "sem_r": None,
        }
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else None
    return {
        "n": int(n),
        "mean_r": round(mean, 4),
        "median_r": round(median, 4),
        "std_r": round(std, 4),
        "sem_r": round(sem, 4) if sem is not None else None,
    }


def bootstrap_ci(
    r_values: list[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap percentile confidence interval for mean R.

    Assumption: rows are exchangeable samples of the event population. This
    does NOT hold perfectly for overlapping time windows; the CI is therefore
    a research heuristic, not a strict statistical guarantee.

    Deterministic: seed is fixed.
    """
    arr = _finite(r_values)
    if len(arr) == 0:
        return {"n": 0, "ci_lower": None, "ci_upper": None, "mean_r": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(means, alpha * 100.0))
    hi = float(np.percentile(means, (1.0 - alpha) * 100.0))
    return {
        "n": int(n),
        "mean_r": round(float(np.mean(arr)), 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def cohens_d(r_values: list[float], null_mean: float = 0.0) -> float | None:
    """Cohen's d effect size of R vs a null mean.

    ``d = (mean - null) / pooled_std``. Assumption: approximately normal R.
    Used as a research heuristic for effect magnitude.
    """
    arr = _finite(r_values)
    if len(arr) < 2:
        return None
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return None
    return round((mean - null_mean) / std, 4)


# ── Stability ────────────────────────────────────────────────────────────────

def stability_by_halves(r_values: list[float]) -> dict[str, Any]:
    """Compare first half vs second half expectancy (chronological stability).

    Assumption: events are in chronological order in ``r_values``.
    """
    arr = _finite(r_values)
    n = len(arr)
    if n < 4:
        return {
            "n": int(n),
            "first_half_mean_r": None,
            "second_half_mean_r": None,
            "degradation": None,
            "degradation_ok": None,
        }
    half = n // 2
    first = arr[:half]
    second = arr[half:]
    f_mean = float(np.mean(first))
    s_mean = float(np.mean(second))
    # Degradation: positive if second half is worse than first.
    degradation = f_mean - s_mean if f_mean > 0 else 0.0
    return {
        "n": int(n),
        "first_half_mean_r": round(f_mean, 4),
        "second_half_mean_r": round(s_mean, 4),
        "degradation": round(degradation, 4),
        "degradation_ok": bool(s_mean >= 0.0),
    }


def stability_by_group(
    r_by_group: dict[str, list[float]],
) -> dict[str, Any]:
    """Stability across groups (symbols / regimes / sessions).

    ``r_by_group`` maps group key -> R values.
    Reports per-group mean/median and the fraction of groups with
    positive mean R.
    """
    per_group = {}
    positive_groups = 0
    for key, values in sorted(r_by_group.items()):
        arr = _finite(values)
        if len(arr) == 0:
            continue
        m = float(np.mean(arr))
        med = float(np.median(arr))
        n = int(len(arr))
        per_group[key] = {
            "n": n,
            "mean_r": round(m, 4),
            "median_r": round(med, 4),
        }
        if m > 0:
            positive_groups += 1
    total = len(per_group)
    return {
        "groups_tested": total,
        "positive_group_fraction": round(positive_groups / total, 4) if total else 0.0,
        "per_group": per_group,
    }


# ── Multiple testing ────────────────────────────────────────────────────────

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Returns a list of booleans (True = survives FDR at level ``alpha``).
    Assumption: p-values are computed correctly for each hypothesis.
    """
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    threshold = np.arange(1, n + 1) / n * alpha
    survives = np.zeros(n, dtype=bool)
    largest = -1
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= threshold[i]:
            largest = i
            break
    if largest >= 0:
        survives[order[: largest + 1]] = True
    return [bool(s) for s in survives]


def adjusted_p_values_bh(p_values: list[float]) -> list[float]:
    """Return BH-adjusted p-values (q-values)."""
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    q = np.ones(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        qi = sorted_p[i] * n / (i + 1)
        running_min = min(running_min, qi)
        q[order[i]] = running_min
    return [round(float(v), 6) for v in q]