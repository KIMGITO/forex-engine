"""Research metrics aggregation and warnings.

Provides cross-window and cross-symbol stability summaries from raw backtest
metric dicts, plus explicit minimum-sample warnings. Metrics are reported
verbatim; insufficient samples are flagged, never silently treated as
meaningful.
"""


__all__ = ["aggregate_metrics", "cross_symbol_summary", "warnings_for"]


def aggregate_metrics(metric_dicts: list[dict]) -> dict:
    """Aggregate a list of per-window/per-symbol metric dicts (mean/std)."""
    if not metric_dicts:
        return {}
    keys: set[str] = set()
    for d in metric_dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and isinstance(k, str):
                keys.add(k)
    out: dict[str, float] = {}
    for k in sorted(keys):
        values: list[float] = [
            float(d[k]) for d in metric_dicts if k in d and isinstance(d[k], (int, float))
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = var ** 0.5
    return out


def warnings_for(metrics: dict, min_trades: int, min_bars: int, n_bars: int) -> list[str]:
    """Return explicit research warnings for insufficient samples.

    Warnings are always surfaced, never silently suppressed.
    """
    warnings: list[str] = []
    trades = int(metrics.get("trade_count", 0))
    if trades < min_trades:
        warnings.append(
            f"WARNING: only {trades} trades in this period; "
            f"statistical confidence is insufficient (min {min_trades})."
        )
    if n_bars < min_bars:
        warnings.append(
            f"WARNING: only {n_bars} bars of data; "
            f"insufficient for robust evaluation (min {min_bars})."
        )
    if trades < 2:
        warnings.append(
            "WARNING: win-rate, profit factor and expectancy are unreliable "
            "with fewer than 2 trades."
        )
    return warnings


def cross_symbol_summary(results_by_symbol: dict[str, dict]) -> dict:
    """Summarise how a strategy's metrics generalize across symbols.

    Reports per-symbol expectancy/trades, plus the mean/std and the fraction
    of symbols with positive expectancy — evidence, not a guarantee.
    """
    summaries: list[dict[str, float | int | str]] = []
    for sym, metrics in results_by_symbol.items():
        exp = float(metrics.get("expectancy") or 0.0)
        trades = int(metrics.get("trade_count", 0))
        summaries.append({"symbol": sym, "expectancy": exp, "trades": trades})

    if not summaries:
        return {}

    exp_values: list[float] = [float(s["expectancy"]) for s in summaries]
    mean = sum(exp_values) / len(exp_values)
    var = sum((v - mean) ** 2 for v in exp_values) / max(len(exp_values) - 1, 1)
    std = var ** 0.5
    positive = sum(1 for v in exp_values if v > 0)
    return {
        "symbols_tested": len(summaries),
        "expectancy_mean": mean,
        "expectancy_std": std,
        "positive_fraction": positive / len(summaries),
        "per_symbol": summaries,
    }
