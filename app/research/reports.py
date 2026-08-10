"""Research report assembly.

Combines dataset provenance, strategy config, training/validation/out-of-sample
results, cross-symbol/window stability, cost assumptions, warnings, and
limitations into a structured, machine-readable :class:`ResearchReport`.

IN-SAMPLE, VALIDATION, and OUT-OF-SAMPLE blocks are kept strictly separate and
explicitly labelled — never mixed.
"""

from collections.abc import Iterable

from app.research.config import ResearchConfig
from app.research.metrics import aggregate_metrics, cross_symbol_summary
from app.research.models import ResearchReport

__all__ = ["build_research_report"]


def _summary_block(
    label: str,
    metrics_list: Iterable[dict],
) -> dict:
    """Build an explicitly-labelled result block with warnings preserved."""
    metrics = list(metrics_list)
    agg = aggregate_metrics(metrics)
    return {
        "label": label,
        "periods_evaluated": len(metrics),
        "aggregate": agg,
        "per_period": metrics,
    }


def build_research_report(
    *,
    provider: str,
    symbols: list[str],
    timeframes: list[str],
    strategy_name: str,
    config: ResearchConfig,
    date_range: dict,
    training_metrics: list[dict],
    validation_metrics: list[dict],
    oos_metrics: list[dict],
    cross_window_metrics: list[dict],
    results_by_symbol: dict[str, dict],
    cost_assumptions: dict,
    warnings: list[str],
    limitations: list[str],
) -> ResearchReport:
    """Assemble a ResearchReport with strictly separated result blocks.

    Labels are explicit: IN-SAMPLE, VALIDATION, OUT-OF-SAMPLE.
    """
    return ResearchReport(
        provider=provider,
        symbols=symbols,
        timeframes=timeframes,
        strategy=strategy_name,
        config_dump=config.to_dict(),
        date_range=date_range,
        training=_summary_block("IN-SAMPLE (TRAIN)", training_metrics),
        validation=_summary_block("VALIDATION", validation_metrics),
        out_of_sample=_summary_block("OUT-OF-SAMPLE (TEST)", oos_metrics),
        cross_symbol=cross_symbol_summary(results_by_symbol),
        cross_window=aggregate_metrics(cross_window_metrics),
        cost_assumptions=cost_assumptions,
        warnings=warnings,
        limitations=limitations,
    )