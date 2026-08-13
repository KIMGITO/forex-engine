"""Candidate artifact persistence for Step 13 Alpha Discovery.

Writes ``research_candidate.json`` per discovery candidate. Each artifact
clearly identifies itself as DISCOVERY_CANDIDATE — never PROFITABLE_STRATEGY.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.research.step13.hypotheses import Hypothesis
from app.research.step13.persist import atomic_write_json, read_json_if_valid


def build_candidate_artifact(
    *,
    hypothesis: Hypothesis,
    evaluator_stats: dict[str, Any],
    discovery_score: dict[str, Any],
    sample_count: int,
    overfit_warnings: list[str],
    data_hash: str,
    engine_version: str,
    configuration_hash: str,
    data_range: dict[str, str] | None = None,
    recommended_validation: str = "step13b",
) -> dict[str, Any]:
    """Assemble a ``research_candidate.json`` payload.

    The artifact is explicitly labeled DISCOVERY_CANDIDATE.
    """
    return {
        "candidate_id": hypothesis.hypothesis_id,
        "strategy_family": hypothesis.strategy_family,
        "hypothesis": hypothesis.hypothesis_description,
        "symbol": hypothesis.symbol,
        "timeframe": hypothesis.timeframe,
        "event_definition": {
            "type": hypothesis.event_type,
            "direction": hypothesis.direction,
        },
        "conditions": list(hypothesis.conditions),
        "entry_definition": hypothesis.entry_rule,
        "stop_definition": {
            "rule": hypothesis.stop_rule,
            "atr_multiple": hypothesis.stop_atr_multiple,
        },
        "exit_definition": {
            "rule": hypothesis.exit_rule,
            "atr_multiple": hypothesis.exit_atr_multiple,
            "max_holding_bars": hypothesis.max_holding_bars,
        },
        "sample_count": sample_count,
        "win_rate": evaluator_stats.get("win_rate"),
        "expectancy_R": (
            evaluator_stats.get("expectancy", {}).get("mean_r")
            if isinstance(evaluator_stats.get("expectancy"), dict)
            else None
        ),
        "average_R": (
            evaluator_stats.get("expectancy", {}).get("mean_r")
            if isinstance(evaluator_stats.get("expectancy"), dict)
            else None
        ),
        "median_R": (
            evaluator_stats.get("expectancy", {}).get("median_r")
            if isinstance(evaluator_stats.get("expectancy"), dict)
            else None
        ),
        "profit_factor": None,  # set by ranking stage
        "max_drawdown": evaluator_stats.get("max_drawdown_r"),
        "losing_streak": None,
        "holding_time": evaluator_stats.get("avg_holding_bars"),
        "regime_metrics": (
            evaluator_stats.get("stability_groups", {}).get("per_group", {})
        ),
        "session_metrics": {},
        "symbol_metrics": {},
        "statistical_metrics": evaluator_stats.get("expectancy", {}),
        "bootstrap_metrics": evaluator_stats.get("bootstrap", {}),
        "FDR": None,
        "stability_metrics": {
            "halves": evaluator_stats.get("stability_halves", {}),
            "groups": evaluator_stats.get("stability_groups", {}),
        },
        "discovery_score": discovery_score,
        "overfit_warnings": overfit_warnings,
        "data_range": data_range or {},
        "data_hash": data_hash,
        "engine_version": engine_version,
        "configuration_hash": configuration_hash,
        "recommended_validation": recommended_validation,
        "status": "DISCOVERY_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_candidate_artifact(
    output_root: str | Path,
    symbol: str,
    timeframe: str,
    artifact: dict[str, Any],
) -> Path:
    """Atomically write a research_candidate.json artifact."""
    root = Path(output_root)
    out_dir = root / symbol.upper() / timeframe.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"research_candidate_{artifact.get('candidate_id', 'unknown')}.json"
    atomic_write_json(path, artifact)
    return path


def load_candidate_artifact(path: str | Path) -> dict[str, Any] | None:
    """Load a research_candidate.json artifact."""
    return read_json_if_valid(Path(path))