"""Step 15 CLI runner and report generator.

Loads Step 13 candidate artifacts + base candles, runs the walk-forward
validation, computes baselines/stability/breakdowns, assesses data
sufficiency, and persists reproducible artifacts under
``research/results/step15``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.step13.hypotheses import Hypothesis
from app.research.step13.persist import read_parquet_if_valid
from app.research.step15.baselines import compute_baselines
from app.research.step15.breakdowns import full_breakdown
from app.research.step15.config import Step15Config
from app.research.step15.models import Step15Fold
from app.research.step15.splits import partition_candidates
from app.research.step15.stability import fold_stability_report
from app.research.step15.walk_forward import Step15WalkForwardEngine

ENGINE_VERSION = "15.0.0"


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", prefix=".step15_tmp_", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _data_hash(df: pd.DataFrame) -> str:
    buf = df[["open", "high", "low", "close"]].sort_index().to_parquet(engine="pyarrow")
    return hashlib.sha256(buf).hexdigest()


def _load_base_candles(storage_root: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load base OHLC candles; timestamp may be a column or the index."""
    raw = pd.read_parquet(
        Path(storage_root) / symbol.upper() / timeframe.upper() / "data.parquet"
    )
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        df = raw[["open", "high", "low", "close", "timestamp"]].sort_values("timestamp")
        df = df.set_index("timestamp")[["open", "high", "low", "close"]]
    else:
        # Timestamp is the index.
        df = raw[["open", "high", "low", "close"]].sort_index()
    return df


def _load_candidates(step13_root: str, symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load candidate_events.parquet / candidate_labels.parquet from a Step 13 output dir."""
    base = Path(step13_root) / symbol.upper() / timeframe.upper()
    events = read_parquet_if_valid(base / "candidate_events.parquet")
    labels = read_parquet_if_valid(base / "candidate_labels.parquet")
    if events is None or events.empty:
        raise ValueError(f"no candidate_events at {base}")
    if labels is None or labels.empty:
        raise ValueError(f"no candidate_labels at {base}")
    events = events.copy()
    labels = labels.copy()
    # Preserve tz-aware datetimes for chronological slicing.
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    meta = {}
    summary_path = base / "summary.json"
    if summary_path.exists():
        meta = json.loads(summary_path.read_text("utf-8"))
    return events, labels, meta


def _hyp_from_artifact(artifact: dict) -> Hypothesis:
    """Reconstruct a Hypothesis from a research_candidate artifact dict."""
    ev = artifact.get("event_definition", {}) or {}
    entry_rule = str(artifact.get("entry_definition", "displacement_confirmation"))
    stop_def = artifact.get("stop_definition", {}) or {}
    exit_def = artifact.get("exit_definition", {}) or {}
    return Hypothesis(
        symbol=str(artifact.get("symbol", "EURUSD")),
        timeframe=str(artifact.get("timeframe", "M15")),
        strategy_family=str(artifact.get("strategy_family", "liquidity_sweep")),
        event_type=str(ev.get("type", "liquidity_sweep")),
        direction=str(ev.get("direction", "long")),
        conditions=tuple(artifact.get("conditions") or ()),
        entry_rule=entry_rule,
        stop_rule=str(stop_def.get("rule", "atr")),
        exit_rule=str(exit_def.get("rule", "fixed_rr_2.0")),
        stop_atr_multiple=float(stop_def.get("atr_multiple", 1.0)),
        exit_atr_multiple=float(exit_def.get("atr_multiple", 2.0)),
        max_holding_bars=int(exit_def.get("max_holding_bars", 0) or 0),
    )


def data_sufficiency_report(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    folds: list[Step15Fold],
    config: Step15Config,
) -> dict[str, Any]:
    """Assess whether the sample is sufficient for meaningful validation."""
    ev_ts = pd.to_datetime(events["timestamp"])
    total_days = (ev_ts.max() - ev_ts.min()).total_seconds() / 86400.0

    per_fold = []
    for f in folds:
        tr = f.get("test_results") if isinstance(f, dict) else f.test_results
        tr = tr or {}
        per_fold.append(int(tr.get("trades", 0)))

    max_holding = 0
    if labels is not None and not labels.empty and "label_holding_bars" in labels.columns:
        max_holding = int(labels["label_holding_bars"].fillna(0).max())

    verdict = "ADEQUATE"
    total_oos = sum(per_fold)
    if total_oos < 100:
        verdict = "INSUFFICIENT"
    elif total_oos < 200:
        verdict = "MARGINAL"

    return {
        "total_historical_duration_days": round(total_days, 1),
        "total_candidates": len(events),
        "candidates_per_fold": per_fold,
        "min_oos_sample": min(per_fold) if per_fold else 0,
        "max_oos_sample": max(per_fold) if per_fold else 0,
        "max_holding_bars": max_holding,
        "number_of_folds": len(folds),
        "min_train_sample": config.min_train_sample,
        "verdict": verdict,
        "assessment": (
            "The OOS sample is too small to separate real edge from noise; "
            "see the per-fold sample table and do not treat any positive "
            "OOS result as statistically meaningful."
            if verdict == "INSUFFICIENT"
            else (
                "OOS sample is marginal — results carry high variance and "
                "should be reproduced on a longer dataset before any claim."
                if verdict == "MARGINAL"
                else "OOS sample is adequate for a preliminary research read."
            )
        ),
    }


def run_step15(
    *,
    step13_root: str = "research/results/step13",
    storage_root: str = "data/research",
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    output_root: str = "research/results/step15",
    config: Step15Config | None = None,
) -> dict[str, Any]:
    """Full Step 15 pipeline over the persisted Step 13 artifacts."""
    config = config or Step15Config(output_root=output_root)
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    events, labels, step13_meta = _load_candidates(step13_root, symbol, timeframe)
    candles = _load_base_candles(storage_root, symbol, timeframe)
    source_data_hash = _data_hash(candles)

    # NOTE: the candles frame must extend past the candidate horizon so the
    # execution model has future bars for label recomputation.
    engine = Step15WalkForwardEngine(
        config,
        candles=candles,
        bar_minutes=15 if timeframe.upper() == "M15" else 60,
    )
    report = engine.run(events, labels)

    # Baselines on the CANONICAL split TEST window (frozen hypothesis).
    canonical = report["canonical_results"]
    selected_hid = canonical.get("selected_hypothesis")
    baselines: dict[str, Any] = {}
    breakdowns: dict[str, Any] = {}
    if selected_hid:
        # Rebuild the frozen hypothesis from the canonical selection audit.
        hyp = _reconstruct_selected_hypothesis(selected_hid, report)
        # Re-partition the canonical split test window.
        from app.research.step15.splits import make_single_split, partition_candidates

        split = make_single_split(
            pd.to_datetime(events["timestamp"]).min(),
            pd.to_datetime(events["timestamp"]).max(),
            config,
        )
        parts = partition_candidates(
            events, labels, split,
            purge_horizon_bars=config.purge_horizon_bars,
            purge_enabled=False,
        )
        test_ev = parts["test_events"]
        test_lab = parts["test_labels"]
        if hyp is not None and not test_ev.empty:
            baselines = compute_baselines(
                hyp,
                test_ev,
                test_lab,
                candles=candles,
                costs=engine.costs,
                random_seed=config.baseline_random_seed,
                random_size=100,
            )
            breakdowns = full_breakdown(test_ev, test_lab)

    # Stability report from folds.
    stability = fold_stability_report(report["folds"])

    # Data sufficiency.
    sufficiency = data_sufficiency_report(events, labels, report["folds"], config)

    artifact = {
        "engine_version": ENGINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "source_data_hash": source_data_hash,
        "step13_artifact_hash": step13_meta.get("data_hash", ""),
        "canonical_split": report["canonical_split"],
        "canonical_results": report["canonical_results"],
        "walk_forward_splits": report["walk_forward_splits"],
        "folds": report["folds"],
        "aggregate_oos": report["aggregate_oos"],
        "hypotheses_audit": report["hypotheses_audit"],
        "selection_audit": report["selection_audit"],
        "leakage_audit": report["leakage_audit"],
        "baselines": baselines,
        "breakdowns": breakdowns,
        "stability": stability,
        "data_sufficiency": sufficiency,
        "costs": engine.costs,
        "cost_model": {
            "spread_pips": config.spread_pips,
            "slippage_pips": config.slippage_pips,
            "commission_per_lot": config.commission_per_lot,
            "applies_at": "single R deduction at exit (spread + slippage + commission)",
        },
        "purge_policy": {
            "enabled": config.purge_enabled,
            "horizon_bars": config.purge_horizon_bars,
            "policy": "training candidates whose label horizon crosses validation are EXCLUDED",
        },
    }

    _atomic_write_json(out / "step15_report.json", artifact)
    _atomic_write_json(out / "manifest.json", {
        "engine_version": ENGINE_VERSION,
        "created_at": artifact["created_at"],
        "symbol": symbol,
        "timeframe": timeframe,
        "config_hash": config.config_hash(),
        "source_data_hash": source_data_hash,
        "artifacts": [
            "step15_report.json",
            "manifest.json",
        ],
    })
    _atomic_write_json(out / "human_report.txt", _human_report(artifact))

    return artifact


def _reconstruct_selected_hypothesis(hid: str, report: dict) -> Hypothesis | None:
    """Reconstruct the frozen hypothesis from the canonical audit definition."""
    audit = report.get("canonical_results", {}).get("selection_metrics", {})
    definition = audit.get("selected_hypothesis_definition")
    if definition and definition.get("hypothesis_id") == hid:
        return Hypothesis.from_dict(definition)
    return None


def _human_report(artifact: dict) -> str:
    lines = ["=" * 72, "STEP 15 — WALK-FORWARD / OUT-OF-SAMPLE VALIDATION REPORT", "=" * 72]
    lines.append(f"Engine: {artifact['engine_version']}")
    lines.append(f"Symbol/Timeframe: {artifact['symbol']}/{artifact['timeframe']}")
    lines.append(f"Config hash: {artifact['config_hash']}")
    lines.append(f"Source data hash: {artifact['source_data_hash']}")
    lines.append("")
    lines.append("TEMPORAL ARCHITECTURE:")
    cs = artifact["canonical_split"]
    lines.append(f"  TRAIN      {cs['train_start']} -> {cs['train_end']}")
    lines.append(f"  VALIDATION {cs['validation_start']} -> {cs['validation_end']}")
    lines.append(f"  TEST (OOS) {cs['test_start']} -> {cs['test_end']}")
    lines.append("")
    lines.append("CANONICAL SPLIT RESULTS:")
    cr = artifact["canonical_results"]
    lines.append(f"  selected hypothesis : {cr.get('selected_hypothesis')}")
    lines.append(f"  train samples       : {cr.get('train_samples')}")
    lines.append(f"  purged from train   : {cr.get('purged_from_train')}")
    lines.append(f"  VALIDATION          : {json.dumps(cr.get('validation', {}), default=str)[:200]}")
    lines.append(f"  TEST (OOS)          : {json.dumps(cr.get('test', {}), default=str)[:200]}")
    lines.append("")
    lines.append("WALK-FORWARD FOLDS:")
    for f in artifact["folds"]:
        tr = f.get("test_results", {})
        lines.append(
            f"  fold {f['index']}: hyp={f.get('selected_hypothesis') or 'NONE'} "
            f"train={f.get('train_sample')} test_trades={tr.get('trades', 0)} "
            f"net_r={tr.get('net_r', 0.0)} dd={tr.get('max_drawdown_r', 0.0)}"
        )
    agg = artifact["aggregate_oos"]
    lines.append("")
    lines.append("AGGREGATE OOS:")
    lines.append(f"  folds with trades : {agg.get('folds_with_trades')}")
    lines.append(f"  total trades      : {agg.get('total_trades')}")
    lines.append(f"  total gross R     : {agg.get('total_gross_r')}")
    lines.append(f"  total net R       : {agg.get('total_net_r')}")
    lines.append(f"  sample warning    : {agg.get('sample_warning')}")
    lines.append("")
    su = artifact.get("data_sufficiency", {})
    lines.append("DATA SUFFICIENCY:")
    lines.append(f"  verdict      : {su.get('verdict')}")
    lines.append(f"  duration days: {su.get('total_historical_duration_days')}")
    lines.append(f"  candidates   : {su.get('total_candidates')}")
    lines.append(f"  max holding  : {su.get('max_holding_bars')} bars")
    lines.append(f"  assessment   : {su.get('assessment')}")
    lines.append("")
    stab = artifact.get("stability", {}).get("aggregate_flags", {})
    lines.append("STABILITY:")
    for k, v in stab.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    bas = artifact.get("baselines", {})
    if bas:
        lines.append("BASELINES (canonical TEST):")
        for name, b in bas.items():
            lines.append(
                f"  {name}: trades={b.get('trades', 0)} "
                f"net_r={b.get('net_r', 0.0)} "
                f"expectancy={b.get('expectancy')}"
            )
    lines.append("")
    lines.append("COST MODEL:")
    cm = artifact.get("cost_model", {})
    for k, v in cm.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("LEAKAGE AUDIT:")
    for k, v in artifact.get("leakage_audit", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 15 — Walk-Forward / Out-of-Sample Validation"
    )
    parser.add_argument("--step13-root", default="research/results/step13")
    parser.add_argument("--storage-root", default="data/research")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--output", default="research/results/step15")
    parser.add_argument(
        "--wf-train-days", type=int, default=120,
        help="walk-forward train window (calendar days)",
    )
    parser.add_argument(
        "--wf-val-days", type=int, default=30, dest="wf_validation_days",
        help="walk-forward validation window (calendar days)",
    )
    parser.add_argument(
        "--wf-test-days", type=int, default=30,
        help="walk-forward test window (calendar days)",
    )
    parser.add_argument(
        "--wf-step-days", type=int, default=30,
        help="walk-forward step (calendar days)",
    )
    parser.add_argument(
        "--no-purge", action="store_true",
        help="disable purge of training candidates whose label horizon crosses validation",
    )
    args = parser.parse_args()

    config = Step15Config(
        wf_train_days=args.wf_train_days,
        wf_validation_days=args.wf_validation_days,
        wf_test_days=args.wf_test_days,
        wf_step_days=args.wf_step_days,
        purge_enabled=not args.no_purge,
        output_root=args.output,
    )
    result = run_step15(
        step13_root=args.step13_root,
        storage_root=args.storage_root,
        symbol=args.symbol,
        timeframe=args.timeframe,
        output_root=args.output,
        config=config,
    )
    print(_human_report(result))


if __name__ == "__main__":
    main()