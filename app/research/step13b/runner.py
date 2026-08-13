"""Step 13B main orchestrator.

Pipeline:
  Historical Data
  -> Features
  -> Market Structure
  -> Regime
  -> Signal Generation
  -> Risk Engine
  -> Backtest
  -> Walk-Forward Validation
  -> Robustness Analysis
  -> Strategy Evaluation
  -> Deployable Strategy Configuration

Memory contract (8 GB dev machine):
  * Process one symbol at a time.
  * Process one timeframe at a time.
  * Process one walk-forward window at a time.
  * Never retain MTF context for the entire historical dataset.
  * Release DataFrames and analytical objects after each window.
  * Persist completed research results incrementally (atomic).
  * Abort safely before OOM via configurable memory limit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.dataset import PartitionedResearchRepository
from app.research.step13b.artifacts import ArtifactManager, read_parquet_if_valid
from app.research.step13b.config import Step13BConfig, build_walk_forward_bounds
from app.research.step13b.memory import MemoryGuard, MemoryLimitError, rss_mb
from app.research.step13b.metrics import (
    monthly_returns,
    regime_performance,
    session_performance,
    symbol_performance,
    yearly_returns,
)
from app.research.step13b.models import (
    StrategyStatus,
    StrategyValidation,
    WindowResult,
)
from app.research.step13b.robustness import analyze_robustness
from app.research.step13b.state import ResearchState
from app.research.step13b.validation import (
    compute_validation_score,
    detect_overfit,
)
from app.research.step13b.window import WindowProcessor

_log = logging.getLogger(__name__)

ENGINE_VERSION = "13b.1.0"
STRATEGY_VERSION = "1.0.0"


def _setup_logging(log_file: str | None = None, verbose: bool = True) -> None:
    root = logging.getLogger("step13b")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        root.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)


def log() -> logging.Logger:
    return logging.getLogger("step13b")


@dataclass
class Step13BRunConfig:
    """Runtime configuration for a Step 13B run."""

    symbols: tuple = ("EURUSD",)
    timeframes: tuple = ("M15",)
    storage_root: str = "data/processed"
    max_bars: int = 0  # 0 = all
    resume: bool = False
    log_file: str = ""


def _data_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of the source OHLC data frame."""
    buf = df[["open", "high", "low", "close"]].sort_index().to_parquet(engine="pyarrow")
    return hashlib.sha256(buf).hexdigest()


def _load_partition(
    repo: PartitionedResearchRepository,
    symbol: str,
    timeframe: str,
    max_bars: int = 0,
) -> pd.DataFrame | None:
    """Load a partition; optionally slice to the last N bars."""
    df = repo.load_df(symbol, timeframe)
    if df is None or df.empty:
        return None
    df = df[["open", "high", "low", "close"]].sort_index()
    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:]
    return df


def _run_symbol_timeframe(
    df: pd.DataFrame,
    config: Step13BConfig,
    *,
    symbol: str,
    timeframe: str,
    artifacts: ArtifactManager,
    state: ResearchState,
    resume: bool,
    memory_guard: MemoryGuard,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run walk-forward research for one symbol/timeframe pair."""
    L = log()
    L.info("=== %s/%s START  rss=%.0fMB ===", symbol, timeframe, rss_mb())

    # Data hash for reproducibility.
    data_hash = _data_hash(df)

    # Build walk-forward windows.
    data_start = df.index[0].to_pydatetime()
    data_end = df.index[-1].to_pydatetime()
    windows = build_walk_forward_bounds(
        data_start,
        data_end,
        train_days=config.train_days,
        validation_days=config.validation_days,
        test_days=config.test_days,
        step_days=config.step_days,
    )

    n_windows = len(windows)
    L.info("  %d walk-forward windows over %s -> %s", n_windows, data_start, data_end)

    if n_windows < config.min_windows:
        L.warning(
            "  INSUFFICIENT DATA: only %d windows (min %d)",
            n_windows, config.min_windows,
        )

    # Resume behavior: find first incomplete window.
    start_window = 0
    if resume:
        start_window = state.next_incomplete_window(symbol, timeframe, n_windows)
        if start_window > 0:
            L.info("  resume: skipping %d completed window(s)", start_window)

    # Process each window sequentially (memory-bounded).
    window_results: list[WindowResult] = []
    for w_idx in range(start_window, n_windows):
        L.info("  --- window %d/%d START  rss=%.0fMB ---", w_idx + 1, n_windows, rss_mb())
        state.set_status(symbol, timeframe, w_idx, "running")

        bounds = windows[w_idx]
        processor = WindowProcessor(
            config, bounds, symbol, timeframe,
            artifacts=artifacts,
            memory_guard=memory_guard,
        )
        try:
            result = processor.process(df)
        except MemoryLimitError:
            L.error("  MEMORY LIMIT at window %d; aborting safely", w_idx)
            state.set_status(symbol, timeframe, w_idx, "failed")
            raise
        except Exception:
            L.error(
                "  window %d FAILED:\n%s",
                w_idx, traceback.format_exc(),
            )
            state.set_status(symbol, timeframe, w_idx, "failed")
            raise

        window_results.append(result)
        memory_guard.check(f"after_window_{w_idx}")
        L.info("  --- window %d/%d DONE  rss=%.0fMB ---", w_idx + 1, n_windows, rss_mb())

    # Only mark windows complete AFTER artifacts are persisted (done in
    # WindowProcessor.process). We now verify and mark complete.
    for w_idx in range(start_window, min(n_windows, len(window_results) + start_window)):
        if artifacts.window_artifacts_valid(w_idx):
            state.set_status(symbol, timeframe, w_idx, "complete")
        else:
            L.error("  window %d artifact not valid; not marking complete", w_idx)

    # Re-load completed results from disk (resumable/reproducible).
    completed_results = _load_window_results(artifacts, n_windows)

    # Aggregate window metrics frame.
    _write_accumulated_metrics(artifacts, completed_results)

    # Read the accumulated trade log for regime/session/symbol aggregation.
    trade_log = read_parquet_if_valid(artifacts.trade_log_path())
    if trade_log is None or trade_log.empty:
        trade_log = _rebuild_trade_log(artifacts, completed_results)
        # Always persist (even empty, so the artifact exists).
        if trade_log is None:
            trade_log = pd.DataFrame(
                columns=[
                    "timestamp", "symbol", "timeframe", "trend", "regime",
                    "volatility", "structure_bias", "liquidity_state",
                    "sweep_detected", "break_of_structure", "displacement",
                    "mtf_bias", "session", "direction", "entry", "stop_loss",
                    "take_profit", "risk_distance", "reward_distance",
                    "risk_reward_ratio", "risk_rejected", "risk_rejection_reason",
                    "risk_percent", "position_size", "result", "r_multiple",
                    "exit_reason", "param_set", "score", "strength",
                ]
            )
        artifacts.write_trade_log(trade_log)

    # Monthly / yearly metrics.
    if trade_log is not None and not trade_log.empty and "entry" in trade_log.columns:
        # Trade log is signal snapshots, not raw backtest trades. We can't
        # compute monthly P&L without exit prices. Instead we aggregate
        # by "result" counts and net R.
        monthly_df = _monthly_from_trade_log(trade_log)
        artifacts.write_monthly_metrics(monthly_df)
    else:
        artifacts.write_monthly_metrics(
            pd.DataFrame(columns=["month", "symbol", "timeframe", "net_r", "trade_count"])
        )

    # Regime / session / symbol performance frames.
    if trade_log is not None and not trade_log.empty:
        artifacts.write_regime_metrics(regime_performance(trade_log))
    else:
        artifacts.write_regime_metrics(
            pd.DataFrame(
                columns=["regime", "trade_count", "wins", "losses", "net_r", "win_rate"]
            )
        )

    # Robustness analysis.
    robustness = analyze_robustness(
        completed_results,
        symbol=symbol,
        timeframe=timeframe,
        min_trades_per_window=config.min_trades_per_window,
    )

    # Validation scoring (uses TEST metrics; config gates).
    validation_score = compute_validation_score(
        window_results=completed_results,
        robustness=robustness,
        config=config,
        positive_symbol_fraction=1.0,  # single-symbol run
    )

    # Overfit detection.
    train_metrics = [
        m for w in completed_results for m in (w.train_metrics or [])
    ]
    val_metrics = [
        m for w in completed_results for m in (w.validation_metrics or [])
    ]
    test_metrics = [
        m for w in completed_results for m in (w.test_metrics or [])
    ]
    is_overfit, overfit_reasons = detect_overfit(
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        robustness=robustness,
    )

    # Determine final status (overfit takes precedence).
    final_status = validation_score.status
    if is_overfit and final_status in (
        StrategyStatus.VALIDATED,
        StrategyStatus.PROMISING,
    ):
        final_status = StrategyStatus.OVERFIT
        validation_score = validation_score  # keep score but override status

    # Best parameters from the selected config across windows.
    best_params = _recommend_parameters(completed_results, config)

    # Risk recommendations.
    risk_recommendation = _risk_recommendation(completed_results)

    # Metrics summary.
    metrics_summary = _metrics_summary(completed_results)

    # Symbol and timeframe metrics.
    symbol_metrics = _symbol_metrics(trade_log, symbol)
    regime_metrics = _regime_metrics(regime_performance(trade_log))
    risk_metrics = _risk_summary(completed_results)

    validation = StrategyValidation(
        strategy_name=config.strategy_name,
        strategy_version=STRATEGY_VERSION,
        engine_version=ENGINE_VERSION,
        data_hash=data_hash,
        configuration_hash=config.config_hash(),
        validation_status=final_status,
        metrics=metrics_summary,
        walk_forward_metrics=robustness.to_dict(),
        regime_metrics=regime_metrics,
        symbol_metrics=symbol_metrics,
        risk_metrics=risk_metrics,
        recommended_parameters=best_params,
        risk_recommendation=risk_recommendation,
        validation_score={
            **validation_score.to_dict(),
            "is_overfit": is_overfit,
            "overfit_reasons": overfit_reasons,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        notes=[
            "Step 13B alternative research path: bounded-memory, walk-forward, "
            "resumable validation. Full MTF object graph never retained in memory.",
            "Validation score formula documented in app/research/step13b/validation.py.",
            "TEST data never used to optimize parameters.",
        ],
    )

    # Atomic write of the final strategy validation artifact.
    artifacts.write_strategy_validation(validation.to_dict())

    # Write research summary.
    artifacts.write_summary(
        {
            "strategy": config.strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": final_status.value,
            "n_windows": n_windows,
            "completed_windows": len(completed_results),
            "validation_score": validation_score.total,
            "created_at": validation.timestamp,
        }
    )

    L.info("=== %s/%s DONE  rss=%.0fMB ===", symbol, timeframe, rss_mb())
    return validation.to_dict()


def _load_window_results(
    artifacts: ArtifactManager, n_windows: int
) -> list[WindowResult]:
    """Load persisted window results from disk (resumable).

    Metrics are stored as JSON dicts; this reloads them back into
    ``WindowMetrics`` objects so downstream analysis works identically
    whether results came from the in-memory pass or a resumed run.
    """
    from app.research.step13b.models import WindowMetrics

    def _metrics(data: list) -> list[WindowMetrics]:
        out: list[WindowMetrics] = []
        for m in data:
            try:
                if isinstance(m, WindowMetrics):
                    out.append(m)
                elif isinstance(m, dict):
                    out.append(WindowMetrics(**m))
            except Exception:  # noqa: BLE001 - skip malformed entries
                continue
        return out

    out: list[WindowResult] = []
    for idx in range(n_windows):
        raw = artifacts.window_json_path(idx)
        if not raw.exists():
            continue
        try:
            data = json.loads(raw.read_text("utf-8"))
            out.append(
                WindowResult(
                    index=int(data.get("index", idx)),
                    symbol=str(data.get("symbol", "")),
                    timeframe=str(data.get("timeframe", "")),
                    bounds=data.get("bounds", {}),
                    selected_params=data.get("selected_params", {}),
                    candidate_metrics=_metrics(data.get("candidate_metrics", [])),
                    train_metrics=_metrics(data.get("train_metrics", [])),
                    validation_metrics=_metrics(data.get("validation_metrics", [])),
                    test_metrics=_metrics(data.get("test_metrics", [])),
                    trade_count=int(data.get("trade_count", 0)),
                    status=str(data.get("status", "complete")),
                    warnings=list(data.get("warnings", [])),
                )
            )
        except Exception:  # noqa: BLE001 - corrupt window file skipped
            continue
    return out


def _write_accumulated_metrics(
    artifacts: ArtifactManager,
    window_results: list[WindowResult],
) -> None:
    """Write the cumulative window metrics parquet from result dicts."""
    rows = []
    for w in window_results:
        for phase in ("train", "validation", "test"):
            m_list = getattr(w, f"{phase}_metrics", []) or []
            for m in m_list:
                rows.append(m)
    if rows:
        from app.research.step13b.models import WindowMetrics

        df = pd.DataFrame(
            [r if isinstance(r, dict) else r.to_dict() for r in rows]
        )
        artifacts.write_window_metrics(df)
    else:
        artifacts.write_window_metrics(
            pd.DataFrame(columns=["window_index", "phase", "param_set"])
        )


def _rebuild_trade_log(
    artifacts: ArtifactManager,
    window_results: list[WindowResult],
) -> pd.DataFrame | None:
    """Rebuild the cumulative trade log from per-window parquet files."""
    frames: list[pd.DataFrame] = []
    import glob

    for p in sorted(artifacts.windows_dir.glob("window_*_trades.parquet")):
        df = read_parquet_if_valid(p)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp") if "timestamp" in combined.columns else combined


def _monthly_from_trade_log(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate net R and trade counts by month from the trade log."""
    if trade_log is None or trade_log.empty or "timestamp" not in trade_log.columns:
        return pd.DataFrame(
            columns=["month", "symbol", "timeframe", "net_r", "trade_count"]
        )
    df = trade_log.copy()
    df["month"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
    grouped = (
        df.groupby(["month", "symbol", "timeframe"])
        .agg(
            net_r=("r_multiple", "sum"),
            trade_count=("r_multiple", "count"),
        )
        .reset_index()
    )
    return grouped.sort_values("month")


def _recommend_parameters(
    window_results: list[WindowResult],
    config: Step13BConfig,
) -> dict[str, Any]:
    """Recommend the most frequently selected parameter set.

    Parameter selection is based on VALIDATION performance per window. The
    recommendation is the modal selection, NOT the highest historical profit.
    """
    from collections import Counter

    selections: Counter = Counter()
    for w in window_results:
        if w.status == "complete" and w.selected_params:
            selections[tuple(sorted(w.selected_params.items()))] += 1
    if not selections:
        return dict(config.param_grid[0]) if config.param_grid else {}
    best = selections.most_common(1)[0][0]
    return dict(best)


def _risk_recommendation(window_results: list[WindowResult]) -> dict[str, Any]:
    """Provide risk recommendations based on observed risk rejections."""
    total_rejected = 0
    total_approved = 0
    for w in window_results:
        for m in w.test_metrics:
            total_rejected += m.risk_rejected_count
            total_approved += m.risk_approved_count
    total = total_rejected + total_approved
    rejection_rate = total_rejected / total if total > 0 else 0.0
    return {
        "risk_rejected_trades": total_rejected,
        "risk_approved_trades": total_approved,
        "risk_rejection_rate": rejection_rate,
        "recommendation": (
            "keep current risk limits"
            if rejection_rate < 0.30
            else "review risk limits — high rejection rate may filter out valid signals"
        ),
    }


def _metrics_summary(window_results: list[WindowResult]) -> dict[str, Any]:
    """Aggregate metrics summary across all completed windows (TEST only)."""
    test_ms = [
        m for w in window_results if w.status == "complete"
        for m in (w.test_metrics or [])
    ]
    if not test_ms:
        return {}
    vals = {
        "net_profit": [m.net_profit for m in test_ms],
        "expectancy_r": [m.expectancy_r for m in test_ms if m.expectancy_r is not None],
        "profit_factor": [m.profit_factor for m in test_ms if m.profit_factor is not None],
        "win_rate": [m.win_rate for m in test_ms if m.win_rate is not None],
        "max_drawdown": [m.max_drawdown for m in test_ms],
        "sharpe": [m.sharpe for m in test_ms if m.sharpe is not None],
        "sortino": [m.sortino for m in test_ms if m.sortino is not None],
        "trade_count": [m.trade_count for m in test_ms],
        "max_consecutive_losses": [m.max_consecutive_losses for m in test_ms],
    }

    def _sum(k):
        return round(sum(vals[k]), 4) if vals[k] else None

    def _mean(k):
        return round(sum(vals[k]) / len(vals[k]), 4) if vals[k] else None

    return {
        "total_trades": _sum("trade_count"),
        "total_net_profit": _sum("net_profit"),
        "median_expectancy_r": _median(vals["expectancy_r"]),
        "mean_win_rate": _mean("win_rate"),
        "mean_profit_factor": _mean("profit_factor"),
        "max_drawdown": _max(vals["max_drawdown"]),
        "mean_sharpe": _mean("sharpe"),
        "mean_sortino": _mean("sortino"),
        "mean_max_consecutive_losses": _mean("max_consecutive_losses"),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return round(float(s[mid]), 4)
    return round((s[mid - 1] + s[mid]) / 2.0, 4)


def _max(values: list[float]) -> float | None:
    return round(max(values), 4) if values else None


def _symbol_metrics(trade_log: pd.DataFrame | None, symbol: str) -> dict[str, Any]:
    """Per-symbol metrics from the trade log."""
    if trade_log is None or trade_log.empty:
        return {
            "symbol": symbol,
            "trade_count": 0,
            "net_r": 0.0,
            "win_rate": None,
        }
    df = trade_log[trade_log["symbol"] == symbol] if "symbol" in trade_log.columns else trade_log
    n = len(df)
    wins = int((df["result"] == "win").sum()) if "result" in df.columns else 0
    return {
        "symbol": symbol,
        "trade_count": int(n),
        "net_r": round(float(df["r_multiple"].sum()), 4) if "r_multiple" in df.columns else 0.0,
        "win_rate": round(wins / n, 4) if n > 0 else None,
    }


def _regime_metrics(regime_df: pd.DataFrame | None) -> dict[str, Any]:
    """Convert regime performance frame to a dict."""
    if regime_df is None or regime_df.empty:
        return {}
    out = {}
    for _, row in regime_df.iterrows():
        out[str(row["regime"])] = {
            "trade_count": int(row["trade_count"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "net_r": round(float(row["net_r"]), 4),
            "win_rate": round(float(row["win_rate"]), 4),
        }
    return out


def _risk_summary(window_results: list[WindowResult]) -> dict[str, Any]:
    """Aggregate risk counters across all windows (all phases)."""
    counters = {
        "risk_rejected_trades": 0,
        "risk_approved_trades": 0,
        "daily_loss_breaches": 0,
        "drawdown_limit_breaches": 0,
        "position_limit_breaches": 0,
        "exposure_breaches": 0,
    }
    for w in window_results:
        for m in w.candidate_metrics:
            counters["risk_rejected_trades"] += m.risk_rejected_count
            counters["risk_approved_trades"] += m.risk_approved_count
            counters["daily_loss_breaches"] += m.daily_loss_breaches
            counters["drawdown_limit_breaches"] += m.drawdown_limit_breaches
            counters["position_limit_breaches"] += m.position_limit_breaches
            counters["exposure_breaches"] += m.exposure_breaches
    return counters


def run_step13b(
    config: Step13BConfig | None = None,
    *,
    symbols: tuple | None = None,
    timeframes: tuple | None = None,
    max_bars: int = 0,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Step 13B research pipeline.

    Parameters
    ----------
    config : optional Step13BConfig
        Research configuration.
    symbols : optional tuple
        Override symbols.
    timeframes : optional tuple
        Override timeframes.
    max_bars : int
        Limit to the last N bars per partition (0 = all).
    resume : bool
        Skip completed walk-forward windows.
    """
    config = config or Step13BConfig()
    if symbols:
        config = _replace_tuple(config, "symbols", symbols)
    if timeframes:
        config = _replace_tuple(config, "timeframes", timeframes)

    out_root = Path(config.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = str(out_root / "step13b_run.log")
    _setup_logging(log_file, verbose=verbose)
    L = log()

    state = ResearchState(out_root / "research_state.json")
    memory_guard = MemoryGuard(config.max_rss_mb)
    repo = PartitionedResearchRepository(config.storage_root)

    L.info("=" * 70)
    L.info("STEP 13B — STRATEGY RESEARCH & WALK-FORWARD VALIDATION ENGINE")
    L.info(f"  symbols={list(config.symbols)}")
    L.info(f"  timeframes={list(config.timeframes)}")
    L.info(f"  windows: train={config.train_days}d val={config.validation_days}d "
           f"test={config.test_days}d step={config.step_days}d")
    L.info(f"  max_rss_mb={config.max_rss_mb}  resume={resume}")
    L.info(f"  output={out_root}")
    L.info(f"  rss_start={rss_mb():.0f}MB")
    L.info("=" * 70)

    all_results: dict[str, Any] = {}
    for sym in config.symbols:
        for tf in config.timeframes:
            try:
                df = _load_partition(repo, sym, tf, max_bars=max_bars)
            except Exception:
                L.error("  %s/%s load failed: %s", sym, tf, traceback.format_exc())
                continue
            if df is None or df.empty:
                L.warning("  %s/%s: no data; skipping", sym, tf)
                continue
            if max_bars > 0 and len(df) > max_bars:
                L.info("  %s/%s: sliced to last %d bars (%d total available)",
                       sym, tf, max_bars, len(df))

            artifacts = ArtifactManager(out_root, sym, tf)
            try:
                result = _run_symbol_timeframe(
                    df,
                    config,
                    symbol=sym,
                    timeframe=tf,
                    artifacts=artifacts,
                    state=state,
                    resume=resume,
                    memory_guard=memory_guard,
                    verbose=verbose,
                )
                all_results[f"{sym}/{tf}"] = result
            except MemoryLimitError:
                L.error("  MEMORY LIMIT reached at %s/%s; aborting safely", sym, tf)
                raise
            except Exception:
                L.error(
                    "  %s/%s run FAILED:\n%s",
                    sym, tf, traceback.format_exc(),
                )
                raise

            # Release symbol/tf data.
            del df
            gc.collect()

    # Write a top-level summary across all runs.
    summary_path = out_root / "research_summary.json"
    top_summary = {
        "runs": [
            {
                "symbol": k.split("/")[0],
                "timeframe": k.split("/")[1],
                "status": v.get("validation_status", "unknown"),
                "validation_score": v.get("validation_score", {}).get("total", 0.0)
                if isinstance(v.get("validation_score"), dict)
                else v.get("validation_score", 0.0),
                "metrics": v.get("metrics", {}),
            }
            for k, v in all_results.items()
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
    }
    from app.research.step13b.artifacts import atomic_write_json

    atomic_write_json(summary_path, top_summary)
    L.info("PIPELINE COMPLETE (Step 13B)")
    return all_results


def _replace_tuple(config: Step13BConfig, field_name: str, values: tuple) -> Step13BConfig:
    """Return a copy of config with one tuple field replaced."""
    data = {k: v for k, v in config.__dict__.items()}
    data[field_name] = tuple(values)
    return Step13BConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 13B — Strategy Research & Walk-Forward Validation"
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--storage-root", default="data/processed")
    parser.add_argument("--output-root", default="research/results/step13b")
    parser.add_argument(
        "--max-bars", type=int, default=0,
        help="dev mode: process only the last N bars (0 = all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip completed walk-forward windows",
    )
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--strategy", default="trend_structure",
                        choices=("trend_structure", "liquidity_reversal"))
    parser.add_argument("--max-rss-mb", type=int, default=3000)
    args = parser.parse_args()

    config = Step13BConfig(
        symbols=tuple(args.symbols) if args.symbols else ("EURUSD",),
        timeframes=tuple(args.timeframes) if args.timeframes else ("M15",),
        storage_root=args.storage_root,
        output_root=args.output_root,
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        strategy_name=args.strategy,
        max_rss_mb=args.max_rss_mb,
    )
    run_step13b(config, max_bars=args.max_bars, resume=args.resume)


if __name__ == "__main__":
    main()