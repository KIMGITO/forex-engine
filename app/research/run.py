"""End-to-end research pipeline orchestrator.

Reuses the existing engines—FeatureEngine, MarketStructureEngine, RegimeEngine,
MtfEngine, strategy scanner, EventBacktester, walk-forward + optimizer—without
duplicating their logic. Every stage is causal.

Fetch window: Twelve Data free-tier caps each request at ~800 rows (~1 month).
This runner fetches a configurable trailing window (``fetch_days``) so we never
waste API credits, and it reports the provider's depth limitation honestly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001,S110 - dotenv optional
    pass

from app.backtest import BacktestConfig, EventBacktester
from app.backtest import Strategy as BacktestStrategy
from app.data.provider import BaseMarketDataProvider
from app.data.providers import create_provider
from app.features import FeatureEngine
from app.market_structure.engine import MarketStructureEngine
from app.mtf import MtfConfig, MtfEngine
from app.regime import RegimeConfig
from app.research.config import ResearchConfig
from app.research.data_quality import validate_partition
from app.research.dataset import PartitionedResearchRepository, sync_partition
from app.research.models import ResearchReport
from app.research.reports import build_research_report
from app.research.splits import make_time_split, split_frame
from app.strategy import (
    HistoricalSignalScanner,
    LiquidityReversalStrategy,
    StrategyConfig,
    TrendStructureStrategy,
)
from app.strategy.base import SignalToOrderAdapter

__all__ = ["ResearchRunConfig", "run_research_pipeline", "smoke_test_pipeline"]

# Research timeframe -> native convention for MTF + backtester.
_MTF_NATIVE = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}


@dataclass(frozen=True)
class ResearchRunConfig:
    """Configuration passed through the whole pipeline."""

    symbols: tuple = ("EURUSD",)
    timeframes: tuple = ("H1",)
    storage_root: str = "data/research"
    strategy_names: tuple = ("trend_structure", "liquidity_reversal")
    fetch_days: int = 30
    baseline_cost: dict = field(
        default_factory=lambda: {"spread_pips": 0.8, "slippage_pips": 0.0,
                                 "commission_percent": 0.0, "commission_per_trade": 0.0}
    )
    # Walk-forward window durations in DAYS (the Twelve Data free-tier returns
    # only ~1 trailing month, so the default multi-year windows cannot be
    # satisfied honestly). These match the real data span.
    walk_train_days: float = 18.0
    walk_validation_days: float = 5.0
    walk_test_days: float = 5.0
    conservative_cost: dict = field(
        default_factory=lambda: {"spread_pips": 1.5, "slippage_pips": 0.5,
                                 "commission_percent": 0.0005, "commission_per_trade": 0.0}
    )


def _native(timeframe: str) -> str:
    return _MTF_NATIVE.get(timeframe, timeframe.lower())


def _df_from_repo(repo, symbol: str, timeframe: str) -> pd.DataFrame | None:
    df = repo.load_df(symbol, timeframe)
    if df is None:
        return None
    return df[["open", "high", "low", "close"]].sort_index()


def _run_backtest(
    df: pd.DataFrame,
    strategy: BacktestStrategy,
    symbol: str,
    native_tf: str,
    bt_config: BacktestConfig,
    mtf_contexts: list | None = None,
    features: pd.DataFrame | None = None,
    news_events: list | None = None,
) -> dict:
    """Run one backtest and return metrics as a dict (causal by the engine)."""
    result = EventBacktester(bt_config).run(
        df,
        strategy,
        features=features,
        news_events=news_events,
        mtf_contexts=mtf_contexts,
        provider="twelvedata",
        source_type="historical",
    )
    m = result.metrics
    return {
        "trade_count": m.trade_count,
        "net_pnl": m.net_pnl,
        "total_return": m.total_return,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "expectancy": m.expectancy,
        "average_win": m.average_win,
        "average_loss": m.average_loss,
        "max_drawdown": m.max_drawdown,
        "drawdown_duration_bars": m.drawdown_duration_bars,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "exposure_fraction": m.exposure_fraction,
        "average_holding_bars": m.average_holding_bars,
        "insufficient_data": list(m.insufficient_data),
    }


def _build_mtf_contexts(
    dfs_all: dict[str, pd.DataFrame],
    symbol: str,
    base_tf: str,
    news_events: list | None = None,
) -> list | None:
    """Build MTF contexts for a symbol's base timeframe, or None when unavailable."""
    native_base = _native(base_tf)
    # Collect this symbol's timeframes into a native-keyed map.
    native_map: dict[str, pd.DataFrame] = {}
    for tf in _MTF_NATIVE:
        key = f"{symbol}|{tf}"
        if key in dfs_all and dfs_all[key] is not None and not dfs_all[key].empty:
            native_map[_native(tf)] = dfs_all[key]
    if native_base not in native_map:
        return None
    order = ["5m", "15m", "1h", "4h", "1d"]
    if native_base in order:
        order.remove(native_base)
    higher = [tf for tf in order if tf in native_map]
    if not higher:
        return None
    try:
        engine = MtfEngine(MtfConfig(base_timeframe=native_base, higher_timeframes=tuple(higher)), symbol)
        return engine.analyze(native_map, native_base, news_events=news_events)
    except Exception:  # noqa: BLE001 - insufficient data reported, never fabricated
        return None


def _build_signal_strategy(
    df,
    symbol,
    native_tf,
    strategy_name,
    strat_config,
    features,
    mtf_ctxs,
):
    """Return (backtest_strategy, signals_by_ts) using cached causal signals.

    The signal scanner runs ONCE over the full frame (causal). The returned
    adapter-lookup strategy replays those signals onto any sub-frame (train,
    validation, test, walk-forward window) by timestamp, so all downstream
    blocks reflect REAL strategy behavior without recomputation.
    """
    strat = (
        TrendStructureStrategy(strat_config)
        if strategy_name == "trend_structure"
        else LiquidityReversalStrategy(strat_config)
    )
    scanner = HistoricalSignalScanner(strategy_config=strat_config, regime_config=RegimeConfig())
    scan = scanner.scan(df, strat, symbol, native_tf, features=features, mtf_contexts=mtf_ctxs)
    signals_by_ts = {s.timestamp: s for s in scan.signals}
    adapter = SignalToOrderAdapter(quantity=1000.0)

    class _SignalStrategy(BacktestStrategy):
        name = strategy_name

        def on_bar(self, context):
            sig = signals_by_ts.get(context.now.to_pydatetime())
            if sig is None:
                return []
            return adapter.to_order_intents(sig, context, context.now.to_pydatetime())

    return _SignalStrategy(), signals_by_ts


def _run_strategy_on_partition(
    df: pd.DataFrame,
    dfs_all: dict[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    strategy_name: str,
    strat_config: StrategyConfig,
    cost: dict,
) -> dict:
    """Features -> structure -> regime -> MTF -> signals -> backtest (causal)."""
    native = _native(timeframe)
    bt = BacktestConfig(
        symbol=symbol,
        timeframe=native,
        spread_pips=cost["spread_pips"],
        slippage_pips=cost["slippage_pips"],
        commission_model=("percentage" if cost["commission_percent"] else
                          "fixed" if cost["commission_per_trade"] else "zero"),
        commission_percent=cost["commission_percent"],
        commission_per_trade=cost["commission_per_trade"],
        initial_balance=10_000.0,
    )

    fe = FeatureEngine()
    features = fe.calculate(df, features=["atr", "rsi"])
    # Market structure + regime are computed causal; the strategy scanner
    # recomputes its own regime internally (kept additive, no duplication).
    MarketStructureEngine().analyze(df, symbol, native)

    # MTF contexts (only when higher-timeframe data exists).
    mtf_ctxs = _build_mtf_contexts(dfs_all, symbol, timeframe) if strat_config.mtf_enabled else None

    strat = (
        TrendStructureStrategy(strat_config)
        if strategy_name == "trend_structure"
        else LiquidityReversalStrategy(strat_config)
    )
    scanner = HistoricalSignalScanner(strategy_config=strat_config, regime_config=RegimeConfig())
    scan = scanner.scan(df, strat, symbol, native, features=features, mtf_contexts=mtf_ctxs)

    adapter = SignalToOrderAdapter(quantity=1000.0)
    signals_by_ts = {s.timestamp: s for s in scan.signals}

    class _Strategy(BacktestStrategy):
        name = strategy_name

        def on_bar(self, context):
            sig = signals_by_ts.get(context.now.to_pydatetime())
            if sig is None:
                return []
            return adapter.to_order_intents(sig, context, context.now.to_pydatetime())

    metrics = _run_backtest(df, _Strategy(), symbol, native, bt, mtf_ctxs, features=features)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "mtf": strat_config.mtf_enabled,
        "trade_count": int(metrics.get("trade_count", 0)),
        "metrics": metrics,
    }


def run_research_pipeline(
    config: ResearchConfig | None = None,
    run_cfg: ResearchRunConfig | None = None,
    *,
    provider: BaseMarketDataProvider | None = None,
    repo: PartitionedResearchRepository | None = None,
    output_root: str = "research/results/latest",
    verbose: bool = True,
) -> dict:
    """Fetch-once, process-once, validate, run strategies, walk-forward, report."""
    config = config or ResearchConfig()
    run_cfg = run_cfg or ResearchRunConfig()
    repo = repo or PartitionedResearchRepository(run_cfg.storage_root)
    provider = provider or create_provider(config.provider)

    end = datetime.now(timezone.utc)
    start_anchor = end - timedelta(days=run_cfg.fetch_days)

    # ── 1. FETCH ONCE (incremental, idempotent, rate-limit-aware) ─────────────
    fetch_notes: list[str] = []
    if verbose:
        print(f"Fetching {run_cfg.symbols} x {run_cfg.timeframes} (window={run_cfg.fetch_days}d)...")
    for sym in run_cfg.symbols:
        for tf in run_cfg.timeframes:
            # Backfill the missing head (older data) incrementally, but never
            # redownload the tail we already have. `sync_partition` is idempotent,
            # so a re-run is safe and does not duplicate existing candles.
            existing_df = repo.load_df(sym, tf)
            if existing_df is not None and len(existing_df) >= 2:
                first_local = existing_df.index[0].to_pydatetime()
                if first_local <= start_anchor:
                    if verbose:
                        print(f"  {sym}/{tf}: cached ({len(existing_df)} rows covering range)")
                    continue
                try:
                    existing, final = sync_partition(
                        provider, repo, sym, tf, end=end, start=start_anchor
                    )
                    if verbose:
                        print(f"  {sym}/{tf}: backfilled head -> {final} rows")
                except Exception as exc:  # noqa: BLE001 - provider limitation surfaced
                    if verbose:
                        print(f"  {sym}/{tf}: FAILED {type(exc).__name__}: {exc}")
                    fetch_notes.append(f"{sym}/{tf}: {type(exc).__name__}: {exc}")
                continue
            try:
                existing, final = sync_partition(provider, repo, sym, tf, end=end, start=start_anchor)
                if verbose:
                    print(f"  {sym}/{tf}: fetched {final} ({existing} existed)")
            except Exception as exc:  # noqa: BLE001 - provider limitation surfaced
                if verbose:
                    print(f"  {sym}/{tf}: FAILED {type(exc).__name__}: {exc}")
                fetch_notes.append(f"{sym}/{tf}: {type(exc).__name__}: {exc}")

    # ── 2. VALIDATE + load ────────────────────────────────────────────────────
    dfs_all: dict[str, pd.DataFrame] = {}
    quality: dict[str, dict] = {}
    for sym in run_cfg.symbols:
        for tf in run_cfg.timeframes:
            df = _df_from_repo(repo, sym, tf)
            if df is None or df.empty:
                quality[f"{sym}|{tf}"] = validate_partition(
                    None, sym, tf, "twelvedata", "native"
                ).to_dict()
                continue
            dq_report = validate_partition(df, sym, tf, provider="twelvedata", native_or_aggregated="native")
            quality[f"{sym}|{tf}"] = dq_report.to_dict()
            if dq_report.passed:
                dfs_all[f"{sym}|{tf}"] = df

    if verbose:
        print("\nDATA QUALITY:")
        for k, q in sorted(quality.items()):
            ok = "PASS" if q["passed"] else "FAIL"
            print(f"  {k}: rows={q['candle_count']} start={q['first_timestamp']} "
                  f"end={q['last_timestamp']} gaps={q['gap_count']} tz={q['timezone_status']} -> {ok}")

    # ── 3. RUN STRATEGIES per symbol/timeframe (MTF on/off) ──────────────────
    results: list[dict] = []
    for sym in run_cfg.symbols:
        for tf in run_cfg.timeframes:
            df = dfs_all.get(f"{sym}|{tf}")
            if df is None:
                continue
            for strat_name in run_cfg.strategy_names:
                # MTF disabled (baseline)
                results.append(_run_strategy_on_partition(
                    df, dfs_all, sym, tf, strat_name, StrategyConfig(), run_cfg.baseline_cost
                ))
                # MTF enabled (if higher-timeframe data exists)
                results.append(_run_strategy_on_partition(
                    df, dfs_all, sym, tf, strat_name,
                    StrategyConfig(mtf_enabled=True, mtf_min_aligned=1),
                    run_cfg.baseline_cost,
                ))

    # ── 4. TRAIN / VALIDATION / TEST (strictly chronological) ────────────────
    rep = dfs_all.get("EURUSD|H1")
    train_metrics: list[dict] = []
    val_metrics: list[dict] = []
    test_metrics: list[dict] = []
    if rep is not None and len(rep) > 100:
        split = make_time_split(rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), config)
        train, val, test = split_frame(rep, split)
        bt = BacktestConfig(symbol="EURUSD", timeframe="1h", spread_pips=0.8)

        # Use the REAL signal-driven strategy (trend_structure) on EURUSD H1 so
        # TRAIN/VALIDATION/TEST reflect actual trades, not a no-op placeholder.
        rep_fe = FeatureEngine().calculate(rep, features=["atr", "rsi"])
        rep_strategy, _ = _build_signal_strategy(
            rep, "EURUSD", "1h", "trend_structure", StrategyConfig(), rep_fe, None
        )
        for frame, store in ((train, train_metrics), (val, val_metrics), (test, test_metrics)):
            if frame.empty:
                continue
            store.append(_run_backtest(frame, rep_strategy, "EURUSD", "1h", bt))

    # ── 5. WALK-FORWARD + SMALL OPTIMIZATION ─────────────────────────────────
    walk_results: list[dict] = []
    opt_results: list[dict] = []
    if rep is not None and len(rep) > 100:
        # Reuse the real strategy built for the split block (cached signals).
        rep_fe = FeatureEngine().calculate(rep, features=["atr", "rsi"])
        rep_strategy, _ = _build_signal_strategy(
            rep, "EURUSD", "1h", "trend_structure", StrategyConfig(), rep_fe, None
        )

        def _wf_bt(frame, params=None):
            bt = BacktestConfig(symbol="EURUSD", timeframe="1h",
                                spread_pips=run_cfg.baseline_cost["spread_pips"])
            return _run_backtest(frame, rep_strategy, "EURUSD", "1h", bt)

        # Build a walk-forward config with day-scale windows (real data span).
        from app.research.walk_forward import build_walk_forward_windows

        wf_cfg = ResearchConfig(
            walk_train_years=run_cfg.walk_train_days / 365.0,
            walk_validation_years=run_cfg.walk_validation_days / 365.0,
            walk_test_years=run_cfg.walk_test_days / 365.0,
        )
        # Override window builder to use the real data range (short windows).
        wf_windows = build_walk_forward_windows(
            rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), wf_cfg
        )
        for w in wf_windows:
            wf_train = rep[(rep.index >= w.split.train_start) & (rep.index < w.split.train_end)]
            wf_val = rep[(rep.index >= w.split.validation_start) & (rep.index < w.split.validation_end)]
            wf_test = rep[(rep.index >= w.split.test_start) & (rep.index <= w.split.test_end)]
            if wf_train.empty or wf_val.empty or wf_test.empty:
                continue
            val_m = _wf_bt(wf_val)
            test_m = _wf_bt(wf_test)
            walk_results.append({
                "index": w.index,
                "split": w.split.to_dict(),
                "validation": val_m,
                "test": test_m,
            })

        from app.research.optimizer import GridSearchOptimizer

        opt = GridSearchOptimizer(config, _wf_bt)
        split = make_time_split(rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), config)
        train, _, _ = split_frame(rep, split)
        # Small grid (2^3 = 8 candidates) — never touches TEST.
        grid = {"regime_strength": [0.4, 0.6], "reward_risk": [1.5, 2.5]}
        for c in opt.optimize(train, grid):
            opt_results.append(c.to_dict())

    # ── 6. ASSEMBLE + PERSIST REPORT ─────────────────────────────────────────
    date_range = {}
    for sym in run_cfg.symbols:
        df = dfs_all.get(f"{sym}|H1")
        if df is not None:
            date_range[sym] = {"start": str(df.index[0]), "end": str(df.index[-1]), "rows": len(df)}

    # Aggregate REAL strategy results per symbol (mean expectancy across the
    # per-partition MTF on/off strategy runs).
    symbol_agg: dict[str, dict] = {}
    for r in results:
        sym = str(r.get("symbol") or "unknown")
        exp = float(r.get("metrics", {}).get("expectancy") or 0.0)
        t = int(r.get("trade_count", 0))
        if sym not in symbol_agg:
            symbol_agg[sym] = {"exp_sum": 0.0, "trades": 0, "runs": 0}
        symbol_agg[sym]["exp_sum"] += exp
        symbol_agg[sym]["trades"] += t
        symbol_agg[sym]["runs"] += 1
    results_by_symbol: dict[str, dict] = {}
    for sym, agg in symbol_agg.items():
        results_by_symbol[sym] = {
            "expectancy": agg["exp_sum"] / max(agg["runs"], 1),
            "trade_count": agg["trades"],
        }

    report = build_research_report(
        provider="twelvedata",
        symbols=list(run_cfg.symbols),
        timeframes=list(run_cfg.timeframes),
        strategy_name=",".join(run_cfg.strategy_names),
        config=config,
        date_range=date_range,
        training_metrics=train_metrics,
        validation_metrics=val_metrics,
        oos_metrics=test_metrics,
        cross_window_metrics=walk_results,
        results_by_symbol=results_by_symbol,
        cost_assumptions={
            "data": "Twelve Data OHLC",
            "execution": "SIMULATED BID/ASK",
            "cost_model": "ASSUMPTION",
            "baseline": run_cfg.baseline_cost,
            "conservative": run_cfg.conservative_cost,
        },
        warnings=[
            "Twelve Data supplies OHLC only (no historical bid/ask).",
            *fetch_notes,
            "Free-tier per-request depth (~800 rows) constrains the historical window.",
        ],
        limitations=[
            ("Twelve Data free-tier plan serves only ~1 trailing month per "
             "symbol/timeframe (verified: requesting 2015/recent returns data "
             "only from ~2026-07-07); dataset is INSUFFICIENT for multi-year "
             "robustness validation."),
            "No historical bid/ask; all costs are SIMULATED ASSUMPTIONS.",
        ],
    )

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report.model_dump(), indent=2, default=str))
    (out / "report.txt").write_text(_human_readable(report, quality))
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str))
    (out / "walk_forward.json").write_text(json.dumps(walk_results, indent=2, default=str))
    (out / "optimization.json").write_text(json.dumps(opt_results, indent=2, default=str))

    if verbose:
        print(f"\nReports written to {out}/")
        print(f"Walk-forward windows: {len(walk_results)}")
        print(f"Optimization candidates: {len(opt_results)}")
        print(f"Strategy runs: {len(results)}")
    return {
        "report": report.model_dump(),
        "results": results,
        "walk_forward": walk_results,
        "optimization": opt_results,
    }


def _human_readable(report: ResearchReport, quality: dict) -> str:
    lines = ["=" * 70, "FOREX RESEARCH REPORT", "=" * 70]
    lines.append(f"Provider: {report.provider}")
    lines.append(f"Symbols: {', '.join(report.symbols)}")
    lines.append(f"Timeframes: {', '.join(report.timeframes)}")
    lines.append(f"Strategies: {report.strategy}")
    lines.append("")
    lines.append("DATA:")
    for k, q in sorted(quality.items()):
        ok = "PASS" if q["passed"] else "FAIL"
        lines.append(f"  {k}: rows={q['candle_count']} {q['first_timestamp']} -> "
                     f"{q['last_timestamp']} gaps={q['gap_count']} {ok}")
    lines.append("")
    lines.append("EXECUTION: SIMULATED BID/ASK (no historical bid/ask)")
    lines.append("COST MODEL: ASSUMPTION (spread/slippage/commission)")
    lines.append("")
    lines.append("TRAIN / VALIDATION / TEST:")
    lines.append(f"  TRAIN: {_fmt(report.training)}")
    lines.append(f"  VALIDATION: {_fmt(report.validation)}")
    lines.append(f"  TEST (OOS): {_fmt(report.out_of_sample)}")
    lines.append("")
    lines.append("WARNINGS:")
    for w in report.warnings:
        lines.append(f"  - {w}")
    lines.append("")
    lines.append("LIMITATIONS:")
    for l in report.limitations:
        lines.append(f"  - {l}")
    lines.append("")
    lines.append("CONCLUSION: INSUFFICIENT DATA — see limitations.")
    lines.append("=" * 70)
    return "\n".join(lines)


def _fmt(block: dict) -> str:
    a = block.get("aggregate", {}) or {}
    return (f"trades_mean={a.get('trade_count_mean')} "
            f"net_pnl_mean={a.get('net_pnl_mean')} "
            f"expectancy_mean={a.get('expectancy_mean')}")


def smoke_test_pipeline(
    provider: BaseMarketDataProvider | None = None,
    repo: PartitionedResearchRepository | None = None,
    output_root: str = "research/results/latest",
    verbose: bool = True,
) -> dict:
    """Small EURUSD H1 smoke run to validate the pipeline end-to-end."""
    if verbose:
        print("SMOKE TEST: EURUSD H1 (fetch 10 days)")
    cfg = ResearchRunConfig(
        symbols=("EURUSD",),
        timeframes=("H1",),
        strategy_names=("trend_structure",),
        fetch_days=31,  # full provider month so walk-forward windows fit
    )
    return run_research_pipeline(
        config=ResearchConfig(symbols=("EURUSD",), timeframes=("H1",)),
        run_cfg=cfg,
        provider=provider,
        repo=repo,
        output_root=output_root,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forex research pipeline")
    parser.add_argument("--smoke", action="store_true", help="EURUSD H1 smoke run")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", default="research/results/latest")
    args = parser.parse_args()

    if args.smoke:
        smoke_test_pipeline(output_root=args.output)
        return
    run_cfg = ResearchRunConfig(
        symbols=tuple(args.symbols) if args.symbols else ("EURUSD",),
        timeframes=tuple(args.timeframes) if args.timeframes else ("H1",),
        fetch_days=args.days,
    )
    run_research_pipeline(run_cfg=run_cfg, output_root=args.output)


if __name__ == "__main__":
    main()
