"""Walk-forward window processing for Step 13B.

Each window is processed as a bounded chunk:
    [train_start - warmup_bars : test_end]

On this bounded chunk we compute (ONCE):
    features -> market structure -> regime

Then for each parameter variant (small bounded grid):
    signals -> risk engine -> backtest on TRAIN / VALIDATION / TEST

All derived objects are released after the window. Objects are never retained
for the full historical dataset.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest import BacktestConfig, EventBacktester
from app.backtest import Strategy as BacktestStrategy
from app.backtest.models import OrderSide
from app.features import FeatureEngine
from app.market_structure.engine import MarketStructureEngine
from app.market_structure.models import MarketStructureResult
from app.regime import RegimeEngine
from app.regime.models import MarketRegime
from app.research.step13b.artifacts import ArtifactManager
from app.research.step13b.config import Step13BConfig, WalkForwardBounds
from app.research.step13b.memory import MemoryGuard, rss_mb
from app.research.step13b.metrics import (
    compute_window_metrics,
    monthly_returns,
    regime_performance,
    session_performance,
    yearly_returns,
)
from app.research.step13b.models import WindowMetrics, WindowResult
from app.research.step13b.risk import (
    RiskResearchTracker,
    build_research_risk_engine,
    risk_counters_to_metrics,
)
from app.research.step13b.snapshot import signal_to_snapshot, snapshots_to_frame
from app.strategy import (
    HistoricalSignalScanner,
    LiquidityReversalStrategy,
    StrategyConfig,
    TrendStructureStrategy,
)
from app.strategy.base import SignalToOrderAdapter

_log = logging.getLogger(__name__)

_MTF_NATIVE = {"M15": "15m", "H1": "1h"}


def _native(tf: str) -> str:
    return _MTF_NATIVE.get(tf.upper(), tf.lower())


@dataclass
class _PhaseResult:
    """Internal result of evaluating one phase (train/val/test) for one param."""

    metrics: WindowMetrics
    trades: list[Any]
    snapshots: list[dict[str, Any]]
    equity_curve: list[Any]
    signals: list[Any]


class _PhaseStrategy(BacktestStrategy):
    """Backtest strategy adapter driven by pre-scanned signal snapshots."""

    name = "step13b_phase"

    def __init__(
        self,
        signal_lookup: dict[Any, Any],
        adapter: SignalToOrderAdapter,
        risk_tracker: RiskResearchTracker,
    ) -> None:
        super().__init__()
        self.signal_lookup = signal_lookup
        self.adapter = adapter
        self.risk_tracker = risk_tracker

    def on_bar(self, context):
        key = context.now.to_pydatetime()
        sig = self.signal_lookup.get(key)
        if sig is None:
            return []
        # Run through RiskEngine (Step 15) BEFORE converting to order.
        decision = self.risk_tracker.evaluate_signal(
            sig, context, param_set="phase"
        )
        if not decision.approved or decision.position_size is None:
            return []
        # Use the risk-approved position size.
        adapter = self.adapter
        # Patch the adapter quantity to the risk-approved size.
        adapter.quantity = decision.position_size
        return adapter.to_order_intents(sig, context, key)


def _slice_phase(
    df: pd.DataFrame,
    start,
    end,
    *,
    end_inclusive: bool = False,
) -> pd.DataFrame:
    """Slice a DataFrame by timestamp bounds (inclusive start, exclusive end)."""
    if end_inclusive:
        return df[(df.index >= start) & (df.index <= end)]
    return df[(df.index >= start) & (df.index < end)]


class WindowProcessor:
    """Processes one walk-forward window for one symbol/timeframe.

    Memory contract:
        * only one bounded window chunk in memory at a time
        * all heavy analytical objects are released after the window
        * gc.collect() runs after the window when configured
    """

    def __init__(
        self,
        config: Step13BConfig,
        window_bounds: WalkForwardBounds,
        symbol: str,
        timeframe: str,
        *,
        artifacts: ArtifactManager | None = None,
        memory_guard: MemoryGuard | None = None,
    ) -> None:
        self.config = config
        self.bounds = window_bounds
        self.symbol = symbol
        self.timeframe = timeframe
        self.artifacts = artifacts
        self.memory_guard = memory_guard or MemoryGuard(config.max_rss_mb)

    def _warmup_offset(self, df: pd.DataFrame) -> int:
        """Find the row offset for warmup bars before train_start."""
        train_start = pd.Timestamp(self.bounds.train_start)
        mask = df.index < train_start
        warmup_rows = int(mask.sum())
        return min(warmup_rows, self.config.warmup_bars)

    def _load_window_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load ONLY the bounded chunk [train_start - warmup : test_end]."""
        train_start = pd.Timestamp(self.bounds.train_start)
        test_end = pd.Timestamp(self.bounds.test_end)
        # Find the position of train_start to slice warmup.
        all_rows = len(df)
        pre_train = df.index < train_start
        pre_count = int(pre_train.sum())
        warm_start_idx = max(0, pre_count - self.config.warmup_bars)
        window = df.iloc[warm_start_idx:]
        window = window[window.index <= test_end]
        return window

    def process(
        self,
        df: pd.DataFrame,
        *,
        param_grid: list[dict[str, Any]] | None = None,
    ) -> WindowResult:
        """Process this window against the full dataset (window chunk sliced).

        Returns a WindowResult with per-parameter train/validation/test metrics
        and the selected configuration.
        """
        if self.memory_guard:
            self.memory_guard.check(f"window_{self.bounds.index}_start")

        # 1. Load the bounded window chunk.
        chunk = self._load_window_chunk(df)
        if chunk is None or chunk.empty:
            return WindowResult(
                index=self.bounds.index,
                symbol=self.symbol,
                timeframe=self.timeframe,
                bounds=self.bounds.to_dict(),
                status="skipped",
                warnings=["no data in window bounds"],
            )

        # 2. Compute analytical layers ONCE on the window chunk.
        features = FeatureEngine().calculate(chunk, features=["atr", "rsi"])
        structure = MarketStructureEngine().analyze(
            chunk, self.symbol, _native(self.timeframe)
        )
        regimes = RegimeEngine().analyze(
            chunk, self.symbol, _native(self.timeframe), market_structure=structure
        )

        _log.info(
            "  [%s %s win%d] chunk=%d bars rss=%.0fMB",
            self.symbol, self.timeframe, self.bounds.index, len(chunk),
            rss_mb(),
        )

        # 3. Evaluate each parameter variant.
        grid = param_grid or [dict(p) for p in self.config.param_grid]
        candidate_metrics: list[WindowMetrics] = []
        train_states: dict[str, list[WindowMetrics]] = {}
        val_states: dict[str, list[WindowMetrics]] = {}
        test_states: dict[str, list[WindowMetrics]] = {}
        all_snapshots: list[dict[str, Any]] = []
        selected_params: dict[str, Any] = dict(grid[0]) if grid else {}
        best_validation_score = float("-inf")

        for param_set in grid:
            strat_config = StrategyConfig(
                mtf_enabled=False,
                **{k: v for k, v in param_set.items()},
            )
            pname = _param_name(param_set)

            # 3a. Scan signals on the window chunk (causal, per-bar).
            strat = _build_strategy(self.config.strategy_name, strat_config)
            scanner = HistoricalSignalScanner(strategy_config=strat_config)
            scan = scanner.scan(
                chunk,
                strat,
                self.symbol,
                _native(self.timeframe),
                features=features,
                structure=structure,
                regimes=regimes,
            )
            signals = scan.signals
            signal_by_ts = _signals_by_ts(signals)

            # 3b. Evaluate each phase (train/validation/test).
            phase_results: dict[str, _PhaseResult] = {}
            for phase_name, start, end, end_incl in (
                ("train", self.bounds.train_start, self.bounds.train_end, False),
                ("validation", self.bounds.validation_start, self.bounds.validation_end, False),
                ("test", self.bounds.test_start, self.bounds.test_end, True),
            ):
                phase_df = _slice_phase(
                    chunk, start, end, end_inclusive=end_incl
                )
                if phase_df.empty:
                    phase_results[phase_name] = _PhaseResult(
                        metrics=compute_window_metrics(
                            window_index=self.bounds.index,
                            phase=phase_name,
                            symbol=self.symbol,
                            timeframe=self.timeframe,
                            param_set=pname,
                            trades=[],
                            equity_curve=[],
                            bars=0,
                        ),
                        trades=[],
                        snapshots=[],
                        equity_curve=[],
                        signals=[],
                    )
                    continue

                # Build a phase-specific risk tracker (fresh account state).
                risk_engine = build_research_risk_engine(
                    risk_percent=self.config.risk_percent,
                    max_daily_loss_pct=self.config.max_daily_loss_pct,
                    max_drawdown_pct=self.config.max_drawdown_pct,
                    max_open_positions=self.config.max_open_positions,
                )
                risk_tracker = RiskResearchTracker(risk_engine)

                strategy = _PhaseStrategy(
                    signal_lookup=signal_by_ts,
                    adapter=SignalToOrderAdapter(quantity=1.0),
                    risk_tracker=risk_tracker,
                )

                bt_config = BacktestConfig(
                    symbol=self.symbol,
                    timeframe=_native(self.timeframe),
                    spread_pips=self.config.spread_pips,
                    slippage_pips=self.config.slippage_pips,
                    commission_model=(
                        "percentage"
                        if self.config.commission_percent
                        else (
                            "fixed"
                            if self.config.commission_per_trade
                            else "zero"
                        )
                    ),
                    commission_percent=self.config.commission_percent,
                    commission_per_trade=self.config.commission_per_trade,
                    initial_balance=self.config.initial_balance,
                )

                result = EventBacktester(bt_config).run(
                    phase_df,
                    strategy,
                    features=features,
                    market_structure=structure,
                    regime_observations=regimes,
                    provider="local",
                    source_type="historical",
                )

                # Build compact snapshots for the signals in this phase.
                phase_signals = [
                    s for s in signals if _signal_in_phase(s, start, end, end_incl)
                ]
                snapshots = [
                    signal_to_snapshot(
                        s,
                        structure=structure,
                        regimes=regimes,
                        param_set=pname,
                    )
                    for s in phase_signals
                ]
                # Merge risk decision info into snapshots.
                _enrich_snapshots_with_risk(
                    snapshots, phase_signals, risk_tracker
                )

                # Augment snapshots with trade results.
                _merge_trade_results(snapshots, result.trades)

                risk_metrics = risk_counters_to_metrics(risk_tracker)
                metrics = compute_window_metrics(
                    window_index=self.bounds.index,
                    phase=phase_name,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    param_set=pname,
                    trades=result.trades,
                    equity_curve=result.equity_curve,
                    bars=len(phase_df),
                    risk_metrics=risk_metrics,
                )
                phase_results[phase_name] = _PhaseResult(
                    metrics=metrics,
                    trades=result.trades,
                    snapshots=snapshots,
                    equity_curve=result.equity_curve,
                    signals=phase_signals,
                )

            # 3c. Record metrics.
            train_m = phase_results["train"].metrics
            val_m = phase_results["validation"].metrics
            test_m = phase_results["test"].metrics
            candidate_metrics.extend([train_m, val_m, test_m])
            train_states[pname] = [train_m]
            val_states[pname] = [val_m]
            test_states[pname] = [test_m]

            # 3d. Select param set by VALIDATION (never test).
            val_score = _selection_score(val_m)
            if val_score > best_validation_score:
                best_validation_score = val_score
                selected_params = dict(param_set)

            all_snapshots.extend(
                phase_results["train"].snapshots
                + phase_results["validation"].snapshots
                + phase_results["test"].snapshots
            )

            _log.info(
                "    [%s %s win%d %s] train=%d val=%d test=%d trades rss=%.0fMB",
                self.symbol, self.timeframe, self.bounds.index, pname,
                train_m.trade_count, val_m.trade_count, test_m.trade_count,
                rss_mb(),
            )

        # 4. Persist window artifacts (atomic).
        selected_train = train_states.get(_param_name(selected_params), [])
        selected_val = val_states.get(_param_name(selected_params), [])
        selected_test = test_states.get(_param_name(selected_params), [])

        result = WindowResult(
            index=self.bounds.index,
            symbol=self.symbol,
            timeframe=self.timeframe,
            bounds=self.bounds.to_dict(),
            selected_params=selected_params,
            candidate_metrics=candidate_metrics,
            train_metrics=selected_train,
            validation_metrics=selected_val,
            test_metrics=selected_test,
            trade_count=sum(m.trade_count for m in selected_test),
            status="complete",
        )

        if self.artifacts is not None:
            self.artifacts.write_window_result(self.bounds.index, result.to_dict())
            trades_df = snapshots_to_frame(all_snapshots)
            self.artifacts.write_window_trades(self.bounds.index, trades_df)

        # 5. Release window objects.
        del chunk, features, structure, regimes, signals, all_snapshots
        if self.config.gc_after_window:
            gc.collect()

        if self.memory_guard:
            self.memory_guard.check(f"window_{self.bounds.index}_end")
            _log.info(
                "  [%s %s win%d] complete rss=%.0fMB",
                self.symbol, self.timeframe, self.bounds.index, rss_mb(),
            )

        return result


def _param_name(params: dict[str, Any]) -> str:
    """Deterministic short name for a parameter set."""
    return "_".join(f"{k}={v}" for k, v in sorted(params.items()))


def _build_strategy(strategy_name: str, config: StrategyConfig):
    if strategy_name == "liquidity_reversal":
        return LiquidityReversalStrategy(config)
    return TrendStructureStrategy(config)


def _signals_by_ts(signals: list[Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for s in signals:
        out[s.timestamp] = s
    return out


def _signal_in_phase(signal: Any, start, end, end_inclusive: bool) -> bool:
    ts = signal.timestamp
    if end_inclusive:
        return start <= ts <= end
    return start <= ts < end


def _enrich_snapshots_with_risk(
    snapshots: list[dict[str, Any]],
    signals: list[Any],
    risk_tracker: RiskResearchTracker,
) -> None:
    """Merge risk decision data into snapshots."""
    rej_by_sig: dict[str, dict[str, Any]] = {}
    for r in risk_tracker.rejected:
        rej_by_sig[r["signal_id"]] = r
    appr_by_sig: dict[str, dict[str, Any]] = {}
    for a in risk_tracker.approved:
        appr_by_sig[a["signal_id"]] = a

    for snap, sig in zip(snapshots, signals):
        rej = rej_by_sig.get(sig.signal_id)
        appr = appr_by_sig.get(sig.signal_id)
        if rej is not None:
            snap["risk_rejected"] = True
            snap["risk_rejection_reason"] = rej.get("reason")
        if appr is not None:
            snap["risk_percent"] = appr.get("monetary_risk") or 0.0
            snap["position_size"] = appr.get("position_size") or 0.0


def _merge_trade_results(
    snapshots: list[dict[str, Any]], trades: list[Any]
) -> None:
    """Merge backtest trade outcomes into snapshots by entry timestamp."""
    # Build a lookup: entry_time -> list of trade outcomes.
    outcomes: dict[Any, list[Any]] = {}
    for t in trades:
        outcomes.setdefault(t.entry_time, []).append(t)

    for snap in snapshots:
        ts = pd.Timestamp(snap["timestamp"]).to_pydatetime()
        matching = outcomes.get(ts, [])
        if not matching:
            continue
        # Use the first matching trade (one signal -> one position).
        trade = matching[0]
        snap["result"] = (
            "win" if trade.net_pnl > 0 else "loss" if trade.net_pnl < 0 else "breakeven"
        )
        entry = float(snap["entry"])
        stop = float(snap["stop_loss"])
        risk_dist = abs(entry - stop)
        if risk_dist > 0:
            snap["r_multiple"] = trade.net_pnl / (risk_dist * trade.quantity)
        # Infer exit reason from direction + price move.
        # LONG: exit < entry => SL (against), exit > entry => TP (favorable)
        # SHORT: exit > entry => SL (against), exit < entry => TP (favorable)
        direction = snap.get("direction", "")
        if direction == "long":
            snap["exit_reason"] = (
                "sl" if trade.exit_price <= entry else "tp"
            )
        elif direction == "short":
            snap["exit_reason"] = (
                "sl" if trade.exit_price >= entry else "tp"
            )
        else:
            snap["exit_reason"] = "manual"


def _selection_score(metrics: WindowMetrics) -> float:
    """Selection score for choosing a parameter set on VALIDATION data.

    Rewards positive expectancy, sufficient trades, and reasonable drawdown.
    Never uses TEST data.
    """
    if metrics.trade_count < 2:
        return float("-inf")
    exp = metrics.expectancy_r if metrics.expectancy_r is not None else 0.0
    dd = metrics.max_drawdown
    dd_penalty = max(0.0, 1.0 - dd * 5.0)
    trades_bonus = min(metrics.trade_count / 20.0, 1.0)
    return (exp * 2.0 + trades_bonus) * dd_penalty


def aggregate_window_results(window_results: list[WindowResult]) -> dict[str, Any]:
    """Aggregate per-window results into a compact summary dict."""
    summary = {
        "total_windows": len(window_results),
        "completed_windows": sum(1 for w in window_results if w.status == "complete"),
        "skipped_windows": sum(1 for w in window_results if w.status == "skipped"),
        "total_trades": sum(w.trade_count for w in window_results),
        "windows": [],
    }
    for w in window_results:
        test_m = w.test_metrics[0] if w.test_metrics else None
        summary["windows"].append(
            {
                "index": w.index,
                "status": w.status,
                "selected_params": w.selected_params,
                "test_trades": test_m.trade_count if test_m else 0,
                "test_expectancy_r": test_m.expectancy_r if test_m else None,
                "test_net_pnl": test_m.net_profit if test_m else 0.0,
                "test_max_drawdown": test_m.max_drawdown if test_m else 0.0,
            }
        )
    return summary