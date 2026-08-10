"""Historical signal scanner and strategy comparison runner.

Pipeline:
  dataset → features → structure → news → regime → strategy → signals

Every signal is produced causally: each bar's StrategyContext only exposes
information whose ``available_from`` is at or before that bar.
"""

from dataclasses import dataclass, field

import pandas as pd

from app.features import FeatureEngine
from app.market_structure.engine import MarketStructureEngine
from app.regime import RegimeConfig, RegimeEngine
from app.strategy.base import Strategy
from app.strategy.config import StrategyConfig
from app.strategy.context import StrategyContext
from app.strategy.models import Signal, SignalDirection, SignalStatus

__all__ = ["HistoricalSignalScanner", "SignalScanResult", "StrategyComparison"]


@dataclass
class SignalScanResult:
    """Output of scanning a dataset with a single strategy."""

    strategy: str
    symbol: str
    timeframe: str
    bars_processed: int
    signals: list[Signal] = field(default_factory=list)

    def long_count(self) -> int:
        return sum(1 for s in self.signals if s.direction == SignalDirection.LONG)

    def short_count(self) -> int:
        return sum(1 for s in self.signals if s.direction == SignalDirection.SHORT)


@dataclass
class StrategyComparison:
    """Comparison report across strategies (statistics only, no P&L claims)."""

    results: list[SignalScanResult] = field(default_factory=list)

    def summary(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.results:
            out[r.strategy] = {
                "signals": len(r.signals),
                "long": r.long_count(),
                "short": r.short_count(),
                "frequency_per_bar": (
                    len(r.signals) / r.bars_processed if r.bars_processed else 0.0
                ),
                "avg_rr": (
                    sum(s.risk_reward_ratio for s in r.signals) / len(r.signals)
                    if r.signals
                    else 0.0
                ),
                "invalidated": sum(1 for s in r.signals if s.status == SignalStatus.INVALIDATED),
            }
        return out


class HistoricalSignalScanner:
    """Sequentially scans historical data, emitting causal signals."""

    def __init__(
        self,
        strategy_config: StrategyConfig | None = None,
        regime_config: RegimeConfig | None = None,
    ) -> None:
        self.strategy_config = strategy_config or StrategyConfig()
        self.regime_config = regime_config or RegimeConfig()

    def scan(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame | None = None,
        news_events: list | None = None,
        mtf_contexts: list | None = None,
    ) -> SignalScanResult:
        """Process bars sequentially, collecting signals.

        Parameters are the causal inputs; the scanner never exposes future
        rows to the strategy (each StrategyContext is capped at its bar).

        ``mtf_contexts`` (optional) provides one MtfContext per bar (ordered
        same as the frame index). When provided, each emitted Signal's
        metadata is enriched with MTF evidence. When omitted, the scanner
        behaves exactly as before (no MTF wiring).
        """
        sorted_data = data.sort_index()
        mtf_by_ts = None
        if mtf_contexts is not None:
            mtf_by_ts = {
                _normalize_ts(mtf.timestamp): mtf
                for mtf in mtf_contexts
            }

        # Precompute trailing-window features once (causal by construction),
        # then slice per bar. The scanner still only hands the strategy the
        # slice at the current index — never the full frame.
        if features is None:
            features = FeatureEngine().calculate(
                sorted_data, features=["atr", "rsi"]
            )

        # Market structure (available_from filters applied per-bar by context).
        structure = MarketStructureEngine().analyze(sorted_data, symbol, timeframe)

        # Regime observations (available_from = bar timestamp).
        regimes = RegimeEngine(self.regime_config).analyze(
            sorted_data, symbol, timeframe, market_structure=structure
        )

        signals: list[Signal] = []
        for i, ts in enumerate(sorted_data.index):
            mtf_ctx = mtf_by_ts.get(_normalize_ts(ts)) if mtf_by_ts is not None else None
            ctx = StrategyContext(
                symbol=symbol,
                timeframe=timeframe,
                now=ts,
                frame=sorted_data,
                current_index=i,
                features=features,
                structure=structure,
                news_events=news_events,
                regime_observations=regimes,
                config=strategy.config,
                mtf=mtf_ctx,
            )
            signal = strategy.evaluate(ctx)
            if signal is not None:
                # Enforce MTF gate if the strategy has MTF enabled.
                if strategy.config.mtf_enabled and mtf_ctx is None:
                    continue
                if mtf_ctx is not None:
                    base_dir = (
                        "long"
                        if signal.direction == SignalDirection.LONG
                        else "short"
                    )
                    passed, _reasons = strategy.mtf_gates_pass(base_dir, mtf_ctx)
                    if not passed:
                        continue
                    # Embed MTF evidence into signal metadata (serializable).
                    signal = _attach_mtf_evidence(signal, mtf_ctx)
                signals.append(signal)
                strategy.record_signal(signal, ts)

        return SignalScanResult(
            strategy=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            bars_processed=len(sorted_data),
            signals=signals,
        )


def _normalize_ts(value):
    """Normalize a timestamp to a hashable UTC key (pandas-safe)."""
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _attach_mtf_evidence(signal: Signal, mtf_ctx) -> Signal:
    """Return a copy of ``signal`` with MTF evidence in metadata (additive)."""
    evidence = {
        "alignment": mtf_ctx.alignment.value,
        "alignment_reasons": list(mtf_ctx.alignment_reasons),
        "available_htf_tiers": mtf_ctx.metadata.get("available_htf_tiers", 0),
        "tiers": [
            {
                "timeframe": t.timeframe,
                "trend": t.trend_state,
                "volatility": t.volatility_state,
                "market_state": t.market_state,
                "structural_bias": t.structural_bias,
                "present": t.present,
                "available_from": t.available_from.isoformat() if t.available_from else None,
            }
            for t in mtf_ctx.hierarchy
        ],
    }
    metadata = dict(signal.metadata)
    metadata["mtf"] = evidence
    return signal.model_copy(update={"metadata": metadata})


def compare_strategies(
    data: pd.DataFrame,
    strategies: list[Strategy],
    symbol: str,
    timeframe: str,
    news_events: list | None = None,
    strategy_config: StrategyConfig | None = None,
) -> StrategyComparison:
    """Run multiple strategies over the same dataset and compare statistics.

    Statistics are signal counts / frequencies / R:R — NOT profitability.
    Profitability must come from the backtest engine.
    """
    comparison = StrategyComparison()
    for strat in strategies:
        scanner = HistoricalSignalScanner(
            strategy_config=strategy_config or strat.config
        )
        result = scanner.scan(data, strat, symbol, timeframe, news_events=news_events)
        comparison.results.append(result)
    return comparison