"""Multi-timeframe research engine.

Orchestrates per-timeframe analysis (features, market structure, regime, news)
and, for each base-bar observation, builds a unified :class:`MtfContext`
with every higher-timeframe tier aligned strictly by the completed-candle
rule. Consumes existing public APIs only — no duplicated algorithms.
"""

from typing import Any

import pandas as pd

from app.market_structure.engine import MarketStructureEngine
from app.market_structure.models import MarketStructureResult
from app.mtf.alignment import classify_alignment
from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.mtf.models import MtfAlignmentState, MtfContext, TimeframeContext
from app.regime.engine import RegimeEngine
from app.regime.models import MarketRegime

__all__ = ["MtfAnalysis", "MtfEngine"]


class MtfAnalysis:
    """Precomputed analysis for a single timeframe's candles."""

    def __init__(
        self,
        timeframe: str,
        frame: pd.DataFrame,
        structure: MarketStructureResult | None,
        regimes: list[MarketRegime] | None,
    ) -> None:
        self.timeframe = timeframe
        self.frame = frame
        self.structure = structure
        self.regimes = regimes or []


class MtfEngine:
    """Produces MtfContext observations across a configurable hierarchy."""

    def __init__(
        self,
        config: MtfConfig | None = None,
        symbol: str = "EURUSD",
    ) -> None:
        self.config = config or MtfConfig()
        self.symbol = symbol

    def analyze(
        self,
        dataframes: dict[str, pd.DataFrame],
        base_timeframe: str | None = None,
        news_events: list[Any] | None = None,
    ) -> list[MtfContext]:
        """Build one MtfContext per base bar.

        Parameters
        ----------
        dataframes : {timeframe: OHLC DataFrame indexed by tz-aware opens}
            Must contain at least the base timeframe; higher timeframes optional
            (missing tiers are surfaced as ``present=False``).
        base_timeframe : optional override for the base (acting) timeframe.
        news_events : optional list of EconomicEvent (availability-filtered).

        Returns
        -------
        List[MtfContext] — one per base bar, each with ``available_from``.
        """
        if not dataframes:
            raise ValueError("dataframes must contain at least one timeframe")

        base_tf = base_timeframe or self.config.base_timeframe
        base_df = dataframes.get(base_tf)
        if base_df is None or base_df.empty:
            raise ValueError(f"base timeframe {base_tf} missing or empty")

        # Precompute per-timeframe analysis (causal by construction).
        analysis: dict[str, MtfAnalysis] = {}
        all_tfs = [base_tf] + list(self.config.higher_timeframes)
        for tf in all_tfs:
            df = dataframes.get(tf)
            if df is None or df.empty:
                analysis[tf] = MtfAnalysis(tf, pd.DataFrame(), None, [])
                continue
            df = df.sort_index()
            structure = None
            regimes: list[MarketRegime] = []
            try:
                structure = MarketStructureEngine().analyze(df, self.symbol, tf)
                regimes = RegimeEngine().analyze(
                    df, self.symbol, tf, market_structure=structure
                )
            except Exception:  # noqa: BLE001 - insufficient bars handled
                # Insufficient bars / analysis failure (e.g. <44 bars for
                # range detection) → the tier remains present for candle
                # alignment, but has NO structure/regime evidence. Data is
                # never fabricated; look-ahead discipline is preserved.
                structure = None
                regimes = []
            analysis[tf] = MtfAnalysis(tf, df, structure, regimes)

        builder = MtfContextBuilder(self.config, self.symbol)

        contexts: list[MtfContext] = []
        base_sorted = base_df.sort_index()
        for i, ts in enumerate(base_sorted.index):
            now = ts
            hierarchy: list[TimeframeContext] = []

            # Build each higher-timeframe tier, strictly aligned.
            available_count = 0
            for tf in self.config.higher_timeframes:
                ana = analysis.get(tf, MtfAnalysis(tf, pd.DataFrame(), None, []))
                tier = builder.build(
                    timeframe=tf,
                    timestamp=now,
                    frame=ana.frame,
                    features=None,  # optional features; not required for context
                    structure=ana.structure,
                    regimes=ana.regimes,
                    news_events=news_events,
                )
                hierarchy.append(tier)
                if tier.present:
                    available_count += 1

            # The base tier (current bar) context, present by default.
            base_ana = analysis[base_tf]
            base_tier = TimeframeContext(
                timeframe=base_tf,
                timestamp=now,
                candle_open=now,
                candle_close=now,
                trend_state=(
                    base_ana.regimes[-1].trend_state.value
                    if base_ana.regimes
                    else None
                ),
                volatility_state=(
                    base_ana.regimes[-1].volatility_state.value
                    if base_ana.regimes
                    else None
                ),
                market_state=(
                    base_ana.regimes[-1].market_state.value if base_ana.regimes else None
                ),
                structural_bias=builder.structural_bias(base_ana.structure, now),
                liquidity_zones=(
                    [
                        z
                        for z in base_ana.structure.liquidity_zones
                        if z.available_from <= now
                    ]
                    if base_ana.structure
                    else []
                ),
                sweeps=(
                    [s for s in base_ana.structure.sweeps if s.available_from <= now]
                    if base_ana.structure
                    else []
                ),
                news_risk_max=builder._news_risk_max(news_events or [], now),
                present=True,
                available_from=now,
            )

            # Base direction (from regime trend if known; else None).
            base_dir = None
            if base_tier.trend_state == "bullish":
                base_dir = "long"
            elif base_tier.trend_state == "bearish":
                base_dir = "short"

            if base_dir is None:
                alignment = MtfAlignmentState.UNKNOWN
                reasons = ["base regime direction unknown"]
            else:
                alignment, reasons, _ = classify_alignment(
                    base_dir,
                    hierarchy,
                    min_aligned=self.config.min_aligned,
                    require_no_htf_conflict=self.config.require_no_htf_conflict,
                )

            # Aggregated news-risk across tiers.
            news_max = (
                max(
                    (t.news_risk_max for t in [base_tier] + hierarchy if t.news_risk_max),
                    key=lambda v: {"low": 1, "medium": 2, "high": 3}.get(v, 0),
                    default=None,
                )
            )

            contexts.append(
                MtfContext(
                    symbol=self.symbol,
                    base_timeframe=base_tf,
                    timestamp=now,
                    hierarchy=[base_tier] + hierarchy,
                    alignment=alignment,
                    alignment_reasons=reasons,
                    min_aligned=float(self.config.min_aligned),
                    news_risk_max=news_max,
                    metadata={
                        "available_htf_tiers": available_count,
                        "hierarchy": list(self.config.higher_timeframes),
                    },
                    available_from=now,
                )
            )

        return contexts