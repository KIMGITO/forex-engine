"""RegimeEngine: sequential, causal market-regime classification.

Pipeline:
  Market Data
  -> Feature Engine (consumed via app.features)
  -> Market Structure (consumed via app.market_structure)
  -> News Context (optional, via app.news)
  -> RegimeEngine
  -> list[MarketRegime]

Every regime observation is computed causally: only information available at or
before the bar timestamp is used. Look-ahead regression tests enforce this.
"""


import pandas as pd

from app.market_structure.models import MarketStructureResult
from app.news.models import PairRiskContext
from app.regime.classifier import classify_regime
from app.regime.config import RegimeConfig
from app.regime.models import MarketRegime, NewsRiskState
from app.regime.structure import (
    build_structure_query_cache,
    range_active_at,
    structure_bias_at,
)
from app.regime.trend import classify_trend_series
from app.regime.volatility import classify_volatility_series

__all__ = ["RegimeEngine"]


class RegimeEngine:
    """Central orchestrator for causal regime detection."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def analyze(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_structure: MarketStructureResult | None = None,
        news_context: PairRiskContext | None = None,
    ) -> list[MarketRegime]:
        """Compute one MarketRegime per bar, causally.

        Parameters
        ----------
        data : pd.DataFrame
            OHLC indexed by tz-aware timestamps (columns: open/high/low/close).
        symbol, timeframe : str
            Metadata propagated to every regime.
        market_structure : optional MarketStructureResult
            Precomputed structure result. Its events are still filtered by
            ``available_from`` per bar to prevent look-ahead.
        news_context : optional PairRiskContext
            Event-risk metadata (does not drive direction).

        Returns
        -------
        List[MarketRegime]
            One observation per bar, each with ``available_from == bar.timestamp``.
        """
        sorted_data = data.sort_index()

        trend_series = classify_trend_series(sorted_data, self.config)
        vol_states, vol_ratios = classify_volatility_series(sorted_data, self.config)

        # Transition-vol ratio: ATR/SMA(ATR) (causal), used to detect expansion.
        vol_ratio_series = _transition_ratio_series(sorted_data, self.config)

        struct_cache = build_structure_query_cache(market_structure)

        regimes: list[MarketRegime] = []
        for i, ts in enumerate(sorted_data.index):
            trend = trend_series.iloc[i]
            vol = vol_states.iloc[i]

            struct_bias, struct_count = structure_bias_at(
                struct_cache, self.config.structure_lookback, ts
            )
            range_active = range_active_at(struct_cache, ts)

            news_risk = self._news_risk(news_context, ts)

            trans_ratio = vol_ratio_series.iloc[i]

            metrics = {
                "atr_ratio": float(vol_ratios.iloc[i]) if pd.notna(vol_ratios.iloc[i]) else float("nan"),
                "vol_ratio": float(trans_ratio) if pd.notna(trans_ratio) else float("nan"),
            }

            regime = classify_regime(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts.to_pydatetime(),
                trend=trend,
                volatility=vol,
                structure_bias=struct_bias,
                structure_count=struct_count,
                range_active=range_active,
                transition_vol_ratio=float(trans_ratio) if pd.notna(trans_ratio) else float("nan"),
                news_risk=news_risk,
                metrics=metrics,
                min_structure_points=self.config.min_structure_points,
            )
            regimes.append(regime)

        return regimes

    @staticmethod
    def _news_risk(ctx: PairRiskContext | None, ts) -> NewsRiskState:
        """Map news context to metadata state at a timestamp (never directional)."""
        if ctx is None:
            return NewsRiskState.UNKNOWN
        if not ctx.active_events:
            return NewsRiskState.CALM
        highest = ctx.highest_active_importance
        from app.news.models import EventImportance

        if highest == EventImportance.HIGH:
            return NewsRiskState.ACTIVE_HIGH
        if highest == EventImportance.MEDIUM:
            return NewsRiskState.ACTIVE_MEDIUM
        return NewsRiskState.CALM


def _transition_ratio_series(data: pd.DataFrame, config: RegimeConfig) -> pd.Series:
    """ATR / SMA(ATR, range_window) — causal volatility-expansion indicator."""
    from app.features.trend import sma
    from app.features.volatility import atr

    a = atr(data, window=config.atr_window).sort_index()
    atr_sma = sma(
        data.assign(_atr=a),
        period=config.range_window,
        price_col="_atr",
    ).sort_index()
    return a / atr_sma
