"""Per-timeframe MTF context builder.

Consumes the existing public APIs of ``app/features``, ``app/market_structure``,
``app/regime``, and ``app/news`` — it does NOT duplicate their implementations.
Every consumed event is filtered by ``available_from <= timestamp`` so the
strict causal guarantee is preserved.
"""

from typing import Any

import pandas as pd

from app.market_structure.models import MarketStructureResult
from app.mtf.availability import latest_completed_candle_open, resolve_window
from app.mtf.config import MtfConfig
from app.mtf.models import TimeframeContext
from app.regime.models import MarketRegime

__all__ = ["MtfContextBuilder"]


class MtfContextBuilder:
    """Builds a single TimeframeContext for one timeframe at a timestamp."""

    def __init__(self, config: MtfConfig, symbol: str) -> None:
        self.config = config
        self.symbol = symbol

    def structural_bias(
        self,
        structure: MarketStructureResult | None,
        now=None,
    ) -> str | None:
        """Structural bias from structure points with available_from <= now.

        STRICT CAUSALITY: structure points whose ``available_from`` is after
        ``now`` are EXCLUDED. Without this, a precomputed structure result
        could leak future swings into earlier observations — a look-ahead
        violation.
        """
        if structure is None or not structure.structure:
            return None
        up = 0
        down = 0
        for p in structure.structure:
            if now is not None and p.available_from is not None and p.available_from > now:
                continue  # future event — not yet legal
            t = p.structure_type.value
            if t in ("higher_high", "higher_low"):
                up += 1
            elif t in ("lower_high", "lower_low"):
                down += 1
        if up == down:
            return "neutral"
        return "bullish" if up > down else "bearish"

    def _news_risk_max(
        self, events: list[Any], timestamp: pd.Timestamp
    ) -> str | None:
        """Highest active news-risk state among events available at now."""
        rank = {"low": 1, "medium": 2, "high": 3}
        best_rank = 0
        best: str | None = None
        for e in events:
            av = getattr(e, "available_from", None)
            if av is not None and av > timestamp:
                continue  # not yet available
            imp = getattr(e, "importance", None)
            if imp is None:
                continue
            r = rank.get(imp.value, 0)
            if r > best_rank:
                best_rank = r
                best = imp.value
        return best

    def build(
        self,
        timeframe: str,
        timestamp,
        frame: pd.DataFrame,
        features: pd.DataFrame | None,
        structure: MarketStructureResult | None,
        regimes: list[MarketRegime] | None,
        news_events: list[Any] | None,
    ) -> TimeframeContext:
        """Build the context for ``timeframe`` at ``timestamp``.

        ``timestamp`` is the observation moment (base bar open/close). Only
        the LAST FULLY COMPLETED higher-timeframe candle is used (unless
        ``timeframe`` == the base timeframe in which case the current bar is
        the observation itself).
        """
        now = timestamp
        # Determine the completed candle open for this timeframe.
        candle_open = latest_completed_candle_open(timeframe, now, frame, self.config)
        if candle_open is None:
            # No completed candle (e.g. before any data). Report present=False.
            return TimeframeContext(
                timeframe=timeframe,
                timestamp=now,
                trend_state=None,
                volatility_state=None,
                market_state=None,
                structural_bias=None,
                present=False,
                available_from=now,
            )
        candle_close = resolve_window(timeframe, now, self.config).available_from

        # Latest regime whose available_from <= now.
        regime: MarketRegime | None = None
        if regimes:
            for r in regime_observe_sorted(regimes):
                if r.available_from <= now:
                    regime = r

        # Structural bias from the structure result, strictly timestamp-filtered.
        bias = self.structural_bias(structure, now)

        liquidity_zones = (
            [z for z in structure.liquidity_zones if z.available_from <= now]
            if structure
            else []
        )
        sweeps = (
            [s for s in structure.sweeps if s.available_from <= now]
            if structure
            else []
        )

        return TimeframeContext(
            timeframe=timeframe,
            timestamp=now,
            candle_open=candle_open,
            candle_close=candle_close,
            trend_state=regime.trend_state.value if regime else None,
            volatility_state=regime.volatility_state.value if regime else None,
            market_state=regime.market_state.value if regime else None,
            structural_bias=bias,
            liquidity_zones=liquidity_zones,
            sweeps=sweeps,
            news_risk_max=self._news_risk_max(news_events or [], now),
            present=True,
            available_from=candle_close,
        )


def regime_observe_sorted(regimes: list[MarketRegime]) -> list[MarketRegime]:
    """Causal order: regimes sorted by available_from (earliest first)."""
    return sorted(regimes, key=lambda r: r.available_from)