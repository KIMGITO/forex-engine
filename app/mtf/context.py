"""Per-timeframe MTF context builder.

Consumes the existing public APIs of ``app/features``, ``app/market_structure``,
``app/regime``, and ``app/news`` — it does NOT duplicate their implementations.
Every consumed event is filtered by ``available_from <= timestamp`` so the
strict causal guarantee is preserved.
"""

from typing import Any

import pandas as pd

from app._causal_index import available_count, available_prefix, build_causal_index
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
        self._regime_cache = {}
        self._structure_cache = {}

    def _structure_index(self, structure: MarketStructureResult):
        """Cached (structure, liquidity_zones, sweeps) causal-index tuple.

        The structure entry is ``(sorted_pts, keys, up_prefix, down_prefix)``
        where ``up_prefix[i]`` / ``down_prefix[i]`` are cumulative counts of
        bullish/bearish structure points in the first ``i`` sorted items —
        enabling O(log S) bias queries without a per-bar O(k) scan.
        """
        cache = self._structure_cache.get(id(structure))
        if cache is None:
            sorted_struct, keys_struct = build_causal_index(structure.structure)
            up_prefix = [0] * (len(sorted_struct) + 1)
            down_prefix = [0] * (len(sorted_struct) + 1)
            for i, p in enumerate(sorted_struct):
                t = p.structure_type.value
                up_prefix[i + 1] = up_prefix[i] + (
                    1 if t in ("higher_high", "higher_low") else 0
                )
                down_prefix[i + 1] = down_prefix[i] + (
                    1 if t in ("lower_high", "lower_low") else 0
                )
            cache = (
                (sorted_struct, keys_struct, up_prefix, down_prefix),
                build_causal_index(structure.liquidity_zones),
                build_causal_index(structure.sweeps),
            )
            self._structure_cache[id(structure)] = cache
        return cache

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
        if now is not None:
            (
                (sorted_struct, keys_struct, up_prefix, down_prefix),
                _,
                _,
            ) = self._structure_index(structure)
            n = available_count(sorted_struct, keys_struct, now)
            up = up_prefix[n]
            down = down_prefix[n]
        else:
            up = 0
            down = 0
            for p in structure.structure:
                t = p.structure_type.value
                if t in ("higher_high", "higher_low"):
                    up += 1
                elif t in ("lower_high", "lower_low"):
                    down += 1
        if up == down:
            return "neutral"
        return "bullish" if up > down else "bearish"

    def liquidity_zones_at(
        self,
        structure: MarketStructureResult | None,
        now,
    ) -> list[Any]:
        """Liquidity zones with available_from <= now (O(log L) + O(k))."""
        if structure is None or not structure.liquidity_zones:
            return []
        _, (sorted_zones, keys_zones), _ = self._structure_index(structure)
        return available_prefix(sorted_zones, keys_zones, now)

    def sweeps_at(
        self,
        structure: MarketStructureResult | None,
        now,
    ) -> list[Any]:
        """Sweeps with available_from <= now (O(log S) + O(k))."""
        if structure is None or not structure.sweeps:
            return []
        _, _, (sorted_sweeps, keys_sweeps) = self._structure_index(structure)
        return available_prefix(sorted_sweeps, keys_sweeps, now)

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
            cache = self._regime_cache.get(id(regimes))
            if cache is None:
                cache = build_causal_index(regimes)
                self._regime_cache[id(regimes)] = cache
            sorted_regimes, keys_regimes = cache
            available_regimes = available_prefix(sorted_regimes, keys_regimes, now)
            if available_regimes:
                regime = available_regimes[-1]

        # Structural bias from the structure result, strictly timestamp-filtered.
        bias = self.structural_bias(structure, now)

        liquidity_zones = self.liquidity_zones_at(structure, now)
        sweeps = self.sweeps_at(structure, now)

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
