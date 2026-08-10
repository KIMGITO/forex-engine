"""Restricted causal strategy context.

Strategies must never see future data. This context exposes only information
whose ``available_from`` is at or before the current timestamp. It reuses the
same causal slices already proven by the backtest layer (Step 8) — the strategy
context is intentionally identical in discipline to ``BacktestContext``.
"""

from typing import Any

import pandas as pd

from app.market_structure.models import MarketStructureResult
from app.regime.models import MarketRegime
from app.strategy.config import StrategyConfig


class StrategyContext:
    """Causal, timestamp-restricted view consumed by strategies."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        now: pd.Timestamp,
        frame: pd.DataFrame,
        current_index: int,
        features: pd.DataFrame | None = None,
        structure: MarketStructureResult | None = None,
        news_events: list[Any] | None = None,
        regime_observations: list[MarketRegime] | None = None,
        config: StrategyConfig | None = None,
        portfolio: Any = None,
        mtf: Any | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.now = now
        self._frame = frame
        self._i = current_index
        self._features = features
        self._structure = structure
        self._news_events = news_events
        self._regime_observations = regime_observations
        self.config = config or StrategyConfig()
        self._portfolio = portfolio
        self._mtf = mtf

    # ── current bar ───────────────────────────────────────────────────────────

    def current_candle(self) -> pd.Series:
        """Current bar only. Never a future bar."""
        return self._frame.iloc[self._i]

    def current_features(self) -> pd.Series | None:
        if self._features is None:
            return None
        return self._features.iloc[self._i]

    def history(self, bars: int | None = None) -> pd.DataFrame:
        """History + current candle, capped at the current bar (no future)."""
        if bars is None:
            return self._frame.iloc[: self._i + 1]
        return self._frame.iloc[max(0, self._i + 1 - bars) : self._i + 1]

    def features_history(self) -> pd.DataFrame | None:
        if self._features is None:
            return None
        return self._features.iloc[: self._i + 1]

    # ── structure (available <= now) ──────────────────────────────────────────

    def structure_points(self) -> list[Any]:
        """Structure points whose available_from <= now (causal)."""
        if self._structure is None:
            return []
        return [
            p
            for p in self._structure.structure
            if p.available_from is None or p.available_from <= self.now
        ]

    def liquidity_zones(self) -> list[Any]:
        if self._structure is None:
            return []
        return [
            z
            for z in self._structure.liquidity_zones
            if z.available_from is None or z.available_from <= self.now
        ]

    def sweeps(self) -> list[Any]:
        if self._structure is None:
            return []
        return [
            s
            for s in self._structure.sweeps
            if s.available_from is None or s.available_from <= self.now
        ]

    def displacement_events(self) -> list[Any]:
        if self._structure is None:
            return []
        return [
            d
            for d in self._structure.displacement
            if d.available_from is None or d.available_from <= self.now
        ]

    def structure_breaks(self) -> list[Any]:
        if self._structure is None:
            return []
        return [
            b
            for b in self._structure.breaks
            if b.available_from is None or b.available_from <= self.now
        ]

    def active_ranges(self) -> list[Any]:
        if self._structure is None:
            return []
        return [
            r
            for r in self._structure.ranges
            if r.available_from is None or r.available_from <= self.now
        ]

    # ── news (available <= now) ───────────────────────────────────────────────

    def news_available(self) -> list[Any]:
        if not self._news_events:
            return []
        return [
            e
            for e in self._news_events
            if getattr(e, "available_from", None) is None
            or e.available_from <= self.now
        ]

    def maximum_news_risk(self) -> str | None:
        """Highest active news-risk state among events available at now.

        Returns None when there is no news risk (calm market).
        """
        events = self.news_available()
        if not events:
            return None
        rank = {"low": 1, "medium": 2, "high": 3}
        best = None
        best_rank = 0
        for e in events:
            imp = getattr(e, "importance", None)
            if imp is None:
                continue
            r = rank.get(imp.value, 0)
            if r > best_rank:
                best_rank = r
                best = imp.value
        return best

    # ── regime (available <= now) ─────────────────────────────────────────────

    def regime_available(self) -> list[MarketRegime]:
        if not self._regime_observations:
            return []
        return [
            r for r in self._regime_observations if r.available_from <= self.now
        ]

    def latest_regime(self) -> MarketRegime | None:
        regs = self.regime_available()
        return regs[-1] if regs else None

    # ── portfolio (current values only) ───────────────────────────────────────

    def equity(self, mid: float) -> float:
        if self._portfolio is None:
            return 0.0
        return float(self._portfolio.equity(mid))

    def open_positions(self) -> list[Any]:
        if self._portfolio is None:
            return []
        return list(self._portfolio.positions.values())

    # ── multi-timeframe (MTF) context ─────────────────────────────────────────

    def mtf_context(self):
        """Return the causal MTF context for this bar, or None when MTF is not
        enabled/provided. The MTF context carries its own ``available_from``
        invariant: a strategy may only ever act on MTF tiers whose candle was
        fully completed before this bar."""
        return self._mtf
