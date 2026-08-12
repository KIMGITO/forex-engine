"""Restricted causal strategy context.

Strategies must never see future data. This context exposes only information
whose ``available_from`` is at or before the current timestamp. It reuses the
same causal slices already proven by the backtest layer (Step 8) — the strategy
context is intentionally identical in discipline to ``BacktestContext``.
"""

from typing import Any

import pandas as pd

from app._causal_index import available_prefix, build_causal_index
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
        _causal_bundle: dict[str, tuple[list[Any], list[Any]]] | None = None,
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

        # Precompute sorted-by-available_from lists and keys once, so per-bar
        # queries are O(log n) via binary search instead of O(n) full scans.
        # The scanner/backtester build the index once and share it here via
        # ``_causal_bundle`` (avoids re-sorting per bar). Direct construction
        # (tests) falls back to building the index here.
        if _causal_bundle is not None:
            self._structure_sorted, self._structure_keys = _causal_bundle["structure"]
            self._liquidity_sorted, self._liquidity_keys = _causal_bundle["liquidity"]
            self._sweeps_sorted, self._sweeps_keys = _causal_bundle["sweeps"]
            self._displacement_sorted, self._displacement_keys = _causal_bundle["displacement"]
            self._breaks_sorted, self._breaks_keys = _causal_bundle["breaks"]
            self._ranges_sorted, self._ranges_keys = _causal_bundle["ranges"]
            self._regime_sorted, self._regime_keys = _causal_bundle["regime"]
            self._news_sorted, self._news_keys = _causal_bundle["news"]
        else:
            self._structure_sorted, self._structure_keys = build_causal_index(
                structure.structure if structure else []
            )
            self._liquidity_sorted, self._liquidity_keys = build_causal_index(
                structure.liquidity_zones if structure else []
            )
            self._sweeps_sorted, self._sweeps_keys = build_causal_index(
                structure.sweeps if structure else []
            )
            self._displacement_sorted, self._displacement_keys = build_causal_index(
                structure.displacement if structure else []
            )
            self._breaks_sorted, self._breaks_keys = build_causal_index(
                structure.breaks if structure else []
            )
            self._ranges_sorted, self._ranges_keys = build_causal_index(
                structure.ranges if structure else []
            )
            self._regime_sorted, self._regime_keys = build_causal_index(
                regime_observations or []
            )
            self._news_sorted, self._news_keys = build_causal_index(
                news_events or []
            )

    def _available(self, items: list[Any], keys: list[Any]) -> list[Any]:
        """Return the causally-available prefix (available_from <= now)."""
        return available_prefix(items, keys, self.now)

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
        return self._available(self._structure_sorted, self._structure_keys)

    def liquidity_zones(self) -> list[Any]:
        return self._available(self._liquidity_sorted, self._liquidity_keys)

    def sweeps(self) -> list[Any]:
        return self._available(self._sweeps_sorted, self._sweeps_keys)

    def displacement_events(self) -> list[Any]:
        return self._available(self._displacement_sorted, self._displacement_keys)

    def structure_breaks(self) -> list[Any]:
        return self._available(self._breaks_sorted, self._breaks_keys)

    def active_ranges(self) -> list[Any]:
        return self._available(self._ranges_sorted, self._ranges_keys)

    # ── news (available <= now) ───────────────────────────────────────────────

    def news_available(self) -> list[Any]:
        return self._available(self._news_sorted, self._news_keys)

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
        return self._available(self._regime_sorted, self._regime_keys)

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