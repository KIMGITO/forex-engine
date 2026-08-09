"""Central market-structure engine.

The engine processes validated OHLC data and returns a typed
:class:`MarketStructureResult` containing swings, structure, breaks, liquidity
zones, sweeps, displacement, and ranges.

This is a research/analysis layer. It describes the market; it does **not**
generate trading signals, entries, or risk decisions.
"""

from dataclasses import dataclass

import pandas as pd

from app.market_structure.displacement import compute_displacement
from app.market_structure.errors import MarketStructureError
from app.market_structure.liquidity import detect_liquidity_zones, detect_sweeps
from app.market_structure.models import MarketStructureResult
from app.market_structure.ranges import detect_ranges
from app.market_structure.structure import build_structure, detect_breaks
from app.market_structure.swings import detect_swings


@dataclass(frozen=True)
class MarketStructureConfig:
    """Configuration for the market-structure engine."""

    # Swing detection
    swing_left: int = 3
    swing_right: int = 3
    # Break detection
    confirm_bars: int = 2
    min_move_pct: float = 0.0
    # Liquidity zones
    tolerance_pct: float = 0.05
    min_swings: int = 2
    # Sweeps
    sweep_bars: int = 3
    # Displacement
    atr_window: int = 14
    p_extreme: float = 95.0
    p_large: float = 80.0
    p_small: float = 20.0
    # Ranges
    compression_threshold: float = 0.85
    range_window: int = 30
    min_range_bars: int = 10


class MarketStructureEngine:
    """Central orchestrator for market-structure analysis."""

    def __init__(self, config: MarketStructureConfig | None = None) -> None:
        self.config = config or MarketStructureConfig()

    def analyze(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> MarketStructureResult:
        """Analyze validated OHLC data and return structured market context.

        Parameters
        ----------
        data : pd.DataFrame
            OHLC data indexed by UTC timestamps. Must contain ``open``,
            ``high``, ``low``, ``close``.
        symbol, timeframe : str
            Symbol (e.g. EURUSD) and timeframe (e.g. 1h) metadata propagated to
            every event.

        Returns
        -------
        MarketStructureResult
            Typed aggregate of all detected events.
        """
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise MarketStructureError(f"Missing required OHLC columns: {missing}")

        cfg = self.config

        swings = detect_swings(
            data,
            symbol=symbol,
            timeframe=timeframe,
            left=cfg.swing_left,
            right=cfg.swing_right,
        )

        structure = build_structure(swings, symbol, timeframe)

        breaks = detect_breaks(
            data,
            swings,
            symbol=symbol,
            timeframe=timeframe,
            confirm_bars=cfg.confirm_bars,
            min_move_pct=cfg.min_move_pct,
        )

        zones = detect_liquidity_zones(
            swings,
            symbol=symbol,
            timeframe=timeframe,
            tolerance_pct=cfg.tolerance_pct,
            min_swings=cfg.min_swings,
        )

        sweeps = detect_sweeps(
            data,
            zones,
            symbol=symbol,
            timeframe=timeframe,
            sweep_bars=cfg.sweep_bars,
        )

        displacement = compute_displacement(
            data,
            symbol=symbol,
            timeframe=timeframe,
            atr_window=cfg.atr_window,
            p_extreme=cfg.p_extreme,
            p_large=cfg.p_large,
            p_small=cfg.p_small,
        )

        ranges = detect_ranges(
            data,
            symbol=symbol,
            timeframe=timeframe,
            atr_window=cfg.atr_window,
            compression_threshold=cfg.compression_threshold,
            range_window=cfg.range_window,
            min_range_bars=cfg.min_range_bars,
        )

        return MarketStructureResult(
            symbol=symbol,
            timeframe=timeframe,
            swings=swings,
            structure=structure,
            breaks=breaks,
            liquidity_zones=zones,
            sweeps=sweeps,
            displacement=displacement,
            ranges=ranges,
        )