"""Bounded MTF context extraction for Step 13.

Uses the authoritative ``MtfEngine.analyze_chunks`` with ``clip_htf=True`` so
HTF frames are causally windowed to the base period (never the full dataset).
Only the current chunk's contexts are held in memory; each is flattened to a
compact row for ``mtf_context.parquet``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.mtf import MtfConfig, MtfEngine
from app.research.step13.schema import MTF_CONTEXT_COLUMNS


class MtfExtractor:
    """Streams MTF context rows for one symbol/base-timeframe."""

    def __init__(
        self,
        symbol: str,
        base_timeframe: str,
        htf_timeframes: tuple = ("1h", "4h", "1d"),
        chunk_size: int = 5000,
        rss_limit_mb: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.base_timeframe = base_timeframe
        self.config = MtfConfig(
            base_timeframe=base_timeframe,
            higher_timeframes=tuple(htf_timeframes),
        )
        self.chunk_size = chunk_size
        self.rss_limit_mb = rss_limit_mb

    def rows_for_chunk(
        self,
        base_df: pd.DataFrame,
        htf_frames: dict[str, pd.DataFrame],
        base_start_idx: int,
        base_end_idx: int,
    ) -> list[dict[str, Any]]:
        """Return compact MTF rows for ``base_df[base_start_idx:base_end_idx]``.

        ``htf_frames`` maps native HTF timeframe (e.g. "1h") to its DataFrame.
        The engine internally clips HTF frames via its causal lookback.
        """
        base_sorted = base_df.sort_index()
        chunk_df = base_sorted.iloc[base_start_idx:base_end_idx]
        if chunk_df.empty:
            return []

        dataframes = {self.base_timeframe: base_sorted}
        for tf, df in htf_frames.items():
            if df is not None and not df.empty:
                dataframes[tf] = df.sort_index()

        engine = MtfEngine(self.config, self.symbol)
        rows: list[dict[str, Any]] = []
        # Only the bars in [base_start_idx, base_end_idx) are requested.
        # analyze_chunks processes chunk boundaries; we take the intersection.
        for start, end, contexts in engine.analyze_chunks(
            dataframes,
            self.base_timeframe,
            chunk_size=self.chunk_size,
            rss_limit_mb=self.rss_limit_mb,
            clip_htf=True,
        ):
            # Intersect with requested range.
            for ctx in contexts:
                ts = ctx.timestamp
                if not (base_sorted.index[base_start_idx] <= ts <= base_sorted.index[min(base_end_idx - 1, len(base_sorted) - 1)]):
                    continue
                for tier in ctx.hierarchy[1:]:  # skip base tier
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": self.symbol,
                            "base_timeframe": self.base_timeframe,
                            "htf_timeframe": tier.timeframe,
                            "htf_tier": tier.timeframe,
                            "candle_open": tier.candle_open.isoformat() if tier.candle_open else None,
                            "candle_close": tier.candle_close.isoformat() if tier.candle_close else None,
                            "htf_trend_state": tier.trend_state,
                            "htf_volatility_state": tier.volatility_state,
                            "htf_market_state": tier.market_state,
                            "htf_structural_bias": tier.structural_bias,
                            "available_from": tier.available_from.isoformat() if tier.available_from else ts.isoformat(),
                        }
                    )
        return rows


def mtf_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert MTF rows to a stable DataFrame."""
    if not rows:
        return pd.DataFrame(columns=MTF_CONTEXT_COLUMNS)
    df = pd.DataFrame(rows)
    for c in MTF_CONTEXT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[MTF_CONTEXT_COLUMNS]