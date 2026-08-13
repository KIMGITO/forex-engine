"""Step 13 — Market Event Research & Candidate Generation Engine configuration.

The redesigned Step 13 produces COMPACT COLUMNAR EVENT DATASETS and CANDIDATE
records that Step 13B consumes for walk-forward validation. It does NOT decide
profitability. It does NOT duplicate Step 13B's strategy-research
responsibilities.

The pipeline is bounded-memory:
* one symbol, one timeframe, one chunk at a time
* pre-computation RSS guard (MemAvailable)
* HTF data causally clipped to the base window
* atomic parquet writes + resumable chunk state
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step13Config:
    """Configuration for the Step 13 event/candidate engine."""

    # ── Scope ───────────────────────────────────────────────────────────────
    symbols: tuple = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
    timeframes: tuple = ("M15", "H1")
    storage_root: str = "data/processed"

    # ── Bounded processing ──────────────────────────────────────────────────
    chunk_size: int = 5000  # base bars per chunk
    max_bars: int = 0  # 0 = all (dev-safe default in CLI)
    overlap_bars: int = 200  # warm-up overlap between successive chunks

    # ── Candidate generation controls ───────────────────────────────────────
    # Lookback (in base bars) after a sweep for a confirming displacement.
    sweep_displacement_lookback: int = 5
    # Require displacement classification at least this strong (large/extreme).
    min_displacement_class: tuple = ("large", "extreme")
    # HTF alignment is a HYPOTHESIS CONDITION, not a global hard filter.
    # Default False so the same event population can be compared WITH vs
    # WITHOUT HTF alignment (discovery architecture).
    require_htf_alignment: bool = False
    # Max bars after candidate to compute labels (MFE/MAE/TP/SL).
    label_lookback_bars: int = 100

    # ── Trading costs (research model — explicit, config-hashed) ────────────
    spread_pips: float = 0.8
    slippage_pips: float = 0.0
    commission_per_lot: float = 0.0  # account currency per standard lot (100k)

    # ── Discovery controls ──────────────────────────────────────────────────
    min_sample_size: int = 30  # minimum events for a discovery candidate
    max_hypotheses: int = 200  # hard cap on generated hypotheses

    # ── MTF hierarchy (same as MtfConfig default but explicit) ──────────────
    htf_timeframes: tuple = ("1h", "4h", "1d")

    # ── Memory guard ────────────────────────────────────────────────────────
    rss_limit_mb: float = 2500.0  # hard limit before heavy precompute
    min_mem_available_mb: float = 256.0

    # ── Output ──────────────────────────────────────────────────────────────
    output_root: str = "research/results/step13"

    # ── Engine versions (bump when detection semantics change) ──────────────
    engine_version: str = "13.2.0"

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "storage_root": self.storage_root,
            "chunk_size": self.chunk_size,
            "max_bars": self.max_bars,
            "overlap_bars": self.overlap_bars,
            "sweep_displacement_lookback": self.sweep_displacement_lookback,
            "min_displacement_class": list(self.min_displacement_class),
            "require_htf_alignment": self.require_htf_alignment,
            "label_lookback_bars": self.label_lookback_bars,
            "spread_pips": self.spread_pips,
            "slippage_pips": self.slippage_pips,
            "commission_per_lot": self.commission_per_lot,
            "min_sample_size": self.min_sample_size,
            "max_hypotheses": self.max_hypotheses,
            "htf_timeframes": list(self.htf_timeframes),
            "rss_limit_mb": self.rss_limit_mb,
            "min_mem_available_mb": self.min_mem_available_mb,
            "output_root": self.output_root,
            "engine_version": self.engine_version,
        }

    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()