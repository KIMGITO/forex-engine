"""Multi-timeframe research engine.

Orchestrates per-timeframe analysis (features, market structure, regime, news)
and, for each base-bar observation, builds a unified :class:`MtfContext`
with every higher-timeframe tier aligned strictly by the completed-candle
rule. Consumes existing public APIs only — no duplicated algorithms.

Memory architecture
-------------------
Production datasets (e.g. 196K M15 bars x 4 tiers) previously accumulated
every ``MtfContext`` in one Python list — multi-GB RSS and OOM on
memory-constrained hosts. The engine now precomputes all analytical state ONCE
(immutable for the run) and builds contexts per-bar via a single shared
``_build_contexts`` method. Monolithic :meth:`analyze` remains for small
datasets / tests; :meth:`analyze_chunks` yields ``(start, end, contexts)``
batches so memory stays bounded as bar count grows (identical semantics).
"""

from typing import Any

import pandas as pd

from app.market_structure.engine import MarketStructureEngine
from app.market_structure.models import MarketStructureResult
from app.mtf.alignment import classify_alignment
from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.mtf.errors import MtfError
from app.mtf.models import MtfAlignmentState, MtfContext, TimeframeContext
from app.regime.config import RegimeConfig
from app.regime.engine import RegimeEngine
from app.regime.models import MarketRegime

__all__ = ["MtfAnalysis", "MtfEngine", "RssLimitExceeded"]


class RssLimitExceeded(MtfError):
    """Raised when the research process exceeds its configured RSS guard."""


def _rss_mb() -> float:
    """Return current process RSS in megabytes from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - best-effort guard
        return -1.0
    return -1.0


def _tf_period_minutes(timeframe: str) -> int:
    """Resolve a native timeframe string to minutes (same mapping as MTF)."""
    _map = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "1w": 10080,
    }
    return _map.get(timeframe.lower(), 60)


def _mem_available_mb() -> float:
    """Return system MemAvailable in MB from /proc/meminfo (Linux), or -1."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - non-Linux / unreadable
        return -1.0
    return -1.0


def _require_rss_headroom(rss_limit_mb: float) -> None:
    """Phase-0 RSS guard: fail BEFORE heavy precomputation is started.

    Raises :class:`RssLimitExceeded` with an actionable message when current
    RSS already exceeds the configured limit, or when system memory is nearly
    exhausted. This keeps the OS from being the first to detect OOM.
    """
    rss = _rss_mb()
    if rss_limit_mb > 0.0 and rss > rss_limit_mb:
        avail = _mem_available_mb()
        raise RssLimitExceeded(
            f"MTF phase-0 RSS guard blocked the run BEFORE precomputation: "
            f"current RSS {rss:.0f}MB exceeds configured limit "
            f"{rss_limit_mb:.0f}MB (MemAvailable {avail:.0f}MB). "
            f"Reduce --max-bars and/or --mtf-chunk-size and retry on a quieter host."
        )
    # If the system is down to a tiny amount of free memory, also fail early.
    avail = _mem_available_mb()
    if avail > 0.0 and rss_limit_mb > 0.0 and avail < 256.0:
        raise RssLimitExceeded(
            f"MTF phase-0 RSS guard blocked the run: system MemAvailable is only "
            f"{avail:.0f}MB (RSS {rss:.0f}MB, limit {rss_limit_mb:.0f}MB). "
            f"Close memory-heavy applications or reduce --max-bars before retrying."
        )


def _causal_htf_lookback_bars(config: MtfConfig | None = None) -> int:
    """Derive a safe historical lookback (in HTF bars) from REAL engine windows.

    The MTF precompute only needs enough HTF history to produce structure and
    regime values that are identical to a full-dataset analysis at the base
    window timestamps. The largest causal warmup among every consumer is:
      - regime trend slow EMA        : RegimeConfig.ema_slow (50)
      - regime volatility percentile : RegimeConfig.percentile_window (100)
      - regime range / ATR SMA       : RegimeConfig.range_window (30)
      - market-structure ATR         : MarketStructureConfig.atr_window (14)
      - market-structure ranges      : MarketStructureConfig.range_window (30)
      - swing confirmation right     : MarketStructureConfig.swing_right (3)
      - MTF gap bridging             : MtfConfig.max_gap_lookback (5)

    A 30% safety slack is added on top of the true maximum so edge data windows
    (weekends, missing bars) cannot silently degrade results.
    """
    rc = RegimeConfig()
    warmups = [
        int(rc.ema_slow),
        int(rc.percentile_window),
        int(rc.range_window),
        int(rc.atr_window),
        int(getattr(rc, "range_min_bars", 10)),
    ]
    try:
        from app.market_structure.engine import MarketStructureConfig

        ms = MarketStructureConfig()
        warmups += [
            int(ms.atr_window),
            int(ms.range_window),
            int(ms.swing_right),
            int(ms.sweep_bars),
        ]
    except Exception:  # noqa: BLE001,S110 - optional config import; fallback defaults
        pass
    gap = int(config.max_gap_lookback) if config else 5
    warmups += [gap]
    base = max(warmups)
    return int(base * 1.3) + 1


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

    def _precompute_analysis(
        self,
        dataframes: dict[str, pd.DataFrame],
        base_timeframe: str | None,
        rss_limit_mb: float = 0.0,
        clip_htf: bool = True,
    ) -> tuple[str, pd.DataFrame, dict[str, MtfAnalysis]]:
        """Validate inputs and precompute per-timeframe analysis once.

        ``rss_limit_mb`` arms the phase-0 RSS guard BEFORE heavy
        MarketStructure/Regime objects are allocated.

        ``clip_htf`` (default True) restricts each higher-timeframe frame to
        the causal window required by the base frame plus a safe historical
        lookback margin derived from real engine warm-up windows. No future
        bars are added and the completed-candle rule is preserved, so per-bar
        outputs at base-window timestamps are identical to a full-dataset
        analysis (this is asserted by tests).

        Returns ``(base_tf, base_sorted, analysis)``. Heavy intermediate HTF
        DataFrames are released per-timeframe so we never hold all source
        frames simultaneously.
        """
        if not dataframes:
            raise ValueError("dataframes must contain at least one timeframe")

        # Phase-0 RSS guard: fail before expensive allocation, not after.
        _require_rss_headroom(rss_limit_mb)

        base_tf = base_timeframe or self.config.base_timeframe
        base_df = dataframes.get(base_tf)
        if base_df is None or base_df.empty:
            raise ValueError(f"base timeframe {base_tf} missing or empty")
        base_df = base_df.sort_index()

        lookback = _causal_htf_lookback_bars(self.config) if clip_htf else 0

        analysis: dict[str, MtfAnalysis] = {}
        all_tfs = [base_tf] + list(self.config.higher_timeframes)
        for tf in all_tfs:
            df = dataframes.get(tf)
            if df is None or df.empty:
                analysis[tf] = MtfAnalysis(tf, pd.DataFrame(), None, [])
                continue
            df = df.sort_index()

            if tf != base_tf and clip_htf:
                # Causal HTF window: bars >= base_first - lookback*HTF_period.
                # This supplies every warm-up (slow EMA, percentile rank,
                # ATR SMA, swing confirmation) so structure/regime values are
                # identical to full-dataset analysis. No future bars are added.
                cutoff = base_df.index[0] - pd.Timedelta(
                    minutes=_tf_period_minutes(tf) * lookback
                )
                window = df.loc[df.index >= cutoff]
                if window.empty:
                    # Too little history: keep the full frame (old behaviour).
                    window = df
                df = window

            structure: MarketStructureResult | None = None
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

            # Keep only the frame slice MtfContextBuilder needs; release the
            # original (possibly very large) HTF DataFrame.
            analysis[tf] = MtfAnalysis(tf, df, structure, regimes)
            del df
            if tf != base_tf:
                import gc as _gc

                _gc.collect()

        return base_tf, base_df, analysis

    def _build_contexts(
        self,
        builder: MtfContextBuilder,
        analysis: dict[str, MtfAnalysis],
        base_tf: str,
        base_sorted: pd.DataFrame,
        start: int,
        end: int,
        news_events: list[Any] | None,
    ) -> list[MtfContext]:
        """Build MtfContext for base bars ``[start, end)``.

        This is the ONLY place per-bar context logic lives — shared verbatim
        by monolithic ``analyze`` and chunked ``analyze_chunks`` so outputs
        are identical.
        """
        contexts: list[MtfContext] = []
        for ts in base_sorted.index[start:end]:
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
                liquidity_zones=builder.liquidity_zones_at(base_ana.structure, now),
                sweeps=builder.sweeps_at(base_ana.structure, now),
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

    def analyze(
        self,
        dataframes: dict[str, pd.DataFrame],
        base_timeframe: str | None = None,
        news_events: list[Any] | None = None,
        rss_limit_mb: float = 0.0,
        clip_htf: bool = True,
    ) -> list[MtfContext]:
        """Build one MtfContext per base bar (monolithic, backward-compatible).

        Parameters
        ----------
        dataframes : {timeframe: OHLC DataFrame indexed by tz-aware opens}
            Must contain at least the base timeframe; higher timeframes optional
            (missing tiers are surfaced as ``present=False``).
        base_timeframe : optional override for the base (acting) timeframe.
        news_events : optional list of EconomicEvent (availability-filtered).
        rss_limit_mb : arms the phase-0 RSS guard BEFORE precomputation.
        clip_htf : restrict higher-timeframe analysis to the causal window.

        Returns
        -------
        List[MtfContext] — one per base bar, each with ``available_from``.

        For production-scale datasets use :meth:`analyze_chunks` which keeps
        memory bounded as bar count grows.
        """
        base_tf, base_sorted, analysis = self._precompute_analysis(
            dataframes, base_timeframe,
            rss_limit_mb=rss_limit_mb, clip_htf=clip_htf,
        )
        builder = MtfContextBuilder(self.config, self.symbol)
        return self._build_contexts(
            builder, analysis, base_tf, base_sorted, 0, len(base_sorted), news_events
        )

    def analyze_chunks(
        self,
        dataframes: dict[str, pd.DataFrame],
        base_timeframe: str | None = None,
        news_events: list[Any] | None = None,
        chunk_size: int = 5000,
        rss_limit_mb: float = 0.0,
        clip_htf: bool = True,
    ):
        """Yield ``(start, end, contexts)`` per chunk with bounded memory.

        Precomputes per-timeframe analysis ONCE (structure / regime / causal
        state is never reset), then processes base bars in fixed-size chunks.
        Only the current chunk's ``MtfContext`` list is held in memory.

        Parameters
        ----------
        dataframes, base_timeframe, news_events : same as :meth:`analyze`.
        chunk_size : base-bar rows per chunk (default 5000).
        rss_limit_mb : arms the phase-0 RSS guard before precomputation and
            checks process RSS after each chunk, raising
            :class:`RssLimitExceeded` once the limit is crossed (the already
            yielded complete chunk remains valid to the caller).
        clip_htf : restrict higher-timeframe analysis to the causal window.

        Yields
        ------
        (start, end, contexts) where ``contexts`` covers base bars ``[start,
        end)``. The output is semantically identical to ``analyze``.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        base_tf, base_sorted, analysis = self._precompute_analysis(
            dataframes, base_timeframe,
            rss_limit_mb=rss_limit_mb, clip_htf=clip_htf,
        )
        builder = MtfContextBuilder(self.config, self.symbol)
        n = len(base_sorted)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            contexts = self._build_contexts(
                builder, analysis, base_tf, base_sorted, start, end, news_events
            )
            yield start, end, contexts
            # Release references to the chunk now that it has been yielded so
            # garbage collection can reclaim it before the next chunk.
            del contexts
            if rss_limit_mb > 0.0:
                rss = _rss_mb()
                if rss > rss_limit_mb:
                    raise RssLimitExceeded(
                        f"RSS {rss:.0f}MB exceeded limit {rss_limit_mb:.0f}MB "
                        f"after chunk ending at bar {end} of {n}"
                    )