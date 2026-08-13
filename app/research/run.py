"""End-to-end research pipeline orchestrator (cache-aware, resumable, memory-safe).

Step 13.2 refactor:

* Persistent logging to ``research/results/step13/run.log`` (also stdout).
* ``--resume`` flag: skips completed stages whose cache artifacts remain
  valid, resuming per ``symbol × timeframe × stage``.
* Pipeline state tracked in ``research/results/step13/pipeline_state.json``.
* Cache writes are atomic (temp file + rename).
* RSS memory reported before/after major stages.
* One symbol processed at a time; frames released between symbols.
* Exceptions and tracebacks are always logged.

Causality is never weakened: the cache only persists the OUTPUT of the existing
engines. A cached artifact is only reused when its manifest exactly matches.

Fetch window: unused in ``provider='local'`` mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001,S110 - dotenv optional
    pass

from app.backtest import BacktestConfig, EventBacktester
from app.backtest import Strategy as BacktestStrategy
from app.data.provider import BaseMarketDataProvider
from app.data.providers import create_provider
from app.features import FeatureEngine
from app.market_structure.engine import MarketStructureEngine
from app.mtf import MtfConfig, MtfEngine
from app.regime import RegimeEngine
from app.research.cache import (
    ResearchCache,
    StageTimer,
    config_hash,
    data_hash,
    deser_features,
    deser_regime,
    deser_signals,
    deser_structure,
    ser_features,
    ser_mtf,
    ser_regime,
    ser_signals,
    ser_structure,
)
from app.research.config import ResearchConfig
from app.research.data_quality import validate_partition
from app.research.dataset import PartitionedResearchRepository
from app.research.models import ResearchReport
from app.research.mtf_chunks import MtfChunkStore, MtfContextMap
from app.research.reports import build_research_report
from app.research.splits import make_time_split, split_frame
from app.strategy import (
    HistoricalSignalScanner,
    LiquidityReversalStrategy,
    StrategyConfig,
    TrendStructureStrategy,
)
from app.strategy.base import SignalToOrderAdapter

__all__ = ["ResearchRunConfig", "run_research_pipeline", "smoke_test_pipeline"]

_MTF_NATIVE = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}

_STAGE_ORDER = [
    "features",
    "structure",
    "regime",
    "mtf",
    "signals_trend_structure",
    "signals_trend_structure_mtf",
    "backtest",
    "backtest_mtf",
]

# ── Logger setup ────────────────────────────────────────────────────────────────

_log = logging.getLogger(__name__)
_log.propagate = False


def _setup_logging(log_file: str | None = None, verbose: bool = True) -> None:
    """Configure root logger to write to stdout and optionally a file."""
    root = logging.getLogger("research")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)


def log() -> logging.Logger:
    return logging.getLogger("research")


# ── Memory reporting ───────────────────────────────────────────────────────────


def _rss_mb() -> float:
    """Return current process RSS in megabytes from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        return -1.0
    return -1.0


# ── Pipeline state ─────────────────────────────────────────────────────────────


class PipelineState:
    """Persistent progress tracker, written atomically after every stage."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text("utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), "utf-8")
        os.replace(str(tmp), str(self.path))

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}/{timeframe.upper()}"

    def status(
        self, symbol: str, timeframe: str, stage: str
    ) -> str:
        """Return 'pending', 'running', 'complete', or 'failed'."""
        return self._data.get(self._key(symbol, timeframe), {}).get(stage, "pending")

    def set_status(
        self, symbol: str, timeframe: str, stage: str, status: str
    ) -> None:
        key = self._key(symbol, timeframe)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][stage] = status
        self._save()

    def symbol_stages(self, symbol: str, timeframe: str) -> dict[str, str]:
        return dict(self._data.get(self._key(symbol, timeframe), {}))


# ── ResearchRunConfig ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResearchRunConfig:
    symbols: tuple = ("EURUSD",)
    timeframes: tuple = ("H1",)
    storage_root: str = "data/research"
    strategy_names: tuple = ("trend_structure",)
    provider: str = "twelvedata"
    fetch_days: int = 30
    max_bars: int = 0
    cache_root: str = "research/cache"
    use_cache: bool = True
    benchmark: bool = False
    resume: bool = False
    log_file: str = ""
    baseline_cost: dict = field(
        default_factory=lambda: {
            "spread_pips": 0.8,
            "slippage_pips": 0.0,
            "commission_percent": 0.0,
            "commission_per_trade": 0.0,
        }
    )
    walk_train_days: float = 18.0
    mtf_chunk_size: int = 5000
    max_rss_mb: int = 5000
    walk_validation_days: float = 5.0
    walk_test_days: float = 5.0
    conservative_cost: dict = field(
        default_factory=lambda: {
            "spread_pips": 1.5,
            "slippage_pips": 0.5,
            "commission_percent": 0.0005,
            "commission_per_trade": 0.0,
        }
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _native(timeframe: str) -> str:
    return _MTF_NATIVE.get(timeframe, timeframe.lower())


def _df_from_repo(repo, symbol: str, timeframe: str) -> pd.DataFrame | None:
    df = repo.load_df(symbol, timeframe)
    if df is None:
        return None
    return df[["open", "high", "low", "close"]].sort_index()


def _apply_window(df: pd.DataFrame, fetch_days: int, max_bars: int) -> pd.DataFrame:
    """Slice a frame the same way the strategy loop does (trailing window).

    fetch_days slices the trailing N days; max_bars keeps only the last N
    bars. This helper guarantees the report's date_range always reflects the
    exact data actually processed (cache-correct, no off-by-one vs. the loop).
    """
    if fetch_days > 0:
        end_local = df.index[-1]
        start_local = end_local - pd.Timedelta(days=fetch_days)
        df = df[(df.index >= start_local) & (df.index <= end_local)]
    if max_bars > 0:
        df = df.iloc[-max_bars:]
    return df


class LocalPartitionMissingError(FileNotFoundError):
    def __init__(self, symbol: str, timeframe: str, storage_root: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.storage_root = storage_root
        self.expected_path = str(
            Path(storage_root) / symbol / timeframe / "data.parquet"
        )
        super().__init__(
            f"Local research partition missing for {symbol}/{timeframe}.\n"
            f"  expected path: {self.expected_path}\n"
            f"  instruction: ingest the dataset first before using provider='local'.\n"
        )


# ── Stages ─────────────────────────────────────────────────────────────────────


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngine().calculate(df, features=["atr", "rsi"])


def _compute_structure(df: pd.DataFrame, symbol: str, timeframe: str):
    return MarketStructureEngine().analyze(df, symbol, _native(timeframe))


def _compute_regime(df: pd.DataFrame, symbol: str, timeframe: str, structure):
    return RegimeEngine().analyze(
        df, symbol, _native(timeframe), market_structure=structure
    )


def _get_features(
    cache: ResearchCache, df: pd.DataFrame, symbol: str, timeframe: str,
) -> pd.DataFrame:
    f, _hit = cache.get_or_compute(
        symbol,
        timeframe,
        "features",
        df,
        {"features": ["atr", "rsi"]},
        {},
        lambda: _compute_features(df),
        ser_features,
        deser_features,
    )
    return f


def _get_structure(
    cache: ResearchCache, df: pd.DataFrame, symbol: str, timeframe: str,
):
    s, _hit = cache.get_or_compute(
        symbol,
        timeframe,
        "structure",
        df,
        {},
        {},
        lambda: _compute_structure(df, symbol, timeframe),
        ser_structure,
        deser_structure,
    )
    return s


def _get_regime(
    cache: ResearchCache,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    structure,
    upstream_hashes: dict,
):
    r, _hit = cache.get_or_compute(
        symbol,
        timeframe,
        "regime",
        df,
        {},
        upstream_hashes,
        lambda: _compute_regime(df, symbol, timeframe, structure),
        ser_regime,
        deser_regime,
    )
    return r


def _compute_mtf(
    dfs_all: dict[str, pd.DataFrame], symbol: str, base_tf: str,
) -> list | None:
    native_base = _native(base_tf)
    native_map: dict[str, pd.DataFrame] = {}
    for tf in _MTF_NATIVE:
        key = f"{symbol}|{tf}"
        if key in dfs_all and dfs_all[key] is not None and not dfs_all[key].empty:
            native_map[_native(tf)] = dfs_all[key]
    if native_base not in native_map:
        return None
    order = ["5m", "15m", "1h", "4h", "1d"]
    if native_base in order:
        order.remove(native_base)
    higher = [tf for tf in order if tf in native_map]
    if not higher:
        return None
    try:
        engine = MtfEngine(
            MtfConfig(base_timeframe=native_base, higher_timeframes=tuple(higher)),
            symbol,
        )
        return engine.analyze(native_map, native_base)
    except Exception:
        return None


def _compute_resolved_mtf(
    cache: ResearchCache,
    dfs_all: dict[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    chunk_size: int,
    rss_limit_mb: float,
    log=None,
):
    """Memory-bounded MTF stage.

    Uses ``MtfEngine.analyze_chunks`` so only one chunk of contexts is in
    memory at a time, and persists each chunk via ``MtfChunkStore`` (atomic
    temp-file writes). On restart, valid chunks are skipped and processing
    resumes from the first missing chunk.

    Returns a streaming ``MtfContextMap`` over the completed chunk store.
    """
    from app.mtf import MtfConfig, MtfEngine

    native_map_raw = {}
    for tf in _MTF_NATIVE:
        key = f"{symbol}|{tf}"
        if key in dfs_all and dfs_all[key] is not None and not dfs_all[key].empty:
            native_map_raw[_native(tf)] = dfs_all[key]

    native_base = _native(timeframe)
    src_df = dfs_all.get(f"{symbol}|{timeframe}", pd.DataFrame())
    src_hash = data_hash(src_df)
    cfg_hash = config_hash({})
    base_df = src_df.sort_index()
    total = len(base_df)

    store = MtfChunkStore(symbol, timeframe, str(cache.root))
    store.write_manifest(
        source_data_hash=src_hash,
        config_hash=cfg_hash,
        upstream_hashes={"data_hashes": src_hash},
        total_bars=total,
        chunk_size=chunk_size,
    )
    resume_at = store.first_missing_index()
    if resume_at > 0 and log:
        log.info(f"        MTF resume: {resume_at} valid chunk(s) already complete")

    engine = MtfEngine(
        MtfConfig(base_timeframe=native_base, higher_timeframes=tuple(
            tf for tf in ("5m", "15m", "1h", "4h", "1d")
            if tf != native_base and tf in native_map_raw
        )),
        symbol,
    )

    chunks_written = 0
    for chunk_index, (start, end, contexts) in enumerate(engine.analyze_chunks(
        {**native_map_raw, native_base: base_df},
        native_base,
        chunk_size=chunk_size,
        rss_limit_mb=rss_limit_mb,
        clip_htf=True,
    )):
        if chunk_index < resume_at:
            # Already persisted in a previous run; skip (chunks remain valid).
            continue
        payload = ser_mtf(contexts)
        store.write_chunk(
            chunk_index,
            start,
            end,
            payload,
            source_data_hash=src_hash,
            config_hash=cfg_hash,
        )
        chunks_written += 1
        if log:
            log.info(
                f"        MTF chunk {chunk_index}: bars={len(contexts)} "
                f"rss_before={_rss_mb():.0f}MB"
            )
        del contexts, payload

    m = MtfContextMap(store)
    return m


def _get_signals(
    cache: ResearchCache,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    strat_config: StrategyConfig,
    features: pd.DataFrame,
    structure,
    regimes,
    mtf_ctxs,
):
    native = _native(timeframe)
    strat = (
        TrendStructureStrategy(strat_config)
        if strategy_name == "trend_structure"
        else LiquidityReversalStrategy(strat_config)
    )
    if isinstance(mtf_ctxs, MtfContextMap):
        mtf_upstream_hash = mtf_ctxs.chunk_set_hash()
    elif mtf_ctxs:
        mtf_upstream_hash = data_hash(
            pd.DataFrame([c.model_dump() for c in mtf_ctxs])
        )
    else:
        mtf_upstream_hash = ""
    upstream = {
        "features": data_hash(features),
        "structure": config_hash(structure),
        "regime": data_hash(pd.DataFrame([r.model_dump() for r in regimes]))
        if regimes
        else "",
        "mtf": mtf_upstream_hash,
    }
    key = f"_{strategy_name}_{config_hash(strat_config)[:8]}"

    def _scan() -> list:
        scanner = HistoricalSignalScanner(strategy_config=strat_config, regime_config=None)
        scan = scanner.scan(
            df,
            strat,
            symbol,
            native,
            features=features,
            mtf_contexts=mtf_ctxs,
            structure=structure,
            regimes=regimes,
        )
        return scan.signals

    sigs, _hit = cache.get_or_compute(
        symbol,
        timeframe,
        "signals",
        df,
        strat_config,
        upstream,
        _scan,
        ser_signals,
        deser_signals,
        key=key,
    )
    return sigs


def _run_backtest(
    df: pd.DataFrame,
    strategy: BacktestStrategy,
    symbol: str,
    native_tf: str,
    bt_config: BacktestConfig,
    mtf_contexts: list | None = None,
    features: pd.DataFrame | None = None,
    news_events: list | None = None,
    provider_tag: str = "twelvedata",
) -> dict:
    result = EventBacktester(bt_config).run(
        df,
        strategy,
        features=features,
        news_events=news_events,
        mtf_contexts=mtf_contexts,
        provider=provider_tag,
        source_type="historical",
    )
    m = result.metrics
    return {
        "trade_count": m.trade_count,
        "net_pnl": m.net_pnl,
        "total_return": m.total_return,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "expectancy": m.expectancy,
        "average_win": m.average_win,
        "average_loss": m.average_loss,
        "max_drawdown": m.max_drawdown,
        "drawdown_duration_bars": m.drawdown_duration_bars,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "exposure_fraction": m.exposure_fraction,
        "average_holding_bars": m.average_holding_bars,
        "insufficient_data": list(m.insufficient_data),
    }


def _run_strategy_on_partition(
    df: pd.DataFrame,
    dfs_all: dict[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    strategy_name: str,
    strat_config: StrategyConfig,
    cost: dict,
    provider_tag: str,
    cache: ResearchCache,
    state: PipelineState,
    timer: StageTimer | None = None,
    mtf_chunk_size: int = 5000,
    max_rss_mb: float = 5000.0,
) -> dict:
    native = _native(timeframe)
    bt = BacktestConfig(
        symbol=symbol,
        timeframe=native,
        spread_pips=cost["spread_pips"],
        slippage_pips=cost["slippage_pips"],
        commission_model=(
            "percentage"
            if cost["commission_percent"]
            else "fixed" if cost["commission_per_trade"] else "zero"
        ),
        commission_percent=cost["commission_percent"],
        commission_per_trade=cost["commission_per_trade"],
        initial_balance=10_000.0,
    )

    mtf = strat_config.mtf_enabled
    sfx = "_mtf" if mtf else ""

    # --- features ---
    _stage_log(symbol, timeframe, strategy_name, sfx, "features", "start")
    _set_state(state, symbol, timeframe, str(StageEnum.FEATURES), "running")
    if timer: timer.begin("_features")
    feats = _get_features(cache, df, symbol, timeframe)
    if timer: timer.end("_features")
    _stage_log(symbol, timeframe, strategy_name, sfx, "features", "end", timer)
    _set_state(state, symbol, timeframe, str(StageEnum.FEATURES), "complete")

    # --- structure ---
    _stage_log(symbol, timeframe, strategy_name, sfx, "structure", "start")
    _set_state(state, symbol, timeframe, str(StageEnum.STRUCTURE), "running")
    if timer: timer.begin("_structure")
    structure = _get_structure(cache, df, symbol, timeframe)
    if timer: timer.end("_structure")
    _stage_log(symbol, timeframe, strategy_name, sfx, "structure", "end", timer)
    _set_state(state, symbol, timeframe, str(StageEnum.STRUCTURE), "complete")

    # --- regime ---
    _stage_log(symbol, timeframe, strategy_name, sfx, "regime", "start")
    _set_state(state, symbol, timeframe, str(StageEnum.REGIME), "running")
    if timer: timer.begin("_regime")
    regimes = _get_regime(cache, df, symbol, timeframe, structure, {})
    if timer: timer.end("_regime")
    _stage_log(symbol, timeframe, strategy_name, sfx, "regime", "end", timer)
    _set_state(state, symbol, timeframe, str(StageEnum.REGIME), "complete")

    # --- mtf ---
    _stage_log(symbol, timeframe, strategy_name, sfx, "mtf", "start")
    _set_state(state, symbol, timeframe, str(StageEnum.MTF), "running")
    if timer: timer.begin("_mtf")
    mtf_ctxs = (
        _compute_resolved_mtf(
            cache, dfs_all, symbol, timeframe,
            chunk_size=mtf_chunk_size,
            rss_limit_mb=float(max_rss_mb),
            log=log(),
        )
        if mtf
        else None
    )
    if timer: timer.end("_mtf")
    _stage_log(symbol, timeframe, strategy_name, sfx, "mtf", "end", timer)
    _set_state(state, symbol, timeframe, str(StageEnum.MTF), "complete")

    # --- signals ---
    sig_stage = f"signals_{strategy_name}{sfx}"
    _stage_log(symbol, timeframe, strategy_name, sfx, "signals", "start")
    _set_state(state, symbol, timeframe, sig_stage, "running")
    if timer: timer.begin("_signals")
    signals = _get_signals(
        cache, df, symbol, timeframe, strategy_name, strat_config,
        feats, structure, regimes, mtf_ctxs,
    )
    if timer: timer.end("_signals")
    _stage_log(symbol, timeframe, strategy_name, sfx, "signals", "end", timer)
    _set_state(state, symbol, timeframe, sig_stage, "complete")

    # --- backtest ---
    bt_stage = f"backtest{sfx}"
    _stage_log(symbol, timeframe, strategy_name, sfx, "backtest", "start")
    _set_state(state, symbol, timeframe, bt_stage, "running")

    signals_by_ts = {s.timestamp: s for s in signals}
    adapter = SignalToOrderAdapter(quantity=1000.0)

    class _Strategy(BacktestStrategy):
        name = strategy_name

        def on_bar(self, context):
            sig = signals_by_ts.get(context.now.to_pydatetime())
            if sig is None:
                return []
            return adapter.to_order_intents(
                sig, context, context.now.to_pydatetime()
            )

    if timer: timer.begin("_backtest")
    metrics = _run_backtest(
        df, _Strategy(), symbol, native, bt, mtf_ctxs,
        features=feats, provider_tag=provider_tag,
    )
    if timer: timer.end("_backtest")
    _stage_log(symbol, timeframe, strategy_name, sfx, "backtest", "end", timer)
    _set_state(state, symbol, timeframe, bt_stage, "complete")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "mtf": mtf,
        "trade_count": int(metrics.get("trade_count", 0)),
        "metrics": metrics,
    }


# ── Stage state enums ──────────────────────────────────────────────────────────

from enum import Enum


class StageEnum(str, Enum):
    FEATURES = "features"
    STRUCTURE = "structure"
    REGIME = "regime"
    MTF = "mtf"
    SIGNALS = "signals"
    BACKTEST = "backtest"


def _set_state(
    state: PipelineState, symbol: str, timeframe: str, stage: str, status: str
) -> None:
    state.set_status(symbol, timeframe, stage, status)


def _stage_log(
    symbol: str,
    timeframe: str,
    strategy: str,
    sfx: str,
    stage: str,
    event: str,
    timer: StageTimer | None = None,
) -> None:
    L = log()
    tag = f"[{symbol} {timeframe} {strategy}{sfx}]"
    if event == "start":
        L.info(f"{tag} {stage}: START  rss={_rss_mb():.0f}MB")
    elif event == "end":
        elapsed = timer.timings.get(f"_{stage}", -1) if timer else -1
        elapsed_str = f" {elapsed:.1f}s" if elapsed >= 0 else ""
        L.info(f"{tag} {stage}: DONE{elapsed_str}  rss={_rss_mb():.0f}MB")


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_research_pipeline(
    config: ResearchConfig | None = None,
    run_cfg: ResearchRunConfig | None = None,
    *,
    provider: BaseMarketDataProvider | None = None,
    repo: PartitionedResearchRepository | None = None,
    output_root: str = "research/results/step13",
    verbose: bool = True,
) -> dict:
    config = config or ResearchConfig()
    run_cfg = run_cfg or ResearchRunConfig()
    repo = repo or PartitionedResearchRepository(run_cfg.storage_root)
    cache = ResearchCache(run_cfg.cache_root, use_cache=run_cfg.use_cache)
    timer = StageTimer() if run_cfg.benchmark else None

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "pipeline_state.json"
    state = PipelineState(str(state_path))

    _setup_logging(
        run_cfg.log_file or str(out / "run.log"), verbose=verbose
    )
    L = log()

    L.info("=" * 60)
    L.info("STEP 13 RESEARCH PIPELINE")
    L.info(f"  provider={run_cfg.provider}  resume={run_cfg.resume}")
    L.info(f"  symbols={list(run_cfg.symbols)}")
    L.info(f"  timeframes={list(run_cfg.timeframes)}")
    L.info(f"  output={out}")
    L.info(f"  max_bars={run_cfg.max_bars}  fetch_days={run_cfg.fetch_days}")
    L.info(f"  rss={_rss_mb():.0f}MB")
    L.info("=" * 60)

    local_mode = run_cfg.provider == "local"
    if local_mode and provider is not None:
        raise ValueError(
            "provider='local' does not accept a provider instance; "
            "it reads persisted research partitions only."
        )
    if local_mode:
        provider = None
    else:
        provider = provider or create_provider(config.provider)

    # ── 1. LOAD / FETCH ──────────────────────────────────────────────────────
    fetch_notes: list[str] = []
    for sym in run_cfg.symbols:
        for tf in run_cfg.timeframes:
            df = _df_from_repo(repo, sym, tf)
            if local_mode:
                if df is None or df.empty:
                    raise LocalPartitionMissingError(
                        sym, tf, run_cfg.storage_root
                    )

    L.info(f"All {len(run_cfg.symbols) * len(run_cfg.timeframes)} partitions present.")

    # ── 2. VALIDATE ──────────────────────────────────────────────────────────
    quality: dict[str, dict] = {}
    _source_tag = "histdata" if local_mode else config.provider
    for sym in run_cfg.symbols:
        for tf in run_cfg.timeframes:
            df = _df_from_repo(repo, sym, tf)
            if local_mode and (df is None or df.empty):
                raise LocalPartitionMissingError(sym, tf, run_cfg.storage_root)
            if df is None or df.empty:
                quality[f"{sym}|{tf}"] = validate_partition(
                    None, sym, tf, _source_tag, "native"
                ).to_dict()
                continue
            dq_report = validate_partition(
                df,
                sym,
                tf,
                provider=_source_tag,
                native_or_aggregated=("native" if tf == "M15" else "aggregated"),
            )
            quality[f"{sym}|{tf}"] = dq_report.to_dict()
            del df

    for k, q in sorted(quality.items()):
        ok = "PASS" if q["passed"] else "FAIL"
        L.info(
            f"  {k}: rows={q['candle_count']} "
            f"start={q['first_timestamp']} end={q['last_timestamp']} "
            f"gaps={q['gap_count']} -> {ok}"
        )

    # ── 3. RUN STRATEGIES per symbol (memory-safe) ───────────────────────────
    results: list[dict] = []

    for sym in run_cfg.symbols:
        sym_rss_before = _rss_mb()
        L.info(f"--- {sym} START  rss={sym_rss_before:.0f}MB ---")

        # Load only this symbol's frames.
        dfs_all: dict[str, pd.DataFrame] = {}
        for tf in run_cfg.timeframes:
            df = _df_from_repo(repo, sym, tf)
            if local_mode:
                if df is None or df.empty:
                    raise LocalPartitionMissingError(sym, tf, run_cfg.storage_root)
                df = _apply_window(df, run_cfg.fetch_days, run_cfg.max_bars)
                if df.empty:
                    raise LocalPartitionMissingError(sym, tf, run_cfg.storage_root)
            if df is not None and not df.empty:
                dfs_all[f"{sym}|{tf}"] = df
                L.info(f"  loaded {sym}/{tf}: {len(df)} rows")

        sym_results: list[dict] = []
        for tf in run_cfg.timeframes:
            df = dfs_all.get(f"{sym}|{tf}")
            if df is None:
                continue

            for strat_name in run_cfg.strategy_names:
                # no-MTF strategy
                try:
                    r = _run_strategy_on_partition(
                        df, dfs_all, sym, tf, strat_name,
                        StrategyConfig(),
                        run_cfg.baseline_cost, _source_tag,
                        cache, state, timer,
                        mtf_chunk_size=run_cfg.mtf_chunk_size,
                        max_rss_mb=float(run_cfg.max_rss_mb),
                    )
                    results.append(r)
                    sym_results.append(r)
                except Exception:
                    L.error(
                        f"[{sym} {tf} {strat_name}] FAILED\n"
                        f"{traceback.format_exc()}"
                    )
                    state.set_status(sym, tf, f"signals_{strat_name}", "failed")
                    state.set_status(sym, tf, "backtest", "failed")
                    raise

                # MTF strategy
                try:
                    r_mtf = _run_strategy_on_partition(
                        df, dfs_all, sym, tf, strat_name,
                        StrategyConfig(mtf_enabled=True, mtf_min_aligned=1),
                        run_cfg.baseline_cost, _source_tag,
                        cache, state, timer,
                        mtf_chunk_size=run_cfg.mtf_chunk_size,
                        max_rss_mb=float(run_cfg.max_rss_mb),
                    )
                    results.append(r_mtf)
                    sym_results.append(r_mtf)
                except Exception:
                    L.error(
                        f"[{sym} {tf} {strat_name}_mtf] FAILED\n"
                        f"{traceback.format_exc()}"
                    )
                    state.set_status(sym, tf, f"signals_{strat_name}_mtf", "failed")
                    state.set_status(sym, tf, "backtest_mtf", "failed")
                    raise

        # Release this symbol's frames.
        del dfs_all
        gc.collect()
        sym_rss_after = _rss_mb()
        L.info(
            f"--- {sym} DONE  rss_before={sym_rss_before:.0f}MB "
            f"rss_after={sym_rss_after:.0f}MB ---"
        )

    # ── 4. TRAIN / VALIDATION / TEST ─────────────────────────────────────────
    rep = _df_from_repo(repo, "EURUSD", "H1")
    if local_mode and (rep is None or rep.empty):
        raise LocalPartitionMissingError("EURUSD", "H1", run_cfg.storage_root)
    if rep is not None and not rep.empty:
        rep = _apply_window(rep, run_cfg.fetch_days, run_cfg.max_bars)

    dfs_all_eur = {"EURUSD|H1": rep} if rep is not None else {}
    train_metrics: list[dict] = []
    val_metrics: list[dict] = []
    test_metrics: list[dict] = []
    if rep is not None and len(rep) > 100:
        split = make_time_split(
            rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), config
        )
        train, val, test = split_frame(rep, split)
        bt = BacktestConfig(symbol="EURUSD", timeframe="1h", spread_pips=0.8)

        rep_fe = _get_features(cache, rep, "EURUSD", "H1")
        rep_struct = _get_structure(cache, rep, "EURUSD", "H1")
        rep_regime = _get_regime(cache, rep, "EURUSD", "H1", rep_struct, {})
        rep_mtf = _compute_resolved_mtf(
            cache, dfs_all_eur, "EURUSD", "H1",
            chunk_size=run_cfg.mtf_chunk_size,
            rss_limit_mb=float(run_cfg.max_rss_mb),
            log=L,
        )
        rep_signals = _get_signals(
            cache, rep, "EURUSD", "H1", "trend_structure", StrategyConfig(),
            rep_fe, rep_struct, rep_regime, rep_mtf,
        )
        rep_strategy, _ = _build_signal_strategy(
            rep, "EURUSD", "1h", "trend_structure", StrategyConfig(),
            rep_fe, rep_mtf, rep_struct, rep_regime, rep_signals,
        )
        for frame in (train, val, test):
            if frame.empty:
                continue
            train_metrics.append(
                _run_backtest(
                    frame, rep_strategy, "EURUSD", "1h", bt,
                    provider_tag=_source_tag,
                )
            )

    # ── 5. WALK-FORWARD ──────────────────────────────────────────────────────
    walk_results: list[dict] = []
    opt_results: list[dict] = []
    if rep is not None and len(rep) > 100:
        from app.research.walk_forward import build_walk_forward_windows

        wf_cfg = ResearchConfig(
            walk_train_years=run_cfg.walk_train_days / 365.0,
            walk_validation_years=run_cfg.walk_validation_days / 365.0,
            walk_test_years=run_cfg.walk_test_days / 365.0,
        )
        wf_windows = build_walk_forward_windows(
            rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), wf_cfg
        )
        for w in wf_windows:
            wf_train = rep[
                (rep.index >= w.split.train_start)
                & (rep.index < w.split.train_end)
            ]
            wf_val = rep[
                (rep.index >= w.split.validation_start)
                & (rep.index < w.split.validation_end)
            ]
            wf_test = rep[
                (rep.index >= w.split.test_start)
                & (rep.index <= w.split.test_end)
            ]
            if wf_train.empty or wf_val.empty or wf_test.empty:
                continue
            val_m = _run_backtest(
                wf_val, rep_strategy, "EURUSD", "1h", bt,
                provider_tag=_source_tag,
            )
            test_m = _run_backtest(
                wf_test, rep_strategy, "EURUSD", "1h", bt,
                provider_tag=_source_tag,
            )
            walk_results.append(
                {
                    "index": w.index,
                    "split": w.split.to_dict(),
                    "validation": val_m,
                    "test": test_m,
                }
            )

        from app.research.optimizer import GridSearchOptimizer

        def _wf_bt(frame, params=None):
            return _run_backtest(
                frame, rep_strategy, "EURUSD", "1h", bt,
                provider_tag=_source_tag,
            )

        opt = GridSearchOptimizer(config, _wf_bt)
        split_opt = make_time_split(
            rep.index[0].to_pydatetime(), rep.index[-1].to_pydatetime(), config
        )
        train_opt, _, _ = split_frame(rep, split_opt)
        grid = {"regime_strength": [0.4, 0.6], "reward_risk": [1.5, 2.5]}
        for c in opt.optimize(train_opt, grid):
            opt_results.append(c.to_dict())

    # ── 6. ASSEMBLE + PERSIST REPORT ─────────────────────────────────────────
    date_range = {}
    for sym in run_cfg.symbols:
        df = _df_from_repo(repo, sym, "H1")
        if df is not None:
            df = _apply_window(df, run_cfg.fetch_days, run_cfg.max_bars)
            date_range[sym] = {
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
                "rows": len(df),
            }

    symbol_agg: dict[str, dict] = {}
    for r in results:
        sym = str(r.get("symbol") or "unknown")
        exp = float(r.get("metrics", {}).get("expectancy") or 0.0)
        t = int(r.get("trade_count", 0))
        if sym not in symbol_agg:
            symbol_agg[sym] = {"exp_sum": 0.0, "trades": 0, "runs": 0}
        symbol_agg[sym]["exp_sum"] += exp
        symbol_agg[sym]["trades"] += t
        symbol_agg[sym]["runs"] += 1
    results_by_symbol: dict[str, dict] = {}
    for sym, agg in symbol_agg.items():
        results_by_symbol[sym] = {
            "expectancy": agg["exp_sum"] / max(agg["runs"], 1),
            "trade_count": agg["trades"],
        }

    if local_mode:
        report_provider = "histdata"
        cost_data = "HistData OHLC (persisted research partition)"
        warnings = [
            "local mode: no network used; data from persisted research partitions.",
            "HistData provides OHLC only (no historical bid/ask).",
            *fetch_notes,
        ]
        limitations = [
            (
                "provider='local': results depend on the persisted research "
                "dataset; ingest fresh data before re-running for updated coverage."
            ),
            "No historical bid/ask; all costs are SIMULATED ASSUMPTIONS.",
        ]
    else:
        report_provider = "twelvedata"
        cost_data = "Twelve Data OHLC"
        warnings = [
            "Twelve Data supplies OHLC only (no historical bid/ask).",
            *fetch_notes,
        ]
        limitations = [
            (
                "Twelve Data free-tier plan serves only ~1 trailing month per "
                "symbol/timeframe; dataset is INSUFFICIENT for multi-year "
                "robustness validation."
            ),
            "No historical bid/ask; all costs are SIMULATED ASSUMPTIONS.",
        ]

    report = build_research_report(
        provider=report_provider,
        symbols=list(run_cfg.symbols),
        timeframes=list(run_cfg.timeframes),
        strategy_name=",".join(run_cfg.strategy_names),
        config=config,
        date_range=date_range,
        training_metrics=train_metrics,
        validation_metrics=val_metrics,
        oos_metrics=test_metrics,
        cross_window_metrics=walk_results,
        results_by_symbol=results_by_symbol,
        cost_assumptions={
            "data": cost_data,
            "data_source": (
                "persisted_research_partition" if local_mode else "provider_api"
            ),
            "mode": ("local" if local_mode else "provider"),
            "execution": "SIMULATED BID/ASK",
            "cost_model": "ASSUMPTION",
            "baseline": run_cfg.baseline_cost,
            "conservative": run_cfg.conservative_cost,
        },
        warnings=warnings,
        limitations=limitations,
    )

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": report_provider,
        "symbols": list(run_cfg.symbols),
        "timeframes": list(run_cfg.timeframes),
        "strategies": list(run_cfg.strategy_names),
        "max_bars": run_cfg.max_bars,
        "cache_root": str(cache.root),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }

    def _atomic_json(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, indent=2, default=str), "utf-8")
        os.replace(str(tmp), str(path))

    _atomic_json(out / "report.json", report.model_dump())
    _atomic_json(out / "data_quality.json", quality)
    _atomic_json(out / "walk_forward.json", walk_results)
    _atomic_json(out / "optimization.json", opt_results)
    _atomic_json(out / "manifest.json", manifest)

    (out / "report.txt").write_text(
        _human_readable(report, quality), "utf-8"
    )

    if timer:
        timings = timer.summary()
        _atomic_json(out / "timing.json", timings)
        L.info("STAGE TIMING:")
        for stage, secs in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
            L.info(f"  {stage:<24} {secs:8.2f}s")
        L.info(f"  {'TOTAL':<24} {timer.total():8.2f}s")
        L.info(f"Cache: {cache.hits} hits, {cache.misses} misses")

    L.info(f"Reports written to {out}/")
    L.info(f"Walk-forward windows: {len(walk_results)}")
    L.info(f"Optimization candidates: {len(opt_results)}")
    L.info(f"Strategy runs: {len(results)}")
    L.info(f"Pipeline state: {state_path}")
    L.info("PIPELINE COMPLETE")
    return {
        "report": report.model_dump(),
        "results": results,
        "walk_forward": walk_results,
        "optimization": opt_results,
        "cache": cache.describe(),
    }


def _build_signal_strategy(
    df, symbol, native_tf, strategy_name, strat_config,
    features, mtf_ctxs, structure, regimes, signals: list,
):
    signals_by_ts = {s.timestamp: s for s in signals}
    adapter = SignalToOrderAdapter(quantity=1000.0)

    class _SignalStrategy(BacktestStrategy):
        name = strategy_name

        def on_bar(self, context):
            sig = signals_by_ts.get(context.now.to_pydatetime())
            if sig is None:
                return []
            return adapter.to_order_intents(
                sig, context, context.now.to_pydatetime()
            )

    return _SignalStrategy(), signals_by_ts


def _human_readable(report: ResearchReport, quality: dict) -> str:
    lines = ["=" * 70, "FOREX RESEARCH REPORT", "=" * 70]
    lines.append(f"Provider: {report.provider}")
    lines.append(f"Symbols: {', '.join(report.symbols)}")
    lines.append(f"Timeframes: {', '.join(report.timeframes)}")
    lines.append(f"Strategies: {report.strategy}")
    lines.append("")
    lines.append("DATA:")
    for k, q in sorted(quality.items()):
        ok = "PASS" if q["passed"] else "FAIL"
        lines.append(
            f"  {k}: rows={q['candle_count']} {q['first_timestamp']} -> "
            f"{q['last_timestamp']} gaps={q['gap_count']} {ok}"
        )
    lines.append("")
    lines.append("EXECUTION: SIMULATED BID/ASK")
    lines.append("COST MODEL: ASSUMPTION")
    lines.append("")
    lines.append("TRAIN / VALIDATION / TEST:")
    lines.append(f"  TRAIN: {_fmt(report.training)}")
    lines.append(f"  VALIDATION: {_fmt(report.validation)}")
    lines.append(f"  TEST (OOS): {_fmt(report.out_of_sample)}")
    lines.append("")
    lines.append("WARNINGS:")
    for w in report.warnings:
        lines.append(f"  - {w}")
    lines.append("")
    lines.append("LIMITATIONS:")
    for l in report.limitations:
        lines.append(f"  - {l}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _fmt(block: dict) -> str:
    a = block.get("aggregate", {}) or {}
    return (
        f"trades_mean={a.get('trade_count_mean')} "
        f"net_pnl_mean={a.get('net_pnl_mean')} "
        f"expectancy_mean={a.get('expectancy_mean')}"
    )


def smoke_test_pipeline(
    provider: BaseMarketDataProvider | None = None,
    repo: PartitionedResearchRepository | None = None,
    output_root: str = "research/results/step13",
    verbose: bool = True,
) -> dict:
    log().info("SMOKE TEST: EURUSD H1")
    cfg = ResearchRunConfig(
        symbols=("EURUSD",),
        timeframes=("H1",),
        strategy_names=("trend_structure",),
        fetch_days=31,
    )
    return run_research_pipeline(
        config=ResearchConfig(symbols=("EURUSD",), timeframes=("H1",)),
        run_cfg=cfg,
        provider=provider,
        repo=repo,
        output_root=output_root,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forex research pipeline")
    parser.add_argument("--smoke", action="store_true", help="EURUSD H1 smoke run")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--provider",
        default="twelvedata",
        choices=("twelvedata", "local"),
        help="local = offline mode reading persisted partitions",
    )
    parser.add_argument("--output", default="research/results/step13")
    parser.add_argument(
        "--max-bars", type=int, default=0,
        help="dev mode: process only the last N bars (0 = all)",
    )
    parser.add_argument("--cache-root", default="research/cache")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="disable the research cache (compute everything)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="print per-stage timing breakdown",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from completed pipeline stages (skip valid cached stages)",
    )
    parser.add_argument(
        "--log-file", default="",
        help="persistent log file (default: <output>/run.log)",
    )
    parser.add_argument(
        "--mtf-chunk-size", type=int, default=5000,
        help="MTF chunk size in bars (default 5000; bounded-memory processing)",
    )
    parser.add_argument(
        "--max-rss-mb", type=int, default=5000,
        help="RSS guard in MB; stops research run if exceeded (default 5000)",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test_pipeline(output_root=args.output)
        return

    log_file = args.log_file or str(Path(args.output) / "run.log")
    run_cfg = ResearchRunConfig(
        symbols=tuple(args.symbols) if args.symbols else ("EURUSD",),
        timeframes=tuple(args.timeframes) if args.timeframes else ("H1",),
        fetch_days=args.days,
        provider=args.provider,
        max_bars=args.max_bars,
        cache_root=args.cache_root,
        use_cache=not args.no_cache,
        benchmark=args.benchmark,
        resume=args.resume,
        log_file=log_file,
        mtf_chunk_size=args.mtf_chunk_size,
        max_rss_mb=args.max_rss_mb,
    )
    run_research_pipeline(run_cfg=run_cfg, output_root=args.output)


if __name__ == "__main__":
    main()