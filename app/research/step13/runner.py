"""Step 13 event-pipeline orchestrator.

For each symbol/timeframe:
  1. Load the base candles partition (optionally sliced to last max_bars).
  2. Load HTF frames once, and clip them to the causal window via the
     derived warm-up (never load the full HTF history per chunk).
  3. Process the base bars in bounded chunks.
  4. Per chunk: run EventExtractor + MtfExtractor + CandidateGenerator +
     LabelComputation, emit compact rows, persist chunk atomically.
  5. Release chunk + analytical objects; gc.collect().
  6. After all chunks, merge to per-dataset parquet and write the manifest.

State is persisted per chunk for ``--resume``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.research.dataset import PartitionedResearchRepository
from app.research.mtf_chunks import MtfChunkStore
from app.research.step13.candidates import CandidateGenerator, candidates_to_frame
from app.research.step13.candidates_io import (
    build_candidate_artifact,
    write_candidate_artifact,
)
from app.research.step13.config import Step13Config
from app.research.step13.discovery import compute_discovery_score, rank_candidates
from app.research.step13.evaluator import FastResearchEvaluator
from app.research.step13.extract import EventExtractor, extract_rows_to_frame
from app.research.step13.guard import require_rss_headroom, rss_mb
from app.research.step13.hypotheses import (
    HypothesisGridLimits,
    generate_hypotheses,
)
from app.research.step13.labels import compute_labels, labels_to_frame
from app.research.step13.mtf import MtfExtractor
from app.research.step13.persist import Step13Artifacts, atomic_write_json
from app.research.step13.schema import (
    CANDIDATE_EVENTS_COLUMNS,
    CANDIDATE_LABELS_COLUMNS,
    DISPLACEMENT_COLUMNS,
    FEATURES_COLUMNS,
    LIQUIDITY_ZONES_COLUMNS,
    REGIME_COLUMNS,
    STRUCTURE_COLUMNS,
    SWEEPS_COLUMNS,
    MTF_CONTEXT_COLUMNS,
    validate_candidate_events,
    validate_candidate_labels,
    validate_feature_label_separation,
)
from app.research.step13.state import Step13State
from app.research.step13.warmup import causal_htf_lookback_bars, clip_htf_frame

_log = logging.getLogger(__name__)

ENGINE_VERSION = "13.2.0"

_NATIVE = {"M15": "15m", "H1": "1h"}
_SUPPORTED_HTF = {"M15": ("1h", "4h", "1d"), "H1": ("4h", "1d")}


def _native(tf: str) -> str:
    return _NATIVE.get(tf.upper(), tf.lower())


def _setup_logging(log_file: str | None = None, verbose: bool = True) -> None:
    root = logging.getLogger("step13")
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
    return logging.getLogger("step13")


def _load_partition(
    repo: PartitionedResearchRepository,
    symbol: str,
    timeframe: str,
    max_bars: int = 0,
) -> pd.DataFrame | None:
    df = repo.load_df(symbol, timeframe)
    if df is None or df.empty:
        return None
    df = df[["open", "high", "low", "close"]].sort_index()
    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:]
    return df


def _make_chunks(n: int, chunk_size: int, overlap: int) -> list[tuple[int, int, int]]:
    """Return list of (chunk_index, start, end) with overlap.

    Each chunk includes ``overlap`` warm-up bars before its start so sweep
    detection, ATR, and structure rolling windows are not truncated at chunk
    boundaries. Row indices are in the full-frame coordinate system.
    """
    chunks: list[tuple[int, int, int]] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        warm_start = max(0, start - overlap)
        chunks.append((idx, warm_start, end))
        if end >= n:
            break
        start = end  # next chunk starts where previous ended
        idx += 1
    return chunks


def _run_symbol_timeframe(
    base_df: pd.DataFrame,
    htf_frames: dict[str, pd.DataFrame],
    config: Step13Config,
    *,
    symbol: str,
    timeframe: str,
    artifacts: Step13Artifacts,
    state: Step13State,
    resume: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    """Process one symbol/timeframe through the event pipeline."""
    L = log()
    L.info("=== %s/%s START  rss=%.0fMB ===", symbol, timeframe, rss_mb())

    # Data hash for reproducibility.
    src_df = base_df[["open", "high", "low", "close"]].sort_index()
    data_hash = hashlib.sha256(src_df.to_parquet(engine="pyarrow")).hexdigest()
    cfg_hash = config.config_hash()

    # Pre-computation RSS guard.
    require_rss_headroom(
        rss_limit_mb=config.rss_limit_mb,
        min_mem_available_mb=config.min_mem_available_mb,
        stage=f"{symbol}/{timeframe} event extraction",
    )

    # Build chunks.
    n = len(base_df)
    chunks = _make_chunks(n, config.chunk_size, config.overlap_bars)
    n_chunks = len(chunks)
    L.info("  %d chunks of max %d bars (overlap %d)", n_chunks, config.chunk_size, config.overlap_bars)

    # Resume point.
    start_chunk = 0
    if resume:
        start_chunk = state.next_incomplete_chunk(symbol, timeframe, n_chunks)
        if start_chunk > 0:
            L.info("  resume: skipping %d complete chunk(s)", start_chunk)

    extractor = EventExtractor(symbol, timeframe)
    htfs = _SUPPORTED_HTF.get(timeframe.upper(), ("1h",))
    mtf_extractor = MtfExtractor(
        symbol,
        _native(timeframe),
        htf_timeframes=htfs,
        chunk_size=config.chunk_size,
        rss_limit_mb=config.rss_limit_mb,
    )
    candidate_gen = CandidateGenerator(
        symbol,
        timeframe,
        sweep_displacement_lookback=config.sweep_displacement_lookback,
        min_displacement_class=config.min_displacement_class,
        require_htf_alignment=config.require_htf_alignment,
    )

    # HTF frames clipped once (still includes enough warm-up for MTF).
    htf_native_frames: dict[str, pd.DataFrame] = {}
    for htf in htfs:
        raw = htf_frames.get(htf)
        if raw is None or raw.empty:
            continue
        htf_native_frames[htf] = raw.sort_index()

    total_candidates = 0
    for chunk_index, warm_start, end in chunks:
        if chunk_index < start_chunk:
            continue
        L.info("  --- chunk %d/%d [%d:%d] rss=%.0fMB ---",
               chunk_index + 1, n_chunks, warm_start, end, rss_mb())
        state.set_status(symbol, timeframe, chunk_index, "running")

        chunk = base_df.iloc[warm_start:end]

        # 1. Event extraction (authoritative engines).
        rows = extractor.extract(chunk)

        # 2. MTF rows (causal HTF window).
        mtf_rows = mtf_extractor.rows_for_chunk(
            base_df, htf_native_frames, warm_start, end
        )

        # 3. Candidate generation (strictly causal).
        candidates = candidate_gen.generate(
            sweeps=rows["sweeps"],
            displacements=rows["displacement"],
            regimes=rows["regime"],
            features=rows["features"],
            mtf_rows=mtf_rows,
            structure_rows=rows["structure_events"],
        )

        # 4. Labels (separate dataset).
        labels = compute_labels(
            candidates,
            base_df.iloc[warm_start:min(end + config.label_lookback_bars + 1, n)],
            lookback_bars=config.label_lookback_bars,
        )

        # 5. Convert to compact frames + validate separation.
        frames = {
            "features": extract_rows_to_frame(rows["features"], FEATURES_COLUMNS),
            "structure_events": extract_rows_to_frame(rows["structure_events"], STRUCTURE_COLUMNS),
            "liquidity_zones": extract_rows_to_frame(rows["liquidity_zones"], LIQUIDITY_ZONES_COLUMNS),
            "sweeps": extract_rows_to_frame(rows["sweeps"], SWEEPS_COLUMNS),
            "displacement": extract_rows_to_frame(rows["displacement"], DISPLACEMENT_COLUMNS),
            "regime": extract_rows_to_frame(rows["regime"], REGIME_COLUMNS),
            "mtf_context": extract_rows_to_frame(mtf_rows, MTF_CONTEXT_COLUMNS),
            "candidate_events": candidates_to_frame(candidates),
            "candidate_labels": labels_to_frame(labels),
        }
        # Feature/label separation enforced.
        validate_candidate_events(frames["candidate_events"])
        validate_candidate_labels(frames["candidate_labels"])
        validate_feature_label_separation(frames["candidate_events"])

        # 6. Persist chunk atomically.
        artifacts.write_chunk(chunk_index, frames)
        state.set_status(symbol, timeframe, chunk_index, "running")

        # Only mark complete after artifacts valid.
        if artifacts.chunk_valid(chunk_index):
            state.set_status(symbol, timeframe, chunk_index, "complete")
        else:
            L.error("  chunk %d artifact invalid; not marking complete", chunk_index)

        total_candidates += len(candidates)
        L.info("  --- chunk %d/%d DONE  candidates=%d rss=%.0fMB ---",
               chunk_index + 1, n_chunks, len(candidates), rss_mb())

        # Release chunk + analytical objects (L1: always cleanup, not verbose-only).
        del chunk, rows, candidates, labels, frames
        gc.collect()

    # Merge all valid chunks into per-dataset parquet.
    datasets = [
        "features", "structure_events", "liquidity_zones", "sweeps",
        "displacement", "regime", "mtf_context",
        "candidate_events", "candidate_labels",
    ]
    artifacts.merge_chunks_to_datasets(datasets, data_hash, cfg_hash, ENGINE_VERSION)

    # ── DISCOVERY PHASE ─────────────────────────────────────────────────────
    # Evaluate a controlled set of deterministic hypotheses over the compact
    # candidate event dataset. No claim of profitability — only discovery
    # scoring + overfit warnings. Step 13B performs the walk-forward validation.
    from app.research.step13.persist import read_parquet_if_valid

    # Discovery-phase memory guard (M4) BEFORE loading merged candidate data.
    require_rss_headroom(
        rss_limit_mb=config.rss_limit_mb,
        min_mem_available_mb=config.min_mem_available_mb,
        stage=f"{symbol}/{timeframe} discovery",
    )

    cand_events = read_parquet_if_valid(artifacts.dataset_path("candidate_events"))
    cand_labels = read_parquet_if_valid(artifacts.dataset_path("candidate_labels"))

    # Deduplicate by candidate_id (C4): overlapping chunks must not inflate
    # the statistical sample.
    if cand_events is not None and not cand_events.empty and "candidate_id" in cand_events.columns:
        dedup = cand_events.drop_duplicates(subset=["candidate_id"])
        if len(dedup) < len(cand_events):
            L.info("  DISCOVERY: deduplicated %d overlapping candidates",
                   len(cand_events) - len(dedup))
        cand_events = dedup
    if cand_labels is not None and not cand_labels.empty and "candidate_id" in cand_labels.columns:
        cand_labels = cand_labels.drop_duplicates(subset=["candidate_id"])

    n_hypotheses = 0
    n_discovery_candidates = 0
    if cand_events is not None and not cand_events.empty:
        limits = HypothesisGridLimits(
            max_hypotheses=config.max_hypotheses,
            max_conditions_per_hypothesis=2,
            min_sample_size=config.min_sample_size,
        )
        # CONTROLLED expanded search (C2): event + condition + entry/stop/exit
        # combinations, strictly bounded by max_hypotheses.
        hypotheses = generate_hypotheses(
            symbols=(symbol,),
            timeframes=(timeframe,),
            entry_rules=("immediate", "displacement_confirmation", "retest"),
            exit_rules=("fixed_rr_1.5", "fixed_rr_2.0", "atr"),
            structure_biases=("bullish", "bearish"),
            regimes=("trending", "ranging"),
            sessions=("europe", "newyork"),
            htf_alignments=("bullish", "bearish"),
            limits=limits,
        )
        evaluator = FastResearchEvaluator()
        costs = {
            "spread_pips": config.spread_pips,
            "slippage_pips": config.slippage_pips,
            "commission_per_lot": config.commission_per_lot,
        }
        scores: dict[str, Any] = {}

        for hyp in hypotheses:
            # C1: the evaluator simulates the hypothesis's actual entry/stop/
            # exit rules against OHLC with the cost model, so resulting R
            # corresponds exactly to the recorded hypothesis.
            result = evaluator.evaluate(
                hyp, cand_events, cand_labels, base_df, costs=costs
            )
            n_hypotheses += 1
            if result.sample_count == 0:
                continue
            score = compute_discovery_score(
                result,
                min_sample=config.min_sample_size,
            )
            if result.sample_count < config.min_sample_size:
                continue
            n_discovery_candidates += 1
            scores[hyp.hypothesis_id] = score
            artifact = build_candidate_artifact(
                hypothesis=hyp,
                evaluator_stats=score.stats,
                discovery_score=score.to_dict(),
                sample_count=result.sample_count,
                overfit_warnings=score.overfit_warnings,
                data_hash=data_hash,
                engine_version=ENGINE_VERSION,
                configuration_hash=cfg_hash,
            )
            try:
                write_candidate_artifact(
                    config.output_root, symbol, timeframe, artifact
                )
            except Exception:  # noqa: BLE001 - artifact write failure logged
                L.error("  candidate artifact write failed for %s: %s",
                        hyp.hypothesis_id, traceback.format_exc())

        ranked = rank_candidates(scores)
        L.info("  DISCOVERY: %d hypotheses evaluated, %d candidates passed min-sample",
               n_hypotheses, n_discovery_candidates)
        for i, (hid, sc) in enumerate(ranked[:5]):
            L.info("    %d. %s score=%.4f warnings=%d",
                   i + 1, hid, sc.total, len(sc.overfit_warnings))

        discovery_summary = {
            "hypotheses_evaluated": n_hypotheses,
            "candidates_passed_min_sample": n_discovery_candidates,
            "ranked": [
                {
                    "candidate_id": hid,
                    "score": sc.total,
                    "warnings": sc.overfit_warnings,
                }
                for hid, sc in ranked
            ],
        }
        atomic_write_json(artifacts.dir / "discovery_summary.json", discovery_summary)
    else:
        L.info("  DISCOVERY: no candidate events to evaluate")
        atomic_write_json(
            artifacts.dir / "discovery_summary.json",
            {
                "hypotheses_evaluated": 0,
                "candidates_passed_min_sample": 0,
                "ranked": [],
            },
        )

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "chunks": n_chunks,
        "completed_chunks": sum(
            1 for i in range(n_chunks)
            if artifacts.chunk_valid(i)
        ),
        "candidates": total_candidates,
        "data_hash": data_hash,
        "config_hash": cfg_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(artifacts.dir / "summary.json", summary)

    L.info("=== %s/%s DONE  candidates=%d rss=%.0fMB ===",
           symbol, timeframe, total_candidates, rss_mb())
    return summary


def run_step13(
    config: Step13Config | None = None,
    *,
    symbols: tuple | None = None,
    timeframes: tuple | None = None,
    max_bars: int = 0,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Step 13 event/candidate pipeline."""
    config = config or Step13Config()
    if symbols:
        config = Step13Config(**{**config.to_dict(), "symbols": tuple(symbols)})
    if timeframes:
        config = Step13Config(**{**config.to_dict(), "timeframes": tuple(timeframes)})
    if max_bars:
        config = Step13Config(**{**config.to_dict(), "max_bars": max_bars})

    out_root = Path(config.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = str(out_root / "step13_run.log")
    _setup_logging(log_file, verbose=verbose)
    L = log()

    state = Step13State(out_root / "step13_state.json")
    repo = PartitionedResearchRepository(config.storage_root)

    L.info("=" * 70)
    L.info("STEP 13 — MARKET EVENT RESEARCH & CANDIDATE GENERATION")
    L.info(f"  symbols={list(config.symbols)}")
    L.info(f"  timeframes={list(config.timeframes)} engine={ENGINE_VERSION}")
    L.info(f"  chunk_size={config.chunk_size} max_bars={config.max_bars}")
    L.info(f"  rss_limit_mb={config.rss_limit_mb} resume={resume}")
    L.info(f"  output={out_root}")
    L.info("=" * 70)

    all_summaries: dict[str, Any] = {}
    for sym in config.symbols:
        for tf in config.timeframes:
            try:
                df = _load_partition(repo, sym, tf, config.max_bars)
            except Exception:
                L.error("  %s/%s load failed: %s", sym, tf, traceback.format_exc())
                continue
            if df is None or df.empty:
                L.warning("  %s/%s: no data; skipping", sym, tf)
                continue

            # Load HTF frames (clipped once to causal window).
            htf_frames: dict[str, pd.DataFrame] = {}
            htf_names = _SUPPORTED_HTF.get(tf.upper(), ("1h",))
            for htf_name in htf_names:
                htf_df = _load_partition(repo, sym, htf_name)
                if htf_df is not None and not htf_df.empty:
                    htf_frames[htf_name] = htf_df

            artifacts = Step13Artifacts(out_root, sym, tf)
            try:
                summary = _run_symbol_timeframe(
                    df, htf_frames, config,
                    symbol=sym, timeframe=tf,
                    artifacts=artifacts, state=state,
                    resume=resume, verbose=verbose,
                )
                all_summaries[f"{sym}/{tf}"] = summary
            except Exception:
                L.error("  %s/%s FAILED:\n%s", sym, tf, traceback.format_exc())
                raise
            finally:
                del df, htf_frames
                gc.collect()

    # Top-level summary.
    top_summary = {
        "runs": all_summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": ENGINE_VERSION,
    }
    atomic_write_json(out_root / "research_summary.json", top_summary)
    L.info("PIPELINE COMPLETE (Step 13)")
    return all_summaries


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 13 — Market Event Research & Candidate Generation"
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--storage-root", default="data/processed")
    parser.add_argument("--output-root", default="research/results/step13")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rss-limit-mb", type=float, default=2500.0)
    args = parser.parse_args()

    config = Step13Config(
        symbols=tuple(args.symbols) if args.symbols else ("EURUSD",),
        timeframes=tuple(args.timeframes) if args.timeframes else ("M15",),
        storage_root=args.storage_root,
        output_root=args.output_root,
        chunk_size=args.chunk_size,
        max_bars=args.max_bars,
        rss_limit_mb=args.rss_limit_mb,
    )
    run_step13(config, resume=args.resume)