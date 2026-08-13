# Step 13 Redesign — Bounded-Memory Event Pipeline

## A. CURRENT ARCHITECTURE ANALYSIS

### Event engines (authoritative, must be reused — no duplicates)

| Component | Location | Output |
|-----------|----------|--------|
| FeatureEngine | `app/features/engine.py` | vectorized feature DataFrame |
| MarketStructureEngine | `app/market_structure/engine.py` | `MarketStructureResult` (swings, structure, breaks, zones, sweeps, displacement, ranges) |
| Liquidity zones + sweeps | `app/market_structure/liquidity.py` | `LiquidityZone[]`, `SweepEvent[]` (active-zone cursor, O(n+z)) |
| Displacement | `app/market_structure/displacement.py` | `DisplacementEvent[]` (Fenwick order-statistic, causal) |
| RegimeEngine | `app/regime/engine.py` | `MarketRegime[]` per bar (causal) |
| MtfEngine | `app/mtf/engine.py` | `MtfContext[]` or chunked (has `analyze_chunks`, `clip_htf`, RSS phase-0 guard, `_causal_htf_lookback_bars`) |
| MtfContextBuilder | `app/mtf/context.py` | `TimeframeContext` (completed-candle rule via `available_from`) |
| Causal index | `app/_causal_index.py` | `build_causal_index`, `available_prefix` (O(log n) queries) |

### Research infra (reusable)

| Component | Location | Purpose |
|-----------|----------|---------|
| ResearchCache | `app/research/cache.py` | versioned artifact cache (hashes, atomic) |
| MtfChunkStore / MtfContextMap | `app/research/mtf_chunks.py` | chunked MTF persistence + streaming lookup |
| PartitionedResearchRepository | `app/research/dataset.py` | symbol/timeframe parquet partitions + provenance |
| Step 13B | `app/research/step13b/` | walk-forward validation (NOT to be rewritten) |

### Old Step 13 problem

`app/research/run.py` is a monolithic pipeline that:
- loads every timeframe for one symbol, computes all stages across every bar, and retains large Pydantic object lists (signals, regimes, structure)
- produces `report.json`, `walk_forward.json`, `optimization.json`
- does NOT produce compact columnar event datasets for Step 13B / candidate research
- precomputes HTF structure/regime before clipping (mitigated for MTF by `clip_htf` but not for the event-research path)

## B. GAP ANALYSIS

1. **No columnar event datasets.** The old Step 13 produces reports, not compact parquet event tables that Step 13B can join.
2. **No candidate generation layer.** Nobody identifies "sweep + displacement + structure + HTF alignment + regime + session" candidate records structurally.
3. **No feature/label separation.** All event data is mixed; there is no `feature_*` / `label_*` schema enforcement.
4. **No per-chunk resumability.** The old pipeline is per-stage resumable but not per-chunk; partial chunks cannot be skipped.
5. **No general pre-computation RSS guard.** The MTF engine has one; the research run doesn't expose a MemAvailable guard before heavy analysis.
6. **No stable schema contract for Step 13B.** Step 13B currently scans signals itself; it should be able to consume a rich `candidate_events.parquet`.
7. **No HTF windowing for general event extraction.** `clip_htf` exists inside `MtfEngine` but event extraction outside MTF still loads whole HTF frames.

## C. DETAILED IMPLEMENTATION PLAN

New module `app/research/step13/` (event computation layer). The old `app/research/run.py` is left untouched.

1. **Step13Config** (symbols, timeframes, chunk_size, max_bars, output root, memory limits, warmup derivation).
2. **Warmup derivation** — reuse `MtfEngine._causal_htf_lookback_bars` semantics; derive HTF warmup from real engine configs.
3. **RSS guard** — pre-computation MemAvailable guard (pattern from `app/mtf/engine.py` `_require_rss_headroom`) exposed as reusable module.
4. **Event extraction** (`extract.py`) — for a bounded chunk: run FeatureEngine + MarketStructureEngine + RegimeEngine once, emit compact dict rows to builders, release objects.
5. **MTF extraction** (`mtf.py`) — use `MtfEngine.analyze_chunks` with `clip_htf=True`; emit compact per-bar MTF context rows.
6. **Candidate generation** (`candidates.py`) — strictly causal event join:
   - for each sweep event, within a configurable lookback (e.g. 5 bars) look for a displacement event in the sweep direction
   - require structural bias + HTF alignment + regime + session
   - emit `CANDIDATE` rows with `feature_*` columns only
7. **Label computation** (`labels.py`) — AFTER candidate timestamp, compute MFE / MAE / TP-hit / SL-hit / excursion within N bars as `label_*` columns (never mixed with feature_*).
8. **Persistence** (`persist.py`) — atomic parquet writes + manifests with schema/engine/config/data hashes + row counts + provenance. Reuse `ArtifactManager`-style atomic pattern.
9. **State/resume** (`state.py`) — per-symbol/timeframe/chunk status file, atomic updates. `--resume` skips complete chunks.
10. **Runner** (`runner.py`) — orchestrates: symbol → timeframe → bounded chunk → extract → emit → persist → release. Cache-aware via `ResearchCache`.
11. **Step 13B contract** — add `app/research/step13b/adapter.py` to load `candidate_events.parquet` (no rewrite of Step 13B).

## D. FILE-BY-FILE CHANGE PLAN

**New files:**
- `app/research/step13/__init__.py`
- `app/research/step13/config.py`
- `app/research/step13/warmup.py`
- `app/research/step13/guard.py`
- `app/research/step13/schema.py`
- `app/research/step13/extract.py`
- `app/research/step13/mtf.py`
- `app/research/step13/candidates.py`
- `app/research/step13/labels.py`
- `app/research/step13/persist.py`
- `app/research/step13/state.py`
- `app/research/step13/runner.py`
- `app/research/step13/README.md`
- `app/research/step13b/adapter.py`
- `tests/research/step13/__init__.py`
- `tests/research/step13/test_candidates.py`
- `tests/research/step13/test_causality.py`
- `tests/research/step13/test_extract_emit.py`
- `tests/research/step13/test_htf_window.py`
- `tests/research/step13/test_persist_resume.py`
- `tests/research/step13/test_guard.py`
- `tests/research/step13/test_contract.py`

**Modified files:**
- `app/research/__init__.py` (documentation)

**Intentionally untouched:**
- `app/research/run.py` (old Step 13)
- `app/research/step13b/` (Step 13B core)
- `app/market_structure/`, `app/regime/`, `app/mtf/`, `app/features/`, `app/backtest/`, `app/strategy/`, `app/risk/`, `app/execution/`

## E. DATA SCHEMA PLAN

One parquet dataset per event type under `research/results/step13/<SYMBOL>/<TIMEFRAME>/`:

| Dataset | Key columns |
|---------|-------------|
| `features.parquet` | timestamp, symbol, timeframe, session, open/high/low/close, atr, rsi, ...
| `structure_events.parquet` | timestamp, structure_type, price, prior_price, available_from
| `liquidity_zones.parquet` | zone_id, zone_type, upper, lower, mid, swing_count, first_ts, last_ts, available_from
| `sweeps.parquet` | timestamp, direction, level, extreme_price, close_price, sweep_type, zone_id, penetration, excursion, session, regime, available_from
| `displacement.parquet` | timestamp, direction, range_ratio, body_ratio, classification, available_from
| `regime.parquet` | timestamp, trend_state, volatility_state, market_state, strength, available_from
| `mtf_context.parquet` | timestamp, htf_tier, htf_timeframe, candle_open, candle_close, trend_state, volatility_state, market_state, structural_bias, available_from
| `candidate_events.parquet` | candidate_id, timestamp, symbol, timeframe, direction, entry_ref, sweep_ref, displacement_ref, structure_ref, htf_ref, regime, session, feature_* columns, available_from
| `candidate_labels.parquet` | candidate_id, timestamp, label_mfe, label_mae, label_tp_hit, label_sl_hit, label_excursion_after_bars

**Feature/label separation is structural**: candidate feature columns are prefixed `feature_*`; label columns are prefixed `label_*`. A SchemaValidator rejects any candidate row that would leak a label into features.

## F. STEP 13 → STEP 13B CONTRACT

Step 13 publishes `candidate_events.parquet` + `candidate_labels.parquet` + support parquet datasets.

Step 13B loads candidates via `app/research/step13b/adapter.py`:
```python
from app.research.step13b.adapter import load_candidate_events
candidates = load_candidate_events("research/results/step13/EURUSD/M15")
```
The adapter returns a compact DataFrame with stable columns — Step 13B doesn't need to know how events were detected.

## G. MEMORY/PERFORMANCE PLAN

- One symbol → one timeframe → one chunk at a time.
- Chunk size default 5000 bars; HTF frames clipped via `MtfEngine` causal lookback.
- After each chunk, emit rows, write parquet atomically, `del` chunk + analytical objects, `gc.collect()`.
- Pre-computation RSS guard reads `MemAvailable`, fails before heavy analysis with actionable message.
- 10K/20K/50K staged validation; safe default `max_bars`.

## H. TEST PLAN

18 required tests grouped into: causal correctness, candidate generation, HTF completed-candle rule, clipped-vs-full equivalence, no future leakage, feature/label separation, chunk equivalence, cache, resume, corrupted artifacts, memory guard, determinism, cross-symbol, schema validation, Step 13→13B contract.

## I. MIGRATION/COMPATIBILITY PLAN

Old `app/research/run.py` remains runnable. New Step 13 pipeline runs in parallel. Step 13B continues to work standalone (its own signal scanning) and gains an optional adapter to consume Step 13 candidates.

## J. ROLLBACK PLAN

- Old Step 13 runner untouched — revert is just deleting `app/research/step13/` + `app/research/step13b/adapter.py`.
- All new code is additive; no existing test is modified.