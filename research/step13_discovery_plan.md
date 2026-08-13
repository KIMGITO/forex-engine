# Step 13 — Strategy Discovery & Alpha Research Engine: Implementation Plan

## 1. Step 13 Objective

Step 13 discovers **which market conditions, event combinations, and rule combinations produce statistically useful trading edges** — producing *candidate hypotheses* as machine-readable artifacts. It does NOT validate strategies (that is Step 13B) and does NOT claim profitability.

The discovery output is `research_candidate.json` per candidate: hypothesis, event definition, entry/exit definition, filters, sample counts, statistics, discovery score, overfit warnings, and recommended validation.

## 2. Existing Functionality Reused (no duplicates)

| Component | Location | Role in Step 13 |
|-----------|----------|-----------------|
| FeatureEngine | `app/features/engine.py` | vectorized causal features |
| MarketStructureEngine | `app/market_structure/engine.py` | swings, structure, breaks, ranges, bias |
| Liquidity zones + sweeps | `app/market_structure/liquidity.py` | sweep events (zone-aware, O(n+z)) |
| Displacement | `app/market_structure/displacement.py` | Fenwick-causal displacement events |
| RegimeEngine | `app/regime/engine.py` | per-bar market regime |
| MtfEngine + MtfContextBuilder | `app/mtf/` | causal multi-timeframe context |
| Causal index | `app/_causal_index.py` | O(log n) availability queries |
| ResearchCache | `app/research/cache.py` | versioned artifact cache |
| PartitionedResearchRepository | `app/research/dataset.py` | parquet partitions + provenance |
| EventBacktester | `app/backtest/engine.py` | entry/exit evaluation (if needed for exit research) |
| Step 13B | `app/research/step13b/` | walk-forward validation consumer |

## 3. What Is Missing

1. **Hypothesis representation** — a typed structure defining event combo / conditions / entry / exit / filters with deterministic hashing.
2. **Fast research evaluator** — evaluates hypotheses against compact event tables WITHOUT the full backtester per hypothesis (vectorized event-to-outcome).
3. **Discovery scoring** — a transparent score combining sample size, expectancy, consistency, and overfit penalization. NOT a profit ranker.
4. **Statistical guards** — bootstrap confidence, multiple-testing control (Bonferroni / Benjamini-Hochberg), stability splits.
5. **Candidate artifact writer** — `research_candidate.json` with data hash, engine version, discovery score, overfit warnings.
6. **Step 13 → Step 13B interface** — candidate definitions consumable by Step 13B as parameterized research runs.

## 4. Proposed Module Structure

```
app/research/step13/
  __init__.py
  config.py            # Step13DiscoveryConfig (bounded grid, limits)
  guard.py             # pre-computation RSS guard (MemAvailable)
  warmup.py            # causal HTF lookback derivation (reuses engine configs)
  schema.py            # stable event/candidate schemas + feature/label separation
  extract.py           # bounded event extraction (features, structure, sweeps, displacement, regime)
  mtf.py               # bounded MTF context extraction (clip_htf=True)
  hypotheses.py        # hypothesis definitions + deterministic hash
  evaluator.py         # fast vectorized event-to-outcome research evaluator
  statistics.py        # expectancy, bootstrap CI, multiple-testing, stability
  discovery.py         # discovery scoring + candidate ranking + overfit screening
  candidates_io.py     # research_candidate.json writer + loader
  state.py             # per-chunk/per-hypothesis resume state
  runner.py            # orchestrator: symbol → timeframe → chunk → extract → evaluate → rank → persist
  README.md
app/research/step13b/adapter.py   # Step 13 → Step 13B interface (load candidates)
```

## 5. Data Flow

```
Historical candles (symbol, timeframe)
  → bounded chunk
  → EventExtractor → features / structure / liquidity_zones / sweeps / displacement / regime
  → MtfExtractor → mtf_context (clip_htf)
  → Hypothesis definitions (deterministic)
  → Fast Research Evaluator (vectorized event-to-outcome)
  → Statistics (expectancy, CI, bootstrap, multiple testing)
  → Discovery Score + Overfit Screen
  → Ranked research_candidate.json
  → STEP 13B (walk-forward validation)
```

## 6. Event Representation

All events are compact columnar rows (parquet) with explicit `available_from`. A row is a dict/DataFrame row — never a retained Pydantic object graph.

- features: timestamp, OHLC, atr, rsi, session
- sweeps: timestamp, direction, level, penetration, excursion, zone_id, regime, session
- displacement: timestamp, direction, range_ratio, classification
- structure_events: timestamp, structure_type, price
- regime: timestamp, trend/volatility/market state
- mtf_context: timestamp, htf_tier, htf_states

## 7. Hypothesis Representation

A candidate is defined by a deterministic hash of:

```yaml
hypothesis_id:      sha256(symbol|timeframe|event|conditions|entry|exit)
event_definition:   { type: liquidity_sweep, direction: long, depth_bucket: deep }
conditions:         [ { feature: displacement_present, value: true },
                      { feature: structure_bias, value: bullish },
                      { feature: regime, value: trending },
                      { feature: session, value: london },
                      { feature: htf_alignment, value: bullish } ]
entry_definition:   { type: displacement_confirmation }
exit_definition:    { type: fixed_rr, rr: 2.0 }
```

Rules: max 5 conditions, min sample 30, bounded condition grid.

## 8. Candidate Generation Method

Controlled grid over:
- event types: liquidity_sweep, displacement, structure_break
- directions: long/short
- conditions (bucketed): sweep depth, displacement present, structure bias, regime, session, volatility, HTF alignment
- entries: immediate / displacement confirmation / retest
- exits: fixed R:R (1.0, 1.5, 2.0, 3.0) / ATR-based / structural

Hard limits: max combinations per family (e.g., 100), max total hypotheses (e.g., 200), min sample 30.

## 9. Evaluation Methodology

The Fast Research Evaluator computes per-hypothesis outcomes from event tables:
- For each qualifying event → project future bars → compute MFE/MAE/R-outcome under entry/exit rules
- Aggregate: sample count, win rate, average R, median R, expectancy, profit factor, max drawdown, losing streak, holding time, regime/session/symbol breakdowns

This is deliberately lighter than the full EventBacktester, which is reserved for Step 13B final validation.

## 10. Statistical Methodology

- Expectancy: mean and median R with standard error
- Bootstrap 95% CI on expectancy (1000 resamples)
- Multiple-testing: Benjamini-Hochberg FDR on p-values; cap hypotheses
- Effect size: Cohen's d on R vs zero
- Stability: split sample by halves / by year / by symbol
- Out-of-sample: initial train half vs second half

## 11. Anti-Overfitting Controls

- Discovery vs validation strictly separated (Step 13 discovers, Step 13B validates)
- Multiple-testing FDR adjustment
- Minimum sample size
- Overfit warning when: train/test expectancy diverges, single-window dependence, FDR-adjusted p not significant
- Never select purely by total profit (reject score component)

## 12. Memory Architecture (8 GB)

- one symbol → one timeframe → one chunk (default 5000 bars)
- bounded HTF warm-up (clipped frames)
- event rows written to parquet per chunk; analytical objects released + gc.collect()
- no giant Pydantic lists; no giant JSON blobs

## 13. Performance Architecture

- Chunked event extraction using existing engines
- Vectorized research evaluator (numpy slices on parquet frames)
- Causal index for lookups
- Cache via ResearchCache (symbol/timeframe/data-hash/config-hash)
- Staged scale: 10K → 20K → 50K

## 14. Artifact Format

```
research/results/step13/<SYMBOL>/<TIMEFRAME>/
  research_candidate.json        # per-hypothesis candidate artifact
  candidate_events.parquet
  candidate_labels.parquet
  features.parquet
  sweeps.parquet
  displacement.parquet
  structure_events.parquet
  regime.parquet
  mtf_context.parquet
  manifest.json
```

`research_candidate.json` contains candidate_id, strategy_family, hypothesis, symbols, timeframes, event_definition, entry_definition, exit_definition, filters, sample_count, win_rate, expectancy_R, profit_factor, average_R, median_R, max_drawdown_R, losing_streak, regime_breakdown, session_breakdown, symbol_breakdown, parameter_definition, data_range, data_hash, engine_version, discovery_score, overfit_warnings, recommended_validation, status (`DISCOVERY_CANDIDATE`).

## 15. Step 13 → Step 13B Interface

Step 13 writes `research_candidate.json` per candidate. Step 13B's adapter reads these and converts them into parameterized Step 13B research configs (walk-forward windows, param grid, strategy name). No detection logic is duplicated.

## 16. CLI Design

```
python3 -m app.research.step13.runner \
    --symbols EURUSD GBPUSD \
    --timeframes M15 H1 \
    --storage-root data/processed \
    --output-root research/results/step13 \
    --chunk-size 5000 \
    --max-bars 50000 \
    --resume \
    --max-hypotheses 200 \
    --min-sample 30 \
    --rss-limit-mb 2000
```

## 17. Resume/Checkpoint Design

- Per-chunk state: `{ "EURUSD/M15": { "chunk_0": "complete", ... } }` (atomic JSON)
- Per-hypothesis state: computed once; cached by hypothesis hash + data hash
- `--resume` skips complete chunks/hypotheses
- Chunk artifacts validated before marked complete

## 18. Test Strategy

Tests for: causal correctness, schema validation, feature/label separation, chunk equivalence, hypothesis hashing, evaluator correctness, statistics (bootstrap/FDR), discovery ranking, overfit screening, resume, atomic writes, memory guard, contract with Step 13B adapter, cross-symbol determinism.

## 19. Benchmark Strategy

- 10K / 20K / 50K staged
- Measure: wall time, peak RSS, rows/events produced, hypotheses evaluated, artifact sizes, cache sizes

## 20. Migration Impact

- All new code is additive under `app/research/step13/`
- Old `app/research/run.py` remains untouched
- Step 13B / Step 15 / Step 16 / BacktestEngine remain untouched

## 21. Files to Create

- `app/research/step13/` package (as in section 4)
- `app/research/step13b/adapter.py`
- `tests/research/step13/` tests
- This plan document

## 22. Files to Modify

- `app/research/__init__.py` (documentation only)

## 23. Files That Must NOT Be Modified

- `app/research/run.py` (old Step 13)
- `app/research/step13b/` (except new adapter.py)
- `app/backtest/`, `app/strategy/`, `app/risk/`, `app/execution/`
- `app/market_structure/`, `app/regime/`, `app/mtf/`, `app/features/`

## 24. Risks and Failure Modes

- Memory: mitigated by chunking + RSS guard + HTF clipping
- Overfitting: controlled by FDR, min sample, discovery/validation separation
- Determinism: enforced via hashes + atomic writes
- Cost model: explicit spread/slippage assumptions documented
- Step 13B integration: adapter isolates interface changes

## 25. Implementation Order

1. Schema + config + guard + warmup (done)
2. Event extraction + MTF extraction (done)
3. Candidate generation + labels (done)
4. Persistence + state + runner (done)
5. Hypothesis module (new)
6. Fast research evaluator (new)
7. Statistics module (new)
8. Discovery scoring + ranking (new)
9. Step 13B adapter (done)
10. Tests for all modules
11. Staged benchmark 10K / 20K / 50K

## 26. Acceptance Criteria

1. 636 existing tests still pass
2. New Step 13 tests pass
3. Event datasets persisted columnar
4. Hypotheses have deterministic hashes
5. Discovery score not profit-based alone
6. FDR / bootstrap statistics implemented
7. `research_candidate.json` generated with all fields
8. Step 13B adapter loads candidates
9. 10K / 20K / 50K complete within configurable RSS limit
10. No profitability claim by Step 13

---

## A. Recommended Step 13 Name

**"Alpha Discovery Engine"** (package: `app.research.step13`).

## B. Proposed Module Tree

(see section 4 — already implemented in part)

## C. Exact Implementation Sequence

1. Schema/config/guard/warmup (DONE)
2. Event + MTF extraction (DONE)
3. Candidate + labels (DONE)
4. Persist + state + runner (DONE)
5. Hypothesis definitions module
6. Fast evaluator + statistics
7. Discovery scoring + research_candidate.json writer
8. Step 13B adapter (DONE)
9. Full test suite
10. Staged benchmarks

## D. Acceptance Tests

- 636 existing pass + new Step 13 tests
- Each candidate output passes schema validation
- Causality tests prove no future leakage
- Memory guard raises before OOM

## E. Performance/Memory Targets

- 10K bars: RSS < 500 MB, complete < 10 min
- 20K bars: RSS < 800 MB
- 50K bars: RSS < 1.5 GB
- Default `--max-bars` safe: 50000
- Default chunk: 5000