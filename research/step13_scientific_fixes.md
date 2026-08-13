# Step 13 — Scientific Fixes: Root-Cause Map & Implementation Plan

## 1. Root-Cause Map

| # | Finding | Root Cause | File(s) |
|---|---------|------------|---------|
| C1 | Hypothesis execution rules don't affect R | `evaluator.py` computes R from label MFE/MAE approximation regardless of `Hypothesis.entry_rule/stop_rule/exit_rule` | `evaluator.py`, `labels.py` |
| C2 | Hypothesis search space too narrow | `runner.py` calls `generate_hypotheses()` with default args (only event_type×direction+single condition) | `hypotheses.py`, `runner.py` |
| C3 | HTF alignment is a hard filter | `candidates.py` `CandidateGenerator` has `require_htf_alignment=True` that drops events BEFORE hypotheses | `candidates.py`, `config.py` |
| C4 | Chunk overlap duplicates candidates | `persist.py` `merge_chunks_to_datasets()` concatenates without dedup | `persist.py`, `runner.py` |
| P1 | No trading-cost model | no config/application of spread/slippage/commission in R | `config.py`, `evaluator.py` |
| P2 | Statistical dependence (overlap/clustering) | no dedup + no dependence documentation | `persist.py`, `statistics.py` |
| M1 | Structure bias may be lifetime-cumulative | `candidates.py` `_structure_bias_before()` counts ALL prior structure points unbounded | `candidates.py` |
| M2 | Displacement lookback counts events not bars | `candidates.py` `_find_displacement()` counts displacement events, not bars | `candidates.py` |
| M3 | Session labels DST-naive | `extract.py` `_session_label()` uses fixed UTC hours | `extract.py`, `schema.py` |
| M4 | No discovery-phase memory guard | guard only before chunk processing | `guard.py`, `runner.py` |
| L1 | GC depends on verbose | `runner.py` `if verbose: gc.collect()` | `runner.py` |
| O1 | Overfit warnings don't affect status | `candidates_io.py` always writes `DISCOVERY_CANDIDATE` | `candidates_io.py`, `discovery.py` |

## 2. Exact Files/Functions Requiring Modification

- `app/research/step13/config.py` — add `spread_pips`, `slippage_pips`, `commission_per_lot`; add `session_timezone` flag.
- `app/research/step13/hypotheses.py` — expand `generate_hypotheses` to iterate entry/stop/exit + condition combos; bounded by limits.
- `app/research/step13/candidates.py` — remove hard HTF gate; apply optional HTF condition in `conditions_pass`; fix structure bias to bounded window; fix displacement lookback to bars not events.
- `app/research/step13/labels.py` — compute R based on hypothesis entry/stop/exit rules (hypothesis-aware labels).
- `app/research/step13/evaluator.py` — use hypothesis-aware R from labels; add cost model deduction.
- `app/research/step13/statistics.py` — document dependence; add block/unique-sample guard (n from `unique_candidate_ids`).
- `app/research/step13/discovery.py` — return status (DISCOVERY_CANDIDATE / DISCOVERY_WARNING / REJECTED) based on overfit warnings.
- `app/research/step13/candidates_io.py` — write status from discovery output.
- `app/research/step13/persist.py` — dedup by candidate_id in `merge_chunks_to_datasets`.
- `app/research/step13/guard.py` — add `require_rss_headroom` callable for discovery phase.
- `app/research/step13/runner.py` — remove hard HTF gate usage; call expanded hypothesis generator; discovery-phase guard; always gc.collect().
- `app/research/step13/extract.py` — document UTC-fixed sessions; produce `session_utc` field.

## 3. Data-Flow Changes

```
Chunk extraction
  → candidates (no HTF gate)
  → hypotheses (expanded: event + conditions + entry/stop/exit)
  → evaluator (uses hypothesis entry/stop/exit rules + costs → R)
  → statistics (dedup unique candidates first)
  → discovery score + status
  → research_candidate.json (status DISCOVERY_CANDIDATE / DISCOVERY_WARNING / REJECTED)
```

## 4. Hypothesis Execution Model

For each hypothesis:

- **Entry**:
  - `immediate`: bar close at candidate timestamp.
  - `displacement_confirmation`: close of first large/extreme displacement bar after event (within lookback).
  - `retest`: close of the first bar after event that retests the event extreme (within lookback).
- **Stop**:
  - `atr`: `entry ∓ stop_atr_multiple * ATR`.
  - `structural`: swing high/low nearest event (direction-dependent).
  - `liquidity`: event's level (sweep level / zone boundary).
- **Exit**:
  - `fixed_rr_<n>`: target at `entry ± n * risk_distance`.
  - `atr`: target at `entry ± exit_atr_multiple * ATR`.
  - `structure`: exit at the next structure point level.
  - `time`: exit after `max_holding_bars`.

Outcome = `R = (exit_price - entry_price) / risk_distance * direction - costs`.

## 5. Cost Model

- `spread_pips` (config; per-symbol overrides allowed)
- `slippage_pips` (config)
- `commission_per_lot` (account currency per standard lot 100k)

Cost deduction in evaluator:
```
cost_in_price = pip_size*(spread_pips+slippage_pips) + commission_units_per_price_unit
R_cost = cost_in_price / risk_distance
R_after_cost = R_before_cost - R_cost  (for longs; adjust for shorts)
```
Config hash recorded in artifacts.

## 6. Statistical Treatment

- Deduplicate by `candidate_id` in merged parquet (unique-sample enforcement).
- Document dependence: overlapping bars create dependent observations; bootstrap is a research heuristic; effective sample = unique candidates.
- Keep chronological order in stability-by-halves.

## 7. Memory Implications

- Discovery-phase RSS guard before loading merged candidate dataset.
- gc.collect() always after chunk.
- Dedup happens on merged (bounded) candidate frame.

## 8. Test Plan

- hypothesis-specific entry/stop/exit produce different R
- different hypotheses → different R/win/expectancy
- HTF candidate optionality + alignment comparison on same population
- candidate dedup regression
- cost model reduces R
- causal bounded structure bias
- displacement lookback bars semantics
- UTC-fixed session documented
- discovery memory guard
- GC cleanup independent of verbose
- overfit status marking
- deterministic ranking
- full regression

## 9. Migration/Compatibility Risks

- Removing `require_htf_alignment` default from config may affect existing integrations; keep config field for backward compat but ignore in candidate stage (documented).
- Expanding hypothesis generation increases runtime but bounded by max_hypotheses.

## 10. Exact Implementation Sequence

1. C1: hypothesis-aware labels/evaluator
2. P1 + C1: cost model
3. C2: expanded hypothesis generator
4. C3: HTF optionality
5. C4: candidate dedup
6. M1/M2: causal structure bias + displacement lookback bars
7. M3: session DST documentation
8. M4+L1: discovery guard + always-gc
9. O1: status marking
10. Tests
11. Full regression + staged benchmark