# Research Layer — Walk-Forward Validation & Leakage-Safe Optimization

Establishes a trustworthy baseline for determining whether strategies have a
genuine statistical edge. **Does NOT implement ML/AI, live execution, or
broker integration.**

## Modules

- `config.py` — `ResearchConfig` (documented development defaults)
- `dataset.py` — `PartitionedResearchRepository` (one parquet per
  symbol/timeframe) + `sync_partition` (incremental, idempotent)
- `splits.py` — strict chronological TRAIN/VALIDATION/TEST splits (no shuffle)
- `walk_forward.py` — rolling leak-free walk-forward windows
- `optimizer.py` — grid-search optimize-on-TRAIN; select-on-VALIDATION
- `metrics.py` — aggregation + minimum-sample warnings
- `reports.py` — `ResearchReport` (IN-SAMPLE / VALIDATION / OUT-OF-SAMPLE)
- `models.py` / `errors.py` — typed models + exceptions

## Dataset persistence

Layout: `data/processed/<SYMBOL>/<TIMEFRAME>/data.parquet` + `meta.json`.
Supports M5/M15/H1/H4/D1 × any symbols (configurable). Provenance records
provider, symbol, timeframe, retrieval time, bounds, row count, timezone, and
gaps. Missing candles are **never fabricated**.

Sync is incremental (fetches only the missing tail) and **idempotent** (running
twice never duplicates). Retries/rate-limits are bounded by the Step-7
HttpClient.

## Chronological splits

`make_time_split` prefers explicit dates; falls back to fractional. The TEST
period is always the most recent block and is **never touched** during
optimization. `split_frame` slices chronologically (inclusive start, exclusive
end).

## Walk-forward

`build_walk_forward_windows` produces rolling train→validation→test windows.
Each window is strictly chronological and disjoint; a future window can never
influence an earlier window.

## Optimization (leakage prevention)

`GridSearchOptimizer.optimize(train_frame, ...)` backtests every candidate on
**TRAIN only**. `select_on_validation(...)` re-evaluates top-N on **VALIDATION**
only. The untouched TEST is used only for final evaluation, outside the
optimizer. The selection score is multi-metric (return, expectancy, profit
factor) with a drawdown penalty — never just max profit, never a probability.

## Minimum-sample warnings

`warnings_for` always surfaces insufficient trades / bars / observations.
Reports are explicit that lower counts make ratios unreliable.

## Research report

`build_research_report` returns a machine-readable `ResearchReport` with
clearly labelled IN-SAMPLE (TRAIN), VALIDATION, and OUT-OF-SAMPLE (TEST)
blocks — never mixed — plus cross-symbol stability, cost assumptions, warnings,
and limitations.

## Cost assumptions

Twelve Data provides OHLC only (no historical bid/ask). Spread/slippage/
commission/swap are explicit ASSUMPTIONS, fully configurable in
`ResearchConfig`. An extension point exists for a future bid/ask provider.

## No profitability claims

Reports are historical research only. No configuration is claimed optimal, and
the short current dataset is explicitly insufficient for validation.