# Step 13B — Strategy Research & Walk-Forward Validation Engine

An **ALTERNATIVE research path** to the giant Step 13 MTF pipeline.

**Purpose**: Determine whether a trading strategy has a robust, repeatable
statistical edge and produce a machine-readable strategy configuration that
can later be consumed by the live trading application.

## Why Step 13B?

The existing Step 13 pipeline attempts to process 6 FX symbols × M15+H1 ×
higher-timeframe MTF analysis across very large historical datasets. While
optimized substantially, production validation exposed a memory problem on
the 8 GB development machine. The system can process bounded datasets, but
the full MTF research pipeline is too memory-heavy.

Step 13B does NOT produce millions of MTF context objects. It processes one
symbol → one timeframe → one walk-forward window at a time, with strict
bounded memory, incremental resumable state, and compact research artifacts.

## Core Pipeline

```
Historical Data
→ Features
→ Market Structure
→ Regime
→ Signal Generation
→ Risk Engine (Step 15)
→ Backtest (existing engine)
→ Walk-Forward Validation
→ Robustness Analysis
→ Strategy Evaluation
→ Deployable Strategy Configuration
```

The pipeline is **causal**: no future information leaks into features,
structure, regime, signals, risk decisions, or validation metrics.

## Memory Architecture (8 GB machine)

- Process **one symbol** at a time
- Process **one timeframe** at a time
- Process **one walk-forward window** at a time
- Never retain MTF context for the entire historical dataset
- Release DataFrames and analytical objects after each window
- `gc.collect()` after each window (configurable)
- Persist completed research results **incrementally** via atomic writes
- Log RSS before/after each major stage
- Configurable memory limit (`max_rss_mb`); abort safely before OOM

## Walk-Forward Windows

Each window is strictly chronological and disjoint:
```
TRAIN → VALIDATION → TEST
```
The window advances by `step_days`. The TEST set is **never** used to
optimize parameters — it is used only for final evaluation.

## Validation Scoring Formula

The validation score is an explicitly weighted combination of components,
each normalized to 0..1:

| Component | Weight | Formula |
|-----------|--------|---------|
| Expectancy | 0.25 | `min(max(median_expectancy_R / 0.20, 0), 1)` |
| Drawdown | 0.20 | `1 - min(max_drawdown / 0.50, 1)` |
| Consistency | 0.20 | `fraction_of_test_windows_with_positive_expectancy` |
| Trade Count | 0.15 | `min(total_trades / 100, 1)` |
| Parameter Stability | 0.10 | `param_stability` |
| Symbol Consistency | 0.10 | `positive_symbol_fraction` |

**Hard gates** (any failure ⇒ status below VALIDATED):
- G1: `total_trades >= min_total_trades`
- G2: `completed_windows >= min_windows`
- G3: `max_drawdown <= max_allowed_drawdown`
- G4: `median_expectancy_R > min_expectancy_r`
- G5: `win_frac >= min_windows_profitable`

Status assignment: `INSUFFICIENT_DATA`, `OVERFIT`, `REJECTED`, `PROMISING`,
`VALIDATED`, `NOT_VALIDATED`.

## Resumability

State file layout:
```json
{
  "EURUSD/M15": {
    "window_1": "complete",
    "window_2": "complete",
    "window_3": "running"
  }
}
```

On `--resume`, completed windows are skipped and processing resumes from the
first incomplete window. A window is only marked complete AFTER its artifacts
are atomically committed and validated.

## Research Artifacts

Per symbol/timeframe:
```
research/results/step13b/<SYMBOL>/<TIMEFRAME>/
    strategy_validation.json
    window_metrics.parquet
    trade_log.parquet
    monthly_metrics.parquet
    regime_metrics.parquet
    research_summary.json
    windows/
        window_000.json
        window_000_trades.parquet
        ...
```

All artifacts are written atomically (temp file + fsync + `os.replace`).

## CLI Commands

### 10K validation
```bash
python3 -m app.research.step13b.runner \
    --symbols EURUSD \
    --timeframes M15 \
    --storage-root data/processed \
    --output-root research/results/step13b \
    --max-bars 10000 \
    --train-days 30 \
    --validation-days 10 \
    --test-days 10 \
    --step-days 20 \
    --max-rss-mb 2000
```

### 50K validation
```bash
python3 -m app.research.step13b.runner \
    --symbols EURUSD \
    --timeframes M15 \
    --storage-root data/processed \
    --output-root research/results/step13b \
    --max-bars 50000 \
    --train-days 90 \
    --validation-days 30 \
    --test-days 30 \
    --step-days 30 \
    --max-rss-mb 2000
```

### Resume
Add `--resume` to skip completed walk-forward windows.

## Reused Components

- `app.features.FeatureEngine` — feature calculation
- `app.market_structure.MarketStructureEngine` — market structure
- `app.regime.RegimeEngine` — regime detection
- `app.strategy.HistoricalSignalScanner` — causal signal scanning
- `app.strategy.TrendStructureStrategy` / `LiquidityReversalStrategy`
- `app.risk.RiskEngine` — Step 15 risk management
- `app.backtest.EventBacktester` — existing backtest engine
- `app._causal_index` — causal index infrastructure
- `app.research.dataset.PartitionedResearchRepository` — data loading

## Honest Validation

A strategy is **NOT** marked validated merely because its backtest made
money. The pipeline requires:
- positive expectancy that survives unseen (TEST) data
- sufficient trade count
- consistency across walk-forward windows
- stability across market regimes
- parameter stability
- acceptable drawdown

If the existing strategy fails validation, the pipeline reports that status
honestly (`REJECTED`, `OVERFIT`, `INSUFFICIENT_DATA`) rather than changing
criteria to make it pass.