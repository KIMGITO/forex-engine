# Multi-Timeframe Research & Signal Context Engine (`app/mtf/`)

Provides strictly causal higher-timeframe context for strategies. Timeframe
alignment follows the **completed-candle rule**, so a lower-timeframe
observation never sees an unfinished higher-timeframe candle.

> **Multi-timeframe context is analytical information, not a prediction guarantee.**

## Timeframe alignment

The MTF engine accepts a configurable hierarchy (never hard-coded):

- `base_timeframe` — the acting timeframe (e.g. `15m`)
- `higher_timeframes` — ordered nearest→farthest (e.g. `1h`, `4h`, `1d`)

For each base-bar observation, every higher-timeframe tier is aligned to the
**last fully completed candle** before that observation moment.

## Completed-candle rule

For a timeframe of period `P` minutes, an observation at `T` may only use a
candle whose slot `[open, open+P)` is fully closed, i.e. `T >= open + P`:

- The currently-open candle is **never** available.
- The last completed slot open = `floor(T/P) * P - P`, close = `floor(T/P)*P`,
  `available_from` = close.

Example (`H4`, P=240):

| Observation | Available H4 candle | Available_from |
|---|---|---|
| 08:15 / 09:00 / 11:45 | 04:00–08:00 | 08:00 |
| 12:00 (or later) | 08:00–12:00 | 12:00 |

## `available_from`

Every `MtfWindow`, `TimeframeContext`, and `MtfContext` carries an explicit
`available_from`. A lower-timeframe strategy may only ever read context whose
`available_from <= its observation moment`. This preserves the platform-wide
look-ahead discipline.

## Missing data behavior

- Missing candles / weekend / holiday gaps: the engine steps back to the
  previous completed slot (bounded by `max_gap_lookback`).
- If no completed candle exists within the lookback, the tier is surfaced as
  `present=False` with `trend=volatility=market=structural_bias=None`.
- **No value is fabricated.** `max_missing_htf_allowed` limits how many tiers
  may be absent before the context is considered unusable.

## MTF hierarchy

`MtfContext.hierarchy` is ordered: `[base_tier] + higher_timeframes`. Each
`TimeframeContext` exposes: trend_state, volatility_state, market_state,
structural_bias, liquidity zones, sweeps, and news-risk maximum — all filtered
by `available_from <= observation`.

## Alignment states

`MtfAlignmentState` values are **explainable**, not derived from any magic
score:

- `ALIGNED_LONG` — base long and all known HTF directions are long.
- `ALIGNED_SHORT` — base short and all known HTF directions are short.
- `CONFLICTED` — at least one known HTF direction opposes the base.
- `UNKNOWN` — insufficient known HTF tiers (< `min_aligned`).

`strength` on each tier is the fraction of known directions agreeing with the
base — **alignment agreement, not probability**.

## Strategy integration (additive, backward compatible)

`StrategyConfig` gains optional MTF fields (all default to disabled/0, so an
existing strategy behaves exactly as before):

```python
StrategyConfig(mtf_enabled=True, mtf_min_aligned=2, mtf_require_no_conflict=True, ...)
```

- `StrategyContext.mtf_context()` returns the causal MTF context (or `None`).
- `Strategy.mtf_gates_pass(base_dir, mtf_ctx)` evaluates configured MTF gates;
  disabled → always passes.
- `HistoricalSignalScanner.scan(..., mtf_contexts=[...])` wires MTF context per
  bar and embeds MTF evidence into `Signal.metadata["mtf"]` (serializable).

## Test coverage

- Alignment math (09:45 H1/H4 case, boundary-at-close, gaps, duplicates, tz).
- `tests/mtf/test_lookahead.py` — all 8 mandatory look-ahead scenarios.
- Determinism, missing-data, and real Twelve Data M15/H1/H4/D1 integration.

## Limitations

- Requires sufficient history per higher timeframe (the engine surfaces
  `present=False` when unavailable — never fabricated).
- The current 120-candle EUR/USD dataset is **insufficient** for profitability
  validation; this engine validates alignment + casual correctness only.
- Development defaults are not optimized for any market.
- Simulated spread/slippage/commission are assumptions (no historical bid/ask).