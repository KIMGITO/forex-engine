# Market Structure & Liquidity Analysis Engine

This module converts validated OHLC market data into measurable market-structure
information. It is a **research/analysis layer** — it describes the market. It
does **not** generate trading signals, entries, stop losses, or risk decisions.

---

## What We Measure

| Concept | Mathematical Rule |
|---|---|
| **Swing High** | Bar `i` where `high[i] > high[j]` for all `j` in `[i-left, i+right]`, excluding `i`. |
| **Swing Low** | Bar `i` where `low[i] < low[j]` for all `j` in `[i-left, i+right]`, excluding `i`. |
| **Higher High** | New confirmed swing high with `price >` prior confirmed swing high. |
| **Lower High** | New confirmed swing high with `price <` prior confirmed swing high. |
| **Higher Low** | New confirmed swing low with `price >` prior confirmed swing low. |
| **Lower Low** | New confirmed swing low with `price <` prior confirmed swing low. |
| **Wick Breach** | `high[i] > level` but `close[i] <= level` (upward) — intrabar penetration only. |
| **Close Break** | `close[i] > level` (upward) — close beyond the level. |
| **Confirmed Break** | A close break sustained for `confirm_bars` subsequent closes (or, when `confirm_bars == 0`, a close beyond the level by at least `min_move_pct` percent). |
| **Equal Highs / Equal Lows** | `>= min_swings` swing highs (lows) within `tolerance_pct` percent of each other. |
| **Liquidity Zone** | Bounded area `[lower, upper]` around a cluster of equal swings. |
| **High Sweep** | `high[i] > zone.upper` against a prior `equal_highs` zone, then a close `<= zone.upper` within `sweep_bars` bars. |
| **Low Sweep** | Mirror against a prior `equal_lows` zone. |
| **Displacement** | `range_ratio = (high - low) / ATR[window]`, body ratio, direction, classified vs trailing percentiles. |
| **Range/Consolidation** | Contiguous run of `>= min_range_bars` bars where `ATR / SMA(ATR, range_window) <= compression_threshold`. |

## Confirmation Behavior

Swing detection is **not causal in real-time**. A swing at bar `i` requires
`right` future bars to confirm. Every swing therefore carries:

- **`timestamp`** — the bar the swing refers to.
- **`confirmation_timestamp`** — the bar at which the swing becomes knowable
  (`i + right`).
- **`available_from`** — the earliest a consumer may legally use the event
  (equals `confirmation_timestamp` for swings).

Break, sweep, displacement, and range events are constructed from confirmed
inputs, so their `available_from` reflects when the underlying information was
actually knowable.

## Look-Ahead Considerations (CRITICAL)

- A consumer must **never** act on an event at its `timestamp` if
  `available_from` is later.
- Swing detection is intended for **historical analysis / replay only**.
- Displacement classification is **causal**: it uses only ratios up to and
  including the current bar.
- Break detection only uses swing levels that are already confirmed at the bar
  being examined.
- Range detection uses only trailing windows.

## Potential Liquidity Zones ≠ Known Order-Book Liquidity

A **potential liquidity zone** is a purely structural statement about price:
multiple swing highs (or lows) clustered within a tolerance. From OHLC data
alone we **cannot** know:

- whether stop orders actually exist at that level,
- who placed them,
- or how much liquidity is present.

We therefore say **"potential liquidity zone"**, never
**"there are definitely institutional stop orders here."** This distinction is
enforced in the documentation and model field descriptions throughout the
module.

## Parameters

All parameters are configurable via `MarketStructureConfig`:

| Parameter | Default | Purpose |
|---|---|---|
| `swing_left` / `swing_right` | 3 / 3 | Swing lookback/lookforward window |
| `confirm_bars` | 2 | Bars a close break must be sustained for confirmation |
| `min_move_pct` | 0.0 | Min percent move for confirmed break when `confirm_bars == 0` |
| `tolerance_pct` | 0.05 | Percent tolerance for grouping equal swings |
| `min_swings` | 2 | Minimum swings to form a liquidity zone |
| `sweep_bars` | 3 | Max bars for the sweep return close |
| `atr_window` | 14 | ATR window for displacement/range compression |
| `p_extreme` / `p_large` / `p_small` | 95 / 80 / 20 | Displacement classification percentiles |
| `compression_threshold` | 0.85 | Max ATR/SMA(ATR) ratio for a compressed bar |
| `range_window` | 30 | Trailing window for ATR average |
| `min_range_bars` | 10 | Min contiguous compressed bars for a range event |

## Usage

```python
from app.market_structure import MarketStructureEngine

engine = MarketStructureEngine()
result = engine.analyze(data, symbol="EURUSD", timeframe="1h")
# result: MarketStructureResult with swings, structure, breaks,
#         liquidity_zones, sweeps, displacement, ranges
```

## Known Limitations

- Swing detection requires `left + right + 1` bars; shorter series raise
  `SwingDetectionError`.
- Displacement `range_ratio` is `NaN` for bars where ATR is not yet defined.
- Sweep detection requires a prior liquidity zone; a wick without a prior level
  is not a sweep.
- Range detection requires `range_window + atr_window` bars.
- No multi-timeframe synchronization is implemented at this stage.