# Market Regime Detection Engine

This module classifies the **current market environment** using measurable
information from the data, feature, market-structure, and news layers. It is an
**analysis layer**: it does NOT generate trade signals, execute trades, or use
machine learning.

## Regime definitions

| State | Enum | Meaning |
|---|---|---|
| Trend | `BULLISH / BEARISH / NEUTRAL / UNKNOWN` | Majority vote of 4 causal signals (fast-vs-slow EMA with margin, fast-EMA slope, price distance from MA, recent-return sign) |
| Volatility | `LOW / NORMAL / HIGH / EXTREME / UNKNOWN` | Causal percentile rank of `ATR / price` over a trailing window |
| MarketState | `TRENDING / RANGING / TRANSITION / UNKNOWN` | Multi-factor combination of trend, volatility, structure, range |
| NewsRisk | `CALM / ACTIVE_MEDIUM / ACTIVE_HIGH / UNKNOWN` | Contextual metadata only — never a directional bias |

A final `MarketRegime` contains: `symbol`, `timeframe`, `timestamp`,
`trend_state`, `volatility_state`, `market_state`, `news_risk`, `strength`
(objective 0..1 internal-agreement score, NOT a probability), `metrics`
(objective inputs used), and `available_from`.

## Classification methodology

- **Trend**: majority vote of four causal signals. NEUTRAL on split votes;
  UNKNOWN when insufficient data. The method is documented in
  `app/regime/trend.py` and deterministic tests.
- **Volatility**: `ATR/price` percentile rank over a trailing window.
  Thresholds configurable (LOW ≤ 25th, HIGH ≥ 75th, EXTREME ≥ 95th by default).
- **Structure**: consumes `app.market_structure` output (HH/HL/LH/LL
  sequence bias) — swing detection is NOT duplicated.
- **Range**: consumes `app.market_structure.ranges` — a market is RANGING when
  a range is active and trend is not decisive.
- **Transition**: trend signals conflict, volatility expands beyond the
  `ATR/SMA(ATR)` ratio threshold, structure actively disagrees with trend, or
  a HIGH-impact news window is active (uncertainty only — never direction).

## Configuration

`RegimeConfig` (all values are DOCUMENTED DEVELOPMENT DEFAULTS, not claimed
optimal):

- Trend: `ema_fast=20`, `ema_slow=50`, `ma_margin_pct=0.10`, `slope_periods=5`,
  `distance_ma="ema"`, `distance_ma_period=20`, `recent_return_bars=5`
- Volatility: `atr_window=14`, `percentile_window=100`,
  `vol_low_pct=25`, `vol_high_pct=75`, `vol_extreme_pct=95`
- Structure: `structure_lookback=12`, `min_structure_points=3`
- Range: `range_min_bars=10`
- Transition: `transition_vol_ratio=1.6`, `range_window=30`,
  `transition_conflict_min=2`

## News integration

`NewsRiskState` is **metadata only**. A HIGH-impact event window increases
uncertainty (may contribute to TRANSITION) but is NEVER used to classify
direction (never BULLISH/BEARISH on news). News risk and market direction are
kept strictly separate. This is validated by deterministic tests.

## Inputs

- `data`: OHLC DataFrame indexed by tz-aware timestamps
- `market_structure`: optional precomputed `MarketStructureResult`
- `news_context`: optional `PairRiskContext`

## Look-ahead protection

- All consumed market-structure events are filtered by `available_from <= T`.
- All feature windows are trailing/causal.
- Every regime observation has `available_from == bar.timestamp`.
- Regression tests perturb future candles and verify earlier regimes are
  unchanged.

## Known limitations

- Regime classification is a **model**, not a prediction guarantee. It
  describes the current environment; future price behavior is not guaranteed.
- `strength` is an internal-agreement score, NOT a probability.
- Percentile-based volatility requires `percentile_window` bars of history.
- No ML/AI; no trade signals; no position sizing; no broker integration.
