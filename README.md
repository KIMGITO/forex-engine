# forex-engine

A modular, production-grade AI-assisted Forex quantitative platform.

## Architecture

```
Market Data API -> Adapter -> Normalization -> Validation -> Internal Candle Model
    -> Repository -> Feature Engine -> downstream modules
```

The platform is built in strictly separated layers. This document covers the
**Quantitative Feature Engine** (`app/features/`).

---

## What is a Feature?

A **feature** is a causally computed numerical transformation of validated OHLC
market data. Examples: simple returns, ATR, RSI, SMA, rolling volatility.

Features are **pure measurements** — they describe what the market *did* or
*is doing*. They are **not** trading signals:

- `RSI = 72` is a feature.
- "RSI = 72 implies SELL" is a strategy decision.

That decision belongs to a future strategy layer, never to the feature engine.

## Why Features Are Separate From Strategies

1. **Reusability** — The same features feed market-structure analysis,
   liquidity analysis, volatility analysis, correlation analysis, regime
   detection, strategy research, backtesting, AI analysis, and risk management.
2. **Zero leakage** — Features are computed with strict look-ahead protection.
   Strategies consume features; they never influence feature computation.
3. **Testability** — Each feature is independently unit-tested against
   deterministic data.
4. **Extensibility** — New features can be added without modifying existing
   consumers or the engine itself.

## Module Layout

```
app/features/
├── __init__.py        # Public re-exports
├── errors.py          # FeatureError, UnknownFeatureError, InsufficientDataError
├── models.py          # Feature model + FeatureDefinition registry
├── _lookahead.py      # Documented look-ahead-bias protection utilities
├── returns.py         # simple_returns, pct_returns, log_returns
├── volatility.py      # rolling_std, annualized_volatility, atr, rolling_atr, volatility_percentile
├── momentum.py        # rate_of_change, price_momentum, rsi
├── trend.py           # sma, ema, distance_from_ma, ma_slope
├── correlation.py     # align_price_dfs, pairwise_correlation, rolling_correlation, correlation_matrix
└── engine.py          # FeatureEngine (central, extensible orchestrator)
```

## Available Features

### Returns (`app/features/returns.py`)

| Function | Formula | Defaults |
|----------|---------|----------|
| `simple_returns` | `(p_t - p_{t-1}) / p_{t-1}` | `price_col="close"` |
| `pct_returns` | `(p_t / p_{t-1} - 1) * 100` | `price_col="close"` |
| `log_returns` | `log(p_t / p_{t-1})` | `price_col="close"` |

All returns are lagged by one period (`shift(1)`) — the return classified at
timestamp `T` uses only prior price information.

### Volatility (`app/features/volatility.py`)

| Function | Description | Defaults |
|----------|-------------|----------|
| `rolling_std_init` | Rolling std of price | `window=20`, `ddof=1` |
| `annualized_volatility` | Rolling std of returns × √(periods/year) | `window=20`, `timeframe="1h"` |
| `atr` | Average True Range (Wilder smoothing) | `window=14` |
| `rolling_atr` | Rolling (simple mean) ATR | `window=14` |
| `volatility_percentile` | Rolling percentile rank of volatility | `window=20`, `rank_window=100` |

`periods_per_year` is derived from the timeframe (`1m`→525600, `1h`→8760,
`1d`→365, etc.). All windows are trailing and right-aligned.

### Momentum (`app/features/momentum.py`)

| Function | Description | Defaults |
|----------|-------------|----------|
| `rate_of_change` | `(p_t / p_{t-period} - 1) * 100` | `period=10` |
| `price_momentum` | `p_t - p_{t-period}` | `period=10` |
| `rsi` | Relative Strength Index (Wilder), [0, 100] | `period=14` |

### Trend (`app/features/trend.py`)

| Function | Description | Defaults |
|----------|-------------|----------|
| `sma` | Simple moving average | `period=20` |
| `ema` | Exponential moving average (`adjust=False`) | `period=20` |
| `distance_from_ma` | `(p / MA - 1) * 100` | `period=20`, `ma_type="sma"` |
| `ma_slope` | ROC of the moving average | `period=20`, `slope_periods=5` |

### Correlation (`app/features/correlation.py`)

| Function | Description |
|----------|-------------|
| `align_price_dfs` | Align multiple instruments to a union timestamp index |
| `pairwise_correlation` | Pearson correlation between two instruments |
| `rolling_correlation` | Trailing rolling Pearson correlation |
| `correlation_matrix` | Symmetric correlation matrix |

Correlation aligns timestamps first — it does **not** assume two instruments
have identical observation times. Missing observations are handled explicitly
as `NaN` and dropped pairwise.

## Feature Engine

The `FeatureEngine` orchestrates computation of selected features:

```python
from app.features import FeatureEngine

engine = FeatureEngine()
features = engine.calculate(
    data=my_dataframe,          # index = UTC timestamps; columns: close/high/low
    features=["rsi", "atr", "sma", "volatility_percentile"],
    params={
        "rsi": {"period": 14},
        "sma": {"period": 50},
    },
)
# features: DataFrame indexed by timestamp, one column per feature
```

### Extensibility

New features are registered without modifying the engine:

```python
from app.features import FeatureEngine, FeatureDefinition

def my_feature(data, **kwargs):
    return data["close"].rolling(window=kwargs["window"]).mean()

engine.register_feature(
    "my_feature",
    FeatureDefinition(
        name="my_feature",
        category="custom",
        description="Custom rolling mean",
        default_params={"window": 5},
    ),
    my_feature,
)
```

Unknown feature names raise `UnknownFeatureError` with the list of available
features.

## Look-Ahead-Bias Rules

**A feature at timestamp `T` must use only information available at or before `T`.**

- All rolling/expanding windows are **trailing and right-aligned** — never centered.
- Returns are computed with a one-period lag (`shift(1)`).
- Future candles must never influence a past feature value.

This invariant is documented in `app/features/_lookahead.py` and enforced by
regression tests in `tests/features/test_lookahead.py`, which verify that
modifying future candles does not change feature values for earlier timestamps.

## Numerical Correctness

- **NaN handling** — Insufficient lookback produces `NaN`; no silent filling.
- **Insufficient data** — Functions raise `InsufficientDataError` when the
  series is too short for the requested window.
- **Floating point** — Vectorized Pandas/NumPy operations; no manual loops.
- **Timezone-aware timestamps** — Preserved throughout; inputs are sorted by index.
- **Missing candles / irregular data** — Handled explicitly (e.g., correlation
  alignment produces `NaN` for missing observations).

## Performance

- Vectorized Pandas/NumPy throughout (`.rolling`, `.ewm`, `.shift`, `.diff`).
- No Python loops in feature math.
- Minimal DataFrame copies.
- No premature C++/GPU/distributed infrastructure.

## Verification

```bash
pytest tests/ -q
ruff check app/features/ tests/
mypy app/features/
```

## Scope Boundaries

The feature engine does **not** implement:

- Trading signals or strategy rules
- Market regime detection
- Machine learning / AI analysis
- News analysis
- Risk management / position sizing
- Broker APIs / order execution
- FastAPI / Supabase / frontend

Those belong to later stages of the platform.