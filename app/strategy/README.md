# Strategy, Signal & Research Engine

This layer produces **research signals** — typed, deterministic, causal
descriptions of what the market is doing. Signals are **NOT orders**. They
are consumed by the backtester (Step 8), a future paper-trading simulation,
and a future API service.

## Research safeguards (documented, not hidden)

- 120 candles is **insufficient** for strategy validation.
- Development parameters are **not optimized** for any market.
- Simulated spread is **not** historical bid/ask.
- Signals are **not** guaranteed predictions.
- Historical performance does **not** guarantee future performance.
- Out-of-sample and walk-forward testing are **required** before any
  production decision.

## Architecture

```
app/strategy/
├── errors.py      domain exceptions
├── models.py      Signal / Setup / SignalDirection / SignalStrength / ... (frozen)
├── config.py      StrategyConfig — documented development defaults
├── context.py     StrategyContext — causal, timestamp-restricted view
├── signals.py     scoring + risk geometry (rule agreement, NOT probability)
├── base.py        Strategy ABC + SignalToOrderAdapter
├── rules.py       reusable transparent rule checks
├── engine.py      HistoricalSignalScanner + StrategyComparison
├── strategies/
│   ├── trend_structure.py
│   └── liquidity_reversal.py
└── README.md
```

## Signal model

A `Signal` carries:
- id, timestamp, symbol, timeframe
- direction (LONG/SHORT)
- strength (WEAK/MODERATE/STRONG — categorical, NOT probability)
- score (rule agreement 0..max_score)
- entry / stop / target / risk_distance / reward_distance / R:R
- strategy name, regime, market_state, reasons, structure evidence,
  news-risk state, setup identity, status, available_from, metadata

The score is a **rule-agreement** model, never a probability of profit.
Thresholds are configurable (`weak_score_threshold`, etc.).

## Strategy interface

```python
class MyStrategy(Strategy):
    name = "my_strategy"
    def evaluate(self, context: StrategyContext) -> Signal | None:
        ...
```

`StrategyContext` exposes only causal data (current bar, history capped at
current, structure/news/regime filtered by `available_from <= now`). Future
data is unreachable through the public API.

## Strategy 1 — TrendStructureStrategy

LONG (conceptual):

1. Regime supports bullish conditions (trend_state=bullish, strength ≥ min)
2. Market structure bullish (recent HH/HL)
3. Volatility acceptable (≤ configured max state)
4. No prohibited high-impact news window
5. Bullish displacement/confirmation exists (large/extreme, up)

SHORT is the inverse. Incomplete conditions → NO SIGNAL.

## Strategy 2 — LiquidityReversalStrategy

LONG (conceptual):

1. A sell-side liquidity zone exists (equal-lows zone available)
2. A sell-side liquidity sweep occurred (low sweep within lookback)
3. Price returned through the zone level
4. Bullish displacement/confirmation follows
5. Market/news conditions permit

SHORT is the inverse. NO SETUP → NO SIGNAL. Not optimized on this dataset.

## Scoring (documented, deterministic)

Points are additive rule agreement:

- regime direction: 1
- structure: 1
- displacement/confirmation: 2
- volatility acceptable: 1
- (liquidity-reversal adds) return-through-level: 1

Thresholds (dev defaults): WEAK < 2; MODERATE 2–4; STRONG ≥ 5. Configurable.

## Cooldown & duplicates

Each setup has a deterministic identity (`Setup.identity()`). The strategy
records emitted setups; duplicate identities are suppressed.

## Backtest adapter

`SignalToOrderAdapter.to_order_intents(...)` converts a completed signal
(status DETECTED/CONFIRMED) into `OrderIntent` objects. The signal itself is
never an order.

## Scanner & comparison

`HistoricalSignalScanner.scan(...)` processes bars sequentially, emitting a
`SignalScanResult`. `compare_strategies(...)` reports counts, frequency, R:R
statistics — **not profitability**.

## API/Frontend readiness

All models are frozen Pydantic with JSON-friendly fields; no FastAPI,
Supabase, or React yet.