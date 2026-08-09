# Research-Grade Event-Driven Backtesting Engine

Deterministic, causal historical simulation. NOT a trading strategy; does NOT
execute real trades. Provides the environment in which future strategies are
tested.

## Architecture

```
app/backtest/
├── models.py      frozen typed models (Order/Position/PortfolioState/BacktestResult/...)
├── config.py      BacktestConfig — documented development defaults
├── clock.py       deterministic sequential BacktestClock
├── costs.py       Spread/Slippage/Commission/Swap models + pip utilities
├── orders.py      OrderIntent → Order lifecycle
├── execution.py   fills, SL/TP resolution, gap handling
├── portfolio.py   long/short accounting, margin, P&L, fees
├── engine.py      EventBacktester (causal loop) + Strategy + BacktestContext
├── metrics.py     drawdown + benchmark curves
├── reports.py     structured text report
└── README.md
```

## Causal discipline (most important property)

The engine processes bars strictly chronologically. The strategy receives
only a `BacktestContext` that exposes:
- current candle (never a future row)
- `history()` capped at the current bar
- features sliced at the current bar
- market-structure events filtered by `available_from <= now`
- news events filtered by availability
- regime observations filtered by `available_from <= now`
- current portfolio values

It is architecturally impossible to access future candles through the public
API.

## Order model

- `MARKET`, `LIMIT`, `STOP` orders with symbol/side/quantity/price/SL/TP.
- `OrderIntent` (strategy-facing) → `Order` (lifecycle:
  pending → submitted → accepted → filled/rejected).

## Execution model

- BUY fills at simulated ask; SELL fills at simulated bid.
- Spread/slippage/commission are SIMULATED assumptions — the underlying
  Twelve Data OHLC dataset has no historical bid/ask. Stated in every report.

## Fill policies

- **SL/TP ambiguity**: when one OHLC bar touches both SL and TP, OHLC alone
  cannot reveal intrabar sequence. Default policy is conservative
  (`CONSERVATIVE_SL_FIRST` = assume stop-loss fills first), configurable.
- **Gaps**: fills through a gap use worse-of(level, open); never assumes a
  better-than-market fill.

## Costs

- `SpreadModel` (baseline `FixedSpreadModel`); extension points for
  historical/time-of-day/volatility spread models.
- `SlippageModel` (baseline deterministic fixed pips; no randomness).
- `CommissionModel` (zero/fixed/percentage).
- `SwapModel` (baseline `NoSwapModel`; real broker swap schedules must be
  supplied later — never invented).

## Portfolio

Long/short positions, FIFO-average entry, realized/unrealized P&L
directionally correct per FX side, fees, financing, balance/equity/margin/free
margin. Leverage affects margin only, never profitability.

## Pip utilities

`pip_size_for_symbol` (0.01 for JPY/HUF/VND-quoted, 0.0001 otherwise),
`pip_distance`, notional/P&L in account currency. Documented assumptions; an
explicit override is available via `BacktestConfig.pip_size`.

## Metrics

Total return, net P&L, gross profit/loss, win/loss rate, profit factor,
average win/loss, expectancy, max drawdown + duration, Sharpe/Sortino
(per-bar, only when ≥3 returns), trade count, exposure, average holding time.
Insufficient-data limitations are reported rather than silently omitted.

## Determinism & look-ahead

Same dataset + config + strategy ⇒ identical results (explicitly tested).
Future candle/news/structure/regime mutations must not change past results
(explicitly tested).

## Scope boundaries

No ML/AI, no live/broker/paper execution, no account credentials, no strategy
engine (only an abstraction), no frontend. Historical simulation only.
