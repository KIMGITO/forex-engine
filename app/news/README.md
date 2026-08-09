# News & Economic Events Intelligence Layer

This module provides **market context** for Forex analysis. It collects,
normalizes, classifies, stores, and analyzes economic events. It does **not**
generate trading signals, place trades, or make autonomous decisions.

## Event Model

`EconomicEvent` is provider-independent:

| Field | Required | Meaning |
|---|---|---|
| `event_id` | yes | Unique provider id |
| `scheduled_at` | yes | Scheduled release time (tz-aware) |
| `country` / `currency` | yes | Primary country/currency |
| `affected_currencies` | no | Multi-currency events |
| `event_name` | yes | Display name |
| `category` | no (default `other`) | Canonical category enum |
| `importance` | yes | `low` / `medium` / `high` / `unknown` |
| `actual` / `forecast` / `previous` | no | Numeric values (or `None`) |
| `unit` / `source` / `url` / `provider` | no | Metadata |
| `received_at` / `released_at` | no | Timestamps |
| `available_from` | no | Look-ahead guard (see below) |

## Provider Abstraction

`BaseEconomicCalendarProvider` is the abstract interface (`fetch_event`,
`fetch_events_between`). `MockEconomicCalendarProvider` is a deterministic
synthetic provider for development/testing only — it is not connected to any
real economic-calendar service. Future integrations (calendar APIs, RSS, news
feeds) must implement the abstract interface without changing the rest of the
layer.

## Normalization

`normalize_provider_event(raw, provider)` maps:

- **Timestamps** — any ISO/date string to a tz-aware datetime (timezone-name aware).
- **Importance** — provider labels (`red`, `3`, `high`, `★★★`, `2`, `green`, ...)
  to the canonical enum. Unrecognized labels map to `unknown` **with a warning**.
- **Category** — keyword matching (`CPI`, `nonfarm`, `rate decision`, `GDP`,
  `retail sales`, ...) to the canonical categories.
- **Currency** — country code (`US`, `EU`, `GB`, `JP`, ...) to a currency list;
  unknown countries map to `[]` (never invented).
- **Numerics** — malformed values become `None` (never guessed).

Missing/malformed data is never silently filled.

## Importance / Categories

Importance: `LOW`, `MEDIUM`, `HIGH`, `UNKNOWN`. Raw provider labels preserved
in `provider_importance`.

Categories: `inflation`, `employment`, `interest_rate`, `GDP`, `manufacturing`,
`services`, `consumer`, `housing`, `trade`, `central_bank`, `speech`, `other`.

## Surprise Calculation

`calculate_surprise(event)` returns an **objective** `surprise = actual - forecast`
(plus a normalized percentage when the forecast is non-zero). It does **not**
claim that a positive surprise is bullish or a negative surprise is bearish —
market reaction depends on many factors outside this module.

## Risk Windows

`RiskWindowConfig` maps importance to pre/post windows in minutes. Defaults are
**DEVELOPMENT values only** (HIGH 30/30, MEDIUM 15/15, LOW 5/5) and are not
claimed to be optimal. Example:

```python
cfg = RiskWindowConfig(high_pre=60, high_post=60)
```

`pair_risk_context(events, "EUR/USD", now, cfg)` considers both EUR and USD
events and returns active/upcoming events, time-until/since, importance, and a
message — answering "is EUR/USD inside a high-impact window?" without making a
trade decision.

## Availability & Look-Ahead

Critical distinction:

- **SCHEDULED information** — the event is scheduled; only `forecast`/`previous`
  may be known before release.
- **RELEASED information** — the `actual` value becomes available only at
  `released_at`.

`available_from` (defaulting to `released_at` when released, else `scheduled_at`)
is the earliest a consumer may legally use an event. A simulated strategy must
never see `actual` before `released_at`. If historical providers do not supply a
reliable release timestamp, this limitation is documented — the system never
invents one.

## Repository

`ParquetEconomicEventRepository` stores events to
`data/processed/economic_events.parquet`. It is provider-independent and
intended for local development; it can later be replaced with Supabase,
PostgreSQL, or another database.

## Scope Boundaries

No trading signals, strategies, AI/ML, broker APIs, order execution, risk
position sizing, FastAPI, Supabase, or frontend in this layer.
