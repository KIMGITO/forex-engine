"""Provider-independent historical dataset ingestion.

Converts a raw historical FX file (CSV/TSV) into our internal :class:`Candle`
model, without touching any analytical engine. Supports OHLC (and optional
bid/ask columns) plus symbol/timeframe normalization and column aliases.

The adapter is schema-tolerant (common Kaggle / HistData / Dukascopy-CSV
conventions) but STRICT about price validity: any row that cannot be parsed to
a valid OHLC record is rejected, never fabricated.

It is NOT a market-data *provider* (no live/feed semantics). It ingests local
raw files the user has placed under ``data/raw/<source>/``. It never downloads
or invents data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data.models import Candle

__all__ = [
    "ALIAS_COLUMNS",
    "HistoricalIngestionConfig",
    "HistoricalIngestionResult",
    "infer_timeframe_from_filename",
    "ingest_csv_file",
]

# Common column aliases -> canonical column.
ALIAS_COLUMNS = {
    "timestamp": ("timestamp", "datetime", "date", "time", "datetime_utc", "gmt_time", "ts", "time_utc"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "vol", "tick_volume", "tickvol"),
    "bid": ("bid", "bid_price"),
    "ask": ("ask", "ask_price"),
}

_TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}


@dataclass(frozen=True)
class HistoricalIngestionConfig:
    """Configuration for ingesting a raw historical file."""

    symbol: str
    timeframe: str
    source: str  # e.g. "histdata", "dukascopy", "kaggle"
    dataset_name: str = ""
    source_url: str = ""
    # None = auto-detect comma / semicolon / tab.
    delimiter: str | None = None
    has_header: bool = True
    timezone: str = "UTC"
    # When the raw file is M1, optionally aggregate to a coarser timeframe.
    aggregate_to: str | None = None


@dataclass
class HistoricalIngestionResult:
    """Summary of a successful (or partial) file ingestion."""

    symbol: str
    timeframe: str
    source: str
    rows_parsed: int = 0
    rows_rejected: int = 0
    candles: list[Candle] = field(default_factory=list)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "rows_parsed": self.rows_parsed,
            "rows_rejected": self.rows_rejected,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "errors": self.errors,
        }


def _resolve_column(header: list[str]) -> dict[str, str]:
    """Map header names to canonical columns using the alias table."""
    mapping: dict[str, str] = {}
    lowered = {h.strip().lower().replace(" ", "_"): i for i, h in enumerate(header)}
    for canonical, aliases in ALIAS_COLUMNS.items():
        for alias in aliases:
            idx = lowered.get(alias)
            if idx is not None:
                mapping[canonical] = header[idx]
                break
    return mapping


def _parse_ts(value: str, tz: str) -> datetime:
    """Parse a timestamp string to a tz-aware UTC datetime."""
    v = str(value).strip()
    parsed = None
    for candidate in (
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y%m%d %H%M%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            parsed = datetime.strptime(v, candidate)  # noqa: DTZ007 - converted to aware below
            break
        except ValueError:
            continue
    if parsed is None:
        # Fall back to pandas (flexible ISO).
        try:
            parsed = pd.Timestamp(v).to_pydatetime()
        except Exception as exc:
            raise ValueError(f"unparseable timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc if tz == "UTC" else None)
    return parsed.astimezone(timezone.utc)


def infer_timeframe_from_filename(filename: str) -> str | None:
    """Best-effort timeframe inference from a raw filename.

    Handles both named tokens (``15m``, ``h1``, ``1day``) and the numeric-minute
    suffix convention used by HistData (e.g. ``EURUSD_15.csv`` = M15,
    ``EURUSD_60.csv`` = H1, ``EURUSD_240.csv`` = H4, ``EURUSD_1440.csv`` = D1).
    """
    low = (filename or "").lower()
    for token, tf in [
        ("1min", "M1"), ("min1", "M1"), ("m1", "M1"), ("1m", "M1"),
        ("5min", "M5"), ("min5", "M5"), ("m5", "M5"), ("5m", "M5"),
        ("15min", "M15"), ("min15", "M15"), ("m15", "M15"), ("15m", "M15"),
        ("30min", "M30"), ("min30", "M30"), ("m30", "M30"), ("30m", "M30"),
        ("1hour", "H1"), ("hour1", "H1"), ("1h", "H1"), ("h1", "H1"),
        ("4hour", "H4"), ("4h", "H4"), ("h4", "H4"),
        ("1day", "D1"), ("daily", "D1"), ("1d", "D1"), ("d1", "D1"),
        ("1week", "W1"), ("weekly", "W1"),
    ]:
        if token in low:
            return tf
    # HistData numeric-minute suffix: <SYMBOL>_<MINUTES>.<ext> -> timeframe.
    import re

    m = re.search(r"_(\d{1,4})(?:\.|$)", low)
    if m:
        minutes = int(m.group(1))
        mapping = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1"}
        return mapping.get(minutes)
    return None


def _to_candle(
    row: dict,
    symbol: str,
    timeframe: str,
    tz: str,
) -> Candle | None:
    """Build a validated Candle from a resolved row dict, or None if invalid."""
    try:
        ts = _parse_ts(row["timestamp"], tz)
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        vol_raw = row.get("volume")
        volume = float(vol_raw) if vol_raw not in (None, "") else None
        # Candle's pydantic validator enforces OHLC sanity at construction.
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=volume,
        )
    except Exception:  # noqa: BLE001 - any malformed row is rejected, not faked
        return None


def _aggregate(candles: list[Candle], timeframe: str) -> list[Candle]:
    """Aggregate M1-scale candles into a coarser timeframe (open-first,
    high-max, low-min, close-last). An incomplete trailing candle is dropped."""
    if not candles:
        return []
    minutes = _TIMEFRAME_MINUTES.get(timeframe.upper())
    if minutes is None:
        return candles
    df = pd.DataFrame([c.model_dump() for c in candles])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    g = df.resample(f"{minutes}min", label="left", closed="left")
    agg = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    agg = agg.dropna(subset=["open", "close"])
    out: list[Candle] = []
    for ts_idx in agg.index:
        r = agg.loc[ts_idx]
        ts_dt = ts_idx.to_pydatetime()
        out.append(Candle(
            symbol=candles[0].symbol,
            timeframe=timeframe,
            timestamp=ts_dt,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=None if pd.isna(r["volume"]) else float(r["volume"]),
        ))
    return out


def ingest_csv_file(
    path: Path,
    config: HistoricalIngestionConfig,
) -> HistoricalIngestionResult:
    """Read a raw CSV/TSV file and produce validated :class:`Candle` objects.

    The candles are NOT persisted here — the caller hands them to the existing
    :class:`app.research.dataset.PartitionedResearchRepository.merge_candles`
    (idempotent). This keeps ingestion pure and the persistence layer unchanged.
    """
    result = HistoricalIngestionResult(
        symbol=config.symbol,
        timeframe=config.timeframe,
        source=config.source,
    )

    delimiter = config.delimiter
    if delimiter is None:
        with open(path, encoding="utf-8", errors="replace") as _fh:
            sample = _fh.read(1000)
        if "\t" in sample:
            delimiter = "\t"
        else:
            delimiter = ";" if sample.count(";") > sample.count(",") else ","

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        result.errors.append("empty file")
        return result

    colmap: dict[str, str | int] = {}
    if config.has_header:
        header = rows[0]
        colmap = _resolve_column(header)  # type: ignore[assignment]
        missing = [c for c in ("timestamp", "open", "high", "low", "close") if c not in colmap]
        if missing:
            result.errors.append(f"missing required columns: {missing}; header={header}")
            return result
        data_rows = rows[1:]
    else:
        # Fixed order: timestamp,open,high,low,close[,volume]
        colmap = {"timestamp": 0, "open": 1, "high": 2, "low": 3, "close": 4}
        data_rows = rows

    candles: list[Candle] = []
    for raw in data_rows:
        if not raw or len(raw) < 5:
            result.rows_rejected += 1
            continue
        row: dict[str, str] = {}
        for canonical, ref in colmap.items():
            if isinstance(ref, int):
                row[canonical] = raw[ref] if ref < len(raw) else ""
            else:
                try:
                    row[canonical] = raw[rows[0].index(ref)]
                except (ValueError, IndexError):
                    row[canonical] = ""
        c = _to_candle(row, config.symbol, config.timeframe, config.timezone)
        if c is None:
            result.rows_rejected += 1
            continue
        candles.append(c)
        result.rows_parsed += 1

    if not candles:
        result.errors.append("no valid candles parsed")
        return result

    # Deduplicate + sort chronologically (same discipline as the rest of the app).
    by_ts: dict[datetime, Candle] = {}
    for c in candles:
        by_ts[c.timestamp] = c
    candles = [by_ts[k] for k in sorted(by_ts)]
    result.candles = candles

    if config.aggregate_to:
        result.candles = _aggregate(candles, config.aggregate_to)
        result.timeframe = config.aggregate_to

    result.first_timestamp = result.candles[0].timestamp
    result.last_timestamp = result.candles[-1].timestamp
    return result