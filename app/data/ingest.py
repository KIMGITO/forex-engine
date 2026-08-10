"""Historical dataset ingestion CLI.

Usage
-----
    python -m app.data.ingest --source kaggle --symbol EURUSD --timeframe M15         data/raw/kaggle/EURUSD_15m.csv

    # aggregate an M1 source up to H1:
    python -m app.data.ingest --source histdata --symbol EURUSD --timeframe M1 \
        --aggregate-to H1 data/raw/histdata/EAUD_1.csv

Reads a raw CSV/TSV, validates every row to a Candle (rejecting malformed rows,
never fabricating), persists through PartitionedResearchRepository (idempotent),
records provenance, and writes data_quality.json / data_quality.txt.

A synthetic deterministic sample is generated with ``--make-sample`` for testing
the ingestion pipeline ONLY — clearly labelled, never real market data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.historical import (
    HistoricalIngestionConfig,
    infer_timeframe_from_filename,
    ingest_csv_file,
)
from app.data.models import Candle
from app.research.data_quality import validate_partition
from app.research.dataset import PartitionedResearchRepository

_SAMPLE_HEADER = "timestamp,open,high,low,close,volume\n"


def _make_sample(path: Path, symbol: str, timeframe: str, rows: int = 5000) -> None:
    """Deterministic synthetic OHLC sample for pipeline testing only."""
    minutes = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h"}[timeframe]
    idx = pd.date_range("2022-01-03", periods=rows, freq=minutes, tz="UTC")
    rng = np.random.default_rng(7)
    price = 1.05 + np.cumsum(rng.normal(0, 3e-4, rows))
    lines = [_SAMPLE_HEADER]
    for ts, p in zip(idx, price):
        o = p
        c = p + rng.normal(0, 2e-4)
        h = max(o, c) + abs(rng.normal(0, 1e-4))
        l = min(o, c) - abs(rng.normal(0, 1e-4))
        lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{o:.5f},{h:.5f},{l:.5f},{c:.5f},{1000}\n")
    path.write_text("".join(lines))
    print(f"[sample] wrote deterministic synthetic sample -> {path}")


def _load_raw_candles(path: Path, config: HistoricalIngestionConfig) -> list[Candle]:
    result = ingest_csv_file(path, config)
    if result.errors:
        raise RuntimeError(f"ingestion failed for {path}: {result.errors}")
    print(f"[ingest] {path.name}: parsed={result.rows_parsed} rejected={result.rows_rejected} "
          f"candles={len(result.candles)} {result.first_timestamp} -> {result.last_timestamp}")
    return result.candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw historical FX CSV into research partitions")
    parser.add_argument("raw_file", help="path to the raw CSV/TSV file")
    parser.add_argument("--source", required=True, help="source name, e.g. kaggle/histdata/dukascopy")
    parser.add_argument("--symbol", required=True, help="canonical symbol, e.g. EURUSD")
    parser.add_argument("--timeframe", default=None, help="timeframe (default: infer from filename)")
    parser.add_argument("--aggregate-to", default=None, help="aggregate M1 source to e.g. H1")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--storage", default="data/research")
    parser.add_argument("--make-sample", action="store_true",
                        help="generate a synthetic deterministic sample to test the pipeline (not real data)")
    parser.add_argument("--output", default="research/results/latest")
    args = parser.parse_args()

    tf = args.timeframe or infer_timeframe_from_filename(Path(args.raw_file).name)
    if not tf:
        raise SystemExit("could not infer timeframe; pass --timeframe")

    raw_path = Path(args.raw_file)
    if args.make_sample:
        if raw_path.suffix != ".csv":
            raise SystemExit("--make-sample requires a .csv target path")
        _make_sample(raw_path, args.symbol, tf)
    if not raw_path.exists():
        raise SystemExit(f"raw file not found: {raw_path}")

    config = HistoricalIngestionConfig(
        symbol=args.symbol,
        timeframe=tf,
        source=args.source,
        dataset_name=args.dataset_name,
        source_url=args.source_url,
        aggregate_to=args.aggregate_to,
    )
    candles = _load_raw_candles(raw_path, config)
    final_tf = config.aggregate_to or tf

    repo = PartitionedResearchRepository(args.storage)
    existing, final = repo.merge_candles(candles, symbol=args.symbol, timeframe=final_tf)
    print(f"[persist] {args.symbol}/{final_tf}: merged to {final} rows (existing {existing})")

    df = repo.load_df(args.symbol, final_tf)
    q = validate_partition(df, args.symbol, final_tf, provider=args.source, native_or_aggregated="native")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dq_path = out / "data_quality.json"
    existing_q = {}
    if dq_path.exists():
        existing_q = json.loads(dq_path.read_text())
    existing_q[f"{args.symbol}|{final_tf}"] = q.to_dict()
    dq_path.write_text(json.dumps(existing_q, indent=2, default=str))

    txt = []
    txt.append("=== DATA QUALITY REPORT ===")
    txt.append(f"source={args.source} dataset={args.dataset_name or 'n/a'}")
    txt.append(f"symbol={args.symbol} timeframe={final_tf}")
    txt.append(f"rows={q.candle_count} first={q.first_timestamp} last={q.last_timestamp}")
    txt.append(f"gaps={q.gap_count} duplicates={q.duplicate_count} invalid_ohlc={q.invalid_ohlc_count}")
    txt.append(f"timezone={q.timezone_status} passed={q.passed}")
    for e in q.errors:
        txt.append(f"  ERROR: {e}")
    (out / "data_quality.txt").write_text("\n".join(txt) + "\n")
    print(f"[report] wrote {dq_path!s} and {out / 'data_quality.txt'!s}")
    print("[done]")


if __name__ == "__main__":
    main()
