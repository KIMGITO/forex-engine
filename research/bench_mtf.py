"""Focused MTF benchmark using real EURUSD research partitions.

Benchmarks MtfEngine.analyze exactly as the Step 13 pipeline invokes it
(base=M15, higher=H1/H4/D1 present in the partitions).

Usage:
    python3 research/bench_mtf.py [--sizes 1000,5000,10000]
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

from app.mtf.config import MtfConfig
from app.mtf.engine import MtfEngine
from app.research.dataset import PartitionedResearchRepository

SYMBOL = "EURUSD"
BASE = "15m"
TIER_FRAME = ("M15", "H1", "H4", "D1")
REPO_ROOT = "data/research"


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


def main() -> None:
    sizes = [1000, 5000, 10000, 25000, 50000, 100000, None]
    if "--sizes" in sys.argv:
        sizes = [int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]

    repo = PartitionedResearchRepository(REPO_ROOT)
    native_map = {}
    for tf in TIER_FRAME:
        df = repo.load_df(SYMBOL, tf)
        native_map[tf] = df[["open", "high", "low", "close"]].sort_index()
        print(f"loaded {tf}: {len(native_map[tf])} rows")

    native_to_native = {
        "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d",
    }
    native_map = {native_to_native[k]: v for k, v in native_map.items()}

    order = ["5m", "15m", "1h", "4h", "1d"]
    order.remove(BASE)
    higher = [tf for tf in order if tf in native_map]
    print(f"higher timeframes: {higher}")

    results = []
    for n in sizes:
        gc.collect()
        engine_map = {
            tf: (df if n is None else df.iloc[-n:])
            for tf, df in native_map.items()
        }
        engine = MtfEngine(
            MtfConfig(base_timeframe=BASE, higher_timeframes=tuple(higher)),
            SYMBOL,
        )
        n_actual = len(engine_map[BASE])
        rss_before = rss_mb()
        t0 = time.monotonic()
        ctxs = engine.analyze(engine_map, BASE)
        elapsed = time.monotonic() - t0
        rss_after = rss_mb()
        rows = {
            "bars": n_actual,
            "elapsed_s": round(elapsed, 3),
            "bars_per_sec": round(n_actual / elapsed, 1),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "htf_count": len(higher),
            "output_count": len(ctxs),
        }
        results.append(rows)
        print(
            f"\nbars={n_actual:>6}  elapsed={elapsed:8.2f}s  "
            f"bars/s={n_actual/elapsed:10.1f}  rss {rss_before:.0f}->{rss_after:.0f}MB"
        )
        del ctxs
        gc.collect()

    Path("research/bench_mtf_results.json").write_text(
        json.dumps(results, indent=2), "utf-8"
    )
    print("\nresults written to research/bench_mtf_results.json")


if __name__ == "__main__":
    main()