"""cProfile trace of the per-bar MTF loop at a small size (200 bars).

Uses cached structure/regime artifacts so we can isolate the per-bar loop cost
and identify the dominant functions with cumulative time.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import pandas as pd

from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.research.cache import deser_regime, deser_structure
from app.research.dataset import PartitionedResearchRepository

SYMBOL = "EURUSD"
BASE = "15m"
CACHE_ROOT = Path("research/cache/EURUSD")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def main() -> None:
    repo = PartitionedResearchRepository("data/research")
    m15 = repo.load_df(SYMBOL, "M15")[["open", "high", "low", "close"]].sort_index()
    h1 = repo.load_df(SYMBOL, "H1")[["open", "high", "low", "close"]].sort_index()

    m15_struct = deser_structure((CACHE_ROOT / "M15" / "structure.json").read_bytes())
    m15_regimes = deser_regime((CACHE_ROOT / "M15" / "regime.parquet").read_bytes())
    h1_struct = deser_structure((CACHE_ROOT / "H1" / "structure.json").read_bytes())
    h1_regimes = deser_regime((CACHE_ROOT / "H1" / "regime.parquet").read_bytes())

    print(f"M15 bars={len(m15)} struct_pts={len(m15_struct.structure)} "
          f"zones={len(m15_struct.liquidity_zones)} sweeps={len(m15_struct.sweeps)} "
          f"regimes={len(m15_regimes)}")
    print(f"H1  bars={len(h1)} struct_pts={len(h1_struct.structure)} "
          f"zones={len(h1_struct.liquidity_zones)} sweeps={len(h1_struct.sweeps)} "
          f"regimes={len(h1_regimes)}")

    base_df = m15.iloc[-N:]
    config = MtfConfig(base_timeframe=BASE, higher_timeframes=("1h",))
    builder = MtfContextBuilder(config, SYMBOL)

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.monotonic()
    for ts in base_df.index:
        # H1 tier build (the high-cost per-bar path in production).
        tier = builder.build(
            timeframe="1h",
            timestamp=ts,
            frame=h1,
            features=None,
            structure=h1_struct,
            regimes=h1_regimes,
            news_events=None,
        )
    elapsed = time.monotonic() - t0
    pr.disable()

    print(f"\nN={N} bars, H1-tier build only: {elapsed:.3f}s total "
          f"({N / elapsed:.0f} bars/s)")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print("\n=== TOP 25 CUMULATIVE ===")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(25)
    print("\n=== TOP 25 TOTTIME ===")
    print(s2.getvalue())


if __name__ == "__main__":
    main()