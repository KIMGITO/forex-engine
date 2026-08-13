"""Isolated per-bar MTF loop benchmark using cached artifacts.

Uses the exact cached structure/regime artifacts that a resumed Step 13 run
would supply (deserialized via app.research.cache), so the per-bar loop cost is
isolated from structure/regime recomputation.

Mirrors MtfEngine.analyze's per-bar loop (engine.py lines 105-213) verbatim.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

from app.mtf.alignment import classify_alignment
from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.mtf.models import MtfAlignmentState, MtfContext, TimeframeContext
from app.research.cache import deser_regime, deser_structure
from app.research.dataset import PartitionedResearchRepository

SYMBOL = "EURUSD"
BASE = "15m"
REPO_ROOT = "data/research"
CACHE_ROOT = Path("research/cache/EURUSD")


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


def load_cached() -> tuple[
    pd.DataFrame, object, list, pd.DataFrame, object, list
]:
    repo = PartitionedResearchRepository(REPO_ROOT)
    m15 = repo.load_df(SYMBOL, "M15")[["open", "high", "low", "close"]].sort_index()
    h1 = repo.load_df(SYMBOL, "H1")[["open", "high", "low", "close"]].sort_index()

    m15_struct = deser_structure(
        (CACHE_ROOT / "M15" / "structure.json").read_bytes()
    )
    m15_regimes = deser_regime((CACHE_ROOT / "M15" / "regime.parquet").read_bytes())
    h1_struct = deser_structure(
        (CACHE_ROOT / "H1" / "structure.json").read_bytes()
    )
    h1_regimes = deser_regime((CACHE_ROOT / "H1" / "regime.parquet").read_bytes())

    print(f"M15: bars={len(m15)} structure_pts={len(m15_struct.structure)} "
          f"zones={len(m15_struct.liquidity_zones)} sweeps={len(m15_struct.sweeps)} "
          f"regimes={len(m15_regimes)}")
    print(f"H1 : bars={len(h1)} structure_pts={len(h1_struct.structure)} "
          f"zones={len(h1_struct.liquidity_zones)} sweeps={len(h1_struct.sweeps)} "
          f"regimes={len(h1_regimes)}")
    return m15, m15_struct, m15_regimes, h1, h1_struct, h1_regimes


def run_loop(
    m15, m15_struct, m15_regimes, h1, h1_struct, h1_regimes,
    n_bars: int | None,
) -> tuple[list[MtfContext], float]:
    config = MtfConfig(base_timeframe=BASE, higher_timeframes=("1h",))
    base_df = m15 if n_bars is None else m15.iloc[-n_bars:]
    base_df = base_df.sort_index()

    builder = MtfContextBuilder(config, SYMBOL)
    analysis = {
        "15m": (base_df, m15_struct, m15_regimes),
        "1h": (h1, h1_struct, h1_regimes),
    }
    contexts: list[MtfContext] = []
    t0 = time.monotonic()

    for ts in base_df.index:
        now = ts
        hierarchy: list[TimeframeContext] = []
        available_count = 0

        ana = analysis["1h"]
        tier = builder.build(
            timeframe="1h",
            timestamp=now,
            frame=ana[0],
            features=None,
            structure=ana[1],
            regimes=ana[2],
            news_events=None,
        )
        hierarchy.append(tier)
        if tier.present:
            available_count += 1

        base_ana = analysis["15m"]
        structure = base_ana[1]
        regimes = base_ana[2]
        base_tier = TimeframeContext(
            timeframe=BASE,
            timestamp=now,
            candle_open=now,
            candle_close=now,
            trend_state=(regimes[-1].trend_state.value if regimes else None),
            volatility_state=(
                regimes[-1].volatility_state.value if regimes else None
            ),
            market_state=(regimes[-1].market_state.value if regimes else None),
            structural_bias=builder.structural_bias(structure, now),
            liquidity_zones=(
                [z for z in structure.liquidity_zones if z.available_from <= now]
                if structure
                else []
            ),
            sweeps=(
                [s for s in structure.sweeps if s.available_from <= now]
                if structure
                else []
            ),
            news_risk_max=builder._news_risk_max([], now),
            present=True,
            available_from=now,
        )

        base_dir = None
        if base_tier.trend_state == "bullish":
            base_dir = "long"
        elif base_tier.trend_state == "bearish":
            base_dir = "short"

        if base_dir is None:
            alignment = MtfAlignmentState.UNKNOWN
            reasons = ["base regime direction unknown"]
        else:
            alignment, reasons, _ = classify_alignment(
                base_dir,
                hierarchy,
                min_aligned=config.min_aligned,
                require_no_htf_conflict=config.require_no_htf_conflict,
            )

        contexts.append(
            MtfContext(
                symbol=SYMBOL,
                base_timeframe=BASE,
                timestamp=now,
                hierarchy=[base_tier] + hierarchy,
                alignment=alignment,
                alignment_reasons=reasons,
                min_aligned=float(config.min_aligned),
                news_risk_max=None,
                metadata={
                    "available_htf_tiers": available_count,
                    "hierarchy": list(config.higher_timeframes),
                },
                available_from=now,
            )
        )

    return contexts, time.monotonic() - t0


def main() -> None:
    sizes = [1000, 5000, 10000, 25000, 50000, 100000, None]
    if "--sizes" in sys.argv:
        sizes = [int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]

    m15, m15_s, m15_r, h1, h1_s, h1_r = load_cached()

    results = []
    for n in sizes:
        gc.collect()
        rss_before = rss_mb()
        ctxs, elapsed = run_loop(
            m15, m15_s, m15_r, h1, h1_s, h1_r, n
        )
        rss_after = rss_mb()
        n_actual = len(ctxs)
        rows = {
            "bars": n_actual,
            "elapsed_s": round(elapsed, 3),
            "bars_per_sec": round(n_actual / elapsed, 1),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "htf_count": 1,
            "output_count": n_actual,
        }
        results.append(rows)
        print(
            f"\nbars={n_actual:>6}  elapsed={elapsed:8.2f}s  "
            f"bars/s={n_actual/elapsed:10.1f}  rss {rss_before:.0f}->{rss_after:.0f}MB"
        )
        del ctxs
        gc.collect()

    Path("research/bench_mtf_loop_results.json").write_text(
        json.dumps(results, indent=2), "utf-8"
    )
    print("\nresults written to research/bench_mtf_loop_results.json")


if __name__ == "__main__":
    main()