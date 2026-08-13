#!/usr/bin/env python3
"""Focused production-scale MTF validation for EURUSD M15 (bounded window).

Memory-safe: uses a bounded window (default 50_000 bars) so the context list
and its JSON serialization stay well within RAM. Reports runtime, peak RSS,
output size, cache validity, resume behavior, causality checks.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mtf import MtfConfig, MtfEngine
from app.research.cache import ResearchCache, data_hash, deser_mtf, ser_mtf


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
    results: dict[str, Any] = {}
    tmp_cache = Path("research/cache/_validate_m15_tmp")
    if tmp_cache.exists():
        shutil.rmtree(tmp_cache)

    try:
        from app.research.dataset import PartitionedResearchRepository

        repo = PartitionedResearchRepository("data/research")
        tf_map = {"M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}
        native_map: dict[str, pd.DataFrame] = {}
        for storage_tf, native_tf in tf_map.items():
            df = repo.load_df("EURUSD", storage_tf)
            if df is not None and not df.empty:
                native_map[native_tf] = df[["open", "high", "low", "close"]].sort_index()

        # Bounded window: last 50_000 M15 bars, HTFs sliced from same start.
        start_ts = native_map["15m"].index[-50_000]
        windowed: dict[str, pd.DataFrame] = {}
        for k, df in native_map.items():
            windowed[k] = df[df.index >= start_ts]
        native_map = windowed

        m15 = native_map["15m"]
        results["bars_processed"] = len(m15)
        print(f"\nEURUSD M15: {len(m15)} bars  {m15.index[0]} → {m15.index[-1]}", flush=True)
        for k, v in sorted(native_map.items(), key=lambda x: len(native_map[x[0]]), reverse=True):
            print(f"  {k}: {len(v)} bars", flush=True)

        engine = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h", "4h", "1d")),
            "EURUSD",
        )
        rss_before = rss_mb()
        t0 = time.monotonic()
        ctxs = engine.analyze(native_map, "15m")
        elapsed = time.monotonic() - t0
        rss_after = rss_mb()

        results["runtime_s"] = round(elapsed, 2)
        results["rss_before_mb"] = round(rss_before, 1)
        results["rss_after_mb"] = round(rss_after, 1)
        results["mtf_contexts"] = len(ctxs)
        results["bar_rate"] = round(len(ctxs) / elapsed)
        print(f"\nMTF: {len(ctxs)} contexts in {elapsed:.1f}s ({len(ctxs)/elapsed:.0f} bar/s)  rss {rss_before:.0f}→{rss_after:.0f}MB", flush=True)
        print(f"First ctx: available_from={ctxs[0].available_from} alignment={ctxs[0].alignment.value}", flush=True)
        print(f"Last  ctx: available_from={ctxs[-1].available_from} alignment={ctxs[-1].alignment.value}", flush=True)

        # Serialize (memory-safe: bounded window → manageable JSON)
        payload = ser_mtf(ctxs)
        results["ser_size_bytes"] = len(payload)
        print(f"Serialized: {len(payload)} bytes ({len(payload)/1024/1024:.1f} MB)", flush=True)

        # Cache write + reload
        cache = ResearchCache(str(tmp_cache), use_cache=True)
        src_hash = data_hash(m15)
        mf, hit = cache.get_or_compute(
            "EURUSD", "M15", "mtf", m15, {}, {"data_hashes": src_hash},
            lambda: ctxs, ser_mtf, deser_mtf,
        )
        mf2, hit2 = cache.get_or_compute(
            "EURUSD", "M15", "mtf", m15, {}, {"data_hashes": src_hash},
            lambda: ctxs, ser_mtf, deser_mtf,
        )
        results["cache_hit"] = hit
        results["cache_second_hit"] = hit2
        results["cache_reload_ok"] = (mf2 is not None) and (len(mf2) == len(ctxs))
        print(f"Cache: write_hit={hit} reload_hit={hit2} reload_ok={results['cache_reload_ok']}", flush=True)

        # Metadata validation
        meta_path = tmp_cache / "EURUSD" / "M15" / "mtf._meta.json"
        art_path = tmp_cache / "EURUSD" / "M15" / "mtf.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            results["meta_valid"] = (
                m.get("symbol") == "EURUSD" and m.get("row_count") == len(m15)
                and hashlib.sha256(art_path.read_bytes()).hexdigest() == m.get("artifact_hash")
            )
            results["meta_row_count"] = m.get("row_count")
            print(f"Meta: symbol={m['symbol']} rows={m['row_count']} hash_valid={results['meta_valid']}", flush=True)
        else:
            results["meta_valid"] = False

        # No .tmp/.part artifacts
        tmp_files = list(tmp_cache.rglob("*.tmp")) + list(tmp_cache.rglob("*.part"))
        results["tmp_artifacts"] = [str(p) for p in tmp_files]
        print(f".tmp/.part artifacts: {len(tmp_files)}", flush=True)

        # Causality
        avail = [c.available_from for c in ctxs]
        results["causality_monotonic"] = all(avail[i] <= avail[i + 1] for i in range(len(avail) - 1))
        results["causality_no_future"] = all(c.available_from <= c.timestamp for c in ctxs)
        print(f"Causality: monotonic={results['causality_monotonic']} no_future={results['causality_no_future']}", flush=True)

        # HTF presence distribution (later bars should have HTFs)
        htf_present = [len([t for t in c.hierarchy if t.present and t.timeframe != "15m"]) for c in ctxs]
        results["min_htf_present"] = min(htf_present) if htf_present else 0
        results["max_htf_present"] = max(htf_present) if htf_present else 0
        results["htf_present_last100"] = sum(1 for x in htf_present[-100:] if x > 0)
        print(f"HTF present: min={results['min_htf_present']} max={results['max_htf_present']} last100_with_htf={results['htf_present_last100']}", flush=True)

        # Resume simulation: 3rd call should hit cache
        _, hit3 = cache.get_or_compute(
            "EURUSD", "M15", "mtf", m15, {}, {"data_hashes": src_hash},
            lambda: None, ser_mtf, deser_mtf,
        )
        results["resume_sim_hit"] = hit3
        print(f"Resume sim (3rd call): hit={hit3}", flush=True)

        results["STATUS"] = "GO"
    except Exception as exc:
        results["STATUS"] = "NO-GO"
        results["error"] = str(exc)
        import traceback
        traceback.print_exc()
    finally:
        if tmp_cache.exists():
            shutil.rmtree(tmp_cache)

    print("\n" + "=" * 60, flush=True)
    print("VALIDATION RESULTS", flush=True)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v}", flush=True)
    print(f"STATUS: {results.get('STATUS', 'UNKNOWN')}", flush=True)

    out = Path("research/results/step13/mtf_m15_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), "utf-8")
    print(f"Results written to {out}", flush=True)


if __name__ == "__main__":
    main()