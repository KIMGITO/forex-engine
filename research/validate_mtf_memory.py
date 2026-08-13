#!/usr/bin/env python3
"""Memory-bounded MTF validation (Step 13.2, chunked path).

Runs the CHUNKED architecture end-to-end:
  MtfEngine.analyze_chunks -> ser_mtf -> MtfChunkStore.write_chunk

Uses an isolated temp cache directory (never touches production cache).
Records RSS per chunk to PROVE memory stays bounded as bar count grows.
"""
import gc, json, shutil, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


def run_validated(n_bars, chunk_size=5000, tmp_cache="research/cache/_memval_tmp"):
    from app.mtf import MtfConfig, MtfEngine
    from app.research.mtf_chunks import MtfChunkStore, MtfContextMap
    from app.research.cache import data_hash, config_hash, ser_mtf
    from app.research.dataset import PartitionedResearchRepository

    repo = PartitionedResearchRepository("data/research")
    tf_map = {"M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}
    native_map = {}
    for storage_tf, native_tf in tf_map.items():
        df = repo.load_df("EURUSD", storage_tf)
        if df is not None and not df.empty:
            native_map[native_tf] = df[["open", "high", "low", "close"]].sort_index()

    # Window to the requested bar count.
    start_ts = native_map["15m"].index[-n_bars]
    for k in list(native_map):
        native_map[k] = native_map[k][native_map[k].index >= start_ts]

    m15 = native_map["15m"]
    src_hash = data_hash(m15)
    cfg_hash = config_hash({})

    if Path(tmp_cache).exists():
        shutil.rmtree(tmp_cache)
    store = MtfChunkStore("EURUSD", "M15", tmp_cache)
    store.write_manifest(source_data_hash=src_hash, config_hash=cfg_hash,
                         upstream_hashes={"data_hashes": src_hash},
                         total_bars=len(m15), chunk_size=chunk_size)

    eng = MtfEngine(MtfConfig(base_timeframe="15m", higher_timeframes=("1h", "4h", "1d")), "EURUSD")

    rss_samples = []
    rss_before = rss_mb()
    t0 = time.monotonic()
    chunks = 0
    for chunk_index, (s, e, ctxs) in enumerate(eng.analyze_chunks(
        native_map, "15m", chunk_size=chunk_size, rss_limit_mb=0.0
    )):
        payload = ser_mtf(ctxs)
        store.write_chunk(chunk_index, s, e, payload,
                          source_data_hash=src_hash, config_hash=cfg_hash)
        r = rss_mb()
        rss_samples.append(round(r, 1))
        chunks += 1
        del ctxs, payload
    elapsed = time.monotonic() - t0
    rss_after = rss_mb()

    m = MtfContextMap(store)
    first_ts = m.get(native_map["15m"].index[0])
    last_valid = len(m._store.valid_chunk_indices())

    return {
        "bars": len(m15),
        "chunks": chunks,
        "chunk_size": chunk_size,
        "elapsed_s": round(elapsed, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_peak_mb": max(rss_samples) if rss_samples else round(rss_after, 1),
        "rss_min_mb": min(rss_samples) if rss_samples else round(rss_after, 1),
        "rss_samples": rss_samples,
        "valid_chunks": last_valid,
        "first_ctx_found": first_ts is not None,
        "bars_per_s": round(len(m15) / elapsed),
    }


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [20000]
    results = []
    for n in sizes:
        print("\n=== VALIDATING %d bars (chunked) ===" % n, flush=True)
        r = run_validated(n)
        results.append(r)
        print(json.dumps(r), flush=True)
    print("\n=== ALL RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    out = Path("research/results/step13/mtf_memory_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nReport: %s" % out, flush=True)
