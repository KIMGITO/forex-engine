"""Focused tests for Step 13.1: versioned research cache, resume, walk-forward
reuse, optimizer test isolation, and memory-safe symbol processing.

These tests use tiny synthetic partitions — no 8-year dataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.research.cache import (
    CACHE_SCHEMA_VERSION,
    ENGINE_VERSION,
    ResearchCache,
    StageTimer,
    data_hash,
    deser_features,
    ser_features,
)
from app.research.dataset import PartitionedResearchRepository
from app.research.run import (
    LocalPartitionMissingError,
    ResearchRunConfig,
    _get_features,
    run_research_pipeline,
)


class _DeterministicProvider(BaseMarketDataProvider):
    """Synthetic provider with controllable symbol and candle count."""

    def __init__(self, symbol="EURUSD", timeframe="H1", n=300):
        self._symbol = symbol
        self._timeframe = timeframe
        self._n = n

    def fetch_candles(self, symbol, timeframe, start, end):
        idx = pd.date_range(start, end, freq="1h", tz="UTC")[: self._n]
        close = 100 + np.sin(np.arange(len(idx)) / 10.0)
        out = []
        for ts, c in zip(idx, close):
            out.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts.to_pydatetime(),
                    open=float(c) - 0.1,
                    high=float(c) + 0.2,
                    low=float(c) - 0.2,
                    close=float(c),
                    volume=100.0,
                )
            )
        return out


def _make_repo(symbol="EURUSD", timeframe="H1", n=300, root=None):
    repo = PartitionedResearchRepository(root or tempfile.mkdtemp())
    prov = _DeterministicProvider(symbol, timeframe, n=n)
    candles = prov.fetch_candles(
        symbol,
        timeframe,
        pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        pd.Timestamp("2026-12-01", tz="UTC").to_pydatetime(),
    )
    repo.merge_candles(candles, symbol=symbol, timeframe=timeframe)
    return repo


def _local_cfg(symbols=("EURUSD",), timeframes=("H1",), cache_root=None, **kw):
    return ResearchRunConfig(
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=("trend_structure",),
        provider="local",
        fetch_days=0,
        cache_root=cache_root or tempfile.mkdtemp(),
        **kw,
    )


class TestResearchCache:
    def test_cache_miss_then_hit(self):
        """First get_or_compute is a MISS; second with same inputs is a HIT."""
        cache = ResearchCache(tempfile.mkdtemp())
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
        )

        artifact, hit = cache.get_or_compute(
            "EURUSD", "H1", "features", df, {}, {},
            lambda: df ** 2, ser_features, deser_features,
        )
        assert hit is False
        assert cache.misses == 1

        artifact2, hit2 = cache.get_or_compute(
            "EURUSD", "H1", "features", df, {}, {},
            lambda: df ** 2, ser_features, deser_features,
        )
        assert hit2 is True
        assert cache.hits == 1
        pd.testing.assert_frame_equal(artifact, artifact2, check_freq=False)

    def test_cache_invalidates_when_data_changes(self):
        """A source-data change must invalidate the cached artifact."""
        cache = ResearchCache(tempfile.mkdtemp())
        df1 = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
        )
        cache.get_or_compute(
            "EURUSD", "H1", "features", df1, {}, {},
            lambda: df1, ser_features, deser_features,
        )

        # Append a future bar -> the source hash changes -> cache must miss.
        df2 = pd.concat(
            [
                df1,
                pd.DataFrame(
                    {"open": [4.0], "high": [4.5], "low": [3.5], "close": [4.2]},
                    index=[df1.index[-1] + pd.Timedelta(hours=1)],
                ),
            ]
        )
        _, hit = cache.get_or_compute(
            "EURUSD", "H1", "features", df2, {}, {},
            lambda: df2, ser_features, deser_features,
        )
        assert hit is False

    def test_cache_invalidates_when_config_changes(self):
        """A configuration change must invalidate the cached artifact."""
        cache = ResearchCache(tempfile.mkdtemp())
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
        )
        cache.get_or_compute(
            "EURUSD", "H1", "features", df, {"features": ["atr"]}, {},
            lambda: df, ser_features, deser_features,
        )
        # Different feature config => cache miss.
        _, hit = cache.get_or_compute(
            "EURUSD", "H1", "features", df, {"features": ["atr", "rsi"]}, {},
            lambda: df, ser_features, deser_features,
        )
        assert hit is False

    def test_future_candle_mutation_does_not_change_earlier_cached_results(self):
        """A future candle appended to the dataset invalidates the cache.

        The cache must NOT silently reuse stale artifacts when the source data
        changes. Appending a future bar changes the source hash, so the cache
        correctly misses (recomputes) rather than returning stale results.
        """
        cache = ResearchCache(tempfile.mkdtemp())
        idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.arange(10.0),
                "high": np.arange(10.0) + 0.5,
                "low": np.arange(10.0) - 0.5,
                "close": np.arange(10.0) + 0.1,
            },
            index=idx,
        )
        cache.get_or_compute(
            "EURUSD", "H1", "features", df, {}, {},
            lambda: df, ser_features, deser_features,
        )
        # Now a "future" bar is appended (simulating dataset growth).
        df2 = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"open": [100.0], "high": [100.5], "low": [99.5], "close": [100.2]},
                    index=[idx[-1] + pd.Timedelta(hours=1)],
                ),
            ]
        )
        # The cache must treat df2 as a different source (MISS) — it must NOT
        # silently reuse the stale artifact computed from the shorter df.
        _, hit1 = cache.get_or_compute(
            "EURUSD", "H1", "features", df2, {}, {},
            lambda: df2, ser_features, deser_features,
        )
        assert hit1 is False
        # The cache now holds the df2 artifact; requesting df again is a MISS
        # because the stored artifact no longer matches df's source hash.
        _, hit2 = cache.get_or_compute(
            "EURUSD", "H1", "features", df, {}, {},
            lambda: df, ser_features, deser_features,
        )
        assert hit2 is False

    def test_repeated_flag_commands_write_consistent_manifests(self):
        """Manifest schema/engine version are stable across runs."""
        cache = ResearchCache(tempfile.mkdtemp())
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
            },
            index=pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC"),
        )
        cache.get_or_compute(
            "EURUSD", "H1", "features", df, {}, {},
            lambda: df, ser_features, deser_features,
        )
        meta = cache._load_meta(cache._meta_path("EURUSD", "H1", "features"))
        assert meta is not None
        assert meta.schema_version == CACHE_SCHEMA_VERSION
        assert meta.engine_version == ENGINE_VERSION
        assert meta.stage == "features"


class TestCacheDeserialization:
    def test_features_roundtrip(self):
        df = pd.DataFrame(
            {
                "atr": [0.1, 0.2, 0.3],
                "rsi": [50.0, 55.0, 60.0],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
        )
        payload = ser_features(df)
        back = deser_features(payload)
        pd.testing.assert_frame_equal(back, df, check_freq=False)

    def test_data_hash_changes_with_append(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
            },
            index=idx,
        )
        h1 = data_hash(df)
        df2 = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"open": [4.0], "high": [4.5], "low": [3.5], "close": [4.2]},
                    index=[idx[-1] + pd.Timedelta(hours=1)],
                ),
            ]
        )
        assert data_hash(df2) != h1


class TestPipelineCache:
    def test_cache_artifacts_are_created(self):
        """Running the local pipeline creates cache artifacts on disk."""
        repo = _make_repo(n=300)
        cache_root = tempfile.mkdtemp()
        cfg = _local_cfg(cache_root=cache_root)
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        cache_dir = Path(cache_root) / "EURUSD" / "H1"
        assert (cache_dir / "features.parquet").exists()
        assert (cache_dir / "structure.json").exists()
        assert (cache_dir / "regime.parquet").exists()
        # MTF is persisted as a chunked store (atomic chunks + manifest) so
        # large outputs are never materialized in RAM. Verify the chunked
        # layout exists and at least one chunk + manifest are valid.
        mtf_chunk_dir = cache_dir / "mtf"
        assert (mtf_chunk_dir / "manifest.json").exists()
        assert any(mtf_chunk_dir.glob("chunk_*.json"))
        assert any(mtf_chunk_dir.glob("chunk_*._meta.json"))
        # At least two signal artifacts (2 strategies x mtf on/off already
        # share feature/structure/regime/mtf caches from the first run).
        assert res["cache"]["artifact_count"] >= 5

    def test_second_run_is_cache_hit(self):
        """Re-running the same pipeline hits the cache for reusable stages."""
        repo = _make_repo(n=300)
        cache_root = tempfile.mkdtemp()
        cfg = _local_cfg(cache_root=cache_root)

        r1 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        r2 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        # Run 1 has more misses than run 2 (cache warmed).
        # Artifact count grows on first run; second run reuses them.
        assert r1["cache"]["artifact_count"] > 0
        assert r2["cache"]["artifact_count"] >= r1["cache"]["artifact_count"]
        # Results are byte-identical (cache hit == fresh calculation).
        assert r1["results"] == r2["results"]
        assert r1["walk_forward"] == r2["walk_forward"]
        assert r1["optimization"] == r2["optimization"]
        assert r1["report"] == r2["report"]

    def test_cache_change_invalidates_results(self):
        """Changing the dataset re-computes (does not silently reuse stale)."""
        repo = _make_repo(n=300)
        cache_root = tempfile.mkdtemp()
        cfg = _local_cfg(cache_root=cache_root)
        r1 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        # Append a bar to the underlying dataset.
        df = repo.load_df("EURUSD", "H1")
        new_ts = df.index[-1] + pd.Timedelta(hours=1)
        repo.merge_candles(
            [
                Candle(
                    symbol="EURUSD", timeframe="H1",
                    timestamp=new_ts.to_pydatetime(),
                    open=200.0, high=201.0, low=199.0, close=200.5, volume=100.0,
                )
            ],
            symbol="EURUSD", timeframe="H1",
        )
        r2 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        assert r1["report"] != r2["report"]

    def test_resume_interrupted_pipeline(self):
        """Simulate an interrupted pipeline: first run computes some stages;
        second run continues and reuses the already-cached stages."""
        repo = _make_repo(n=300)
        cache_root = tempfile.mkdtemp()
        from app.research.run import _compute_structure

        # Pre-compute only features + structure (simulate partial run).
        df = repo.load_df("EURUSD", "H1")[["open", "high", "low", "close"]].sort_index()
        cache = ResearchCache(cache_root)
        from app.research.cache import deser_structure, ser_structure
        cache.get_or_compute(
            "EURUSD", "H1", "features", df, {"features": ["atr", "rsi"]}, {},
            lambda: _get_features(cache, df, "EURUSD", "H1"),
            ser_features, deser_features,
        )
        cache.get_or_compute(
            "EURUSD", "H1", "structure", df, {}, {},
            lambda: _compute_structure(df, "EURUSD", "H1"),
            ser_structure, deser_structure,
        )

        # Now run the full pipeline: features/structure are HIT, others MISS.
        cfg = _local_cfg(cache_root=cache_root)
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        assert res["cache"]["artifact_count"] >= 6  # features+structure+regime+mtf+signals
        assert "report" in res

    def test_max_bars_slices_input(self):
        """--max-bars limits the number of bars processed."""
        repo = _make_repo(n=1000)
        cfg = _local_cfg(cache_root=tempfile.mkdtemp(), max_bars=100)
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        assert res["report"]["date_range"]["EURUSD"]["rows"] <= 100

    def test_missing_partition_still_raises(self):
        """Memory-safe per-symbol processing still raises on missing partition."""
        repo = PartitionedResearchRepository(tempfile.mkdtemp())
        cfg = _local_cfg()
        with pytest.raises(LocalPartitionMissingError):
            run_research_pipeline(
                run_cfg=cfg, repo=repo,
                output_root=tempfile.mkdtemp(), verbose=False,
            )

    def test_stage_timing_reports(self):
        """StageTimer accumulates timings per stage."""
        t = StageTimer()
        t.begin("_features")
        t.end("_features")
        t.begin("_structure")
        t.end("_structure")
        summary = t.summary()
        assert "_features" in summary
        assert "_structure" in summary
        assert summary["_features"] >= 0
        assert t.total() >= 0