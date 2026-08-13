"""Tests for the bounded-memory chunked MTF architecture (Step 13.2)."""
import numpy as np
import pandas as pd
import pytest

from app.mtf import MtfConfig, MtfEngine
from app.mtf.engine import RssLimitExceeded
from app.research.mtf_chunks import MtfChunkStore, MtfContextMap


def _frame(n, freq="15min", seed=1):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 1.08 + np.cumsum(rng.normal(0, 0.0005, n))
    return pd.DataFrame({"open": close, "high": close + 0.001, "low": close - 0.001, "close": close}, index=idx)


def _data(n=500):
    return {"15m": _frame(n, "15min", 7), "1h": _frame(max(20, n // 4), "1h", 8)}


class TestChunkEquivalence:
    def test_monolith_vs_chunks_exact(self):
        data = _data(500)
        eng = MtfEngine(MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)), "EURUSD")
        mono = eng.analyze(data, "15m")
        chunked = []
        for _s, _e, ctxs in eng.analyze_chunks(data, "15m", chunk_size=250):
            chunked.extend(ctxs)
        assert len(chunked) == len(mono) == 500
        for a, b in zip(mono, chunked):
            assert a.model_dump() == b.model_dump()

    def test_chunk_boundary_250_251(self):
        data = _data(500)
        eng = MtfEngine(MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)), "EURUSD")
        mono = eng.analyze(data, "15m")
        chunks = list(eng.analyze_chunks(data, "15m", chunk_size=250))
        assert chunks[0][0] == 0 and chunks[0][1] == 250
        assert chunks[1][0] == 250 and chunks[1][1] == 500
        assert chunks[0][2][-1].model_dump() == mono[249].model_dump()
        assert chunks[1][2][0].model_dump() == mono[250].model_dump()


class TestChunkStore:
    def test_atomic_write_and_validate(self, tmp_path):
        store = MtfChunkStore("EURUSD", "M15", str(tmp_path))
        store.write_manifest(source_data_hash="abc", config_hash="def", upstream_hashes={}, total_bars=10, chunk_size=5)
        store.write_chunk(0, 0, 5, b'[{"x":1}]', source_data_hash="abc", config_hash="def")
        store.write_chunk(1, 5, 10, b'[{"x":2}]', source_data_hash="abc", config_hash="def")
        assert store.is_chunk_complete(0) and store.is_chunk_complete(1)
        assert store.valid_chunk_indices() == [0, 1]
        assert store.first_missing_index() == 2
        assert not list(store.root.glob("*.tmp"))

    def test_incomplete_chunk_rejected(self, tmp_path):
        store = MtfChunkStore("EURUSD", "M15", str(tmp_path))
        store.write_manifest(source_data_hash="abc", config_hash="def", upstream_hashes={}, total_bars=5, chunk_size=5)
        store.chunk_path(0).write_text('[{"x":1}]')
        assert not store.is_chunk_complete(0)
        assert store.valid_chunk_indices() == []

    def test_corrupted_chunk_rejected(self, tmp_path):
        store = MtfChunkStore("EURUSD", "M15", str(tmp_path))
        store.write_manifest(source_data_hash="abc", config_hash="def", upstream_hashes={}, total_bars=5, chunk_size=5)
        store.write_chunk(0, 0, 5, b'[{"x":1}]', source_data_hash="abc", config_hash="def")
        store.chunk_path(0).write_text('[{"x":999}]')
        assert not store.is_chunk_complete(0)

    def test_resume_skips_valid_chunks(self, tmp_path):
        store = MtfChunkStore("EURUSD", "M15", str(tmp_path))
        store.write_manifest(source_data_hash="abc", config_hash="def", upstream_hashes={}, total_bars=15, chunk_size=5)
        store.write_chunk(0, 0, 5, b'[{"x":1}]', source_data_hash="abc", config_hash="def")
        store.write_chunk(1, 5, 10, b'[{"x":2}]', source_data_hash="abc", config_hash="def")
        assert store.first_missing_index() == 2


class TestMtfContextMap:
    def test_streaming_lookup(self, tmp_path):
        from app.mtf.models import MtfAlignmentState, MtfContext, TimeframeContext
        from app.research.cache import ser_mtf

        store = MtfChunkStore("EURUSD", "M15", str(tmp_path))
        store.write_manifest(source_data_hash="abc", config_hash="def", upstream_hashes={}, total_bars=1, chunk_size=1)
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        tier = TimeframeContext(
            timeframe="1h", timestamp=ts.to_pydatetime(), present=True,
            available_from=ts.to_pydatetime(),
        )
        ctx = MtfContext(
            symbol="EURUSD", base_timeframe="15m", timestamp=ts.to_pydatetime(),
            hierarchy=[tier], alignment=MtfAlignmentState.UNKNOWN,
            alignment_reasons=["x"], min_aligned=1.0, metadata={},
            available_from=ts.to_pydatetime(),
        )
        store.write_chunk(0, 0, 1, ser_mtf([ctx]), source_data_hash="abc", config_hash="def")
        m = MtfContextMap(store)
        got = m.get(ts)
        assert got is not None and got.symbol == "EURUSD"
        assert m.get(pd.Timestamp("2099-01-01", tz="UTC")) is None


class TestCausalHtfClipping:
    """HTF causal-window clipping must NOT change MTF outputs.

    A clipped higher-timeframe analysis (only base-window + warm-up history)
    must produce byte-identical MtfContext output to a full-dataset analysis
    for every base-window timestamp.
    """

    def _make_df(self, n, freq="15min", seed=3):
        idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
        rng = np.random.default_rng(seed)
        close = 1.08 + np.cumsum(rng.normal(0, 0.0005, n))
        return pd.DataFrame(
            {"open": close, "high": close + 0.001, "low": close - 0.001, "close": close},
            index=idx,
        )

    def test_clipped_output_equals_full_output(self):
        # H1 has far more history than the base window strictly needs.
        m15 = self._make_df(300, "15min", seed=11)
        h1 = self._make_df(2000, "1h", seed=12)  # 2000 H1 bars (~83 days)
        data_full = {"15m": m15, "1h": h1}

        eng = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)),
            "EURUSD",
        )
        full = eng.analyze(data_full, "15m", clip_htf=False)
        clipped = eng.analyze(data_full, "15m", clip_htf=True)

        assert len(clipped) == len(full) == 300
        for a, b in zip(full, clipped):
            assert a.model_dump() == b.model_dump()

    def test_clipped_never_uses_future(self):
        m15 = self._make_df(120, "15min", seed=21)
        h1 = self._make_df(500, "1h", seed=22)
        data = {"15m": m15, "1h": h1}
        eng = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)),
            "EURUSD",
        )
        outputs = eng.analyze(data, "15m", clip_htf=True)
        # Every per-bar MTF context's available_from must be <= the observation
        # timestamp (no look-ahead) and monotone non-decreasing.
        avails = [c.available_from for c in outputs]
        from itertools import pairwise
        assert all(a <= b for a, b in pairwise(avails))
        assert all(c.available_from <= c.timestamp for c in outputs)


class TestPhase0RssGuard:
    """The phase-0 RSS guard must fail BEFORE heavy precomputation."""

    def test_guard_raises_when_limit_exceeded(self, monkeypatch):
        import app.mtf.engine as eng_mod

        # Force RSS high enough to trip the guard deterministically.
        monkeypatch.setattr(eng_mod, "_rss_mb", lambda: 9999.0)
        monkeypatch.setattr(eng_mod, "_mem_available_mb", lambda: 8000.0)

        m15 = pd.DataFrame(
            {
                "open": [1.0, 1.01, 1.02, 1.01],
                "high": [1.02, 1.03, 1.03, 1.02],
                "low": [0.99, 1.00, 1.01, 1.00],
                "close": [1.01, 1.02, 1.01, 1.01],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
        )
        eng = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)),
            "EURUSD",
        )
        with pytest.raises(RssLimitExceeded, match="phase-0"):
            eng.analyze({"15m": m15}, "15m", rss_limit_mb=1000.0)

    def test_guard_passes_when_under_limit(self, monkeypatch):
        import app.mtf.engine as eng_mod

        monkeypatch.setattr(eng_mod, "_rss_mb", lambda: 100.0)
        monkeypatch.setattr(eng_mod, "_mem_available_mb", lambda: 6000.0)
        m15 = _frame(50)
        h1 = _frame(20, "1h", seed=9)
        eng = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)),
            "EURUSD",
        )
        out = eng.analyze({"15m": m15, "1h": h1}, "15m", rss_limit_mb=5000.0)
        assert len(out) == 50

    def test_guard_called_before_precompute(self, monkeypatch):
        import app.mtf.engine as eng_mod

        monkeypatch.setattr(eng_mod, "_rss_mb", lambda: 50.0)
        monkeypatch.setattr(eng_mod, "_mem_available_mb", lambda: 5000.0)

        # Phase-0 guard is invoked before MarketStructureEngine.analyze.
        class _Guard:
            called = False

        orig = eng_mod.MarketStructureEngine.analyze

        def _wrapped(self, data, symbol, timeframe, *a, **k):
            assert _Guard.called, "phase-0 guard ran BEFORE precompute"
            return orig(self, data, symbol, timeframe, *a, **k)

        def _phantom_guard(_):
            _Guard.called = True

        monkeypatch.setattr(eng_mod, "_require_rss_headroom", _phantom_guard)
        monkeypatch.setattr(eng_mod.MarketStructureEngine, "analyze", _wrapped)
        m15 = _frame(60)
        h1 = _frame(20, "1h", seed=13)
        eng = eng_mod.MtfEngine(
            eng_mod.MtfConfig(base_timeframe="15m", higher_timeframes=("1h",)),
            "EURUSD",
        )
        out = eng.analyze({"15m": m15, "1h": h1}, "15m", rss_limit_mb=9000.0)
        assert len(out) == 60
        assert _Guard.called
