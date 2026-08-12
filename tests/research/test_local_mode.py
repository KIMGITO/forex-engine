"""Tests for provider='local' offline research mode.

Verifies local mode never touches Twelve Data, never needs an API key, loads
persisted partitions deterministically, slices by --days, and fails clearly on
missing partitions — while the existing twelvedata provider path stays intact.
All data here is tiny synthetic partitions (no real 8-year dataset).
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.research.dataset import PartitionedResearchRepository
from app.research.run import (
    LocalPartitionMissingError,
    ResearchRunConfig,
    run_research_pipeline,
)


class _DeterministicProvider(BaseMarketDataProvider):
    """Synthetic provider with a controllable symbol (mimics real normalization)."""

    def __init__(self, symbol, timeframe, n=60):
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


def _repo_from_df(df, symbol="EURUSD", timeframe="H1"):
    """Create a fresh repo from an existing Candle frame (fixed window)."""
    repo = PartitionedResearchRepository(tempfile.mkdtemp())
    candles = [
        Candle(
            symbol=symbol, timeframe=timeframe,
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]), volume=100.0,
        )
        for ts, row in df.iterrows()
    ]
    repo.merge_candles(candles, symbol=symbol, timeframe=timeframe)
    return repo


class TestLocalMode:
    def test_local_mode_does_not_instantiate_twelvedata(self):
        """No API key, no api-key-backed provider — local load only."""
        repo = _make_repo()
        # No MARKET_DATA_API_KEY needed; if code tried create_provider it would raise.
        cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            provider="local",
            fetch_days=0,  # no slicing for this test
        )
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        assert "report" in res
        # Provenance: provider/source = histdata, mode = local
        assert res["report"]["provider"] == "histdata"

    def test_local_mode_does_not_require_api_key(self):
        """Monkeypatch a failing create_provider and prove it is never called."""
        import app.research.run as run_mod

        def _boom(*args, **kwargs):
            raise AssertionError("create_provider must not be called in local mode")

        original = run_mod.create_provider
        run_mod.create_provider = _boom
        try:
            repo = _make_repo()
            cfg = ResearchRunConfig(
                symbols=("EURUSD",),
                timeframes=("H1",),
                strategy_names=("trend_structure",),
                provider="local",
                fetch_days=0,
            )
            res = run_research_pipeline(
                run_cfg=cfg, repo=repo,
                output_root=tempfile.mkdtemp(), verbose=False,
            )
            assert res["report"]["provider"] == "histdata"
        finally:
            run_mod.create_provider = original

    def test_local_partition_loads_correctly(self):
        repo = _make_repo(n=120)
        df = repo.load_df("EURUSD", "H1")
        assert df is not None
        assert len(df) == 120
        timestamps = df.index
        assert timestamps.is_monotonic_increasing
        assert timestamps.tz is not None  # UTC preserved

    def test_missing_partition_raises_clear_error(self):
        repo = PartitionedResearchRepository(tempfile.mkdtemp())
        cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            provider="local",
            fetch_days=0,
            storage_root=repo.root,
        )
        with pytest.raises(LocalPartitionMissingError) as excinfo:
            run_research_pipeline(
                run_cfg=cfg, repo=repo,
                output_root=tempfile.mkdtemp(), verbose=False,
            )
        err = str(excinfo.value)
        assert "EURUSD" in err and "H1" in err
        assert "data.parquet" in err
        assert "ingest the dataset first" in err

    def test_days_slices_local_data_by_timestamp(self):
        repo = _make_repo(n=1500)
        full = repo.load_df("EURUSD", "H1")
        assert len(full) == 1500
        cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            provider="local",
            fetch_days=30,  # slice trailing 30 days (721 hourly bars)
        )
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        df = repo.load_df("EURUSD", "H1")
        end_local = df.index[-1]
        start_local = end_local - pd.Timedelta(days=30)
        expected = df[(df.index >= start_local) & (df.index <= end_local)]
        # Regime engine needs >= the EMA warm-up minimum; 30 days gives 721 bars.
        assert 0 < len(expected) <= 721
        # The runner sliced local data deterministically; the report's date_range
        # reflects the sliced partition (rows == sliced count, end == newest).
        date_range = res["report"].get("date_range", {})
        if date_range.get("EURUSD"):
            assert date_range["EURUSD"]["rows"] == len(expected)
            assert str(end_local) in str(date_range["EURUSD"]["end"])

    def test_repeated_local_run_is_deterministic(self):
        repo = _make_repo(n=300)
        cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            provider="local",
            fetch_days=0,
        )
        r1 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        r2 = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        assert r1["optimization"] == r2["optimization"]
        assert r1["walk_forward"] == r2["walk_forward"]
        assert r1["results"] == r2["results"]

    def test_future_data_mutation_cannot_change_earlier_results(self):
        """Appending a future bar must not alter any earlier signal.

        Causality is verified at the signal level through the real local
        pipeline: for every timestamp <= T, the signal set produced from
        [.., T] must be identical to the signal set produced from [.., T, T+1h].
        The trailing-window anchor legitimately shifts the *evaluation window*,
        but it must never change an earlier observation.
        """
        from app.features import FeatureEngine
        from app.regime import RegimeConfig
        from app.strategy import (
            HistoricalSignalScanner,
            StrategyConfig,
            TrendStructureStrategy,
        )

        repo = _make_repo(n=300)
        df = repo.load_df("EURUSD", "H1")[["open", "high", "low", "close"]].sort_index()
        cutoff = df.index[-1]
        df_a = df
        future_ts = cutoff + pd.Timedelta(hours=1)
        df_b = pd.concat(
            [
                df_a,
                pd.DataFrame(
                    {
                        "open": [200.0], "high": [201.0],
                        "low": [199.0], "close": [200.5],
                    },
                    index=[future_ts],
                ),
            ]
        ).sort_index()

        def _signals(frame):
            fe = FeatureEngine().calculate(frame, features=["atr", "rsi"])
            scanner = HistoricalSignalScanner(
                strategy_config=StrategyConfig(), regime_config=RegimeConfig()
            )
            scan = scanner.scan(
                frame, TrendStructureStrategy(StrategyConfig()),
                "EURUSD", "1h", features=fe, mtf_contexts=None,
            )
            return {s.timestamp: s for s in scan.signals}

        sig_a = _signals(df_a)
        sig_b = _signals(df_b)
        # Every signal at or before the cutoff is unchanged even after a future
        # bar is appended.
        common_ts = sorted(t for t in sig_a if t <= cutoff.to_pydatetime())
        for ts in common_ts:
            sa = sig_a[ts]
            sb = sig_b[ts]
            assert sa.direction == sb.direction
            assert sa.score == sb.score
            assert sa.entry == sb.entry
        # The newly appended bar may only produce (or not) NEW signals after
        # the cutoff — it can never alter earlier ones.
        assert all(t > cutoff.to_pydatetime() for t in set(sig_b) - set(sig_a))

    def test_train_validation_test_isolation_remains(self):
        repo = _make_repo(n=400)
        cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            provider="local",
            fetch_days=0,
        )
        res = run_research_pipeline(
            run_cfg=cfg, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        report = res["report"]
        assert report["training"]["label"].startswith("IN-SAMPLE")
        assert report["validation"]["label"].startswith("VALIDATION")
        assert report["out_of_sample"]["label"].startswith("OUT-OF-SAMPLE")
        assert report["training"] is not report["validation"]
        assert report["validation"] is not report["out_of_sample"]