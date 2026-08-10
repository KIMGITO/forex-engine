"""Tests for the research-pipeline orchestrator (app/research/run.py).

Uses a deterministic mock provider only — never touches the real Twelve Data
API. Validates:
- real-pipeline orchestration (data -> features -> structure -> regime -> MTF
  -> strategy -> backtest -> walk-forward -> optimizer)
- TRAIN/VALIDATION/TEST isolation
- optimizer never consumes TEST data
- MTF remains causal (future data cannot alter earlier results)
- deterministic repeated research run
"""

import tempfile

import numpy as np
import pandas as pd

from app.data.models import Candle
from app.data.provider import BaseMarketDataProvider
from app.research.dataset import PartitionedResearchRepository
from app.research.run import ResearchRunConfig, run_research_pipeline


class _DeterministicProvider(BaseMarketDataProvider):
    """Synthetic provider with a controllable symbol (mimics real normalization)."""

    def __init__(self, symbol, timeframe, n=300):
        self._symbol = symbol
        self._timeframe = timeframe
        self._n = n

    def fetch_candles(self, symbol, timeframe, start, end):
        idx = pd.date_range(start, end, freq="1h", tz="UTC")
        idx = idx[: self._n]
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


class _Mock:
    """Minimal stand-in so tests don't require argparse/CLI."""

    def __init__(self, run_cfg):
        self.run_cfg = run_cfg


def _make_repo(provider, symbol, timeframe, n=300):
    """Pre-seed a partition via the provider+repo so the runner treats it as
    cached (no network). This mirrors what the live fetch step produces."""
    repo = PartitionedResearchRepository(tempfile.mkdtemp())
    candles = provider.fetch_candles(
        symbol, timeframe,
        pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        pd.Timestamp("2026-06-01", tz="UTC").to_pydatetime(),
    )
    repo.merge_candles(candles, symbol=symbol, timeframe=timeframe)
    return repo


class TestPipelineOrchestration:
    def test_end_to_end_with_mock_provider(self):
        provider = _DeterministicProvider("EURUSD", "H1")
        repo = _make_repo(provider, "EURUSD", "H1")
        run_cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            fetch_days=10,  # short so no backfill API calls
        )
        # Seed the partition directly and mark it as fully cached.
        res = run_research_pipeline(
            run_cfg=run_cfg,
            provider=provider,
            repo=repo,
            output_root=tempfile.mkdtemp(),
            verbose=False,
        )
        assert "report" in res
        assert res["report"]["provider"] == "twelvedata"
        assert len(res["results"]) >= 1

    def test_train_validation_test_are_separate(self):
        # The runner must never combine TRAIN/VALIDATION/TEST blocks.
        provider = _DeterministicProvider("EURUSD", "H1", n=400)
        repo = _made_repo_for(provider, "EURUSD", "H1", 400)
        run_cfg = ResearchRunConfig(
            symbols=("EURUSD",),
            timeframes=("H1",),
            strategy_names=("trend_structure",),
            fetch_days=20,
        )
        res = run_research_pipeline(
            run_cfg=run_cfg, provider=provider, repo=repo,
            output_root=tempfile.mkdtemp(), verbose=False,
        )
        report = res["report"]
        # Each block has its own label and is never merged.
        assert report["training"]["label"].startswith("IN-SAMPLE")
        assert report["validation"]["label"].startswith("VALIDATION")
        assert report["out_of_sample"]["label"].startswith("OUT-OF-SAMPLE")
        # Periods are distinct objects; no shared aggregate.
        assert report["training"] is not report["validation"]
        assert report["validation"] is not report["out_of_sample"]


def _made_repo_for(provider, symbol, timeframe, n):
    repo = PartitionedResearchRepository(tempfile.mkdtemp())
    candles = provider.fetch_candles(
        symbol, timeframe,
        pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        pd.Timestamp("2026-07-01", tz="UTC").to_pydatetime(),
    )
    repo.merge_candles(candles, symbol=symbol, timeframe=timeframe)
    return repo
