"""Isolation, causality, and determinism tests for the research layer."""
import tempfile

import numpy as np
import pandas as pd

from app.research.config import ResearchConfig
from app.research.data_quality import ResearchDataValidator
from app.research.dataset import PartitionedResearchRepository, sync_partition
from app.research.optimizer import GridSearchOptimizer
from app.research.splits import make_time_split, split_frame
from tests.research.test_pipeline_orchestration import (
    _DeterministicProvider,
    _made_repo_for,
)


def _frame(n=400, seed=1):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=idx)

class TestDeterminism:
    def test_repeated_research_run_is_deterministic(self):
        from app.research.run import ResearchRunConfig, run_research_pipeline
        provider = _DeterministicProvider("EURUSD", "H1", n=300)
        repo1 = _made_repo_for(provider, "EURUSD", "H1", 300)
        repo2 = _made_repo_for(provider, "EURUSD", "H1", 300)
        cfg = ResearchRunConfig(symbols=("EURUSD",), timeframes=("H1",), strategy_names=("trend_structure",), fetch_days=10)
        r1 = run_research_pipeline(run_cfg=cfg, provider=provider, repo=repo1, output_root=tempfile.mkdtemp(), verbose=False)
        r2 = run_research_pipeline(run_cfg=cfg, provider=provider, repo=repo2, output_root=tempfile.mkdtemp(), verbose=False)
        assert r1["optimization"] == r2["optimization"]
        assert r1["walk_forward"] == r2["walk_forward"]

    def test_future_data_mutation_cannot_change_earlier_result(self):
        df = _frame(400)
        split = make_time_split(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime(), ResearchConfig(train_fraction=0.6, validation_fraction=0.2))
        train, val, test = split_frame(df, split)
        mutated = df.copy()
        test_mid = mutated.index[int(len(test) * 0.5 + len(train) + len(val))]
        mutated.loc[test_mid, "close"] += 5.0
        train2, val2, test2 = split_frame(mutated, split)
        assert train.equals(train2)
        assert val.equals(val2)
        assert not test.equals(test2)

class TestOptimizerNeverTouchesTest:
    def test_optimizer_only_uses_train(self):
        df = _frame(300)
        split = make_time_split(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime(), ResearchConfig(train_fraction=0.6, validation_fraction=0.2))
        train, _, _ = split_frame(df, split)
        touched = []
        def _bt(frame, params=None):
            touched.append(frame.index[0])
            return {"trade_count": 20, "net_pnl": 1.0, "expectancy": 0.2, "profit_factor": 1.2, "max_drawdown": 0.02}
        opt = GridSearchOptimizer(ResearchConfig(min_trades=10), _bt)
        candidates = opt.optimize(train, {"stop_loss": [10, 20]})
        assert len(candidates) == 2
        assert all(t == train.index[0] for t in touched)

class TestIncrementalSyncIdempotent:
    def test_no_duplicates_after_second_sync(self):
        from datetime import datetime, timezone
        repo = PartitionedResearchRepository(tempfile.mkdtemp())
        provider = _DeterministicProvider("EURUSD", "M15", n=120)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 16, tzinfo=timezone.utc)
        _, final1 = sync_partition(provider, repo, "EURUSD", "M15", start=start, end=end)
        _, final2 = sync_partition(provider, repo, "EURUSD", "M15", start=start, end=end)
        assert final2 == final1
        df = repo.load_df("EURUSD", "M15")
        assert df is not None
        assert not df.index.duplicated().any()

class TestMtfCausal:
    def test_higher_time_never_future(self):
        from app.mtf.availability import completed_slot_close
        obs = pd.Timestamp("2026-08-01 09:45", tz="UTC")
        h1_close = completed_slot_close(obs, 60)
        assert h1_close == pd.Timestamp("2026-08-01 09:00", tz="UTC")
        assert h1_close <= obs

    def test_validator_rejects_naive_timezone(self):
        idx = pd.date_range("2026-01-01", periods=50, freq="1h", tz=None)
        df = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05}, index=idx)
        report = ResearchDataValidator().validate(df, "EURUSD", "H1")
        assert report.passed is False
        assert report.timezone_status == "naive"
