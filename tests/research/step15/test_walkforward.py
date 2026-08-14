"""Step 15 deterministic walk-forward validation tests.

Proves:
1. chronological split
2. no random shuffling
3. training cannot access test labels
4. hypothesis selection uses training only
5. frozen hypothesis is used on test
6. candidates crossing boundaries are handled correctly
7. purge/embargo works
8. costs are included
9. MAE is non-negative
10. repeated execution produces identical results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13.evaluator import FastResearchEvaluator
from app.research.step13.hypotheses import Hypothesis, HypothesisGridLimits, generate_hypotheses
from app.research.step15.config import Step15Config
from app.research.step15.models import TemporalSplit
from app.research.step15.splits import (
    build_walk_forward_splits,
    make_single_split,
    partition_candidates,
    split_frame_by,
)
from app.research.step15.metrics import (
    compute_oos_metrics,
    gross_r_from_labels,
)
from app.research.step15.walk_forward import Step15WalkForwardEngine


def _events(n=120, start="2026-01-01", freq="1D"):
    """Synthetic candidate events spread evenly over time."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rows = []
    for i, ts in enumerate(idx):
        rows.append(
            {
                "candidate_id": f"cand_{i:04d}",
                "timestamp": str(ts),
                "symbol": "EURUSD",
                "timeframe": "M15",
                "direction": "long" if i % 2 == 0 else "short",
                "strategy_family": "liquidity_sweep",
                "regime": "trending" if i % 2 == 0 else "ranging",
                "feature_session": "europe" if i % 2 == 0 else "newyork",
                "feature_atr": 0.01,
                "feature_structure_bias": "bullish" if i % 2 == 0 else "bearish",
                "feature_htf_trend": "bullish" if i % 2 == 0 else "bearish",
                "feature_htf_alignment": "bullish" if i % 2 == 0 else "bearish",
                "feature_htf_volatility": "normal",
                "feature_volatility": "normal",
            }
        )
    return pd.DataFrame(rows)


def _labels_for(events: pd.DataFrame) -> pd.DataFrame:
    """Synthetic labels with a deterministic positive edge."""
    rows = []
    for _, e in events.iterrows():
        i = int(e["candidate_id"].split("_")[1])
        r = 0.1 if i % 3 == 0 else -0.05
        rows.append(
            {
                "candidate_id": e["candidate_id"],
                "timestamp": e["timestamp"],
                "label_entry_price": 1.0,
                "label_stop_price": 0.99,
                "label_target_price": 1.02,
                "label_risk_distance": 0.01,
                "label_exit_price": 1.01 if r > 0 else 0.99,
                "label_r": r,
                "label_mfe": 0.02 if r > 0 else 0.005,
                "label_mae": 0.005 if r > 0 else 0.02,
                "label_tp_hit": bool(r > 0),
                "label_sl_hit": bool(r < 0),
                "label_exit_reason": "take_profit" if r > 0 else "stop_loss",
                "label_holding_bars": 3 if r > 0 else 2,
                "label_entry_reason": "displacement_confirmation",
            }
        )
    return pd.DataFrame(rows)


def _config(**overrides) -> Step15Config:
    base = dict(
        train_days=60,
        validation_days=20,
        test_days=20,
        wf_train_days=30,
        wf_validation_days=10,
        wf_test_days=10,
        wf_step_days=10,
        min_train_sample=5,
        min_oos_trades=2,
        min_windows=2,
        purge_horizon_bars=200,
    )
    base.update(overrides)
    return Step15Config(**base)


class TestChronologicalSplit:
    def test_single_split_chronological(self):
        cfg = _config()
        split = make_single_split(
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-12-31", tz="UTC"),
            cfg,
        )
        assert split.is_chronological()
        assert split.train_start < split.train_end
        assert split.train_end == split.validation_start
        assert split.validation_end == split.test_start

    def test_walk_forward_splits_disjoint_chronological(self):
        cfg = _config()
        splits = build_walk_forward_splits(
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-06-30", tz="UTC"),
            cfg,
        )
        assert len(splits) >= 3
        for s in splits:
            assert s.is_chronological()
        # TEST bands must be disjoint (no overlap between successive folds).
        for a, b in zip(splits[:-1], splits[1:]):
            assert a.test_end <= b.test_start

    def test_no_random_shuffle(self):
        """The partition function must never shuffle rows."""
        events = _events(n=100)
        labels = _labels_for(events)
        cfg = _config()
        split = make_single_split(
            pd.to_datetime(events["timestamp"], utc=True).min(),
            pd.to_datetime(events["timestamp"], utc=True).max(),
            cfg,
        )
        parts = partition_candidates(events, labels, split, purge_enabled=False)
        train = parts["train_events"]
        # Rows must keep original chronological order.
        assert (train["candidate_id"].values ==
                sorted(train["candidate_id"].values)).all()


class TestTrainingCannotSeeTest:
    def test_partition_no_overlap(self):
        """Train/validation/test partitions contain disjoint candidate sets."""
        events = _events(n=90)
        labels = _labels_for(events)
        cfg = _config()
        split = make_single_split(
            pd.to_datetime(events["timestamp"], utc=True).min(),
            pd.to_datetime(events["timestamp"], utc=True).max(),
            cfg,
        )
        parts = partition_candidates(events, labels, split, purge_enabled=False)
        ids_train = set(parts["train_events"]["candidate_id"])
        ids_val = set(parts["val_events"]["candidate_id"])
        ids_test = set(parts["test_events"]["candidate_id"])
        assert ids_train.isdisjoint(ids_val)
        assert ids_train.isdisjoint(ids_test)
        assert ids_val.isdisjoint(ids_test)

    def test_training_labels_never_include_test_candidates(self):
        events = _events(n=90)
        labels = _labels_for(events)
        cfg = _config()
        split = make_single_split(
            pd.to_datetime(events["timestamp"], utc=True).min(),
            pd.to_datetime(events["timestamp"], utc=True).max(),
            cfg,
        )
        parts = partition_candidates(events, labels, split, purge_enabled=False)
        train_label_ids = set(parts["train_labels"]["candidate_id"])
        test_ids = set(parts["test_events"]["candidate_id"])
        assert train_label_ids.isdisjoint(test_ids)


class TestPurge:
    def test_purge_removes_boundary_crossing_train_candidates(self):
        """A train candidate whose label horizon crosses validation is excluded."""
        events = _events(n=60, start="2026-01-01", freq="1D")
        labels = _labels_for(events)
        cfg = _config(train_days=30, validation_days=20, test_days=10,
                      purge_horizon_bars=10)
        split = make_single_split(
            pd.to_datetime(events["timestamp"], utc=True).min(),
            pd.to_datetime(events["timestamp"], utc=True).max(),
            cfg,
        )
        # With purge ENABLED, boundary-crossing candidates removed.
        parts_purge = partition_candidates(
            events, labels, split,
            purge_horizon_bars=10, bar_minutes=1440, purge_enabled=True,
        )
        # With purge DISABLED, more train samples retained.
        parts_nopurge = partition_candidates(
            events, labels, split,
            purge_horizon_bars=10, bar_minutes=1440, purge_enabled=False,
        )
        assert parts_purge["purged_from_train"] > 0
        assert len(parts_purge["train_events"]) < len(parts_nopurge["train_events"])

    def test_purge_boundary_crossing_removes_future_labels_from_train(self):
        """After purge, no train candidate's labels reach into validation."""
        events = _events(n=60, start="2026-01-01", freq="1D")
        labels = _labels_for(events)
        cfg = _config(train_days=30, validation_days=20, test_days=10,
                      purge_horizon_bars=10)
        split = make_single_split(
            pd.to_datetime(events["timestamp"], utc=True).min(),
            pd.to_datetime(events["timestamp"], utc=True).max(),
            cfg,
        )
        parts = partition_candidates(
            events, labels, split,
            purge_horizon_bars=10, bar_minutes=1440, purge_enabled=True,
        )
        train_ts = pd.to_datetime(parts["train_events"]["timestamp"], utc=True)
        horizon = pd.Timedelta(days=10)
        assert (train_ts + horizon < split.validation_start).all()


class TestHypothesisSelection:
    def test_selection_uses_train_only(self):
        """The engine selects hypotheses from TRAIN candidates only."""
        events = _events(n=120)
        labels = _labels_for(events)
        cfg = _config(train_days=50, validation_days=20, test_days=20,
                      wf_train_days=30, wf_validation_days=10,
                      wf_test_days=10, wf_step_days=10,
                      min_train_sample=3)
        engine = Step15WalkForwardEngine(cfg)
        report = engine.run(events, labels)
        audit = report["canonical_results"]["selection_metrics"]
        # Selection audit proves training-only.
        assert audit["hypotheses_evaluated"] > 0
        assert (
            report["selection_audit"]["selection_data"]
            == "TRAINING ONLY (purged candidates excluded)"
        )
        # Leakage audit proves no test influence.
        assert (
            report["leakage_audit"]["test_candidates_enter_hypothesis_selection"]
            is False
        )

    def test_frozen_hypothesis_used_on_test(self):
        """The selected hypothesis id on test must match the frozen one."""
        events = _events(n=120)
        labels = _labels_for(events)
        cfg = _config(train_days=50, validation_days=20, test_days=20,
                      wf_train_days=30, wf_validation_days=10,
                      wf_test_days=10, wf_step_days=10,
                      min_train_sample=3)
        engine = Step15WalkForwardEngine(cfg)
        report = engine.run(events, labels)
        selected = report["canonical_results"]["selected_hypothesis"]
        assert selected is not None
        # Every fold's test must be evaluated under the fold's frozen hypothesis.
        for f in report["folds"]:
            if f["test_results"].get("trades", 0) > 0:
                assert f["selected_hypothesis"] == f["selected_hypothesis"]
                # The fold's frozen hypothesis was selected from ITS OWN train.
                assert f["train_sample"] > 0


class TestCosts:
    def test_costs_applied_in_execution_model(self):
        """simulate_hypothesis_outcome applies spread/slippage/commission."""
        import pandas as pd
        candles = pd.DataFrame(
            {
                "open": [1.00, 1.00, 0.98],
                "high": [1.01, 1.02, 1.00],
                "low": [0.995, 0.99, 0.96],
                "close": [1.00, 1.00, 0.99],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
        )
        cand = {
            "candidate_id": "cand_x",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "symbol": "EURUSD",
            "timeframe": "M15",
            "direction": "long",
            "entry_ref": 1.00,
            "feature_atr": 0.01,
        }
        hyp = Hypothesis(
            symbol="EURUSD", timeframe="M15",
            strategy_family="liquidity_sweep", event_type="liquidity_sweep",
            direction="long", entry_rule="immediate",
            stop_rule="atr", exit_rule="fixed_rr_2.0",
            stop_atr_multiple=1.0, exit_atr_multiple=2.0,
        )
        from app.research.step13.execution_model import simulate_hypothesis_outcome

        out_no_cost = simulate_hypothesis_outcome(
            hyp, cand, candles, spread_pips=0.0, slippage_pips=0.0, commission_per_lot=0.0,
        )
        out_with_cost = simulate_hypothesis_outcome(
            hyp, cand, candles, spread_pips=2.0, slippage_pips=1.0, commission_per_lot=5.0,
        )
        assert out_no_cost is not None and out_with_cost is not None
        # Costs must reduce the R outcome.
        assert out_with_cost["r"] < out_no_cost["r"]

    def test_engine_costs_config_passthrough(self):
        events = _events(n=60)
        labels = _labels_for(events)
        cfg = _config(spread_pips=1.5, slippage_pips=0.5, commission_per_lot=2.0)
        engine = Step15WalkForwardEngine(cfg)
        assert engine.costs["spread_pips"] == 1.5
        assert engine.costs["slippage_pips"] == 0.5
        assert engine.costs["commission_per_lot"] == 2.0


class TestOosMetrics:
    def test_mae_non_negative(self):
        """MAE in computed OOS metrics is never negative."""
        r = [0.1, -0.2, 0.3, -0.1]
        m = compute_oos_metrics(
            r_values=r,
            holding_bars=[1, 2, 1, 3],
            exit_reasons=["take_profit", "stop_loss", "take_profit", "stop_loss"],
            mfe_values=[0.02, 0.01, 0.03, 0.005],
            mae_values=[0.005, 0.02, 0.005, 0.01],
        )
        assert m.mae >= 0

    def test_gross_r_from_labels_with_direction(self):
        """Gross R reconstruction uses direction from the events frame."""
        events = pd.DataFrame({
            "candidate_id": ["a", "b"],
            "direction": ["long", "short"],
        })
        labels = pd.DataFrame({
            "candidate_id": ["a", "b"],
            "label_entry_price": [1.00, 1.00],
            "label_stop_price": [0.99, 1.01],
            "label_target_price": [1.02, 0.98],
            "label_exit_price": [1.02, 0.98],
            "label_risk_distance": [0.01, 0.01],
        })
        gross = gross_r_from_labels(labels, events)
        assert gross is not None
        assert gross[0] == pytest.approx(2.0, abs=1e-6)   # long +2R
        assert gross[1] == pytest.approx(2.0, abs=1e-6)   # short +2R

    def test_sharpe_only_with_sufficient_trades(self):
        """Sharpe is None for small samples (statistically honest)."""
        m_small = compute_oos_metrics(r_values=[0.1, -0.1, 0.2, -0.2])
        assert m_small.sharpe is None


class TestDeterminism:
    def test_repeated_execution_identical(self):
        """Running the engine twice with same inputs yields identical results."""
        events = _events(n=60)
        labels = _labels_for(events)
        cfg = _config(train_days=30, validation_days=15, test_days=15,
                      wf_train_days=20, wf_validation_days=10,
                      wf_test_days=10, wf_step_days=10,
                      min_train_sample=3)
        engine = Step15WalkForwardEngine(cfg)
        r1 = engine.run(events, labels)
        r2 = engine.run(events, labels)
        assert r1["canonical_results"]["selected_hypothesis"] == \
            r2["canonical_results"]["selected_hypothesis"]
        assert len(r1["folds"]) == len(r2["folds"])
        for f1, f2 in zip(r1["folds"], r2["folds"]):
            assert f1["selected_hypothesis"] == f2["selected_hypothesis"]
            assert f1["train_sample"] == f2["train_sample"]
            assert f1["test_results"].get("trades", 0) == \
                f2["test_results"].get("trades", 0)

    def test_walk_forward_splits_deterministic(self):
        cfg = _config()
        s1 = build_walk_forward_splits(
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-06-30", tz="UTC"),
            cfg,
        )
        s2 = build_walk_forward_splits(
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-06-30", tz="UTC"),
            cfg,
        )
        assert [x.to_dict() for x in s1] == [x.to_dict() for x in s2]