"""Tests for Step 13 Alpha Discovery: hypotheses, evaluator, stats, scoring."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.step13.hypotheses import (
    Hypothesis,
    HypothesisGridLimits,
    conditions_pass,
    generate_hypotheses,
)
from app.research.step13.evaluator import FastResearchEvaluator, ResearchEvaluatorResult
from app.research.step13.statistics import (
    adjusted_p_values_bh,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    expectancy_stats,
    stability_by_group,
    stability_by_halves,
)
from app.research.step13.discovery import (
    compute_discovery_score,
    rank_candidates,
)
from app.research.step13.candidates_io import (
    build_candidate_artifact,
    write_candidate_artifact,
    load_candidate_artifact,
)


def _hypothesis(**overrides):
    kwargs = dict(
        symbol="EURUSD",
        timeframe="M15",
        strategy_family="liquidity_sweep",
        event_type="liquidity_sweep",
        direction="long",
        hypothesis_description="test hypothesis",
    )
    kwargs.update(overrides)
    return Hypothesis(**kwargs)


class TestHypothesis:
    def test_id_deterministic(self):
        h1 = _hypothesis()
        h2 = _hypothesis()
        assert h1.hypothesis_id == h2.hypothesis_id

    def test_id_changes_with_conditions(self):
        h1 = _hypothesis(conditions=("session=europe",))
        h2 = _hypothesis(conditions=("session=newyork",))
        assert h1.hypothesis_id != h2.hypothesis_id

    def test_from_dict_roundtrip(self):
        h = _hypothesis(conditions=("regime=trending",))
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        assert h2.hypothesis_id == h.hypothesis_id
        assert h2.conditions == h.conditions

    def test_missing_fields_rejected(self):
        with pytest.raises(Exception):
            Hypothesis.from_dict({"symbol": "EURUSD"})

    def test_generation_bounded(self):
        hs = generate_hypotheses(
            symbols=("EURUSD",),
            timeframes=("M15",),
            limits=HypothesisGridLimits(max_hypotheses=20),
        )
        assert len(hs) <= 20

    def test_conditions_pass(self):
        h = _hypothesis(conditions=("session=europe", "regime=trending"))
        row = {
            "feature_session": "europe",
            "regime": "trending",
            "feature_structure_bias": "bullish",
        }
        assert conditions_pass(h, row)
        row_bad = {
            "feature_session": "newyork",
            "regime": "trending",
        }
        assert not conditions_pass(h, row_bad)


class TestStatistics:
    def test_expectancy_stats(self):
        r = [0.1, -0.05, 0.2, 0.15, -0.1, 0.08]
        e = expectancy_stats(r)
        assert e["n"] == 6
        assert e["mean_r"] == round(np.mean(r), 4)
        assert e["sem_r"] is not None

    def test_expectancy_empty(self):
        e = expectancy_stats([])
        assert e["n"] == 0
        assert e["mean_r"] is None

    def test_bootstrap_ci_deterministic(self):
        r = [0.1, -0.05, 0.2, 0.15, -0.1, 0.08, 0.03, 0.12, -0.02, 0.06]
        b1 = bootstrap_ci(r, n_bootstrap=100, seed=42)
        b2 = bootstrap_ci(r, n_bootstrap=100, seed=42)
        assert b1 == b2
        assert b1["ci_lower"] is not None
        assert b1["ci_lower"] <= b1["mean_r"] <= b1["ci_upper"]

    def test_cohens_d(self):
        r = [0.1, 0.2, 0.15, 0.3, 0.05, 0.12]
        d = cohens_d(r)
        assert d is not None and d > 0

    def test_cohens_d_zero_std(self):
        assert cohens_d([1.0, 1.0]) is None

    def test_stability_halves(self):
        r = [0.1, 0.2, 0.15, 0.05, 0.02, -0.01]
        s = stability_by_halves(r)
        assert s["degradation_ok"] is True  # second half near zero but non-negative

    def test_stability_group(self):
        groups = {
            "session=europe": [0.1, 0.2],
            "session=newyork": [-0.05, -0.02],
        }
        s = stability_by_group(groups)
        assert s["groups_tested"] == 2
        assert 0.0 < s["positive_group_fraction"] < 1.0

    def test_benjamini_hochberg(self):
        p = [0.001, 0.01, 0.04, 0.5, 0.8]
        surv = benjamini_hochberg(p)
        assert surv[0] is True
        assert surv[3] is False

    def test_adjusted_p_values(self):
        p = [0.001, 0.01, 0.04, 0.5, 0.8]
        q = adjusted_p_values_bh(p)
        assert len(q) == len(p)
        assert all(0.0 <= v <= 1.0 for v in q)


class TestEvaluator:
    def _candidate_events(self, n=10, direction="long"):
        rows = []
        for i in range(n):
            rows.append(
                {
                    "candidate_id": f"cand_{i}",
                    "timestamp": f"2024-01-01T{i:02d}:00:00+00:00",
                    "symbol": "EURUSD",
                    "timeframe": "M15",
                    "direction": direction,
                    "strategy_family": "liquidity_sweep",
                    "regime": "trending",
                    "feature_session": "europe",
                    "feature_atr": 0.01,
                    "feature_structure_bias": "bullish",
                    "feature_htf_trend": "bullish",
                    "feature_volatility": "normal",
                }
            )
        return pd.DataFrame(rows)

    def _candidate_labels(self, n=10):
        rows = []
        for i in range(n):
            r = 0.1 if i % 2 == 0 else -0.05
            rows.append(
                {
                    "candidate_id": f"cand_{i}",
                    "label_mfe": 0.02 if r > 0 else 0.005,
                    "label_mae": 0.005 if r > 0 else 0.02,
                    "label_excursion_after_bars": 20,
                }
            )
        return pd.DataFrame(rows)

    def test_evaluate_no_samples(self):
        h = _hypothesis()
        ev = FastResearchEvaluator()
        result = ev.evaluate(h, pd.DataFrame(), None)
        assert result.sample_count == 0

    def test_evaluate_collects_samples(self):
        h = _hypothesis()
        ev = FastResearchEvaluator()
        events = self._candidate_events(n=10)
        labels = self._candidate_labels(n=10)
        result = ev.evaluate(h, events, labels)
        assert result.sample_count > 0
        assert result.win_count > 0
        assert result.loss_count > 0
        assert result.groups  # session/regime/symbol groups populated

    def test_evaluate_filters_by_conditions(self):
        h = _hypothesis(conditions=("session=newyork",))
        ev = FastResearchEvaluator()
        events = self._candidate_events(n=10)  # all are session=europe
        result = ev.evaluate(h, events, None)
        assert result.sample_count == 0

    def test_evaluate_respects_direction(self):
        h = _hypothesis(direction="short")
        ev = FastResearchEvaluator()
        events = self._candidate_events(n=10, direction="long")
        result = ev.evaluate(h, events, None)
        assert result.sample_count == 0


def _candidate_events(n=10, direction="long"):
    rows = []
    for i in range(n):
        rows.append(
            {
                "candidate_id": f"cand_{i}",
                "timestamp": f"2024-01-01T{i:02d}:00:00+00:00",
                "symbol": "EURUSD",
                "timeframe": "M15",
                "direction": direction,
                "strategy_family": "liquidity_sweep",
                "regime": "trending",
                "feature_session": "europe",
                "feature_atr": 0.01,
                "feature_structure_bias": "bullish",
                "feature_htf_trend": "bullish",
                "feature_volatility": "normal",
            }
        )
    return pd.DataFrame(rows)


def _candidate_labels(n=10):
    rows = []
    for i in range(n):
        r = 0.1 if i % 2 == 0 else -0.05
        rows.append(
            {
                "candidate_id": f"cand_{i}",
                "label_mfe": 0.02 if r > 0 else 0.005,
                "label_mae": 0.005 if r > 0 else 0.02,
                "label_excursion_after_bars": 20,
            }
        )
    return pd.DataFrame(rows)


class TestDiscovery:
    def test_score_no_samples(self):
        h = _hypothesis()
        ev = FastResearchEvaluator()
        result = ev.evaluate(h, pd.DataFrame(), None)
        score = compute_discovery_score(result, min_sample=30)
        assert score.total == 0.0
        assert "no samples" in score.overfit_warnings[0]

    def test_score_positive_edge(self):
        h = _hypothesis()
        ev = FastResearchEvaluator()
        events = _candidate_events(n=50)
        labels = _candidate_labels(n=50)
        result = ev.evaluate(h, events, labels)
        score = compute_discovery_score(result, min_sample=10)
        assert score.total > 0.0

    def test_score_warns_small_sample(self):
        h = _hypothesis()
        ev = FastResearchEvaluator()
        events = _candidate_events(n=5)
        labels = _candidate_labels(n=5)
        result = ev.evaluate(h, events, labels)
        score = compute_discovery_score(result, min_sample=30)
        assert any("sample size" in w for w in score.overfit_warnings)

    def test_rank_deterministic(self):
        h1 = _hypothesis()
        h2 = _hypothesis(conditions=("session=europe",))
        ev = FastResearchEvaluator()
        events = _candidate_events(n=50)
        labels = _candidate_labels(n=50)
        scores = {}
        for h in (h1, h2):
            r = ev.evaluate(h, events, labels)
            scores[h.hypothesis_id] = compute_discovery_score(r, min_sample=10)
        ranked = rank_candidates(scores)
        assert len(ranked) == 2
        # Ranking is deterministic.
        ranked2 = rank_candidates(scores)
        assert ranked == ranked2


class TestCandidateArtifact:
    def test_build_artifact(self):
        h = _hypothesis()
        artifact = build_candidate_artifact(
            hypothesis=h,
            evaluator_stats={"n": 100, "win_rate": 0.55, "max_drawdown_r": 1.2},
            discovery_score={"total": 0.65, "components": {}},
            sample_count=100,
            overfit_warnings=[],
            data_hash="abc123",
            engine_version="13.2.0",
            configuration_hash="def456",
        )
        assert artifact["status"] == "DISCOVERY_CANDIDATE"
        assert artifact["candidate_id"] == h.hypothesis_id
        assert "hypothesis" in artifact
        assert "overfit_warnings" in artifact

    def test_artifact_never_claims_profit(self):
        h = _hypothesis()
        artifact = build_candidate_artifact(
            hypothesis=h,
            evaluator_stats={},
            discovery_score={"total": 0.9},
            sample_count=100,
            overfit_warnings=[],
            data_hash="a",
            engine_version="13.2.0",
            configuration_hash="b",
        )
        assert "PROFITABLE_STRATEGY" not in artifact["status"]
        assert artifact["status"] == "DISCOVERY_CANDIDATE"

    def test_write_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _hypothesis()
            artifact = build_candidate_artifact(
                hypothesis=h,
                evaluator_stats={},
                discovery_score={"total": 0.5},
                sample_count=50,
                overfit_warnings=["sample small"],
                data_hash="a",
                engine_version="13.2.0",
                configuration_hash="b",
            )
            path = write_candidate_artifact(tmp, "EURUSD", "M15", artifact)
            assert path.exists()
            loaded = load_candidate_artifact(path)
            assert loaded is not None
            assert loaded["candidate_id"] == artifact["candidate_id"]
            assert loaded["status"] == "DISCOVERY_CANDIDATE"