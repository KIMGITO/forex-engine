"""Scientific-fix tests: hypothesis-aware R, HTF optionality, dedup, costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13.hypotheses import Hypothesis
from app.research.step13.execution_model import simulate_hypothesis_outcome
from app.research.step13.candidates import CandidateGenerator, candidates_to_frame
from app.research.step13.persist import Step13Artifacts, read_parquet_if_valid


def _candles(n=200, seed=7):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.005, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close},
        index=idx,
    )


def _candidate():
    return {
        "candidate_id": "cand_test",
        "timestamp": "2024-01-01T10:00:00+00:00",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "direction": "long",
        "entry_ref": 100.0,
        "displacement_ref": 100.05,
        "level": 99.8,
        "feature_atr": 0.1,
    }


def _hyp(**overrides):
    base = dict(
        symbol="EURUSD", timeframe="M15", strategy_family="liquidity_sweep",
        event_type="liquidity_sweep", direction="long",
    )
    base.update(overrides)
    return Hypothesis(**base)


class TestHypothesisAwareOutcome:
    def test_entry_rule_changes_entry_price(self):
        c = _candidate()
        candles = _candles()
        h1 = _hyp(entry_rule="immediate", exit_rule="fixed_rr_2.0")
        h2 = _hyp(entry_rule="displacement_confirmation", exit_rule="fixed_rr_2.0")
        o1 = simulate_hypothesis_outcome(h1, c, candles)
        o2 = simulate_hypothesis_outcome(h2, c, candles)
        assert o1 is not None and o2 is not None
        assert o1["entry_price"] != o2["entry_price"]

    def test_stop_rule_changes_stop_and_r(self):
        c = _candidate()
        candles = _candles()
        h1 = _hyp(stop_rule="atr", exit_rule="fixed_rr_2.0")
        h2 = _hyp(stop_rule="liquidity", exit_rule="fixed_rr_2.0")
        o1 = simulate_hypothesis_outcome(h1, c, candles)
        o2 = simulate_hypothesis_outcome(h2, c, candles)
        assert o1 is not None and o2 is not None
        assert o1["stop_price"] != o2["stop_price"]
        assert o1["r"] != o2["r"] or o1["target_price"] != o2["target_price"]

    def test_exit_rule_changes_target_and_r(self):
        c = _candidate()
        # Engineered rising futures so both exit rules reach TP at different prices.
        # Frame spans 09:00 -> 12:15, so the 10:00 candidate has future bars.
        idx = pd.date_range("2024-01-01 09:00", periods=30, freq="15min", tz="UTC")
        close = np.linspace(100.0, 103.0, 30)
        candles = pd.DataFrame(
            {"open": close, "high": close + 0.05, "low": close - 0.02, "close": close},
            index=idx,
        )
        h1 = _hyp(exit_rule="fixed_rr_1.5")
        h2 = _hyp(exit_rule="fixed_rr_3.0")
        o1 = simulate_hypothesis_outcome(h1, c, candles)
        o2 = simulate_hypothesis_outcome(h2, c, candles)
        assert o1 is not None and o2 is not None
        assert o1["target_price"] != o2["target_price"]
        # R must differ because the exit rule places different target prices.
        assert o1["r"] != o2["r"]


class TestHtfOptionality:
    def test_same_population_with_and_without_htf(self):
        """HTF alignment must be an optional hypothesis condition, not a hard
        global filter: the same event population can be evaluated both ways."""
        gen = CandidateGenerator("EURUSD", "M15", require_htf_alignment=False)
        sweeps = [{
            "timestamp": "2024-01-01T10:00:00+00:00", "direction": "long",
            "penetration": 0.002, "excursion": 0.002, "level": 99.8,
        }]
        displacements = [{
            "timestamp": "2024-01-01T10:15:00+00:00", "direction": "long",
            "classification": "large", "range_ratio": 2.0,
        }]
        regimes = [{
            "timestamp": "2024-01-01T09:00:00+00:00",
            "market_state": "trending", "volatility_state": "normal",
        }]
        features = [{
            "timestamp": "2024-01-01T09:45:00+00:00",
            "atr": 0.01, "rsi": 55.0, "session": "europe",
        }]
        # Both with and without HTF rows produce candidates (no hard filter).
        no_htf = gen.generate(sweeps, displacements, regimes, features, mtf_rows=[])
        with_htf = gen.generate(sweeps, displacements, regimes, features, mtf_rows=[
            {
                "timestamp": "2024-01-01T09:45:00+00:00",
                "htf_trend_state": "bullish", "htf_volatility_state": "normal",
            }
        ])
        assert len(no_htf) == 1
        assert len(with_htf) == 1
        # HTF trend is recorded on the candidate so hypotheses can filter.
        assert with_htf[0]["feature_htf_trend"] == "bullish"


class TestCandidateDedup:
    def test_overlapping_chunks_do_not_inflate_sample(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            am = Step13Artifacts(tmp, "EURUSD", "M15")
            # Same candidate_id in both chunks (overlap region).
            df = pd.DataFrame({"candidate_id": ["cand_a", "cand_a", "cand_b"]})
            am.write_chunk(0, {"candidate_events": df})
            am.write_chunk(1, {"candidate_events": df})
            am.merge_chunks_to_datasets(
                ["candidate_events"], "hash", "cfg", "13.2.0"
            )
            merged = read_parquet_if_valid(am.dataset_path("candidate_events"))
            assert merged is not None
            # "cand_a" appears TWICE in raw but must be deduped to 1.
            assert len(merged) == 2
            assert merged["candidate_id"].nunique() == 2


class TestTradingCosts:
    def test_costs_reduce_r(self):
        c = _candidate()
        candles = _candles()
        h = _hyp()
        o_no_cost = simulate_hypothesis_outcome(
            h, c, candles, spread_pips=0.0, slippage_pips=0.0,
        )
        o_cost = simulate_hypothesis_outcome(
            h, c, candles, spread_pips=1.0, slippage_pips=1.0,
        )
        assert o_no_cost is not None and o_cost is not None
        # Costs reduce R (or at worst do not increase it).
        assert o_cost["r"] <= o_no_cost["r"]
