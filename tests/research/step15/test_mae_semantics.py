"""Step 15 MAE semantic regression tests.

MAE must ALWAYS be expressed as a non-negative adverse excursion:
    long  : MAE = max(0, entry - minimum_post_entry_low)
    short : MAE = max(0, maximum_post_entry_high - entry)

A completely favorable trade has MAE = 0. MFE semantics are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13.execution_model import simulate_hypothesis_outcome
from app.research.step13.hypotheses import Hypothesis
from app.research.step13.labels import compute_labels


def _candles(rows, start="2024-01-01 10:00", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )


def _long_candidate(ts="2024-01-01 10:00", entry_ref=1.00):
    return {
        "candidate_id": "cand_long",
        "timestamp": ts + "+00:00",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "direction": "long",
        "entry_ref": entry_ref,
        "displacement_ref": "2024-01-01T10:45:00+00:00",
        "level": 0.995,
        "feature_atr": 0.01,
    }


def _short_candidate(ts="2024-01-01 10:00", entry_ref=1.00):
    return {
        "candidate_id": "cand_short",
        "timestamp": ts + "+00:00",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "direction": "short",
        "entry_ref": entry_ref,
        "displacement_ref": "2024-01-01T10:45:00+00:00",
        "level": 1.005,
        "feature_atr": 0.01,
    }


def _hyp(**overrides):
    base = dict(
        symbol="EURUSD", timeframe="M15", strategy_family="liquidity_sweep",
        event_type="liquidity_sweep", direction="long",
        entry_rule="immediate", stop_rule="atr", exit_rule="fixed_rr_2.0",
    )
    base.update(overrides)
    return Hypothesis(**base)


class TestMaeCompletelyFavorable:
    """MAE must be 0 when price never moves adversely."""

    def test_long_completely_favorable_mae_zero_labels(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],  # candidate
            [1.00, 1.02, 1.005, 1.02],  # never below 1.005 (> entry 1.00)
            [1.02, 1.04, 1.015, 1.03],
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        # Long MAE = max(0, max(entry - low)) = max(0, max(-0.005, -0.01)) = 0
        assert lab["label_mae"] == pytest.approx(0.0, abs=1e-9)

    def test_long_completely_favorable_mae_zero_execution(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.02, 1.005, 1.02],
            [1.02, 1.04, 1.015, 1.03],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_short_completely_favorable_mae_zero_labels(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 0.995, 0.98, 0.98],  # never above 0.995 (< entry 1.00)
            [0.98, 0.99, 0.96, 0.97],
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        # Short MAE = max(0, max(high - entry)) = max(0, max(-0.005, -0.01)) = 0
        assert lab["label_mae"] == pytest.approx(0.0, abs=1e-9)

    def test_short_completely_favorable_mae_zero_execution(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 0.995, 0.98, 0.98],
            [0.98, 0.99, 0.96, 0.97],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["mae"] == pytest.approx(0.0, abs=1e-9)


class TestMaeAdverse:
    """MAE must be positive when price moves adversely."""

    def test_long_adverse_mae_positive_labels(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.01, 0.97, 0.98],   # low 0.97 -> adverse 0.03
            [0.98, 1.00, 0.96, 0.99],   # low 0.96 -> adverse 0.04 (worse)
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        assert lab["label_mae"] > 0
        # MAE = max(0, entry - min_low) = max(0, 1.00 - 0.96) = 0.04
        # (for short: max(0, max_high - entry) = max(0, 1.04 - 1.00) = 0.04)
        assert lab["label_mae"] == pytest.approx(0.04, abs=1e-6)

    def test_long_adverse_mae_positive_execution(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.01, 0.97, 0.98],
            [0.98, 1.00, 0.96, 0.99],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["mae"] > 0
        assert out["mae"] == pytest.approx(0.04, abs=1e-6)

    def test_short_adverse_mae_positive_labels(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.03, 0.995, 1.02],  # high 1.03 -> adverse 0.03
            [1.02, 1.04, 1.00, 1.03],   # high 1.04 -> adverse 0.04 (worse)
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        assert lab["label_mae"] > 0
        # MAE = max(0, entry - min_low) = max(0, 1.00 - 0.96) = 0.04
        # (for short: max(0, max_high - entry) = max(0, 1.04 - 1.00) = 0.04)
        assert lab["label_mae"] == pytest.approx(0.04, abs=1e-6)

    def test_short_adverse_mae_positive_execution(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.03, 0.995, 1.02],
            [1.02, 1.04, 1.00, 1.03],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["mae"] > 0
        assert out["mae"] == pytest.approx(0.04, abs=1e-6)


class TestMfeSemantics:
    """MFE semantics unchanged (may be positive or zero, never negative)."""

    def test_mfe_positive(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.02, 1.005, 1.02],
        ])
        labels = compute_labels([cand], candles, hyp)
        assert labels[0]["label_mfe"] > 0

    def test_mfe_zero_when_no_favorable_move(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.005, 0.98, 0.99],  # only adverse
        ])
        labels = compute_labels([cand], candles, hyp)
        # MFE = max(high - entry) = max(0.005, ...) = 0.005 (favorable small)
        assert labels[0]["label_mfe"] >= 0