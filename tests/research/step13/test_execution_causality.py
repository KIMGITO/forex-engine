"""Step 14 causality audit: candidate -> label/execution temporal correctness.

These tests construct tiny hand-crafted OHLC datasets where the expected
answer is obvious and verify the execution/label simulation is causally
valid and directionally symmetric.

CRITICAL LOOK-AHEAD RULE VERIFIED HERE:
    For ``displacement_confirmation`` / ``retest`` entries, the future-bar
    window starts AFTER the confirmation candle. Bars that closed BEFORE the
    entry price was known must NOT be allowed to stop/target the position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13.candidates import CandidateGenerator
from app.research.step13.execution_model import (
    _resolve_entry_bar,
    simulate_hypothesis_outcome,
)
from app.research.step13.hypotheses import Hypothesis
from app.research.step13.labels import compute_labels


def _candles(rows, start="2024-01-01 10:00", freq="15min"):
    """Build OHLC DataFrame from [open, high, low, close] rows."""
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


class TestBasicOutcomes:
    """TEST A/B/C/D: TP and SL reach for long and short."""

    def _verify(self, outcome, direction, expected_reason):
        assert outcome is not None
        assert outcome["exit_reason"] == expected_reason
        assert outcome["holding_bars"] >= 1
        if direction == "long" and expected_reason == "take_profit":
            assert outcome["r"] > 0
        if direction == "long" and expected_reason == "stop_loss":
            assert outcome["r"] < 0
        if direction == "short" and expected_reason == "take_profit":
            assert outcome["r"] > 0
        if direction == "short" and expected_reason == "stop_loss":
            assert outcome["r"] < 0

    def test_a_long_reaches_tp(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.03, 0.995, 1.03],
            [1.03, 1.04, 1.02, 1.04],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        self._verify(out, "long", "take_profit")
        assert out["exit_price"] == pytest.approx(1.02)

    def test_b_long_reaches_sl(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.005, 0.98, 0.99],
            [0.99, 1.02, 0.98, 1.02],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        self._verify(out, "long", "stop_loss")
        assert out["exit_price"] == pytest.approx(0.99)

    def test_c_short_reaches_tp(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.005, 0.97, 0.98],
            [0.98, 1.00, 0.97, 0.99],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        self._verify(out, "short", "take_profit")
        assert out["exit_price"] == pytest.approx(0.98)

    def test_d_short_reaches_sl(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.03, 0.995, 1.02],
            [1.02, 1.04, 1.00, 1.03],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        self._verify(out, "short", "stop_loss")
        assert out["exit_price"] == pytest.approx(1.01)


class TestSameCandleAmbiguity:
    """TEST E: both TP and SL touched in the same candle."""

    def test_conservative_sl_first_long(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.05, 0.98, 1.01],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["exit_reason"] == "conser_sl_first"
        assert out["exit_price"] == pytest.approx(0.99)
        assert out["r"] < 0

    def test_conservative_sl_first_short(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.03, 0.96, 1.00],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["exit_reason"] == "conser_sl_first"
        assert out["exit_price"] == pytest.approx(1.01)
        assert out["r"] < 0


class TestEntryTiming:
    """TEST F: entry at confirmation candle close — the critical audit."""

    def test_displacement_confirmation_does_not_use_pre_entry_bars(self):
        """CRITICAL REGRESSION TEST for the look-ahead fix.

        Scenario: candidate at 10:00; bar at 10:15 has low 0.98 (would hit
        a stop if position existed); actual displacement confirmation at 10:45
        closes at 1.03; bar at 11:00 high 1.06 reaches TP 1.05.

        A causally valid outcome must be take_profit — the 10:15 bar is
        PRE-ENTRY and must not trigger a stop.
        """
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(entry_rule="displacement_confirmation", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],   # 10:00 candidate
            [1.00, 1.005, 0.98, 0.99],   # 10:15 pre-entry SL-level bar
            [0.99, 0.995, 0.985, 0.99],  # 10:30 pre-entry down bar
            [0.99, 1.04, 0.99, 1.03],    # 10:45 UP displacement (entry 1.03)
            [1.03, 1.06, 1.025, 1.05],   # 11:00 TP hit
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["exit_reason"] == "take_profit", (
            f"look-ahead: pre-entry bars leaked into outcome; got {out['exit_reason']}"
        )
        assert out["exit_price"] == pytest.approx(1.05)
        assert out["r"] > 0

    def test_short_displacement_confirmation_symmetry(self):
        """Short side of the same look-ahead scenario."""
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", entry_rule="displacement_confirmation", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],   # 10:00 candidate
            [1.00, 1.02, 0.995, 1.01],    # 10:15 pre-entry SL-level bar
            [1.01, 1.015, 1.005, 1.01],   # 10:30 pre-entry up bar
            [1.01, 1.015, 0.97, 0.98],    # 10:45 DOWN displacement (entry 0.98)
            # post-entry: entry=0.98, stop=0.99, tp=0.96.
            # Bar 4 must touch TP WITHOUT touching SL.
            [0.98, 0.985, 0.95, 0.96],    # 11:00 high 0.985 < 0.99, low 0.95 <= 0.96
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["exit_reason"] == "take_profit", (
            f"short look-ahead: got {out['exit_reason']}"
        )
        assert out["exit_price"] == pytest.approx(0.96)
        assert out["r"] > 0

    def test_immediate_entry_evaluates_from_next_bar(self):
        """Immediate entry at sweep close: bars AFTER the candidate only."""
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(entry_rule="immediate", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],  # candidate bar high 1.01 must NOT be TP
            # post-entry: entry=1.00, stop=0.99, tp=1.02.
            # Bar 1 must touch TP WITHOUT touching SL.
            [1.00, 1.06, 1.005, 1.05],   # next bar: high 1.06 >= 1.02, low 1.005 > 0.99
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["exit_reason"] == "take_profit"
        assert out["exit_price"] == pytest.approx(1.02)


class TestMfeMae:
    """TEST G: MFE/MAE only from post-entry bars."""

    def test_long_mfe_mae(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.03, 0.985, 1.02],
            [1.02, 1.04, 1.01, 1.03],
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        # Long MFE = max(high - entry) = max(0.03, 0.04) = 0.04
        assert lab["label_mfe"] == pytest.approx(0.04, abs=1e-6)
        # Long MAE = max(entry - low) = max(0.015, -0.01) = 0.015 (adverse).
        assert lab["label_mae"] == pytest.approx(0.015, abs=1e-6)

    def test_short_mfe_mae(self):
        cand = _short_candidate(entry_ref=1.00)
        hyp = _hyp(direction="short", exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.015, 0.97, 0.98],
            [0.98, 1.00, 0.96, 0.97],
        ])
        labels = compute_labels([cand], candles, hyp)
        assert len(labels) == 1
        lab = labels[0]
        # Short MFE = max(entry - low) = max(0.03, 0.04) = 0.04
        assert lab["label_mfe"] == pytest.approx(0.04, abs=1e-6)
        # Short MAE = max(high - entry) = max(0.015, 0.0) = 0.015 (adverse).
        assert lab["label_mae"] == pytest.approx(0.015, abs=1e-6)


class TestTimeExit:
    """TEST H: neither TP nor SL reached."""

    def test_time_exit(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0")
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.004, 0.996, 1.00],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles, lookback_bars=2)
        assert out is not None
        assert out["exit_reason"] == "time_exit"

    def test_max_holding_bars(self):
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(exit_rule="fixed_rr_2.0", max_holding_bars=2)
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.005, 0.995, 1.00],
            [1.00, 1.004, 0.996, 1.00],
            [1.00, 1.003, 0.997, 1.00],
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles)
        assert out is not None
        assert out["holding_bars"] == 2
        assert out["exit_reason"] == "time_exit"


class TestSweepConfirmationBarrier:
    """TEST: candidates must not use displacement info before the sweep confirms."""

    def _sweep(self, ts="2024-01-01T10:00:00+00:00", avail="2024-01-01T10:45:00+00:00"):
        return {
            "timestamp": ts, "direction": "long", "close_price": 1.00,
            "penetration": 0.002, "excursion": 0.004,
            "available_from": avail,
        }

    def test_displacement_before_sweep_confirmation_rejected(self):
        """A displacement at 10:15 must NOT confirm a sweep that is only
        knowable at 10:45 (available_from). That would be look-ahead."""
        sweeps = [self._sweep(avail="2024-01-01T10:45:00+00:00")]
        displacements = [{
            "timestamp": "2024-01-01T10:15:00+00:00", "direction": "up",
            "classification": "large", "range_ratio": 2.0,
            "available_from": "2024-01-01T10:15:00+00:00",
        }]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=10)
        cands = gen.generate(
            sweeps=sweeps, displacements=displacements,
            regimes=[{
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending", "volatility_state": "normal",
            }],
            features=[{
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 55.0, "session": "europe",
            }],
        )
        assert len(cands) == 0

    def test_displacement_after_sweep_confirmation_accepted(self):
        """A displacement at 11:00 is valid when the sweep confirms at 10:45."""
        sweeps = [self._sweep(avail="2024-01-01T10:45:00+00:00")]
        displacements = [{
            "timestamp": "2024-01-01T11:00:00+00:00", "direction": "up",
            "classification": "large", "range_ratio": 2.0,
            "available_from": "2024-01-01T11:00:00+00:00",
        }]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=10)
        cands = gen.generate(
            sweeps=sweeps, displacements=displacements,
            regimes=[{
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending", "volatility_state": "normal",
            }],
            features=[{
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 55.0, "session": "europe",
            }],
        )
        assert len(cands) == 1
        decision_ts = pd.Timestamp(cands[0]["timestamp"])
        assert decision_ts >= pd.Timestamp("2024-01-01T11:00:00+00:00")

    def test_decision_timestamp_is_latest_of_sweep_and_displacement_avail(self):
        """When displacement available_from is LATER than sweep available_from,
        the candidate timestamp must be the displacement's availability."""
        sweeps = [self._sweep(avail="2024-01-01T10:45:00+00:00")]
        displacements = [{
            "timestamp": "2024-01-01T10:50:00+00:00", "direction": "up",
            "classification": "large", "range_ratio": 2.0,
            "available_from": "2024-01-01T11:15:00+00:00",
        }]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=10)
        cands = gen.generate(
            sweeps=sweeps, displacements=displacements,
            regimes=[{
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending", "volatility_state": "normal",
            }],
            features=[{
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 55.0, "session": "europe",
            }],
        )
        assert len(cands) == 1
        decision_ts = pd.Timestamp(cands[0]["timestamp"])
        assert decision_ts == pd.Timestamp("2024-01-01T11:15:00+00:00")


class TestNoFutureConfirmation:
    """TEST I/J: candidate cannot use future HTF/structure events."""

    def test_no_confirmation_falls_back_to_immediate(self):
        """With no displacement confirmation, entry is at candidate close.

        This still evaluates only post-entry bars — a conservative fallback.
        """
        cand = _long_candidate(entry_ref=1.00)
        hyp = _hyp(entry_rule="displacement_confirmation", exit_rule="fixed_rr_2.0")
        # No direction-consistent displacement after candidate.
        candles = _candles([
            [1.00, 1.01, 0.995, 1.00],
            [1.00, 1.005, 0.995, 1.00],  # flat
            [1.00, 1.004, 0.996, 1.00],  # flat — no up close > 1.00
        ])
        out = simulate_hypothesis_outcome(hyp, cand, candles, lookback_bars=2)
        assert out is not None
        assert out["entry_price"] == pytest.approx(1.00)