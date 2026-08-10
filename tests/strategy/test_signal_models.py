"""Tests for Signal / Setup / risk-geometry models."""

import datetime

import pytest
from pydantic import ValidationError

from app.strategy.config import StrategyConfig
from app.strategy.models import Setup, Signal, SignalDirection
from app.strategy.signals import (
    SignalScore,
    calculate_risk_geometry,
    classify_strength,
)


def _ts():
    return datetime.datetime(2024, 6, 10, 12, 0, tzinfo=datetime.timezone.utc)


def _make_signal(direction=SignalDirection.LONG, **overrides):
    base = {
        "signal_id": "s1",
        "timestamp": _ts(),
        "symbol": "EURUSD",
        "timeframe": "1h",
        "direction": direction,
        "strength": "moderate",
        "score": 4.0,
        "max_score": 5.0,
        "entry": 1.10000 if direction == SignalDirection.LONG else 1.11000,
        "stop_loss": 1.09500 if direction == SignalDirection.LONG else 1.11500,
        "take_profit": 1.11000 if direction == SignalDirection.LONG else 1.10000,
        "risk_distance": 0.005,
        "reward_distance": 0.010,
        "risk_reward_ratio": 2.0,
        "strategy": "test",
        "available_from": _ts(),
    }
    base.update(overrides)
    return Signal(**base)


class TestSignalValidation:
    def test_valid_long(self):
        s = _make_signal()
        assert s.direction == SignalDirection.LONG
        assert s.risk_reward_ratio == pytest.approx(2.0)

    def test_valid_short(self):
        s = _make_signal(SignalDirection.SHORT)
        assert s.direction == SignalDirection.SHORT

    def test_invalid_long_levels(self):
        with pytest.raises(ValidationError):
            _make_signal(SignalDirection.LONG, stop_loss=1.105, take_profit=1.10)

    def test_invalid_short_levels(self):
        with pytest.raises(ValidationError):
            _make_signal(SignalDirection.SHORT, stop_loss=1.10, take_profit=1.115)

    def test_inconsistent_rr(self):
        with pytest.raises(ValidationError):
            _make_signal(risk_reward_ratio=3.0)


class TestSetupIdentity:
    def test_identity_deterministic(self):
        a = Setup(
            symbol="EURUSD", timeframe="1h", direction=SignalDirection.LONG,
            strategy="s", anchor_timestamp=_ts(), anchor_price=1.1,
            context_key="k",
        )
        b = Setup(
            symbol="EURUSD", timeframe="1h", direction=SignalDirection.LONG,
            strategy="s", anchor_timestamp=_ts(), anchor_price=1.1,
            context_key="k",
        )
        assert a.identity() == b.identity()


class TestRiskGeometry:
    def test_long_geometry(self):
        geo = calculate_risk_geometry(
            SignalDirection.LONG, entry=1.1, atr_value=0.001,
            stop_distance_atr=1.0, reward_risk_target=2.0,
        )
        assert geo["stop"] == pytest.approx(1.099)
        assert geo["target"] == pytest.approx(1.102)
        assert geo["r_r"] == pytest.approx(2.0)

    def test_short_geometry(self):
        geo = calculate_risk_geometry(
            SignalDirection.SHORT, entry=1.1, atr_value=0.001,
            stop_distance_atr=1.5, reward_risk_target=3.0,
        )
        assert geo["stop"] == pytest.approx(1.1015)
        assert geo["target"] == pytest.approx(1.0955)

    def test_zero_atr_raises(self):
        from app.strategy.errors import SignalValidationError

        with pytest.raises(SignalValidationError):
            calculate_risk_geometry(
                SignalDirection.LONG, entry=1.1, atr_value=0.0,
                stop_distance_atr=1.0, reward_risk_target=2.0,
            )


class TestScoring:
    def test_score_totals(self):
        score = SignalScore()
        score.add("a", 1.0)
        score.add("b", 2.0)
        assert score.total == pytest.approx(3.0)

    def test_strength_thresholds(self):
        cfg = StrategyConfig()
        assert classify_strength(1.0, cfg).value == "weak"
        assert classify_strength(3.0, cfg).value == "moderate"
        assert classify_strength(5.0, cfg).value == "strong"
