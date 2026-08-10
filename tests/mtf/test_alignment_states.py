"""Tests for MTF alignment states (explainable, no magic scores)."""

from datetime import datetime, timezone

from app.mtf.alignment import classify_alignment, compute_strength, tier_direction
from app.mtf.models import MtfAlignmentState, TimeframeContext


def _tier(tf, trend, bias, present=True, vol="normal", market="trending"):
    return TimeframeContext(
        timeframe=tf,
        timestamp=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
        trend_state=trend,
        volatility_state=vol,
        market_state=market,
        structural_bias=bias,
        present=present,
        available_from=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
    )


class TestTierDirection:
    def test_long_when_bullish_agree(self):
        assert tier_direction(_tier("1h", "bullish", "bullish")) == "long"

    def test_short_when_bearish_agree(self):
        assert tier_direction(_tier("1h", "bearish", "bearish")) == "short"

    def test_neutral_when_mixed(self):
        assert tier_direction(_tier("1h", "bullish", "bearish")) == "neutral"

    def test_none_when_missing(self):
        assert tier_direction(_tier("1h", None, None)) == "neutral"


class TestClassifyAlignment:
    def test_aligned_long(self):
        state, _, strength = classify_alignment(
            "long",
            [_tier("1h", "bullish", "bullish"), _tier("4h", "bullish", "bullish")],
            min_aligned=1,
        )
        assert state == MtfAlignmentState.ALIGNED_LONG
        assert strength == 1.0

    def test_aligned_short(self):
        state, _, _ = classify_alignment(
            "short",
            [_tier("1h", "bearish", "bearish"), _tier("4h", "bearish", "bearish")],
            min_aligned=1,
        )
        assert state == MtfAlignmentState.ALIGNED_SHORT

    def test_conflicted(self):
        state, _, _ = classify_alignment(
            "long",
            [_tier("1h", "bullish", "bullish"), _tier("4h", "bearish", "bearish")],
            min_aligned=1,
        )
        assert state == MtfAlignmentState.CONFLICTED

    def test_unknown_when_insufficient(self):
        state, _, _ = classify_alignment(
            "long",
            [_tier("1h", "neutral", "neutral")],
            min_aligned=2,
        )
        assert state == MtfAlignmentState.UNKNOWN

    def test_no_hierarchy_unknown(self):
        state, reasons, _ = classify_alignment("long", [], min_aligned=1)
        assert state == MtfAlignmentState.UNKNOWN
        assert "no higher-timeframe tiers" in reasons


class TestComputeStrength:
    def test_full_agreement(self):
        assert compute_strength("long", ["long", "long"]) == 1.0

    def test_partial(self):
        assert compute_strength("long", ["long", "short"]) == 0.5

    def test_ignores_neutral(self):
        assert compute_strength("long", ["long", "neutral", None]) == 1.0

    def test_none_when_unknown(self):
        assert compute_strength("long", [None, "neutral"]) == 0.0