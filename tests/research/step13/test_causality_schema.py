"""Causality, feature/label separation, and schema tests for Step 13."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.step13.schema import (
    CANDIDATE_LABEL_COLUMNS,
    SchemaError,
    validate_candidate_events,
    validate_candidate_labels,
    validate_feature_label_separation,
)
from app.research.step13.candidates import CandidateGenerator, candidates_to_frame
from app.research.step13.labels import compute_labels, labels_to_frame
from app.research.step13.extract import EventExtractor, extract_rows_to_frame
from app.research.step13.warmup import causal_htf_lookback_bars, clip_htf_frame
from app._causal_index import available_prefix, build_causal_index


def _frame(n=400, seed=42, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.005, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        },
        index=idx,
    )


class TestFeatureLabelSeparation:
    def test_mixed_frame_rejected(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "feature_atr": [1.0],
                "label_mfe": [0.5],
            }
        )
        with pytest.raises(SchemaError):
            validate_feature_label_separation(df)

    def test_feature_only_frame_ok(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "feature_atr": [1.0],
                "feature_rsi": [50.0],
            }
        )
        validate_feature_label_separation(df)  # no error

    def test_label_only_frame_ok(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "label_mfe": [0.5],
                "label_mae": [0.2],
            }
        )
        validate_feature_label_separation(df)  # no error

    def test_candidate_events_requires_core_columns(self):
        with pytest.raises(SchemaError):
            validate_candidate_events(pd.DataFrame({"candidate_id": ["a"]}))

    def test_candidate_events_timestamp_tz_aware(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "timestamp": pd.to_datetime(["2024-01-01"]),  # naive
                "symbol": ["EURUSD"],
                "timeframe": ["M15"],
                "direction": ["long"],
                "feature_atr": [1.0],
            }
        )
        with pytest.raises(SchemaError):
            validate_candidate_events(df)

    def test_candidate_labels_no_features(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
                "feature_atr": [1.0],
            }
        )
        with pytest.raises(SchemaError):
            validate_candidate_labels(df)

    def test_candidate_labels_ok(self):
        df = labels_to_frame(
            [
                {
                    "candidate_id": "cand_1",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "label_mfe": 1.0,
                    "label_mae": 0.5,
                    "label_tp_hit": True,
                    "label_sl_hit": False,
                    "label_excursion_after_bars": 10,
                    "label_future_return": 0.01,
                    "label_displacement_after": 0.2,
                }
            ]
        )
        validate_candidate_labels(df)  # no error


class TestCausalIndexReuse:
    def test_available_prefix_respects_available_from(self):
        from datetime import datetime, timezone

        class _E:
            def __init__(self, id, available_from):
                self.id = id
                self.available_from = available_from

        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        items = [
            _E("past", datetime(2024, 1, 1, tzinfo=timezone.utc)),
            _E("future", datetime(2024, 12, 1, tzinfo=timezone.utc)),
            _E("none", None),
        ]
        sorted_items, keys = build_causal_index(items)
        avail = available_prefix(sorted_items, keys, now)
        ids = [e.id for e in avail]
        assert "past" in ids
        assert "none" in ids
        assert "future" not in ids


class TestWarmup:
    def test_causal_lookback_positive(self):
        from app.mtf.config import MtfConfig

        lb = causal_htf_lookback_bars(MtfConfig())
        assert lb >= 100  # at least the slow EMA + slack

    def test_clip_htf_frame(self):
        # Frame spans from 2023 into early 2024; base_first is inside it.
        idx = pd.date_range("2023-01-01", periods=9000, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": np.arange(9000.0)}, index=idx)
        base_first = pd.Timestamp("2024-01-01", tz="UTC")
        clipped = clip_htf_frame(df, base_first, "1h", 100)
        assert len(clipped) < len(df)
        assert clipped.index[0] >= base_first - pd.Timedelta(hours=100)
        # And the clipped frame retains bars at/after the cutoff.
        assert clipped.index[-1] == df.index[-1]


class TestSweepCausality:
    def test_sweep_uses_only_prior_zones(self):
        """Sweep detection must never use a zone that isn't yet available."""
        from app.market_structure.liquidity import detect_sweeps
        from app.market_structure.models import LiquidityZone, SweepType
        from datetime import datetime, timezone, timedelta

        # Two candles: first creates a high level, second sweeps it.
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.10] * 10,
                "high": [1.11, 1.12, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11],
                "low": [1.09] * 10,
                "close": [1.10] * 10,
            },
            index=idx,
        )
        # Zone available after bar 5.
        zone = LiquidityZone(
            symbol="EURUSD", timeframe="M15", zone_type="equal_highs",
            upper=1.12, lower=1.115, mid=1.1175, swing_count=2,
            first_timestamp=idx[0].to_pydatetime(),
            last_timestamp=idx[1].to_pydatetime(),
            available_from=idx[5].to_pydatetime(),
        )
        sweeps = detect_sweeps(df, [zone], "EURUSD", "M15", sweep_bars=3)
        # No sweep BEFORE zone availability even though high[1] > upper.
        assert len(sweeps) == 0

    def test_sweep_detected_after_zone_available(self):
        from app.market_structure.liquidity import detect_sweeps
        from app.market_structure.models import LiquidityZone
        from datetime import datetime, timezone

        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.10] * 10,
                "high": [1.12, 1.12, 1.12, 1.13, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11],
                "low": [1.09] * 10,
                "close": [1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11],
            },
            index=idx,
        )
        zone = LiquidityZone(
            symbol="EURUSD", timeframe="M15", zone_type="equal_highs",
            upper=1.12, lower=1.115, mid=1.1175, swing_count=2,
            first_timestamp=idx[0].to_pydatetime(),
            last_timestamp=idx[1].to_pydatetime(),
            available_from=idx[2].to_pydatetime(),
        )
        sweeps = detect_sweeps(df, [zone], "EURUSD", "M15", sweep_bars=3)
        assert len(sweeps) == 1


class TestDisplacementCausality:
    def test_displacement_available_from(self):
        df = _frame(200)
        extractor = EventExtractor("EURUSD", "M15")
        rows = extractor.extract(df)
        for d in rows["displacement"]:
            # The displacement's available_from == its timestamp (causal).
            assert d["available_from"] == d["timestamp"]


class TestCandidateGeneration:
    def _simple_sweep_rows(self):
        from datetime import datetime, timezone

        return [
            {
                "timestamp": "2024-01-01T10:00:00+00:00",
                "symbol": "EURUSD",
                "timeframe": "M15",
                "direction": "long",
                "sweep_type": "low_sweep",
                "level": 1.09,
                "extreme_price": 1.088,
                "close_price": 1.092,
                "zone_id": "z1",
                "penetration": 0.002,
                "excursion": 0.002,
                "session": "europe",
                "regime": "trending",
                "htf_bias": "bullish",
                "available_from": "2024-01-01T10:15:00+00:00",
            }
        ]

    def test_candidate_requires_displacement(self):
        gen = CandidateGenerator("EURUSD", "M15")
        sweeps = self._simple_sweep_rows()
        # No displacement.
        cands = gen.generate(
            sweeps=sweeps,
            displacements=[],
            regimes=[],
            features=[],
        )
        assert len(cands) == 0

    def test_candidate_generated_with_displacement_and_regime(self):
        gen = CandidateGenerator(
            "EURUSD", "M15",
            sweep_displacement_lookback=10,
            require_htf_alignment=False,
        )
        sweeps = self._simple_sweep_rows()
        displacements = [
            {
                "timestamp": "2024-01-01T10:15:00+00:00",
                "direction": "long",
                "classification": "large",
                "range_ratio": 2.0,
                "available_from": "2024-01-01T10:15:00+00:00",
            }
        ]
        regimes = [
            {
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending",
                "volatility_state": "normal",
                "available_from": "2024-01-01T09:00:00+00:00",
            }
        ]
        features = [
            {
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 55.0, "session": "europe",
            }
        ]
        cands = gen.generate(
            sweeps=sweeps,
            displacements=displacements,
            regimes=regimes,
            features=features,
        )
        assert len(cands) == 1
        df = candidates_to_frame(cands)
        # Candidate dataset is feature_* only.
        assert not any(c.startswith("label_") for c in df.columns)
        assert any(c.startswith("feature_") for c in df.columns)