"""Regression tests for the Step 13 data-integrity fixes.

These tests FAIL if the following bugs return:

1. ``persist.merge_chunks_to_datasets`` produced the UNION of all schemas'
   columns for every dataset (49 columns), padding unrelated columns with
   NaN — e.g. OHLC appearing 100% NULL on structure_events/sweeps/regime.
2. HTF partitions were loaded by native names ("1h"/"4h"/"1d") that do not
   exist on disk (the repo stores H1/H4/D1), silently disabling MTF context.
3. Candidate generation compared sweep direction ("long"/"short") against
   displacement direction ("up"/"down") verbatim, so every candidate was
   silently dropped (0 candidates in the EURUSD/M15 run).
4. Warm-up overlap bars were re-emitted by every chunk, inflating dense
   datasets (features/regime/displacement).
5. ``mtf_context`` rows carried an accidental ``timeframe`` column and NaN
   OHLC because the union-column corruption leaked across schemas.
6. ``entry_ref`` was an ISO date string instead of a price, which crashes
   ``float(entry_ref)`` in labels/execution_model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.step13.candidates import (
    CandidateGenerator,
    _direction_key,
    _find_displacement,
)
from app.research.step13.persist import Step13Artifacts, read_parquet_if_valid
from app.research.step13.runner import _make_chunks, _partition_tf
from app.research.step13.schema import (
    DISPLACEMENT_COLUMNS,
    FEATURES_COLUMNS,
    LIQUIDITY_ZONES_COLUMNS,
    MTF_CONTEXT_COLUMNS,
    REGIME_COLUMNS,
    STRUCTURE_COLUMNS,
    SWEEPS_COLUMNS,
    event_schema_for,
)


# ── 1. Persist schema projection ────────────────────────────────────────────

class TestPersistSchemaProjection:
    """Merged parquet datasets MUST contain exactly their schema columns.

    Regression: ``write_chunk`` uses ``pd.concat`` over all dataset frames,
    which unions every dataset's columns into every other dataset's group
    (padding with NaN). ``merge_chunks_to_datasets`` must project back to the
    stable per-dataset schema or the outputs accumulate 49-column unions with
    100% NULL values in unrelated columns (e.g. OHLC on structure_events).
    """

    def _frames(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        features = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "timeframe": "M15",
                "session": "asia", "open": 1.10, "high": 1.11, "low": 1.09,
                "close": 1.105, "atr": 0.001, "rsi": 55.0, "return_1": 0.001,
                "volume": 100.0,
            }]
        )
        structure = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "timeframe": "M15",
                "structure_type": "higher_high", "price": 1.11,
                "prior_price": 1.09, "available_from": ts,
            }]
        )
        sweeps = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "timeframe": "M15",
                "direction": "long", "sweep_type": "low_sweep", "level": 1.09,
                "extreme_price": 1.088, "close_price": 1.092, "zone_id": "z",
                "penetration": 0.002, "excursion": 0.004, "session": "asia",
                "regime": "trending", "htf_bias": "bullish", "available_from": ts,
            }]
        )
        displacement = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "timeframe": "M15",
                "direction": "up", "range_ratio": 2.0, "body_ratio": 0.8,
                "classification": "large", "available_from": ts,
            }]
        )
        regime = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "timeframe": "M15",
                "trend_state": "bullish", "volatility_state": "normal",
                "market_state": "trending", "strength": 0.7,
                "available_from": ts,
            }]
        )
        zones = pd.DataFrame(
            [{
                "zone_id": "EURUSD_M15_0", "symbol": "EURUSD", "timeframe": "M15",
                "zone_type": "equal_highs", "upper": 1.11, "lower": 1.105,
                "mid": 1.1075, "swing_count": 2, "first_timestamp": ts,
                "last_timestamp": ts, "available_from": ts,
            }]
        )
        mtf = pd.DataFrame(
            [{
                "timestamp": ts, "symbol": "EURUSD", "base_timeframe": "15m",
                "htf_timeframe": "1h", "htf_tier": "1h",
                "candle_open": "2024-01-01T00:00:00+00:00",
                "candle_close": "2024-01-01T01:00:00+00:00",
                "htf_trend_state": "bullish", "htf_volatility_state": "normal",
                "htf_market_state": "trending", "htf_structural_bias": "bullish",
                "available_from": "2024-01-01T01:00:00+00:00",
            }]
        )
        return {
            "features": features,
            "structure_events": structure,
            "liquidity_zones": zones,
            "sweeps": sweeps,
            "displacement": displacement,
            "regime": regime,
            "mtf_context": mtf,
        }

    def test_merged_features_have_only_feature_schema_columns(self, tmp_path):
        am = Step13Artifacts(tmp_path, "EURUSD", "M15")
        frames = self._frames()
        am.write_chunk(0, frames)
        am.merge_chunks_to_datasets(list(frames), "hash", "cfg", "13.2.0")

        merged = read_parquet_if_valid(am.dataset_path("features"))
        assert merged is not None
        # The merged features frame must NOT contain structure/sweep/mtf
        # columns (the union-column corruption).
        assert set(merged.columns) == set(FEATURES_COLUMNS)
        assert merged["open"].notna().all()
        assert merged["close"].notna().all()

    def test_merged_structure_events_have_no_ohlc(self, tmp_path):
        am = Step13Artifacts(tmp_path, "EURUSD", "M15")
        am.write_chunk(0, self._frames())
        am.merge_chunks_to_datasets(list(self._frames()), "hash", "cfg", "13.2.0")

        merged = read_parquet_if_valid(am.dataset_path("structure_events"))
        assert merged is not None
        # The structure schema has NO open/high/low/close columns; the union
        # corruption added them as 100% NaN. Projection must remove them.
        assert set(merged.columns) == set(STRUCTURE_COLUMNS)
        assert "open" not in merged.columns
        assert "close" not in merged.columns

    def test_merged_mtf_context_no_timeframe_column(self, tmp_path):
        am = Step13Artifacts(tmp_path, "EURUSD", "M15")
        frames = self._frames()
        am.write_chunk(0, frames)
        am.merge_chunks_to_datasets(list(frames), "hash", "cfg", "13.2.0")

        merged = read_parquet_if_valid(am.dataset_path("mtf_context"))
        assert merged is not None
        # The MTF schema has NO "timeframe" column. The user observed a
        # corrupted "timeframe" (NaN) column — union-column corruption.
        assert set(merged.columns) == set(MTF_CONTEXT_COLUMNS)
        assert "timeframe" not in merged.columns
        # MTF state fields must be populated.
        assert merged["htf_trend_state"].notna().all()

    def test_liquidity_zones_projected(self, tmp_path):
        am = Step13Artifacts(tmp_path, "EURUSD", "M15")
        frames = self._frames()
        am.write_chunk(0, frames)
        am.merge_chunks_to_datasets(list(frames), "hash", "cfg", "13.2.0")

        merged = read_parquet_if_valid(am.dataset_path("liquidity_zones"))
        assert merged is not None
        assert set(merged.columns) == set(LIQUIDITY_ZONES_COLUMNS)
        # Zone-centric columns populated, no accidental timestamp/OHLC union.
        assert merged["upper"].notna().all()
        assert "timestamp" not in merged.columns


# ── 2. HTF partition naming ─────────────────────────────────────────────────

class TestHtfPartitionNaming:
    def test_native_tf_maps_to_research_partition(self):
        assert _partition_tf("1h") == "H1"
        assert _partition_tf("4h") == "H4"
        assert _partition_tf("1d") == "D1"
        assert _partition_tf("15m") == "M15"

    def test_unknown_tf_passthrough(self):
        assert _partition_tf("custom") == "custom"


# ── 3. Candidate direction normalization ────────────────────────────────────

class TestDirectionNormalization:
    def test_direction_key_maps_up_long_down_short(self):
        assert _direction_key("up") == "long"
        assert _direction_key("down") == "short"
        assert _direction_key("long") == "long"
        assert _direction_key("short") == "short"

    def test_find_displacement_matches_cross_namespace(self):
        """A long sweep (low swept) must confirm on an UP displacement."""
        sweeps = [{
            "timestamp": "2024-01-01T10:00:00+00:00", "direction": "long",
            "close_price": 1.092, "available_from": "2024-01-01T10:15:00+00:00",
        }]
        displacements = [
            {
                "timestamp": "2024-01-01T10:15:00+00:00", "direction": "up",
                "classification": "large", "range_ratio": 2.0,
                "available_from": "2024-01-01T10:15:00+00:00",
            }
        ]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=5)
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
        assert cands[0]["direction"] == "long"
        # entry_ref must be a PRICE (float), not a datetime string.
        assert isinstance(cands[0]["entry_ref"], float)
        assert cands[0]["entry_ref"] == pytest.approx(1.092)
        # refs must be ISO strings (stable identity), never datetimes.
        assert isinstance(cands[0]["sweep_ref"], str)
        assert isinstance(cands[0]["displacement_ref"], str)
        assert isinstance(cands[0]["available_from"], str)

    def test_short_sweep_confirms_on_down_displacement(self):
        sweeps = [{
            "timestamp": "2024-01-01T10:00:00+00:00", "direction": "short",
            "close_price": 1.092, "available_from": "2024-01-01T10:15:00+00:00",
        }]
        displacements = [
            {
                "timestamp": "2024-01-01T10:15:00+00:00", "direction": "down",
                "classification": "large", "range_ratio": 2.0,
                "available_from": "2024-01-01T10:15:00+00:00",
            }
        ]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=5)
        cands = gen.generate(
            sweeps=sweeps, displacements=displacements,
            regimes=[{
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending", "volatility_state": "normal",
            }],
            features=[{
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 45.0, "session": "europe",
            }],
        )
        assert len(cands) == 1
        assert cands[0]["direction"] == "short"

    def test_wrong_direction_displacement_rejected(self):
        """A short sweep must NOT match an UP displacement."""
        sweeps = [{
            "timestamp": "2024-01-01T10:00:00+00:00", "direction": "short",
            "close_price": 1.092, "available_from": "2024-01-01T10:15:00+00:00",
        }]
        displacements = [
            {
                "timestamp": "2024-01-01T10:15:00+00:00", "direction": "up",
                "classification": "large", "range_ratio": 2.0,
                "available_from": "2024-01-01T10:15:00+00:00",
            }
        ]
        gen = CandidateGenerator("EURUSD", "M15", sweep_displacement_lookback=5)
        cands = gen.generate(
            sweeps=sweeps, displacements=displacements,
            regimes=[{
                "timestamp": "2024-01-01T09:00:00+00:00",
                "market_state": "trending", "volatility_state": "normal",
            }],
            features=[{
                "timestamp": "2024-01-01T09:45:00+00:00",
                "atr": 0.01, "rsi": 45.0, "session": "europe",
            }],
        )
        assert len(cands) == 0


# ── 4. Emission-range dedup ─────────────────────────────────────────────────

class TestEmissionDedup:
    def test_chunks_have_two_sided_ownership_windows(self):
        # Chunk structure: (index, warm_start, end). Overlaps are warm-up only.
        chunks = _make_chunks(10000, 5000, 200)
        assert chunks == [(0, 0, 5000), (1, 4800, 10000)]
        # Chunk 0 owns [0, 5000); chunk 1 owns [5000, 10000) — no overlap.
        assert chunks[0][2] == chunks[1][1] + 200  # 5000 == 4800 + 200

    def test_features_row_count_equals_bar_count(self):
        """Run extraction on a small frame with overlap; the emitted features
        must equal the full bar count — not bar_count + overlap duplication."""
        from app.research.step13.extract import EventExtractor, extract_rows_to_frame
        from app.research.step13.schema import FEATURES_COLUMNS

        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0.0005, 0.005, n))
        df = pd.DataFrame(
            {"open": close, "high": close + 0.05, "low": close - 0.05, "close": close},
            index=idx,
        )
        chunk_size, overlap = 200, 50
        chunks = _make_chunks(n, chunk_size, overlap)
        ext = EventExtractor("EURUSD", "M15")
        frames = []
        for ci, warm_start, end in chunks:
            rows = ext.extract(df.iloc[warm_start:end])
            fdf = extract_rows_to_frame(rows["features"], FEATURES_COLUMNS)
            # Apply the same ownership filter the runner applies.
            nominal_start = (warm_start + overlap) if ci > 0 else 0
            lower = df.index[min(nominal_start, len(df) - 1)]
            upper = df.index[end] if end < n else None
            ts_vals = pd.to_datetime(fdf["timestamp"])
            if upper is None:
                fdf = fdf[ts_vals >= lower]
            else:
                fdf = fdf[(ts_vals >= lower) & (ts_vals < upper)]
            frames.append(fdf)

        merged = pd.concat(frames, ignore_index=True)
        assert len(merged) == n, (
            f"emission dedup must output exactly {n} feature rows, got {len(merged)}"
        )
        assert merged["timestamp"].is_unique
