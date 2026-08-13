"""Stable schemas for Step 13 columnar event datasets.

FEATURE vs LABEL SEPARATION is STRUCTURAL:

* feature_* — information known at candidate timestamp (causal)
* label_*   — information observed AFTER candidate timestamp

A SchemaValidator rejects any candidate frame whose columns mix feature_*
and label_* namespaces incorrectly (a label may ONLY exist in the label
dataset, never as a feature).
"""

from __future__ import annotations

import pandas as pd

# ── Support-event schemas (compact columnar rows) ────────────────────────────

FEATURES_COLUMNS = [
    "timestamp", "symbol", "timeframe", "session",
    "open", "high", "low", "close",
    "atr", "rsi", "return_1", "volume",
]

STRUCTURE_COLUMNS = [
    "timestamp", "symbol", "timeframe",
    "structure_type", "price", "prior_price", "available_from",
]

LIQUIDITY_ZONES_COLUMNS = [
    "zone_id", "symbol", "timeframe", "zone_type",
    "upper", "lower", "mid", "swing_count",
    "first_timestamp", "last_timestamp", "available_from",
]

SWEEPS_COLUMNS = [
    "timestamp", "symbol", "timeframe",
    "direction", "sweep_type", "level",
    "extreme_price", "close_price", "zone_id",
    "penetration", "excursion", "session",
    "regime", "htf_bias", "available_from",
]

DISPLACEMENT_COLUMNS = [
    "timestamp", "symbol", "timeframe",
    "direction", "range_ratio", "body_ratio",
    "classification", "available_from",
]

REGIME_COLUMNS = [
    "timestamp", "symbol", "timeframe",
    "trend_state", "volatility_state", "market_state",
    "strength", "available_from",
]

MTF_CONTEXT_COLUMNS = [
    "timestamp", "symbol", "base_timeframe",
    "htf_timeframe", "htf_tier",
    "candle_open", "candle_close",
    "htf_trend_state", "htf_volatility_state", "htf_market_state",
    "htf_structural_bias", "available_from",
]

# ── Candidate schema ─────────────────────────────────────────────────────────

# Fixed reference columns (NOT feature_* but causal identifiers).
CANDIDATE_ID_COLUMNS = [
    "candidate_id", "timestamp", "symbol", "timeframe",
    "direction", "entry_ref", "sweep_ref", "displacement_ref",
    "structure_ref", "htf_ref", "regime", "session", "available_from",
]

# Feature columns observed at candidate timestamp (causal).
CANDIDATE_FEATURE_COLUMNS = [
    "feature_atr",
    "feature_rsi",
    "feature_volatility",
    "feature_structure_bias",
    "feature_sweep_penetration",
    "feature_sweep_excursion",
    "feature_displacement_ratio",
    "feature_htf_alignment",
    "feature_htf_trend",
    "feature_htf_volatility",
    "feature_session",
]

# Label columns observed AFTER candidate timestamp.
CANDIDATE_LABEL_COLUMNS = [
    "label_mfe",
    "label_mae",
    "label_tp_hit",
    "label_sl_hit",
    "label_excursion_after_bars",
    "label_future_return",
    "label_displacement_after",
]

# Full candidate event frame columns (features only, no labels).
CANDIDATE_EVENTS_COLUMNS = CANDIDATE_ID_COLUMNS + CANDIDATE_FEATURE_COLUMNS

# Full candidate label frame columns.
CANDIDATE_LABELS_COLUMNS = ["candidate_id", "timestamp"] + CANDIDATE_LABEL_COLUMNS

# List of event dataset names produced by the pipeline.
EVENT_DATASETS = [
    "features",
    "structure_events",
    "liquidity_zones",
    "sweeps",
    "displacement",
    "regime",
    "mtf_context",
]

CANDIDATE_DATASETS = ["candidate_events", "candidate_labels"]


class SchemaError(ValueError):
    """Raised when a frame violates the stable schema contract."""


def validate_feature_label_separation(df: pd.DataFrame) -> None:
    """Reject any frame that mixes feature_* and label_* namespaces.

    A candidate event frame may ONLY contain feature_* columns (plus the fixed
    reference/id columns). A label frame may ONLY contain label_* columns.
    A single frame that mixes both is a structural look-ahead violation.
    """
    if df is None or df.empty:
        return
    cols = set(df.columns)
    has_features = any(c.startswith("feature_") for c in cols)
    has_labels = any(c.startswith("label_") for c in cols)
    if has_features and has_labels:
        raise SchemaError(
            "Feature/label separation violated: a single frame mixes "
            "feature_* and label_* columns. Labels must live ONLY in the "
            "candidate_labels dataset, never in candidate features."
        )


def validate_candidate_events(df: pd.DataFrame) -> None:
    """Validate a candidate_events frame."""
    validate_feature_label_separation(df)
    required = {"candidate_id", "timestamp", "symbol", "timeframe", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"candidate_events missing required columns: {missing}")
    # Candidate timestamp must be tz-aware.
    if "timestamp" in df.columns and not df.empty:
        try:
            ts = pd.to_datetime(df["timestamp"])
            if getattr(ts.dt.tz, "utcoffset", None) is None and ts.dt.tz is None:
                raise SchemaError("candidate_events timestamp must be tz-aware")
        except Exception as exc:  # noqa: BLE001
            raise SchemaError(f"candidate_events timestamp invalid: {exc}")


def validate_candidate_labels(df: pd.DataFrame) -> None:
    """Validate a candidate_labels frame."""
    if df is None or df.empty:
        return
    required = {"candidate_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"candidate_labels missing required columns: {missing}")
    # Label frame must NOT contain feature_* columns.
    if any(c.startswith("feature_") for c in df.columns):
        raise SchemaError(
            "candidate_labels must not contain feature_* columns"
        )


def event_schema_for(name: str) -> list[str]:
    """Return the expected columns for a named event dataset."""
    schemas = {
        "features": FEATURES_COLUMNS,
        "structure_events": STRUCTURE_COLUMNS,
        "liquidity_zones": LIQUIDITY_ZONES_COLUMNS,
        "sweeps": SWEEPS_COLUMNS,
        "displacement": DISPLACEMENT_COLUMNS,
        "regime": REGIME_COLUMNS,
        "mtf_context": MTF_CONTEXT_COLUMNS,
        "candidate_events": CANDIDATE_EVENTS_COLUMNS,
        "candidate_labels": CANDIDATE_LABELS_COLUMNS,
    }
    return list(schemas.get(name, []))