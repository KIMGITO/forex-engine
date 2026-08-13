"""Step 13 → Step 13B contract adapter.

Step 13 publishes compact columnar event/candidate datasets under
``research/results/step13/<SYMBOL>/<TIMEFRAME>/``. This adapter loads them
for Step 13B WITHOUT duplicating any detection logic. Step 13B can consume
candidates via ``load_candidate_events`` and join ``candidate_labels``.

The adapter never rewrites Step 13B core — it only provides a clean loading
surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.research.step13.persist import read_parquet_if_valid, read_json_if_valid
from app.research.step13.schema import (
    CANDIDATE_EVENTS_COLUMNS,
    CANDIDATE_LABELS_COLUMNS,
)


class Step13DataNotFound(FileNotFoundError):
    """Raised when Step 13 candidate artifacts are missing."""


def load_candidate_events(
    step13_output: str | Path,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
) -> pd.DataFrame:
    """Load the Step 13 candidate_events dataset for a symbol/timeframe.

    Parameters
    ----------
    step13_output : path to the Step 13 output root
    symbol, timeframe : which partition to load

    Returns a compact DataFrame with the stable candidate schema
    (candidate_id + feature_* columns). Raises ``Step13DataNotFound`` when
    the artifact is absent.
    """
    root = Path(step13_output)
    path = root / symbol.upper() / timeframe.upper() / "candidate_events.parquet"
    df = read_parquet_if_valid(path)
    if df is None:
        raise Step13DataNotFound(
            f"Step 13 candidate_events missing at {path}.\n"
            f"Run the Step 13 pipeline first."
        )
    # Ensure stable column ordering.
    for c in CANDIDATE_EVENTS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_EVENTS_COLUMNS]


def load_candidate_labels(
    step13_output: str | Path,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
) -> pd.DataFrame:
    """Load the Step 13 candidate_labels dataset for a symbol/timeframe."""
    root = Path(step13_output)
    path = root / symbol.upper() / timeframe.upper() / "candidate_labels.parquet"
    df = read_parquet_if_valid(path)
    if df is None:
        raise Step13DataNotFound(
            f"Step 13 candidate_labels missing at {path}.\n"
            f"Run the Step 13 pipeline first."
        )
    for c in CANDIDATE_LABELS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CANDIDATE_LABELS_COLUMNS]


def load_event_dataset(
    step13_output: str | Path,
    dataset: str,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
) -> pd.DataFrame:
    """Load any Step 13 event dataset (features, sweeps, displacement, ...)."""
    root = Path(step13_output)
    path = root / symbol.upper() / timeframe.upper() / f"{dataset}.parquet"
    df = read_parquet_if_valid(path)
    if df is None:
        raise Step13DataNotFound(f"Step 13 dataset {dataset} missing at {path}.")
    return df


def manifest_for(
    step13_output: str | Path,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
) -> dict[str, Any] | None:
    """Load the Step 13 manifest for a partition."""
    root = Path(step13_output)
    path = root / symbol.upper() / timeframe.upper() / "manifest.json"
    return read_json_if_valid(path)