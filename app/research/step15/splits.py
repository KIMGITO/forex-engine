"""Strict temporal splits and walk-forward windows for Step 15.

Guarantees:
* TRAIN -> VALIDATION -> TEST are strictly chronological and disjoint.
* No random shuffle anywhere.
* Fold windows advance in time (rolling walk-forward).
* The purge/embargo policy excludes training candidates whose label horizon
  crosses the train/validation boundary (preferred: EXCLUDE, not leak).
* Adaptive candidate-count splits are supported for bursty candidate data;
  they remain strictly chronological.
"""

from __future__ import annotations

import pandas as pd

from app.research.step15.config import Step15Config
from app.research.step15.models import TemporalSplit


def make_single_split(
    data_start,
    data_end,
    config: Step15Config,
    *,
    timestamps: pd.Series | None = None,
    adaptive: bool = False,
) -> TemporalSplit:
    """Build the canonical TRAIN -> VALIDATION -> TEST split.

    Calendar-based (default): train_days / validation_days / test_days.

    Adaptive mode (``timestamps`` + ``adaptive=True``): uses candidate-count
    percentiles so each phase contains a meaningful sample. Strictly
    chronological — the quantile boundaries land between bars, never reorder
    rows. Required when candidate data is bursty and pure calendar windows
    would land in empty periods.
    """
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)

    if adaptive and timestamps is not None and len(timestamps) > 0:
        ts = pd.to_datetime(timestamps, utc=True).sort_values()
        n = len(ts)
        total = config.train_days + config.validation_days + config.test_days
        train_frac = config.train_days / total
        val_frac = config.validation_days / total
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train_end = ts.iloc[n_train - 1]
        val_start = ts.iloc[min(n_train, n - 1)]
        val_end = ts.iloc[min(n_train + n_val - 1, n - 1)]
        test_start = ts.iloc[min(n_train + n_val, n - 1)]
        test_end = end
        split = TemporalSplit(
            train_start=start,
            train_end=train_end,
            validation_start=val_start,
            validation_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        if not split.is_chronological():
            raise ValueError(
                f"adaptive temporal split violates chronology: {split.to_dict()}"
            )
        return split

    train_end = start + pd.Timedelta(days=config.train_days)
    val_start = train_end
    val_end = val_start + pd.Timedelta(days=config.validation_days)
    test_start = val_end
    test_end = test_start + pd.Timedelta(days=config.test_days)
    split = TemporalSplit(
        train_start=start,
        train_end=train_end,
        validation_start=val_start,
        validation_end=val_end,
        test_start=test_start,
        test_end=min(test_end, end),
    )
    if not split.is_chronological():
        raise ValueError(
            f"temporal split violates chronology: {split.to_dict()}"
        )
    return split


def build_walk_forward_splits(
    data_start,
    data_end,
    config: Step15Config,
) -> list[TemporalSplit]:
    """Build rolling walk-forward splits.

    Each window:
        TRAIN  -> VALIDATION -> TEST
    advances by ``wf_step_days``. Windows are chronologically ordered and
    disjoint (no overlapping test periods across the fold sequence).

    A window is only included when its TEST period starts at-or-before the
    data end.
    """
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)
    out: list[TemporalSplit] = []
    for idx in range(config.max_windows):
        offset = pd.Timedelta(days=config.wf_step_days * idx)
        w_train_start = start + offset
        w_train_end = w_train_start + pd.Timedelta(days=config.wf_train_days)
        w_val_start = w_train_end
        w_val_end = w_val_start + pd.Timedelta(days=config.wf_validation_days)
        w_test_start = w_val_end
        w_test_end = w_test_start + pd.Timedelta(days=config.wf_test_days)

        # Stop when the TEST period starts after the data end.
        if w_test_start >= end:
            break

        split = TemporalSplit(
            train_start=w_train_start,
            train_end=w_train_end,
            validation_start=w_val_start,
            validation_end=w_val_end,
            test_start=w_test_start,
            test_end=min(w_test_end, end),
        )
        if not split.is_chronological():
            raise ValueError(
                f"walk-forward split violates chronology at index {idx}: "
                f"{split.to_dict()}"
            )
        out.append(split)
    return out


# ── Partitioning helpers ─────────────────────────────────────────────────────


def slice_frame(
    frame: pd.DataFrame,
    start,
    end,
    *,
    end_inclusive: bool = False,
) -> pd.DataFrame:
    """Slice a DataFrame by timestamp bounds.

    Inclusive start; exclusive end normally (prevents boundary leakage). The
    final TEST block uses end_inclusive=True to include the last data bar.
    """
    if end_inclusive:
        return frame[(frame.index >= start) & (frame.index <= end)]
    return frame[(frame.index >= start) & (frame.index < end)]


def split_frame_by(
    frame: pd.DataFrame,
    split: TemporalSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a frame into (train, validation, test) chronologically."""
    train = slice_frame(frame, split.train_start, split.train_end)
    val = slice_frame(frame, split.validation_start, split.validation_end)
    test = slice_frame(frame, split.test_start, split.test_end, end_inclusive=True)
    return train, val, test


def partition_candidates(
    events: pd.DataFrame,
    labels: pd.DataFrame | None,
    split: TemporalSplit,
    *,
    purge_horizon_bars: int = 200,
    bar_minutes: int = 15,
    purge_enabled: bool = True,
) -> dict[str, pd.DataFrame]:
    """Partition candidate events + labels into train/val/test with PURGE.

    Purge policy (documented):
    A training candidate whose LABEL HORIZON (``timestamp`` +
    ``purge_horizon_bars`` bars of ``bar_minutes``) crosses into the
    VALIDATION period is EXCLUDED from the training set. This prevents future
    test outcomes from influencing training discovery/selection. The
    validation/test partitions never purge (they are evaluated on real labels
    whose horizon naturally reaches forward).

    Returns dict with keys: train_events, val_events, test_events,
    train_labels, val_labels, test_labels, purged_from_train (count),
    purge_boundary (iso timestamp).
    """
    if events is None or events.empty:
        return {
            "train_events": pd.DataFrame(),
            "val_events": pd.DataFrame(),
            "test_events": pd.DataFrame(),
            "train_labels": pd.DataFrame(),
            "val_labels": pd.DataFrame(),
            "test_labels": pd.DataFrame(),
            "purged_from_train": 0,
            "purge_boundary": None,
        }

    ts = pd.to_datetime(events["timestamp"], utc=True)

    # For the final TEST block, the label horizon reaches past the data end;
    # that is fine (labels were computed from the full frame). Only the
    # TRAIN partition gets purged.
    train_mask = (ts >= split.train_start) & (ts < split.train_end)
    val_mask = (ts >= split.validation_start) & (ts < split.validation_end)
    test_mask = (ts >= split.test_start) & (ts <= split.test_end)

    train_events = events[train_mask].copy()
    val_events = events[val_mask].copy()
    test_events = events[test_mask].copy()

    purged = 0
    purge_boundary: str | None = None
    if purge_enabled and not train_events.empty:
        # A training candidate is purged when its label horizon crosses into
        # the validation period.
        train_ts = pd.to_datetime(train_events["timestamp"], utc=True)
        horizon = pd.Timedelta(minutes=purge_horizon_bars * bar_minutes)
        label_end = train_ts + horizon
        crossing = label_end >= split.validation_start
        purged = int(crossing.sum())
        if purged > 0:
            train_events = train_events[~crossing]
        purge_boundary = split.validation_start.isoformat()

    def _labels_for(events_sub: pd.DataFrame) -> pd.DataFrame:
        if labels is None or labels.empty or events_sub.empty:
            return pd.DataFrame()
        ids = set(events_sub["candidate_id"])
        lab = labels[labels["candidate_id"].isin(ids)]
        return lab.copy()

    return {
        "train_events": train_events,
        "val_events": val_events,
        "test_events": test_events,
        "train_labels": _labels_for(train_events),
        "val_labels": _labels_for(val_events),
        "test_labels": _labels_for(test_events),
        "purged_from_train": purged,
        "purge_boundary": purge_boundary,
    }