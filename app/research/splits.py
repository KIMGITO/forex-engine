"""Chronological TRAIN / VALIDATION / TEST splitting.

Financial time series are NEVER randomly shuffled. Splits are strictly
chronological, and the TEST period is never touched during optimization.
"""

from datetime import datetime, timedelta

import pandas as pd

from app.research.config import ResearchConfig
from app.research.errors import SplitConfigurationError
from app.research.models import TimeSplit

__all__ = ["is_leak_free", "make_time_split", "split_frame"]


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def make_time_split(
    data_start: datetime,
    data_end: datetime,
    config: ResearchConfig,
) -> TimeSplit:
    """Build a chronological split.

    Prefers explicit dates from config; falls back to fractional split when
    the dates are absent. The TEST period is the LAST (most recent) block.
    """
    # Explicit dates path.
    if config.test_start and config.test_end:
        train_start = _parse_dt(config.train_start) or data_start
        train_end = _parse_dt(config.train_end)
        val_start = _parse_dt(config.validation_start)
        val_end = _parse_dt(config.validation_end)
        test_start = _parse_dt(config.test_start)
        test_end = _parse_dt(config.test_end)

        if train_start is None or test_start is None or test_end is None:
            raise SplitConfigurationError("explicit dates require at least train_start/test_start/test_end")
        if train_end is None:
            train_end = val_start if val_start is not None else test_start
        if val_start is None or val_end is None:
            val_start = train_end
            val_end = test_start

        if not (train_start < train_end <= val_start < val_end <= test_start < test_end):
            raise SplitConfigurationError("chronological order violated in explicit split dates")
        return TimeSplit(
            train_start=train_start, train_end=train_end,
            validation_start=val_start, validation_end=val_end,
            test_start=test_start, test_end=test_end,
        )

    # Fractional path over the full data range.
    total = (data_end - data_start).total_seconds()
    if total <= 0:
        raise SplitConfigurationError("data range must be positive")
    if not (0 < config.train_fraction < 1) or not (0 < config.validation_fraction < 1 - config.train_fraction):
        raise SplitConfigurationError("train_fraction/validation_fraction invalid")

    train_seconds = total * config.train_fraction
    val_seconds = total * config.validation_fraction

    train_end = data_start + timedelta(seconds=train_seconds)
    val_start = train_end
    val_end = val_start + timedelta(seconds=val_seconds)
    test_start = val_end

    return TimeSplit(
        train_start=data_start, train_end=train_end,
        validation_start=val_start, validation_end=val_end,
        test_start=test_start, test_end=data_end,
    )


def split_frame(
    frame: pd.DataFrame,
    split: TimeSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into (train, validation, test) chronologically.

    The frame is sliced by timestamp boundaries (inclusive start, exclusive end
    semantics via < on end boundaries to avoid leakage).
    """
    def _slice(start, end, end_inclusive=False) -> pd.DataFrame:
        # Inclusive start; exclusive end normally (avoids boundary leakage
        # between train/validation/test). The final (test) block includes the
        # last data bar via end_inclusive.
        if end_inclusive:
            mask = (frame.index >= start) & (frame.index <= end)
        else:
            mask = (frame.index >= start) & (frame.index < end)
        return frame[mask]

    return (
        _slice(split.train_start, split.train_end, end_inclusive=False),
        _slice(split.validation_start, split.validation_end, end_inclusive=False),
        _slice(split.test_start, split.test_end, end_inclusive=True),
    )


def is_leak_free(train_end: datetime, validation_start: datetime, test_start: datetime) -> bool:
    """Verify chronological ordering (no overlap/leak)."""
    return train_end <= validation_start <= test_start