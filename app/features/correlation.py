"""Correlation features between multiple instruments.

Supports pairwise correlation, rolling correlation, and correlation matrices.
All calculations align timestamps before computing — they do not assume that
two instruments have identical observation times.
"""


import numpy as np
import pandas as pd

from app.features.errors import FeatureError, InsufficientDataError

__all__ = ["align_price_dfs", "correlation_matrix", "pairwise_correlation", "rolling_correlation"]


def align_price_dfs(
    data_map: dict[str, pd.DataFrame],
    price_col: str = "close",
) -> pd.DataFrame:
    """Align multiple instrument DataFrames to a common timestamp index.

    Parameters
    ----------
    data_map : dict of str -> DataFrame
        A mapping of instrument names to their price DataFrames. Each DataFrame
        must contain ``price_col`` and will be sorted by index.
    price_col : str
        The column name to extract from each DataFrame.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with one column per instrument, indexed by the union
        of all timestamps. Missing observations are filled with ``NaN``.
    """
    if not data_map:
        raise ValueError("At least one instrument is required for alignment.")

    aligned: pd.DataFrame = pd.DataFrame(index=pd.Index([]))
    for name, df in data_map.items():
        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in instrument '{name}'.")
        series = df[price_col].sort_index()
        s = series.copy()
        s.name = name
        aligned = aligned.join(s, how="outer")

    # For multiple instruments, require at least one timestamp where two or
    # more instruments have data (a single instrument is always valid).
    if len(aligned.columns) >= 2:
        overlap_count = aligned.notna().sum(axis=1)
        if (overlap_count >= 2).sum() == 0:
            raise FeatureError("No overlapping timestamps found across instruments.")

    return aligned


def pairwise_correlation(
    aligned: pd.DataFrame,
    instrument_a: str,
    instrument_b: str,
    min_periods: int = 20,
) -> float:
    """Pearson correlation between two instruments' aligned price series.

    Parameters
    ----------
    aligned : DataFrame
        Already aligned by :func:`align_price_dfs`.
    instrument_a, instrument_b : str
        Column names in ``aligned``.
    min_periods : int
        Minimum number of overlapping observations required.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    if instrument_a not in aligned.columns:
        raise ValueError(f"Instrument '{instrument_a}' not found in aligned data.")
    if instrument_b not in aligned.columns:
        raise ValueError(f"Instrument '{instrument_b}' not found in aligned data.")

    # A series is perfectly correlated with itself.
    if instrument_a == instrument_b:
        return 1.0

    # Use only overlapping, non-NaN pairs
    combined = aligned[[instrument_a, instrument_b]].dropna()
    if len(combined) < min_periods:
        raise InsufficientDataError(
            f"Need at least {min_periods} overlapping observations for pairwise correlation; "
            f"got {len(combined)}."
        )
    return float(combined[instrument_a].corr(combined[instrument_b]))


def rolling_correlation(
    aligned: pd.DataFrame,
    instrument_a: str,
    instrument_b: str,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling Pearson correlation between two instruments.

    Parameters
    ----------
    aligned : DataFrame
        Already aligned by :func:`align_price_dfs`.
    instrument_a, instrument_b : str
        Column names in ``aligned``.
    window : int
        Trailing window size.
    min_periods : int, optional
        Minimum overlapping observations required in the window. If ``None``,
        uses ``window``.

    Returns
    -------
    pd.Series
        Rolling correlation indexed by timestamp.
    """
    if instrument_a not in aligned.columns:
        raise ValueError(f"Instrument '{instrument_a}' not found in aligned data.")
    if instrument_b not in aligned.columns:
        raise ValueError(f"Instrument '{instrument_b}' not found in aligned data.")

    mp = min_periods or window
    return (
        aligned[instrument_a]
        .rolling(window=window, min_periods=mp)
        .corr(aligned[instrument_b])
    )


def correlation_matrix(
    aligned: pd.DataFrame,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Correlation matrix for all instruments in the aligned data.

    Parameters
    ----------
    aligned : DataFrame
        Already aligned by :func:`align_price_dfs`.
    min_periods : int
        Minimum overlapping observations required for each pair.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix (NaN if insufficient observations for a pair).
    """
    if aligned.empty:
        raise FeatureError("Cannot compute correlation matrix from empty data.")

    result = pd.DataFrame(index=aligned.columns, columns=aligned.columns, dtype=float)
    for a in aligned.columns:
        for b in aligned.columns:
            if a == b:
                result.loc[a, b] = 1.0
                continue
            try:
                result.loc[a, b] = pairwise_correlation(
                    aligned, a, b, min_periods=min_periods
                )
            except InsufficientDataError:
                result.loc[a, b] = np.nan
    return result
