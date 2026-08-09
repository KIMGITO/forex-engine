"""Look-ahead-bias protection utilities.

Principle
---------
A feature computed at timestamp ``T`` must use only information available at or
before ``T``. Concretely:

* All feature calculations must be causal (`backward-looking`), never centered.
* Return series must be computed with a one-period lag (``shift(1)``) so that a
  return classified at ``T`` reflects only prior price information.
* Rolling/expanding windows must be trailing and right-aligned.
* Future candles must never influence a past feature value.

Please do not violate this when adding new feature modules. If you are unsure
whether a calculation is causal, verify it with the regression test provided in
:mod:`tests.features.test_lookahead`.
"""


import pandas as pd

__all__ = ["suffix_ratio", "truncate_prefix"]


def truncate_prefix(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the first ``n`` rows of ``df`` (keeps timestamps intact).

    Used by the look-ahead regression test to assert that dropping future data
    does not change feature values for earlier timestamps.
    """
    return df.iloc[:n]


def suffix_ratio(min_len: int, periods: int, max_rows: int) -> int:
    """Compute a safe trailing-window suffix length for alignment logic.

    Returns ``min_len - periods`` (the number of full lookback windows
    available), capped at ``max_rows`` and never below zero.
    """
    return max(0, min(min_len - periods, max_rows))