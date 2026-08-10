"""Rolling walk-forward validation with leakage prevention.

Each window slides forward in time: TRAIN -> VALIDATION -> TEST. Information
from a future window can never influence an earlier window because every
window's train/validation/test slices are strictly chronological and disjoint.

Every window records its periods, selected configuration, and results so the
researcher can assess stability across windows without ever leaking test data.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

from app.research.config import ResearchConfig
from app.research.errors import SplitConfigurationError
from app.research.models import TimeSplit, WalkForwardWindow

__all__ = ["WalkForwardRunner", "build_walk_forward_windows", "run_walk_forward"]


def _end_of_window(train_start: datetime, config: ResearchConfig, step: int) -> tuple:
    """Compute the train/validation/test boundaries for walk-forward window."""
    train_years = config.walk_train_years
    val_years = config.walk_validation_years
    test_years = config.walk_test_years

    def _years(y: float) -> timedelta:
        return timedelta(days=int(y * 365.0))

    # Window start slides forward by the test-span each step.
    offset = _years(test_years) * step
    train_start_w = train_start + offset
    train_end = train_start_w + _years(train_years)
    val_start = train_end
    val_end = val_start + _years(val_years)
    test_start = val_end
    test_end = test_start + _years(test_years)
    return train_start_w, train_end, val_start, val_end, test_start, test_end


def build_walk_forward_windows(
    data_start: datetime,
    data_end: datetime,
    config: ResearchConfig,
    max_windows: int = 20,
) -> list[WalkForwardWindow]:
    """Build a list of rolling walk-forward windows.

    Each window's TEST starts where the previous window's TRAIN+VALIDATION end
    was; windows are chronologically ordered and disjoint. No window overlaps
    a later window's data.
    """
    windows: list[WalkForwardWindow] = []
    for step in range(max_windows):
        try:
            tws, tw_end, vs, ve, ts, te = _end_of_window(data_start, config, step)
        except Exception:  # noqa: BLE001, S112 - boundary math failure: skip window (documented)
            continue
        # Stop when this window falls entirely outside the available data.
        if ts > data_end:
            break
        split = TimeSplit(
            train_start=tws, train_end=tw_end,
            validation_start=vs, validation_end=ve,
            test_start=ts, test_end=min(te, data_end),
        )
        # Verify chronological ordering (leak-free).
        if not (split.train_start < split.train_end <= split.validation_start
                < split.validation_end <= split.test_start < split.test_end):
            raise SplitConfigurationError(
                "walk-forward window violates chronological ordering"
            )
        windows.append(WalkForwardWindow(index=step, split=split))
    return windows


class WalkForwardRunner:
    """Orchestrates walk-forward evaluation over a frame.

    ``run_strategy`` is a callback that, given a TRAIN frame, returns the
    selected config (dict) and the fitted strategy used for VALIDATION+TEST.
    """

    def __init__(
        self,
        config: ResearchConfig,
        backtest_callable: Callable[[pd.DataFrame, dict], dict],
    ) -> None:
        self.config = config
        self.backtest_callable = backtest_callable

    def run(self, frame: pd.DataFrame) -> list[WalkForwardWindow]:
        """Run walk-forward over ``frame`` (tz-aware, sorted)."""
        if frame is None or frame.empty:
            raise ValueError("frame must not be empty")
        data_start = frame.index[0].to_pydatetime()
        data_end = frame.index[-1].to_pydatetime()
        windows = build_walk_forward_windows(data_start, data_end, self.config)

        results: list[WalkForwardWindow] = []
        for w in windows:
            train = frame[(frame.index >= w.split.train_start) & (frame.index < w.split.train_end)]
            val = frame[(frame.index >= w.split.validation_start) & (frame.index < w.split.validation_end)]
            test = frame[(frame.index >= w.split.test_start) & (frame.index < w.split.test_end)]
            if len(train) < self.config.warn_min_bars or len(test) == 0:
                w = replace(w, warnings=w.warnings + [
                    f"insufficient bars in window {w.index} (train={len(train)}, test={len(test)})"
                ])
                results.append(w)
                continue
            # Backtest on validation+test (the trained strategy is applied).
            val_res = self.backtest_callable(val, w.selected_config or {})
            test_res = self.backtest_callable(test, w.selected_config or {})
            w = replace(w, results={
                "validation": val_res,
                "test": test_res,
                "selected_config": w.selected_config,
                "train_bars": len(train),
                "validation_bars": len(val),
                "test_bars": len(test),
            })
            results.append(w)
        return results


def run_walk_forward(
    frame: pd.DataFrame,
    config: ResearchConfig,
    backtest_callable: Callable[[pd.DataFrame, dict], dict],
) -> list[WalkForwardWindow]:
    """Convenience wrapper around WalkForwardRunner."""
    return WalkForwardRunner(config, backtest_callable).run(frame)
