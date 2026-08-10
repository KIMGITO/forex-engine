"""Lightweight research-data validation.

Validates each symbol/timeframe partition structurally (timestamps, OHLC,
ordering, timezone) and produces a machine-readable report. A research run
fails on malformed/duplicate timestamps, invalid OHLC, non-chronological
ordering, or invalid timezone. Legitimate market-session gaps are warnings.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from app.data.exceptions import ValidationError
from app.research.errors import ResearchError

__all__ = ["DataQualityReport", "ResearchDataValidator", "validate_partition"]

_TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


def _expected_minutes(timeframe: str) -> int:
    return _TIMEFRAME_MINUTES.get(str(timeframe).upper(), 60)


class DataQualityReport:
    """Structured validation output for one partition."""

    def __init__(self) -> None:
        self.symbol: str = ""
        self.timeframe: str = ""
        self.first_timestamp: str | None = None
        self.last_timestamp: str | None = None
        self.candle_count: int = 0
        self.expected_interval: int = 0
        self.gap_count: int = 0
        self.duplicate_count: int = 0
        self.invalid_ohlc_count: int = 0
        self.timezone_status: str = "ok"
        self.provider: str = "unknown"
        self.native_or_aggregated: str = "native"
        self.passed: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "candle_count": self.candle_count,
            "expected_interval": self.expected_interval,
            "gap_count": self.gap_count,
            "duplicate_count": self.duplicate_count,
            "invalid_ohlc_count": self.invalid_ohlc_count,
            "timezone_status": self.timezone_status,
            "provider": self.provider,
            "native_or_aggregated": self.native_or_aggregated,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ResearchDataValidator:
    """Validates a research partition DataFrame (tz-aware, OHLC indexed)."""

    def validate(self, df: pd.DataFrame | None, symbol: str, timeframe: str) -> DataQualityReport:
        report = DataQualityReport()
        report.symbol = symbol
        report.timeframe = timeframe
        report.expected_interval = _expected_minutes(timeframe)
        report.candle_count = len(df) if df is not None else 0

        if df is None or df.empty:
            report.passed = False
            report.errors.append("no data")
            return report

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            report.passed = False
            report.errors.append(f"missing OHLC columns: {sorted(missing)}")
            return report

        # Index / timestamp integrity.
        if not isinstance(df.index, pd.DatetimeIndex):
            report.passed = False
            report.errors.append("index is not a DatetimeIndex")
            return report
        if getattr(df.index, "tz", None) is None:
            report.timezone_status = "naive"
            report.errors.append("timestamps are not timezone-aware")
            report.passed = False

        if not df.index.is_monotonic_increasing:
            report.errors.append("data is not chronologically ordered")
            report.passed = False
        dups = int(df.index.duplicated().sum())
        report.duplicate_count = dups
        if dups > 0:
            report.errors.append(f"{dups} duplicate timestamps remain")
            report.passed = False

        # OHLC validity (vector check).
        try:
            self._validate_price_cols(df, report)
        except ValidationError as exc:
            report.errors.append(str(exc))
            report.passed = False

        # Gaps (legitimate market closures are warnings, not failures).
        report.gap_count = self._count_gaps(df, report.expected_interval)
        report.first_timestamp = _utc_iso(df.index[0])
        report.last_timestamp = _utc_iso(df.index[-1])
        return report

    @staticmethod
    def _validate_price_cols(df: pd.DataFrame, report: DataQualityReport) -> None:
        price_cols = ["open", "high", "low", "close"]
        if df[price_cols].isna().any().any():
            raise ValidationError("dataset contains missing (NaN) price values")
        if np.isinf(df[price_cols].to_numpy()).any():
            raise ValidationError("dataset contains infinite price values")
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        invalid = int((invalid_high | invalid_low).sum())
        report.invalid_ohlc_count = invalid
        if invalid > 0:
            raise ValidationError(f"{invalid} invalid OHLC relationship rows")

    @staticmethod
    def _count_gaps(df: pd.DataFrame, expected_minutes: int, tolerance_pct: float = 0.02) -> int:
        """Count timestamp gaps exceeding the expected interval.

        Uses an explicit pairwise loop (like the dataset repo's detect_gaps)
        and never treats legitimate market-session closures as validation
        failures — they are surfaced separately as warnings.
        """
        idx = list(df.index)
        if len(idx) < 2:
            return 0
        threshold_min = expected_minutes * (1.0 + tolerance_pct)
        gaps = 0
        for a, b in pairwise(idx):
            delta_min = (b - a).total_seconds() / 60.0
            if delta_min > threshold_min:
                gaps += 1
        return gaps


def _utc_iso(ts) -> str:
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        from datetime import timezone

        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def validate_partition(
    df: pd.DataFrame | None,
    symbol: str,
    timeframe: str,
    provider: str = "unknown",
    native_or_aggregated: str = "native",
) -> DataQualityReport:
    """Validate a loaded partition and annotate provenance.

    Raises :class:`ResearchError` when validation fails. Legitimate market
    session gaps are surfaced as warnings, not errors.
    """
    validator = ResearchDataValidator()
    report = validator.validate(df, symbol, timeframe)
    report.provider = provider
    report.native_or_aggregated = native_or_aggregated
    if not report.passed:
        raise ResearchError(
            f"Data quality validation failed for {symbol} {timeframe}: "
            + "; ".join(report.errors)
        )
    return report
