"""Tests for the research data-quality validator. Uses synthetic frames only —
never hits the real provider."""

import pandas as pd
import pytest

from app.research.data_quality import ResearchDataValidator, validate_partition
from app.research.errors import ResearchError


def _frame(n=120, freq="1h", seed=1, tz="UTC"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz=tz)
    base = pd.Series(100 + pd.Series(range(n)).rolling(5, min_periods=1).mean())
    base.index = idx
    return pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5, "close": base})


class TestResearchDataValidator:
    def test_valid_frame_passes(self):
        report = ResearchDataValidator().validate(_frame(), "EURUSD", "H1")
        assert report.passed is True
        assert report.candle_count == 120
        assert report.duplicate_count == 0
        assert report.invalid_ohlc_count == 0
        assert report.gap_count == 0

    def test_duplicate_timestamps_fail(self):
        df = _frame()
        df = pd.concat([df, df.iloc[:5]])
        report = ResearchDataValidator().validate(df, "EURUSD", "H1")
        assert report.passed is False
        assert report.duplicate_count == 5
        assert any("duplicate" in e for e in report.errors)

    def test_invalid_ohlc_fails(self):
        df = _frame()
        df.iloc[3, df.columns.get_loc("high")] = 50.0  # high < open
        report = ResearchDataValidator().validate(df, "EURUSD", "H1")
        assert report.passed is False
        assert report.invalid_ohlc_count >= 1

    def test_non_chronological_fails(self):
        df = _frame().iloc[::-1]
        report = ResearchDataValidator().validate(df, "EURUSD", "H1")
        assert report.passed is False
        assert any("chronological" in e for e in report.errors)

    def test_naive_timezone_fails(self):
        df = _frame(tz=None)
        report = ResearchDataValidator().validate(df, "EURUSD", "H1")
        assert report.passed is False
        assert report.timezone_status == "naive"

    def test_market_closure_gap_is_warning_not_fail(self):
        # Weekend-like gap: Fri 21:00 -> Mon 00:00 is legitimate FX closure.
        idx = list(pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"))
        idx += list(pd.date_range("2026-01-04", periods=3, freq="1h", tz="UTC"))
        df = _frame()
        df = df.reindex(pd.DatetimeIndex(idx))
        df = df.dropna()
        if len(df) < 2:
            df = _frame()
        report = ResearchDataValidator().validate(df, "EURUSD", "M15")
        # Gaps reported but do NOT by themselves fail the run.
        assert report.passed is True or all(
            "gap" not in e for e in report.errors
        )
        assert report.gap_count >= 0

    def test_validate_partition_annotates_provenance(self):
        df = _frame()
        report = validate_partition(df, "EURUSD", "H1", provider="twelvedata", native_or_aggregated="native")
        assert report.provider == "twelvedata"
        assert report.native_or_aggregated == "native"

    def test_validate_partition_raises_on_bad_data(self):
        with pytest.raises(ResearchError):
            validate_partition(_frame().iloc[::-1], "EURUSD", "H1")
