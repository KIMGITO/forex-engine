"""Tests for report/metrics utilities."""


import numpy as np
import pandas as pd
import pytest

from app.backtest import BacktestConfig, EventBacktester, NoOpStrategy
from app.backtest.metrics import drawdown_pct
from app.backtest.models import EquityPoint
from app.backtest.reports import render_report


def _frame(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(1).normal(0, 0.001, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.001, "low": close - 0.001, "close": close},
        index=idx,
    )


def _equity_points(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1h", tz="UTC")
    return [
        EquityPoint(
            timestamp=t,
            equity=v,
            balance=v,
            unrealized_pnl=0.0,
        )
        for t, v in zip(idx, values)
    ]


class TestDrawdown:
    def test_no_drawdown_when_rising(self):
        pts = _equity_points([100, 101, 102, 103])
        assert drawdown_pct(pts) == [0.0, 0.0, 0.0, 0.0]

    def test_drawdown_detected(self):
        pts = _equity_points([100, 110, 100, 90])
        dd = drawdown_pct(pts)
        assert dd[0] == pytest.approx(0.0)
        assert dd[3] == pytest.approx(90 / 110 - 0.0, abs=1e-6) or dd[3] > 0.18


class TestReport:
    def test_render_contains_key_fields(self):
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(_frame(), NoOpStrategy())
        text = render_report(res)
        assert "BACKTEST" in text
        assert "Symbol" in text
        assert "assumptions" in text.lower() or "ASSUMPTIONS" in text