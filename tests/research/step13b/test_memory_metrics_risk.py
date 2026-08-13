"""Memory guard, metric calculations, and risk integration tests for Step 13B."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.step13b.memory import MemoryGuard, MemoryLimitError, rss_mb
from app.research.step13b.metrics import (
    compute_window_metrics,
    monthly_returns,
    regime_performance,
    session_performance,
    symbol_performance,
    yearly_returns,
)
from app.research.step13b.risk import (
    RiskResearchTracker,
    build_research_risk_engine,
    risk_counters_to_metrics,
)
from app.strategy.models import Signal, SignalDirection, SignalStrength


def _mock_trade(entry, exit, qty=1000.0, side="buy"):
    from datetime import datetime, timezone

    import pandas as pd

    class _T:
        def __init__(self, entry_time, exit_time, entry_price, exit_price):
            self.entry_time = entry_time
            self.exit_time = exit_time
            self.entry_price = entry_price
            self.exit_price = exit_price
            self.quantity = qty
            self.net_pnl = (exit_price - entry_price) * qty if side == "buy" else (entry_price - exit_price) * qty
            self.stop_loss = entry_price * 0.99

    start = pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime()
    return _T(start, start, entry, exit)


class TestMemoryGuard:
    def test_rss_mb_positive(self):
        assert rss_mb() > 0

    def test_guard_allows_under_limit(self):
        guard = MemoryGuard(max_rss_mb=100000.0)
        assert guard.check() > 0

    def test_guard_raises_over_limit(self):
        guard = MemoryGuard(max_rss_mb=0.000001)  # tiny limit
        with pytest.raises(MemoryLimitError):
            guard.check()

    def test_guard_validates_limit(self):
        with pytest.raises(ValueError):
            MemoryGuard(max_rss_mb=0)


class TestMetrics:
    def test_compute_window_metrics_empty(self):
        m = compute_window_metrics(
            window_index=0,
            phase="test",
            symbol="EURUSD",
            timeframe="M15",
            param_set="baseline",
            trades=[],
            equity_curve=[],
            bars=100,
        )
        assert m.trade_count == 0
        assert m.net_profit == 0.0
        assert m.expectancy is None
        assert m.win_rate is None

    def test_compute_window_metrics_with_trades(self):
        trades = [
            _mock_trade(100.0, 105.0, qty=1000.0),  # win
            _mock_trade(100.0, 98.0, qty=1000.0),   # loss
            _mock_trade(100.0, 102.0, qty=1000.0),  # win
            _mock_trade(100.0, 99.0, qty=1000.0),   # loss
        ]
        # Equity curve with downside dips so Sortino is computable.
        from app.backtest.models import EquityPoint

        from datetime import datetime, timezone

        curve = [
            EquityPoint(timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), equity=10000.0, balance=10000.0, unrealized_pnl=0.0),
            EquityPoint(timestamp=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc), equity=9950.0, balance=9950.0, unrealized_pnl=0.0),
            EquityPoint(timestamp=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc), equity=10030.0, balance=10030.0, unrealized_pnl=0.0),
            EquityPoint(timestamp=datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc), equity=9980.0, balance=9980.0, unrealized_pnl=0.0),
            EquityPoint(timestamp=datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc), equity=10200.0, balance=10200.0, unrealized_pnl=0.0),
        ]
        m = compute_window_metrics(
            window_index=0,
            phase="test",
            symbol="EURUSD",
            timeframe="M15",
            param_set="baseline",
            trades=trades,
            equity_curve=curve,
            bars=100,
        )
        assert m.trade_count == 4
        assert m.win_rate == 0.5
        assert m.net_profit > 0
        assert m.max_consecutive_wins == 1
        assert m.max_consecutive_losses == 1
        assert m.sharpe is not None
        assert m.sortino is not None

    def test_monthly_returns(self):
        trades = [_mock_trade(100.0, 105.0), _mock_trade(100.0, 98.0)]
        df = monthly_returns(trades, "EURUSD", "M15")
        assert len(df) == 1  # 1 unique month
        assert df["trade_count"].iloc[0] == 2
        assert "month" in df.columns

    def test_yearly_returns(self):
        trades = [_mock_trade(100.0, 105.0)]
        df = yearly_returns(trades, "EURUSD", "M15")
        assert len(df) == 1
        assert "year" in df.columns

    def test_regime_performance(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
                "regime": ["trending", "trending", "ranging", "trending"],
                "result": ["win", "loss", "win", "win"],
                "r_multiple": [1.0, -1.0, 0.5, 2.0],
            }
        )
        out = regime_performance(df)
        assert len(out) == 2  # trending + ranging
        trending = out[out["regime"] == "trending"].iloc[0]
        assert trending["trade_count"] == 3
        assert trending["wins"] == 2

    def test_session_performance(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
                "session": ["asia", "asia", "newyork"],
                "result": ["win", "loss", "win"],
                "r_multiple": [1.0, -1.0, 0.5],
            }
        )
        out = session_performance(df)
        assert len(out) == 2

    def test_symbol_performance(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
                "symbol": ["EURUSD", "EURUSD", "GBPUSD"],
                "result": ["win", "loss", "win"],
                "r_multiple": [1.0, -1.0, 2.0],
            }
        )
        out = symbol_performance(df)
        assert len(out) == 2
        assert out["net_r"].sum() == 2.0


class TestRiskIntegration:
    def _make_signal(self, direction="long"):
        from datetime import datetime, timezone

        return Signal(
            signal_id="sig-001",
            timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            symbol="EURUSD",
            timeframe="M15",
            direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
            strength=SignalStrength.MODERATE,
            score=3.0,
            max_score=5.0,
            entry=1.1000,
            stop_loss=1.0900,
            take_profit=1.1200,
            risk_distance=0.0100,
            reward_distance=0.0200,
            risk_reward_ratio=2.0,
            strategy="trend_structure",
            available_from=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        )

    def test_risk_engine_approves_valid_trade(self):
        engine = build_research_risk_engine()
        tracker = RiskResearchTracker(engine)
        sig = self._make_signal("long")
        from app.risk import AccountState, ProposedTrade, PositionSide

        account = AccountState(
            balance=10000.0,
            equity=10000.0,
            peak_equity=10000.0,
            daily_pnl=0.0,
            open_positions=[],
        )
        trade = ProposedTrade(
            symbol="EURUSD",
            side=PositionSide.BUY,
            entry_price=sig.entry,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
            signal_id=sig.signal_id,
        )
        decision = engine.evaluate(trade, account)
        assert decision.approved
        assert decision.position_size is not None and decision.position_size > 0

    def test_risk_counters_to_metrics(self):
        engine = build_research_risk_engine(
            max_daily_loss_pct=0.0  # daily loss limit already hit
        )
        tracker = RiskResearchTracker(engine)
        sig = self._make_signal("long")
        # Force rejection by hitting daily loss limit.
        from app.risk import AccountState, ProposedTrade, PositionSide

        account = AccountState(
            balance=10000.0,
            equity=9000.0,
            peak_equity=10000.0,
            daily_pnl=-800.0,  # > 0.0 limit means already exceeded
            open_positions=[],
        )
        trade = ProposedTrade(
            symbol="EURUSD",
            side=PositionSide.BUY,
            entry_price=sig.entry,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
            signal_id=sig.signal_id,
        )
        decision = engine.evaluate(trade, account)
        assert not decision.approved
        # Manually record.
        tracker._record(decision, sig, "baseline")
        metrics = risk_counters_to_metrics(tracker)
        assert metrics["risk_rejected_count"] == 1
        assert metrics["daily_loss_breaches"] == 1