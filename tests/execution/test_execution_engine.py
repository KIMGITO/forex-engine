"""Tests for the ExecutionEngine risk-gate integration (Step 16)."""

from datetime import datetime, timezone

import pytest

from app.execution.engine import ExecutionEngine
from app.execution.paper import PaperBroker
from app.risk import (
    AccountState,
    InstrumentSpec,
    PositionSide,
    ProposedTrade,
    RiskConfig,
    RiskEngine,
)


def _ts() -> datetime:
    return datetime(2026, 2, 1, 9, 30, tzinfo=timezone.utc)


def _instruments() -> dict[str, InstrumentSpec]:
    return {
        "EURUSD": InstrumentSpec(symbol="EURUSD"),
        "GBPUSD": InstrumentSpec(symbol="GBPUSD"),
    }


def _engine(**cfg_kw) -> tuple[ExecutionEngine, PaperBroker]:
    broker = PaperBroker()
    risk = RiskEngine(RiskConfig(**cfg_kw), _instruments())
    return ExecutionEngine(risk, broker), broker


def _trade(
    symbol: str = "EURUSD",
    side: PositionSide = PositionSide.BUY,
    entry: float = 1.1000,
    stop: float = 1.0950,
    take: float | None = 1.1100,
) -> ProposedTrade:
    return ProposedTrade(
        symbol=symbol,
        side=side,
        entry_price=entry,
        stop_loss=stop,
        take_profit=take,
        signal_id="sig-1",
    )


def _acct(
    balance=10_000.0,
    daily_pnl=0.0,
    equity=None,
    drawdown_pct=None,
    open_positions=None,
) -> AccountState:
    return AccountState(
        balance=balance,
        equity=equity if equity is not None else balance,
        daily_pnl=daily_pnl,
        peak_equity=balance,
        drawdown_pct=drawdown_pct,
        open_positions=open_positions or [],
    )


class TestApprovedOrdersReachBroker:
    def test_approved_order_reaches_broker(self):
        eng, broker = _engine()
        out = eng.place_order(_trade(), _acct(), timestamp=_ts())
        assert out["approved"] is True
        assert out["order_id"] == "ord-1"
        assert out["position_id"] == "pos-1"
        assert len(broker.get_open_positions()) == 1
        # Broker order carries the approved risk decision.
        req_seen = broker.get_order("ord-1")
        assert req_seen is not None
        assert req_seen.status.value == "filled"
        # Its associated position filled at requested price.
        pos = broker.get_open_positions()[0]
        assert pos.entry_price == 1.1000

    def test_approved_order_uses_risk_position_size(self):
        eng = _engine()[0]
        # No quantity supplied -> uses RiskDecision.position_size (20k for 1% of 10k).
        eng.place_order(_trade(), _acct(), timestamp=_ts())
        pos = eng.broker.get_open_positions()[0]
        assert pos.quantity == pytest.approx(20_000)

    def test_second_order_ids_increment(self):
        eng, broker = _engine()
        eng.place_order(_trade(), _acct(), timestamp=_ts())
        eng.place_order(_trade(symbol="GBPUSD", entry=1.27, stop=1.265),
                        _acct(), timestamp=_ts())
        assert [o.order_id for o in broker.get_order_history()] == ["ord-1", "ord-2"]


class TestRiskRejections:
    def test_daily_loss_rejection_prevents_execution(self):
        eng, broker = _engine()
        acct = _acct(daily_pnl=-600.0)  # exceeds 3% of 10k
        out = eng.place_order(_trade(), acct, timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "daily_loss_limit"
        assert len(broker.get_open_positions()) == 0
        # Audit record exists but broker saw nothing.
        assert len(broker.get_order_history()) == 0

    def test_drawdown_rejection_prevents_execution(self):
        eng, broker = _engine()
        acct = _acct(equity=8_000, drawdown_pct=0.20)
        out = eng.place_order(_trade(), acct, timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "drawdown_limit"
        assert len(broker.get_open_positions()) == 0

    def test_emergency_stop_rejection(self):
        eng, broker = _engine(emergency_stop=True)
        out = eng.place_order(_trade(), _acct(), timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "emergency_stop"
        assert len(broker.get_open_positions()) == 0

    def test_max_positions_rejection(self):
        eng, broker = _engine(max_open_positions=1)
        existing = [
            {"symbol": "GBPUSD", "side": "sell", "quantity": 1000, "entry_price": 1.27}
        ]
        out = eng.place_order(_trade(), _acct(open_positions=existing), timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "max_open_positions"
        assert len(broker.get_open_positions()) == 0

    def test_duplicate_position_rejection(self):
        eng, broker = _engine(prevent_duplicate_position=True)
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05}
        ]
        out = eng.place_order(_trade(), _acct(open_positions=existing), timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "duplicate_position"
        assert len(broker.get_open_positions()) == 0

    def test_per_trade_risk_cap_rejection(self):
        eng, broker = _engine(max_risk_per_trade=20.0)  # risk is $100
        out = eng.place_order(_trade(), _acct(), timestamp=_ts())
        assert out["approved"] is False
        assert out["rejection_code"] == "per_trade_risk"
        assert len(broker.get_open_positions()) == 0


class TestRiskGateIntegrity:
    def test_rejected_order_never_reaches_broker(self):
        eng, broker = _engine()
        acct = _acct(daily_pnl=-1000.0)
        out = eng.place_order(_trade(), acct, timestamp=_ts())
        assert out["approved"] is False
        assert broker.get_order_history() == []

    def test_audit_trails_rejected_orders(self):
        eng = _engine()[0]
        eng.place_order(_trade(), _acct(daily_pnl=-1000.0), timestamp=_ts())
        audit = eng.get_audit()
        assert len(audit) == 1
        assert audit[0]["status"] == "rejected"
        assert audit[0]["rejection_code"] == "daily_loss_limit"
        assert audit[0]["position_id"] is None

    def test_audit_trails_approved_orders(self):
        eng = _engine()[0]
        eng.place_order(_trade(), _acct(), timestamp=_ts())
        audit = eng.get_audit()
        assert len(audit) == 1
        assert audit[0]["status"] == "filled"
        assert audit[0]["position_id"] == "pos-1"
        assert audit[0]["risk_decision"].approved is True

    def test_broker_rejects_order_without_risk_gate(self):
        # Direct broker wall: a request with no risk decision is refused.
        from app.execution.errors import RiskGateViolationError
        from app.execution.models import OrderRequest

        broker = PaperBroker()
        with pytest.raises(RiskGateViolationError):
            broker.submit_order(
                OrderRequest(
                    order_id="ord-1",
                    symbol="EURUSD",
                    side=PositionSide.BUY,
                    quantity=10_000,
                    requested_price=1.1,
                    timestamp=_ts(),
                    risk_decision=None,
                )
            )

    def test_invalid_quantity_from_risk_rejected(self):
        # A ProposedTrade with negative quantity is caught by Pydantic; here we
        # verify the engine rejects a trade whose risk approval yields no size.
        eng, broker = _engine(risk_percent=0.0)  # invalid config would be caught upstream
        acct = _acct()
        # risk_percent=0 is invalid for RiskEngine -> it raises RiskError caught
        # by engine? Actually RiskEngine.evaluate calls position_size_for_risk
        # which raises; the engine catches broad Exception. We verify no broker
        # order is placed.
        out = eng.place_order(_trade(), acct, timestamp=_ts())
        assert out["approved"] is False
        assert broker.get_order_history() == []


class TestPositionManagement:
    def test_close_position_through_engine(self):
        eng, broker = _engine()
        out = eng.place_order(_trade(), _acct(), timestamp=_ts())
        pos_id = out["position_id"]
        res = eng.close_position(pos_id, 1.1010, _ts())
        assert res["closed"] is True
        assert res["realized_pnl"] == pytest.approx((1.1010 - 1.1000) * 20_000)
        assert len(broker.get_open_positions()) == 0

    def test_account_state_via_engine(self):
        eng = _engine()[0]
        eng.place_order(_trade(), _acct(), timestamp=_ts())
        eng.set_mid_prices({"EURUSD": 1.1050})
        state = eng.get_account_state()
        assert len(state.open_positions) == 1
        assert state.equity == pytest.approx(10_000 + (1.1050 - 1.1000) * 20_000)

    def test_multiple_simultaneous_positions(self):
        eng, broker = _engine(max_open_positions=5)
        eng.place_order(_trade(), _acct(), timestamp=_ts())
        eng.place_order(
            _trade(symbol="GBPUSD", entry=1.27, stop=1.265),
            _acct(),
            timestamp=_ts(),
        )
        assert len(broker.get_open_positions()) == 2

    def test_isolation_across_engines(self):
        broker_a = PaperBroker()
        broker_b = PaperBroker(initial_balance=50_000)
        risk_a = RiskEngine(RiskConfig(), _instruments())
        risk_b = RiskEngine(RiskConfig(), _instruments())
        eng_a = ExecutionEngine(risk_a, broker_a)
        ExecutionEngine(risk_b, broker_b)
        eng_a.place_order(_trade(), _acct(), timestamp=_ts())
        assert len(broker_a.get_open_positions()) == 1
        assert len(broker_b.get_open_positions()) == 0
        assert broker_b.balance == 50_000