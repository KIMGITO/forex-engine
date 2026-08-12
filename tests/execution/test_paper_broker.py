"""Tests for the PaperBroker (Step 16)."""

from datetime import datetime, timezone

import pytest

from app.execution.errors import (
    InvalidOrderError,
    OrderNotFoundError,
    PositionNotFoundError,
    RiskGateViolationError,
)
from app.execution.models import (
    CancelRequest,
    ModifyRequest,
    OrderRequest,
    OrderStatus,
    PositionStatus,
    RejectionCode,
)
from app.execution.paper import PaperBroker
from app.risk.models import PositionSide, RiskDecision, RiskDecisionType


def _approved_decision() -> RiskDecision:
    return RiskDecision(
        type=RiskDecisionType.APPROVED,
        position_size=10_000,
        monetary_risk=50.0,
        risk_percent=0.005,
    )


def _ts(second: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, second, tzinfo=timezone.utc)


def _req(
    symbol: str = "EURUSD",
    side: PositionSide = PositionSide.BUY,
    quantity: float = 10_000,
    requested_price: float = 1.1000,
    stop_loss: float | None = 1.0950,
    take_profit: float | None = 1.1050,
    order_id: str = "ord-1",
    risk: RiskDecision | None = None,
) -> OrderRequest:
    return OrderRequest(
        order_id=order_id,
        client_order_id="sig-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        requested_price=requested_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        timestamp=_ts(),
        risk_decision=risk if risk is not None else _approved_decision(),
    )


class TestSubmit:
    def test_valid_market_buy(self):
        broker = PaperBroker()
        result = broker.submit_order(_req())
        assert result.status == OrderStatus.FILLED
        assert result.executed_price == 1.1000
        assert result.position_id == "pos-1"
        assert len(broker.get_open_positions()) == 1
        pos = broker.get_open_positions()[0]
        assert pos.symbol == "EURUSD"
        assert pos.side == PositionSide.BUY
        assert pos.quantity == 10_000
        assert pos.status == PositionStatus.OPEN

    def test_valid_market_sell(self):
        broker = PaperBroker()
        result = broker.submit_order(
            _req(side=PositionSide.SELL, stop_loss=1.1050, take_profit=1.0950)
        )
        assert result.status == OrderStatus.FILLED
        assert result.position_id == "pos-1"
        assert broker.get_open_positions()[0].side == PositionSide.SELL

    def test_invalid_quantity_blocked_by_model(self):
        # Pydantic's gt=0.0 rejects non-positive quantity at the model boundary.
        with pytest.raises(ValueError):
            _req(quantity=-100)

    def test_invalid_quantity_rejected_by_broker_defensive_check(self):
        # A request constructed with model_construct bypasses Pydantic; the
        # broker must still reject it defensively.
        broker = PaperBroker()
        request = OrderRequest.model_construct(
            order_id="ord-1",
            client_order_id="sig-1",
            symbol="EURUSD",
            side=PositionSide.BUY,
            quantity=-100,
            requested_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1050,
            timestamp=_ts(),
            risk_decision=_approved_decision(),
        )
        result = broker.submit_order(request)
        assert result.status == OrderStatus.REJECTED
        assert result.rejection_code == RejectionCode.INVALID_QUANTITY

    def test_invalid_symbol_rejected(self):
        broker = PaperBroker()
        result = broker.submit_order(_req(symbol="BTC"))
        assert result.status == OrderStatus.REJECTED
        assert result.rejection_code == RejectionCode.INVALID_SYMBOL

    def test_invalid_stop_rejected(self):
        broker = PaperBroker()
        result = broker.submit_order(_req(stop_loss=1.1050))  # stop above entry for BUY
        assert result.status == OrderStatus.REJECTED
        assert result.rejection_code == RejectionCode.INVALID_STOP

    def test_missing_risk_decision_raises(self):
        broker = PaperBroker()
        # Construct directly with risk_decision=None (Pydantic allows None).
        request = OrderRequest(
            order_id="ord-1",
            client_order_id="sig-1",
            symbol="EURUSD",
            side=PositionSide.BUY,
            quantity=10_000,
            requested_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1050,
            timestamp=_ts(),
            risk_decision=None,
        )
        with pytest.raises(RiskGateViolationError):
            broker.submit_order(request)

    def test_non_approved_risk_raises(self):
        broker = PaperBroker()
        with pytest.raises(RiskGateViolationError):
            broker.submit_order(
                _req(risk=RiskDecision(type=RiskDecisionType.REJECTED))
            )

    def test_deterministic_order_ids(self):
        broker = PaperBroker()
        r1 = broker.submit_order(_req(order_id="ord-1"))
        r2 = broker.submit_order(
            _req(order_id="ord-2", symbol="GBPUSD", requested_price=1.27,
                 stop_loss=1.265, take_profit=1.275)
        )
        assert r1.order_id == "ord-1"
        assert r2.order_id == "ord-2"
        assert r1.position_id == "pos-1"
        assert r2.position_id == "pos-2"

    def test_no_sl_tp_allowed(self):
        broker = PaperBroker()
        result = broker.submit_order(_req(stop_loss=None, take_profit=None))
        assert result.status == OrderStatus.FILLED


class TestPositionLifecycle:
    def test_position_opening_and_closing(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(quantity=10_000))
        pos_id = r.position_id
        assert pos_id == "pos-1"
        closed = broker.close_position(pos_id, 1.1020, _ts(5))
        assert closed.status == PositionStatus.CLOSED
        assert closed.realized_pnl == pytest.approx((1.1020 - 1.1000) * 10_000)
        assert len(broker.get_open_positions()) == 0

    def test_realized_pnl_short(self):
        broker = PaperBroker()
        r = broker.submit_order(
            _req(side=PositionSide.SELL, stop_loss=1.1050, take_profit=1.0950)
        )
        closed = broker.close_position(r.position_id, 1.0950, _ts(5))  # profit
        assert closed.realized_pnl == pytest.approx((1.1000 - 1.0950) * 10_000)

    def test_sell_loss_position(self):
        broker = PaperBroker()
        r = broker.submit_order(
            _req(side=PositionSide.SELL, stop_loss=1.1050, take_profit=1.0950)
        )
        closed = broker.close_position(r.position_id, 1.1040, _ts(5))  # loss
        assert closed.realized_pnl == pytest.approx((1.1000 - 1.1040) * 10_000)
        assert broker.balance == pytest.approx(10_000 + (1.1000 - 1.1040) * 10_000)

    def test_partial_close(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(quantity=10_000))
        pos = broker.close_position(r.position_id, 1.1010, _ts(5), quantity=4_000)
        assert pos.status == PositionStatus.OPEN
        assert pos.quantity == pytest.approx(6_000)
        assert pos.closed_quantity == pytest.approx(4_000)
        assert pos.realized_pnl == pytest.approx((1.1010 - 1.1000) * 4_000)

    def test_close_unknown_position_raises(self):
        broker = PaperBroker()
        with pytest.raises(PositionNotFoundError):
            broker.close_position("pos-999", 1.1, _ts())

    def test_close_already_closed_raises(self):
        broker = PaperBroker()
        r = broker.submit_order(_req())
        broker.close_position(r.position_id, 1.1010, _ts(5))
        with pytest.raises(InvalidOrderError):
            broker.close_position(r.position_id, 1.1020, _ts(6))

    def test_sl_execution(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(stop_loss=1.0950, take_profit=1.1050))
        closed = broker.mark_position_sl_tp(r.position_id, 1.0940, 1.0940, _ts(5))
        assert closed.status == PositionStatus.CLOSED
        assert closed.closed_at is not None
        # PnL = (1.0950 - 1.1000) * 10_000 = -50
        assert closed.realized_pnl == pytest.approx(-50.0)
        assert broker.balance == pytest.approx(10_000 - 50.0)

    def test_tp_execution(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(stop_loss=1.0950, take_profit=1.1050))
        closed = broker.mark_position_sl_tp(r.position_id, 1.1060, 1.0990, _ts(5))
        assert closed.status == PositionStatus.CLOSED
        assert closed.realized_pnl == pytest.approx((1.1050 - 1.1000) * 10_000)

    def test_sl_tp_both_hit_sl_first(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(stop_loss=1.0950, take_profit=1.1050))
        # Bar touches both: low 1.0940 (SL) and high 1.1060 (TP). SL-first.
        closed = broker.mark_position_sl_tp(r.position_id, 1.1060, 1.0940, _ts(5))
        assert closed.realized_pnl == pytest.approx(-50.0)

    def test_no_hit_keeps_position_open(self):
        broker = PaperBroker()
        r = broker.submit_order(_req(stop_loss=1.0950, take_profit=1.1050))
        pos = broker.mark_position_sl_tp(r.position_id, 1.1020, 1.0980, _ts(5))
        assert pos.status == PositionStatus.OPEN
        assert len(broker.get_open_positions()) == 1

    def test_modify_position_sl(self):
        broker = PaperBroker()
        r = broker.submit_order(_req())
        result = broker.modify_order(
            ModifyRequest(order_id=r.order_id, stop_loss=1.0970, timestamp=_ts(5))
        )
        assert result.status == OrderStatus.MODIFIED
        pos = broker.get_open_positions()[0]
        assert pos.stop_loss == 1.0970
        assert pos.take_profit == 1.1050

    def test_modify_order_unknown_raises(self):
        broker = PaperBroker()
        with pytest.raises(OrderNotFoundError):
            broker.modify_order(
                ModifyRequest(order_id="ord-999", stop_loss=1.0970, timestamp=_ts())
            )

    def test_cancel_filled_order_returns_cancelled(self):
        broker = PaperBroker()
        r = broker.submit_order(_req())
        res = broker.cancel_order(CancelRequest(order_id=r.order_id, timestamp=_ts(5)))
        assert res.status == OrderStatus.CANCELLED
        assert res.rejection_code == RejectionCode.INVALID_REQUEST

    def test_cancel_unknown_raises(self):
        broker = PaperBroker()
        with pytest.raises(OrderNotFoundError):
            broker.cancel_order(CancelRequest(order_id="ord-999", timestamp=_ts()))


class TestAccountState:
    def test_account_state_initial(self):
        broker = PaperBroker(initial_balance=20_000)
        state = broker.get_account_state()
        assert state.balance == 20_000
        assert state.equity == 20_000
        assert state.peak_equity == 20_000
        assert state.daily_pnl == 0.0
        assert state.open_positions == []

    def test_account_state_with_open_position(self):
        broker = PaperBroker()
        broker.submit_order(_req(quantity=10_000))
        state = broker.get_account_state(mid_prices={"EURUSD": 1.1010})
        assert len(state.open_positions) == 1
        assert state.open_positions[0]["symbol"] == "EURUSD"
        assert state.equity == pytest.approx(10_000 + (1.1010 - 1.1000) * 10_000)

    def test_unrealized_pnl_long(self):
        broker = PaperBroker()
        broker.submit_order(_req(quantity=10_000))
        state = broker.get_account_state(mid_prices={"EURUSD": 1.1050})
        assert state.equity == pytest.approx(10_000 + (1.1050 - 1.1000) * 10_000)

    def test_unrealized_pnl_short(self):
        broker = PaperBroker()
        broker.submit_order(
            _req(side=PositionSide.SELL, stop_loss=1.1050, take_profit=1.0950)
        )
        state = broker.get_account_state(mid_prices={"EURUSD": 1.0950})
        assert state.equity == pytest.approx(10_000 + (1.1000 - 1.0950) * 10_000)

    def test_unrealized_pnl_jpy_with_conversion(self):
        broker = PaperBroker(quote_to_account={"USDJPY": 1 / 150.0})
        broker.submit_order(
            _req(symbol="USDJPY", requested_price=150.0, stop_loss=149.5, take_profit=150.5)
        )
        state = broker.get_account_state(mid_prices={"USDJPY": 150.5})
        # PnL in JPY = (150.5-150.0)*10000 = 5000 JPY -> /150 = 33.33 USD
        assert state.equity == pytest.approx(10_000 + 5000 / 150.0)

    def test_no_price_no_unrealized(self):
        broker = PaperBroker(quote_to_account={"USDJPY": 1 / 150.0})
        broker.submit_order(
            _req(symbol="USDJPY", requested_price=150.0, stop_loss=149.5, take_profit=150.5)
        )
        state = broker.get_account_state()  # no mid price
        assert state.equity == pytest.approx(10_000)  # marked at entry

    def test_account_state_drawdown(self):
        broker = PaperBroker()
        broker.submit_order(_req(quantity=10_000))
        state = broker.get_account_state(mid_prices={"EURUSD": 1.0900})
        assert state.equity == pytest.approx(10_000 - (1.1000 - 1.0900) * 10_000)
        assert state.drawdown_pct == pytest.approx(100.0 / 10_000)

    def test_multiple_positions(self):
        broker = PaperBroker(initial_balance=100_000)
        broker.submit_order(_req(order_id="ord-1", quantity=10_000))
        broker.submit_order(
            _req(order_id="ord-2", symbol="GBPUSD", quantity=5_000,
                 requested_price=1.27, stop_loss=1.265, take_profit=1.275)
        )
        assert len(broker.get_open_positions()) == 2
        state = broker.get_account_state(mid_prices={"EURUSD": 1.10, "GBPUSD": 1.27})
        assert len(state.open_positions) == 2
        assert state.exposure == pytest.approx(10_000 * 1.10 + 5_000 * 1.27)


class TestIsolation:
    def test_accounts_are_isolated(self):
        a = PaperBroker(initial_balance=10_000)
        b = PaperBroker(initial_balance=50_000)
        a.submit_order(_req(order_id="ord-1"))
        assert len(a.get_open_positions()) == 1
        assert len(b.get_open_positions()) == 0
        assert b.balance == 50_000
        assert a.balance == 10_000

    def test_order_history(self):
        broker = PaperBroker()
        broker.submit_order(_req(order_id="ord-1"))
        broker.submit_order(
            _req(order_id="ord-2", symbol="GBPUSD", requested_price=1.27,
                 stop_loss=1.265, take_profit=1.275)
        )
        hist = broker.get_order_history()
        assert [o.order_id for o in hist] == ["ord-1", "ord-2"]
        assert all(o.status == OrderStatus.FILLED for o in hist)

    def test_get_order(self):
        broker = PaperBroker()
        broker.submit_order(_req(order_id="ord-1"))
        result = broker.get_order("ord-1")
        assert result is not None
        assert result.order_id == "ord-1"
        assert broker.get_order("ord-999") is None