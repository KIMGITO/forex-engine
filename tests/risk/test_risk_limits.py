"""Tests for risk limits and trade approval/rejection."""

import pytest

from app.risk import (
    AccountState,
    InstrumentSpec,
    PositionSide,
    ProposedTrade,
    RiskConfig,
    RiskEngine,
)


def _eur_spec() -> InstrumentSpec:
    return InstrumentSpec(symbol="EURUSD")


def _engine(**cfg_kw) -> RiskEngine:
    return RiskEngine(
        RiskConfig(**cfg_kw),
        {"EURUSD": _eur_spec()},
    )


def _trade(**kw):
    base = {
        "symbol": "EURUSD",
        "side": PositionSide.BUY,
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
        "take_profit": 1.1100,
        "signal_id": "sig-1",
    }
    base.update(kw)
    return ProposedTrade(**base)


def _acct(
    balance=10_000.0,
    equity=None,
    daily_pnl=0.0,
    peak_equity=None,
    open_positions=None,
    drawdown_pct=None,
    available_margin=0.0,
) -> AccountState:
    return AccountState(
        balance=balance,
        equity=equity if equity is not None else balance,
        daily_pnl=daily_pnl,
        peak_equity=peak_equity if peak_equity is not None else balance,
        open_positions=open_positions or [],
        drawdown_pct=drawdown_pct,
        available_margin=available_margin,
    )


class TestApprovalAndLimits:
    def test_trade_within_limits_approved(self):
        d = _engine().evaluate(_trade(), _acct())
        assert d.approved

    def test_default_daily_loss_not_triggered(self):
        # Default max_daily_loss_pct=0.03. daily_pnl = -$100 is within 3% of $10k.
        d = _engine().evaluate(_trade(), _acct(daily_pnl=-100.0))
        assert d.approved

    def test_daily_loss_limit_exceeded(self):
        d = _engine().evaluate(_trade(), _acct(daily_pnl=-400.0))
        assert not d.approved
        assert d.reason.value == "daily_loss_limit_exceeded"

    def test_drawdown_exceeded(self):
        # Account down 15% from peak; max_drawdown_pct default 0.10.
        d = _engine().evaluate(_trade(), _acct(equity=8_500, peak_equity=10_000))
        assert not d.approved
        assert d.reason.value == "drawdown_limit_exceeded"

    def test_drawdown_within_limit_approved(self):
        d = _engine().evaluate(_trade(), _acct(equity=9_900, peak_equity=10_000))
        assert d.approved

    def test_drawdown_disabled(self):
        eng = _engine(max_drawdown_pct=None)
        d = eng.evaluate(_trade(), _acct(equity=8_000, peak_equity=10_000))
        assert d.approved

    def test_max_open_positions_reached(self):
        cfg = RiskConfig(max_open_positions=1)
        eng = RiskEngine(cfg, {"EURUSD": _eur_spec()})
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05}
        ]
        d = eng.evaluate(_trade(), _acct(open_positions=existing))
        assert not d.approved
        assert d.reason.value == "max_open_positions_reached"

    def test_symbol_exposure_exceeded(self):
        # max_symbol_exposure absolute notional = 25,000. New EURUSD notional
        # ~22,000 (20k units * 1.10). Combined with existing would exceed.
        eng = _engine(max_symbol_exposure=25_000)
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 10_000, "entry_price": 1.05}
        ]
        d = eng.evaluate(_trade(), _acct(open_positions=existing))
        assert not d.approved
        assert d.reason.value == "symbol_exposure_exceeded"

    def test_total_exposure_exceeded(self):
        eng = _engine(max_total_exposure=25_000)
        # Existing GBPUSD notional ~10,500; new EURUSD ~22,000 -> total ~32,500.
        existing = [
            {"symbol": "GBPUSD", "side": "buy", "quantity": 10_000, "entry_price": 1.05}
        ]
        d = eng.evaluate(_trade(), _acct(open_positions=existing))
        assert not d.approved
        assert d.reason.value == "total_exposure_exceeded"

    def test_emergency_stop(self):
        d = _engine(emergency_stop=True).evaluate(_trade(), _acct())
        assert not d.approved
        assert d.reason.value == "emergency_stop"

    def test_duplicate_position(self):
        eng = _engine(prevent_duplicate_position=True)
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05}
        ]
        d = eng.evaluate(_trade(), _acct(open_positions=existing))
        assert not d.approved
        assert d.reason.value == "duplicate_position"

    def test_opposite_side_not_duplicate(self):
        eng = _engine(prevent_duplicate_position=True)
        existing = [
            {"symbol": "EURUSD", "side": "sell", "quantity": 1000, "entry_price": 1.05}
        ]
        d = eng.evaluate(_trade(), _acct(open_positions=existing))
        assert d.approved

    def test_duplicate_protection_disabled_by_default(self):
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05}
        ]
        d = _engine().evaluate(_trade(), _acct(open_positions=existing))
        assert d.approved

    def test_per_trade_risk_cap(self):
        # Risk on this trade = $100. Cap at $50 -> rejected.
        eng = _engine(max_risk_per_trade=50.0)
        d = eng.evaluate(_trade(), _acct())
        assert not d.approved
        assert d.reason.value == "per_trade_risk_exceeded"


class TestInstrumentValidation:
    def test_missing_instrument(self):
        eng = RiskEngine(RiskConfig(), {})
        d = eng.evaluate(_trade(), _acct())
        assert not d.approved
        assert d.reason.value == "invalid_instrument"

    def test_unknown_symbol_rejected(self):
        eng = RiskEngine(RiskConfig(allow_unknown_symbols=False), {"EURUSD": _eur_spec()})
        d = eng.evaluate(
            ProposedTrade(
                symbol="XAUUSD", side=PositionSide.BUY,
                entry_price=1800.0, stop_loss=1790.0,
            ),
            _acct(),
        )
        assert not d.approved
        assert d.reason.value == "invalid_instrument"

    def test_unknown_symbol_allowed_if_configured(self):
        eng = RiskEngine(
            RiskConfig(allow_unknown_symbols=True),
            {"EURUSD": _eur_spec(), "XAUUSD": InstrumentSpec(symbol="XAUUSD")},
        )
        trade = ProposedTrade(
            symbol="XAUUSD", side=PositionSide.BUY,
            entry_price=1800.0, stop_loss=1790.0,
        )
        d = eng.evaluate(trade, _acct())
        # XAUUSD maps to UNKNOWN but a spec exists and allow is on -> approved.
        assert d.approved


class TestTradeValidation:
    def test_missing_stop_loss_rejected_by_model(self):
        # Pydantic requires stop_loss > 0; a missing stop cannot be constructed.
        with pytest.raises(ValueError):
            ProposedTrade(
                symbol="EURUSD", side=PositionSide.BUY,
                entry_price=1.1, stop_loss=None,
            )

    def test_stop_on_wrong_side_buy(self):
        d = _engine().evaluate(
            _trade(entry_price=1.1000, stop_loss=1.1050), _acct()
        )
        assert not d.approved
        assert d.reason.value == "stop_on_wrong_side"

    def test_stop_on_wrong_side_sell(self):
        d = _engine().evaluate(
            _trade(side=PositionSide.SELL, entry_price=1.1000, stop_loss=1.0900),
            _acct(),
        )
        assert not d.approved
        assert d.reason.value == "stop_on_wrong_side"

    def test_sell_trade_approved(self):
        d = _engine().evaluate(
            _trade(side=PositionSide.SELL, entry_price=1.1000, stop_loss=1.1050),
            _acct(),
        )
        assert d.approved

    def test_zero_stop_distance_rejected(self):
        d = _engine().evaluate(
            _trade(entry_price=1.1000, stop_loss=1.1000), _acct()
        )
        assert not d.approved
        assert d.reason.value == "invalid_trade"


class TestDeterminism:
    def test_identical_inputs_identical_decision(self):
        eng = _engine()
        trade = _trade()
        acct = _acct()
        d1 = eng.evaluate(trade, acct)
        d2 = eng.evaluate(trade, acct)
        assert d1.model_dump() == d2.model_dump()
        assert d1.position_size == d2.position_size
        assert d1.monetary_risk == d2.monetary_risk

    def test_determinism_across_engines(self):
        a = _engine().evaluate(_trade(), _acct())
        b = _engine().evaluate(_trade(), _acct())
        assert a.model_dump() == b.model_dump()