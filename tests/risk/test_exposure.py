"""Tests for exposure groups and exposure computation."""

import pytest

from app.risk import (
    AccountState,
    ExposureGroup,
    InstrumentSpec,
    PositionSide,
    ProposedTrade,
    RiskConfig,
    RiskEngine,
)
from app.risk.exposure import exposure_group_for, group_exposure, symbol_exposure


class TestExposureGroupMapping:
    def test_euro_usd_group(self):
        assert ExposureGroup.from_symbol("EURUSD") == ExposureGroup.USD_QUOTE

    def test_gbp_usd_group(self):
        assert ExposureGroup.from_symbol("GBPUSD") == ExposureGroup.USD_QUOTE

    def test_usd_jpy_group(self):
        assert ExposureGroup.from_symbol("USDJPY") == ExposureGroup.USD_BASE_JPY

    def test_usd_chf_group(self):
        assert ExposureGroup.from_symbol("USDCHF") == ExposureGroup.USD_BASE_CHF

    def test_unknown_group(self):
        assert ExposureGroup.from_symbol("EURNOK") == ExposureGroup.UNKNOWN
        assert ExposureGroup.from_symbol("EUR") == ExposureGroup.UNKNOWN

    def test_override(self):
        cfg = RiskConfig(exposure_groups={"USDJPY": "usd_quote"})
        assert exposure_group_for("USDJPY", cfg) == ExposureGroup.USD_QUOTE

    def test_invalid_override_falls_back_to_unknown(self):
        cfg = RiskConfig(exposure_groups={"EURUSD": "bogus"})
        assert exposure_group_for("EURUSD", cfg) == ExposureGroup.UNKNOWN


class TestExposureComputation:
    def test_symbol_exposure(self):
        cfg = RiskConfig()
        specs = {"EURUSD": InstrumentSpec(symbol="EURUSD")}
        positions = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05}
        ]
        assert symbol_exposure(positions, "EURUSD", cfg, specs) == pytest.approx(1050.0)

    def test_total_exposure(self):
        cfg = RiskConfig()
        specs = {
            "EURUSD": InstrumentSpec(symbol="EURUSD"),
            "GBPUSD": InstrumentSpec(symbol="GBPUSD"),
        }
        positions = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05},
            {"symbol": "GBPUSD", "side": "sell", "quantity": 2000, "entry_price": 1.27},
        ]
        total = symbol_exposure(positions, "EURUSD", cfg, specs) + symbol_exposure(
            positions, "GBPUSD", cfg, specs
        )
        assert total == pytest.approx(1050.0 + 2540.0)

    def test_group_exposure(self):
        cfg = RiskConfig()
        specs = {
            "EURUSD": InstrumentSpec(symbol="EURUSD"),
            "GBPUSD": InstrumentSpec(symbol="GBPUSD"),
            "USDJPY": InstrumentSpec(symbol="USDJPY", quote_to_account=1 / 150.0),
        }
        positions = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.05},
            {"symbol": "GBPUSD", "side": "sell", "quantity": 2000, "entry_price": 1.27},
            {"symbol": "USDJPY", "side": "buy", "quantity": 3000, "entry_price": 150.0},
        ]
        usd_quote = group_exposure(positions, ExposureGroup.USD_QUOTE, cfg, specs)
        assert usd_quote == pytest.approx(1050.0 + 2540.0)
        jpy = group_exposure(positions, ExposureGroup.USD_BASE_JPY, cfg, specs)
        assert jpy == pytest.approx(3000 * 150.0 * (1 / 150.0))


class TestExposureGroupLimit:
    def test_group_exposure_limit_rejected(self):
        cfg = RiskConfig(max_exposure_per_group=10_000.0)
        eng = RiskEngine(
            cfg,
            {
                "EURUSD": InstrumentSpec(symbol="EURUSD"),
                "GBPUSD": InstrumentSpec(symbol="GBPUSD"),
            },
        )
        existing = [
            {"symbol": "EURUSD", "side": "buy", "quantity": 5_000, "entry_price": 1.05}
        ]
        trade = ProposedTrade(
            symbol="GBPUSD", side=PositionSide.BUY,
            entry_price=1.27, stop_loss=1.26,
        )
        d = eng.evaluate(trade, AccountState(balance=10_000, equity=10_000, open_positions=existing))
        # EURUSD 5250 + GBPUSD ~12700 = ~17950 > 10000 -> rejected.
        assert not d.approved
        assert d.reason.value == "exposure_group_exceeded"

    def test_group_exposure_within_limit(self):
        cfg = RiskConfig(max_exposure_per_group=10_000.0)
        eng = RiskEngine(
            cfg,
            {
                "EURUSD": InstrumentSpec(symbol="EURUSD"),
                "GBPUSD": InstrumentSpec(symbol="GBPUSD"),
            },
        )
        trade = ProposedTrade(
            symbol="GBPUSD", side=PositionSide.BUY,
            entry_price=1.27, stop_loss=1.26,
        )
        d = eng.evaluate(trade, AccountState(balance=10_000, equity=10_000))
        # GBPUSD alone ~12700 > 10000 -> rejected.
        assert not d.approved