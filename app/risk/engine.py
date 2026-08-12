"""Broker-independent Risk Management Engine.

Pipeline (conceptual):

    ProposedTrade
        -> validate trade
        -> resolve instrument spec
        -> calculate stop distance
        -> calculate monetary risk
        -> calculate position size
        -> check per-trade risk
        -> check daily loss
        -> check drawdown
        -> check open positions / duplicate
        -> check symbol exposure
        -> check total exposure
        -> check exposure group
        -> check emergency stop
        -> RiskDecision

The engine is a pure domain component: no broker API, no HTTP, no UI, no
database. It returns a structured :class:`RiskDecision` with a typed
rejection reason whenever a trade is not allowed.
"""

from __future__ import annotations

from app.risk.config import RiskConfig
from app.risk.exposure import group_exposure, symbol_exposure, total_exposure
from app.risk.instrument import InstrumentSpec, position_size_for_risk
from app.risk.models import (
    AccountState,
    ExposureGroup,
    PositionSide,
    ProposedTrade,
    RejectionReason,
    RiskDecision,
    RiskDecisionType,
)


class RiskEngine:
    """Deterministic risk gate between signals and the execution layer."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        instruments: dict[str, InstrumentSpec] | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.instruments = instruments or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        trade: ProposedTrade,
        account: AccountState,
    ) -> RiskDecision:
        """Evaluate a proposed trade against the account and limits."""
        # 1. Basic trade validation.
        if (
            trade.entry_price <= 0
            or trade.stop_loss <= 0
            or trade.entry_price == trade.stop_loss
        ):
            return self._reject(
                RejectionReason.INVALID_TRADE, "invalid entry/stop levels"
            )

        # 2. Instrument resolution.
        spec = self._instrument_for(trade.symbol)
        if spec is None:
            return self._reject(
                RejectionReason.INVALID_INSTRUMENT,
                f"no instrument spec configured for {trade.symbol}",
            )
        if (
            ExposureGroup.from_symbol(trade.symbol) is ExposureGroup.UNKNOWN
            and not self.config.allow_unknown_symbols
        ):
            return self._reject(
                RejectionReason.INVALID_INSTRUMENT,
                f"unsupported symbol {trade.symbol}",
            )

        # 3. Stop-loss on wrong side.
        if not self._stop_on_correct_side(trade):
            return self._reject(
                RejectionReason.STOP_ON_WRONG_SIDE,
                "stop-loss is on the wrong side of the entry",
            )

        # 4. Daily loss limit.
        if self.config.max_daily_loss_pct is not None:
            daily_limit = account.equity * self.config.max_daily_loss_pct
            if account.daily_pnl <= -daily_limit:
                return self._reject(
                    RejectionReason.DAILY_LOSS_LIMIT_EXCEEDED,
                    "daily loss limit reached",
                )

        # 5. Drawdown limit.
        dd = self._drawdown_pct(account)
        if (
            self.config.max_drawdown_pct is not None
            and dd is not None
            and dd >= self.config.max_drawdown_pct
        ):
            return self._reject(
                RejectionReason.DRAWDOWN_LIMIT_EXCEEDED,
                f"drawdown {dd:.2%} >= limit {self.config.max_drawdown_pct:.2%}",
            )

        # 6. Open positions / duplicate.
        if len(account.open_positions) >= self.config.max_open_positions:
            return self._reject(
                RejectionReason.MAX_OPEN_POSITIONS_REACHED,
                f"open positions {len(account.open_positions)} >= "
                f"{self.config.max_open_positions}",
            )
        if self.config.prevent_duplicate_position and self._has_duplicate(
            trade, account
        ):
            return self._reject(
                RejectionReason.DUPLICATE_POSITION,
                f"duplicate {trade.symbol} {trade.side.value} position",
            )

        # 7. Position sizing.
        try:
            units, monetary_risk = position_size_for_risk(
                account.equity,
                self.config.risk_percent,
                trade.entry_price,
                trade.stop_loss,
                spec,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as structured rejection
            return self._reject(
                RejectionReason.INVALID_TRADE, f"position sizing failed: {exc}"
            )

        if units <= 0:
            return self._reject(
                RejectionReason.POSITION_SIZE_TOO_SMALL,
                "calculated position size is zero or below minimum",
            )
        if (
            self.config.min_position_units > 0
            and units < self.config.min_position_units
        ):
            return self._reject(
                RejectionReason.POSITION_SIZE_TOO_SMALL,
                f"position {units:.0f} units < minimum "
                f"{self.config.min_position_units:.0f}",
            )
        if (
            self.config.max_position_units is not None
            and units > self.config.max_position_units
        ):
            return self._reject(
                RejectionReason.POSITION_SIZE_TOO_LARGE,
                f"position {units:.0f} units > maximum "
                f"{self.config.max_position_units:.0f}",
            )

        # 8. Per-trade risk cap (monetary).
        if (
            self.config.max_risk_per_trade is not None
            and monetary_risk > self.config.max_risk_per_trade
        ):
            return self._reject(
                RejectionReason.PER_TRADE_RISK_EXCEEDED,
                f"risk {monetary_risk:.2f} > cap {self.config.max_risk_per_trade:.2f}",
            )

        # 9. Margin check (only when the account explicitly exposes free margin).
        #    The engine has no broker leverage knowledge, so the check is a
        #    strict "new notional must fit in available margin" comparison when
        #    the caller supplies margin fields.
        if account.available_margin > 0:
            new_notional = units * trade.entry_price * (
                spec.quote_to_account or 1.0
            )
            if new_notional > account.available_margin:
                return self._reject(
                    RejectionReason.INSUFFICIENT_MARGIN,
                    "insufficient free margin for proposed size",
                )

        # 10. Exposure checks.
        notional = units * trade.entry_price * (spec.quote_to_account or 1.0)
        sym_exp = symbol_exposure(
            account.open_positions, trade.symbol, self.config, self.instruments
        )
        new_sym_exp = sym_exp + notional
        if (
            self.config.max_symbol_exposure is not None
            and new_sym_exp > self.config.max_symbol_exposure
        ):
            return self._reject(
                RejectionReason.SYMBOL_EXPOSURE_EXCEEDED,
                f"symbol exposure {new_sym_exp:.2f} > "
                f"{self.config.max_symbol_exposure:.2f}",
            )

        tot_exp = total_exposure(
            account.open_positions, self.config, self.instruments
        ) + notional
        if (
            self.config.max_total_exposure is not None
            and tot_exp > self.config.max_total_exposure
        ):
            return self._reject(
                RejectionReason.TOTAL_EXPOSURE_EXCEEDED,
                f"total exposure {tot_exp:.2f} > {self.config.max_total_exposure:.2f}",
            )

        group = ExposureGroup.from_symbol(trade.symbol)
        if self.config.max_exposure_per_group is not None:
            grp_exp = group_exposure(
                account.open_positions, group, self.config, self.instruments
            ) + notional
            if grp_exp > self.config.max_exposure_per_group:
                return self._reject(
                    RejectionReason.EXPOSURE_GROUP_EXCEEDED,
                    f"group {group.value} exposure {grp_exp:.2f} > "
                    f"{self.config.max_exposure_per_group:.2f}",
                )

        # 11. Emergency stop.
        if self.config.emergency_stop:
            return self._reject(
                RejectionReason.EMERGENCY_STOP,
                "emergency stop is active",
            )

        return RiskDecision(
            type=RiskDecisionType.APPROVED,
            position_size=units,
            monetary_risk=monetary_risk,
            risk_percent=self.config.risk_percent,
            exposure_after=(
                total_exposure(
                    account.open_positions, self.config, self.instruments
                )
                + notional
            ),
            limits={
                "risk_percent": self.config.risk_percent,
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_open_positions": self.config.max_open_positions,
                "max_symbol_exposure": self.config.max_symbol_exposure,
                "max_total_exposure": self.config.max_total_exposure,
                "max_exposure_per_group": self.config.max_exposure_per_group,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reject(reason: RejectionReason, message: str) -> RiskDecision:
        return RiskDecision(
            type=RiskDecisionType.REJECTED,
            reason=reason,
            message=message,
        )

    def _instrument_for(self, symbol: str) -> InstrumentSpec | None:
        return self.instruments.get(symbol.upper())

    @staticmethod
    def _stop_on_correct_side(trade: ProposedTrade) -> bool:
        if trade.side == PositionSide.BUY:
            return trade.stop_loss < trade.entry_price
        return trade.stop_loss > trade.entry_price

    @staticmethod
    def _drawdown_pct(account: AccountState) -> float | None:
        if account.drawdown_pct is not None:
            return account.drawdown_pct
        if account.peak_equity <= 0:
            return None
        dd = (account.peak_equity - account.equity) / account.peak_equity
        return max(0.0, dd)

    @staticmethod
    def _has_duplicate(trade: ProposedTrade, account: AccountState) -> bool:
        for p in account.open_positions:
            if (
                p.get("symbol", "").upper() == trade.symbol.upper()
                and p.get("side", "").lower() == trade.side.value
            ):
                return True
        return False

