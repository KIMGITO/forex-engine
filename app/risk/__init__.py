"""Broker-independent Risk Management Engine (Step 15).

Architecture:

    Signal -> Risk Engine -> RiskDecision (approved/rejected) -> Order

The engine controls exposure and decides whether a proposed trade is allowed.
It never decides whether a signal is profitable.
"""

from app.risk.config import RiskConfig
from app.risk.engine import RiskEngine
from app.risk.errors import RiskError
from app.risk.instrument import InstrumentSpec, default_specs, position_size_for_risk
from app.risk.models import (
    AccountState,
    ExposureGroup,
    PositionSide,
    ProposedTrade,
    RejectionReason,
    RiskDecision,
    RiskDecisionType,
)

__all__ = [
    "AccountState",
    "ExposureGroup",
    "InstrumentSpec",
    "PositionSide",
    "ProposedTrade",
    "RejectionReason",
    "RiskConfig",
    "RiskDecision",
    "RiskDecisionType",
    "RiskEngine",
    "RiskError",
    "default_specs",
    "position_size_for_risk",
]