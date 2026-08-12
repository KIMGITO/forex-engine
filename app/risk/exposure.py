"""Exposure computation and conservative exposure groups.

The risk engine uses a **conservative grouping** mechanism based on the quote
currency of each pair. This is deliberately NOT a statistical correlation
model: currency grouping is not equivalent to historical correlation. Real
statistical correlation is computed by ``app.features.correlation`` and can be
wired in later; this module only provides coarse caps the operator opts into
via ``RiskConfig.max_exposure_per_group``.
"""

from __future__ import annotations

from app.risk.config import RiskConfig
from app.risk.instrument import InstrumentSpec
from app.risk.models import ExposureGroup


def exposure_group_for(
    symbol: str,
    config: RiskConfig,
) -> ExposureGroup:
    """Return the exposure group for a symbol (override-aware).

    ``RiskConfig.exposure_groups`` provides explicit group overrides; when a
    symbol is not overridden, :meth:`ExposureGroup.from_symbol` is used.
    """
    override = config.exposure_groups.get(symbol.upper())
    if override is not None:
        try:
            return ExposureGroup(override)
        except ValueError:
            return ExposureGroup.UNKNOWN
    return ExposureGroup.from_symbol(symbol)


def _position_notional(
    p: dict,
    specs: dict[str, InstrumentSpec],
) -> float:
    """Notional of one position in account currency.

    In the absence of live quote->account conversion for each pair, USD-quote
    pairs (EURUSD, GBPUSD, AUDUSD, USDCAD) use 1.0; USDJPY/USDCHF are computed
    from the supplied instrument spec's ``quote_to_account``. This is a
    conservative approximation: cross-currency math requires a live rate feed
    that the broker-independent engine must not fabricate.
    """
    qty = float(p.get("quantity", 0.0))
    entry = float(p.get("entry_price", 0.0))
    spec = specs.get(str(p.get("symbol", "")).upper())
    factor = spec.quote_to_account if spec else 1.0
    return qty * entry * factor


def symbol_exposure(
    positions: list[dict],
    symbol: str,
    config: RiskConfig,
    specs: dict[str, InstrumentSpec],
) -> float:
    """Current notional exposure to ``symbol`` in account currency."""
    total = 0.0
    for p in positions:
        if p.get("symbol", "").upper() != symbol.upper():
            continue
        total += _position_notional(p, specs)
    return total


def total_exposure(
    positions: list[dict],
    config: RiskConfig,
    specs: dict[str, InstrumentSpec],
) -> float:
    """Total notional exposure across all open positions in account currency."""
    return sum(
        _position_notional(p, specs)
        for p in positions
    )


def group_exposure(
    positions: list[dict],
    group: ExposureGroup,
    config: RiskConfig,
    specs: dict[str, InstrumentSpec],
) -> float:
    """Total notional exposure for all symbols in ``group``."""
    total = 0.0
    for p in positions:
        sym = str(p.get("symbol", ""))
        if exposure_group_for(sym, config) == group:
            total += _position_notional(p, specs)
    return total