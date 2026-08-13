"""Deterministic hypothesis definitions for Step 13 Alpha Discovery.

A hypothesis defines a researchable trading setup:

    event type (e.g. liquidity_sweep)
    + conditions (e.g. bullish displacement, H1 bullish bias, London session)
    + entry rule (immediate / displacement confirmation / retest)
    + stop rule (ATR / structural / liquidity)
    + exit rule (fixed R:R / ATR / trailing / time)

The hypothesis has a deterministic SHA-256 hash: the same definition always
produces the same ``hypothesis_id``. No brute-forcing: controlled generation
enforces hard limits on total hypotheses, conditions per hypothesis, and
minimum sample.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# ── Supported event/condition/entry/stop/exit vocabularies ──────────────────

EVENT_TYPES = ("liquidity_sweep", "displacement", "structure_break")

DIRECTIONS = ("long", "short")

CONDITION_FIELDS = (
    "structure_bias",      # bullish | bearish | neutral
    "regime",              # trending | ranging | transition
    "volatility",          # low | normal | high | extreme
    "session",             # asia | europe | newyork | late
    "htf_alignment",       # bullish | bearish | neutral | none
    "displacement_present",
    "structure_break_present",
)

ENTRY_RULES = ("immediate", "displacement_confirmation", "retest")

STOP_RULES = ("atr", "structural", "liquidity")

EXIT_RULES = ("fixed_rr_1.0", "fixed_rr_1.5", "fixed_rr_2.0", "fixed_rr_3.0", "atr", "trailing", "time")


class HypothesisConfigError(ValueError):
    """Raised when a hypothesis definition is invalid."""


@dataclass(frozen=True)
class Hypothesis:
    """A single deterministic research hypothesis."""

    # ── Identity / scope ────────────────────────────────────────────────────
    symbol: str
    timeframe: str
    strategy_family: str

    # ── Event ───────────────────────────────────────────────────────────────
    event_type: str                 # liquidity_sweep | displacement | structure_break
    direction: str                  # long | short

    # ── Conditions (all must hold; max bounded) ──────────────────────────────
    conditions: tuple = ()

    # ── Entry / stop / exit ─────────────────────────────────────────────────
    entry_rule: str = "displacement_confirmation"
    stop_rule: str = "atr"
    exit_rule: str = "fixed_rr_2.0"
    stop_atr_multiple: float = 1.0
    exit_atr_multiple: float = 2.0
    max_holding_bars: int = 0       # 0 = none (time-based exit disabled)

    # ── Post-hoc metadata (not part of the identity) ─────────────────────────
    hypothesis_description: str = ""

    # ── Identity ────────────────────────────────────────────────────────────

    def _identity_dict(self) -> dict[str, Any]:
        """The identity-contributing fields (everything that defines the setup)."""
        return {
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe.upper(),
            "strategy_family": self.strategy_family,
            "event_type": self.event_type,
            "direction": self.direction,
            "conditions": sorted(self.conditions),
            "entry_rule": self.entry_rule,
            "stop_rule": self.stop_rule,
            "exit_rule": self.exit_rule,
            "stop_atr_multiple": self.stop_atr_multiple,
            "exit_atr_multiple": self.exit_atr_multiple,
            "max_holding_bars": self.max_holding_bars,
        }

    @property
    def hypothesis_id(self) -> str:
        """Deterministic hash identifying this hypothesis."""
        raw = json.dumps(
            self._identity_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"hyp_{hashlib.sha256(raw).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        d = self._identity_dict()
        d["hypothesis_id"] = self.hypothesis_id
        d["hypothesis_description"] = self.hypothesis_description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        """Reconstruct a hypothesis from a dict (idempotent / deterministic)."""
        required = ("symbol", "timeframe", "strategy_family", "event_type", "direction")
        missing = [k for k in required if k not in data]
        if missing:
            raise HypothesisConfigError(f"missing required fields: {missing}")
        return cls(
            symbol=str(data["symbol"]),
            timeframe=str(data["timeframe"]),
            strategy_family=str(data["strategy_family"]),
            event_type=str(data["event_type"]),
            direction=str(data["direction"]),
            conditions=tuple(data.get("conditions", ())),
            entry_rule=str(data.get("entry_rule", "displacement_confirmation")),
            stop_rule=str(data.get("stop_rule", "atr")),
            exit_rule=str(data.get("exit_rule", "fixed_rr_2.0")),
            stop_atr_multiple=float(data.get("stop_atr_multiple", 1.0)),
            exit_atr_multiple=float(data.get("exit_atr_multiple", 2.0)),
            max_holding_bars=int(data.get("max_holding_bars", 0)),
            hypothesis_description=str(data.get("hypothesis_description", "")),
        )

    def __hash__(self) -> int:
        return hash(self.hypothesis_id)


# ── Controlled hypothesis generation ────────────────────────────────────────


@dataclass(frozen=True)
class HypothesisGridLimits:
    """Hard safety limits for controlled hypothesis generation."""

    max_hypotheses: int = 200
    max_conditions_per_hypothesis: int = 4
    min_sample_size: int = 30


def generate_hypotheses(
    *,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    event_types: tuple[str, ...] = EVENT_TYPES,
    directions: tuple[str, ...] = DIRECTIONS,
    structure_biases: tuple = ("bullish", "bearish"),
    regimes: tuple = ("trending", "ranging"),
    sessions: tuple = ("europe", "newyork"),
    htf_alignments: tuple = ("bullish", "bearish"),
    entry_rules: tuple = ("displacement_confirmation",),
    exit_rules: tuple = ("fixed_rr_2.0",),
    limits: HypothesisGridLimits | None = None,
) -> list[Hypothesis]:
    """Generate a CONTROLLED set of hypotheses (no brute force).

    Every combination is deterministic (iteration order is stable) and the
    total count is bounded by ``limits.max_hypotheses``.
    """
    limits = limits or HypothesisGridLimits()
    out: list[Hypothesis] = []

    for symbol in symbols:
        for timeframe in timeframes:
            for event_type in event_types:
                for direction in directions:
                    base = Hypothesis(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_family=event_type,
                        event_type=event_type,
                        direction=direction,
                        entry_rule=(entry_rules[0] if entry_rules else "displacement_confirmation"),
                        exit_rule=(exit_rules[0] if exit_rules else "fixed_rr_2.0"),
                    )
                    if len(out) < limits.max_hypotheses:
                        out.append(base)

                    for bias in structure_biases:
                        h = _with_conditions(base, (f"structure_bias={bias}",), limits)
                        if h and len(out) < limits.max_hypotheses:
                            out.append(h)
                    for regime in regimes:
                        h = _with_conditions(base, (f"regime={regime}",), limits)
                        if h and len(out) < limits.max_hypotheses:
                            out.append(h)
                    for session in sessions:
                        h = _with_conditions(base, (f"session={session}",), limits)
                        if h and len(out) < limits.max_hypotheses:
                            out.append(h)
                    for align in htf_alignments:
                        h = _with_conditions(base, (f"htf_alignment={align}",), limits)
                        if h and len(out) < limits.max_hypotheses:
                            out.append(h)

    return out


def _with_conditions(
    base: Hypothesis,
    conditions: tuple[str, ...],
    limits: HypothesisGridLimits,
) -> Hypothesis | None:
    """Return a copy of ``base`` with conditions if under the limit."""
    if len(conditions) > limits.max_conditions_per_hypothesis:
        return None
    return Hypothesis(
        symbol=base.symbol,
        timeframe=base.timeframe,
        strategy_family=base.strategy_family,
        event_type=base.event_type,
        direction=base.direction,
        conditions=conditions,
        entry_rule=base.entry_rule,
        exit_rule=base.exit_rule,
    )


def conditions_pass(hypothesis: Hypothesis, feature_row: dict[str, Any]) -> bool:
    """Return True when a candidate row satisfies all hypothesis conditions.

    ``feature_row`` is a compact candidate feature dict with keys like
    ``feature_structure_bias``, ``feature_htf_trend``, ``feature_volatility``,
    ``feature_session``, ``regime``, etc.
    """
    for cond in hypothesis.conditions:
        field, _, value = cond.partition("=")
        if not value:
            return False
        if field == "structure_bias":
            key = "feature_structure_bias"
        elif field == "htf_alignment":
            key = "feature_htf_trend"
        elif field == "volatility":
            key = "feature_volatility"
        elif field == "session":
            key = "feature_session"
        elif field == "regime":
            key = "regime"
        else:
            key = field
        if feature_row.get(key) != value:
            return False
    return True