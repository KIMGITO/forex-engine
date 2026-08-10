"""Multi-timeframe alignment logic — explainable, no magic scores.

Alignment is derived strictly from per-timeframe evidence (regime trend +
structural bias per tier) and the base setup direction. Every classification
is explainable from its underlying evidence; ``alignment_reasons`` lists the
exact reasons.

- All known HTF directions == base direction  → ALIGNED_LONG / ALIGNED_SHORT
- Any known HTF direction != base direction   → CONFLICTED
- fewer than ``min_aligned`` known HTF tiers  → UNKNOWN

The ``strength`` on each TimeframeContext is the fraction of known tiers that
agree with the base — it is ALIGNMENT AGREEMENT, never a probability.
"""


from app.mtf.models import MtfAlignmentState, TimeframeContext

__all__ = ["classify_alignment", "compute_strength", "tier_direction"]


def tier_direction(tier: TimeframeContext) -> str | None:
    """Derive a tier's directional bias from regime + structure evidence.

    Returns 'long' / 'short' / 'neutral' / None.
    Both regime trend_state and structural_bias must be known and agree to
    assert a direction; otherwise the tier is neutral/unknown.
    """
    trend = (tier.trend_state or "").lower()
    bias = (tier.structural_bias or "").lower()

    # Both known and agree → bullish/bearish.
    if trend == "bullish" and bias == "bullish":
        return "long"
    if trend == "bearish" and bias == "bearish":
        return "short"

    # Mixed evidence → neutral (not a directional claim).
    return "neutral"


def compute_strength(base_direction: str, known_directions: list[str | None]) -> float:
    """Fraction of known (non-neutral, non-None) tier directions == base.

    Documented as ALIGNMENT AGREEMENT, never probability.
    """
    known = [d for d in known_directions if d in ("long", "short")]
    if not known:
        return 0.0
    matches = sum(1 for d in known if d == base_direction)
    return matches / len(known)


def classify_alignment(
    base_direction: str,
    hierarchy: list[TimeframeContext],
    min_aligned: int,
    require_no_htf_conflict: bool = False,
) -> tuple[MtfAlignmentState, list[str], float]:
    """Classify MTF alignment from the base direction and HTF hierarchy.

    Returns ``(state, reasons, agreement)``.
    """
    htf = hierarchy
    if not htf:
        return MtfAlignmentState.UNKNOWN, ["no higher-timeframe tiers"], 0.0

    directions: list[str | None] = [tier_direction(t) for t in htf]
    known = [d for d in directions if d in ("long", "short")]
    reasons: list[str] = []

    if len(known) < min_aligned:
        reasons.append(
            f"only {len(known)} known HTF direction(s); min_aligned={min_aligned}"
        )
        return (
            MtfAlignmentState.UNKNOWN,
            reasons,
            compute_strength(base_direction, directions),
        )

    agreement = compute_strength(base_direction, directions)

    # Conflict detection: any known HTF direction opposite to base.
    if any(d != base_direction for d in known):
        reasons.append(f"higher-timeframe conflict with base={base_direction}")
        return MtfAlignmentState.CONFLICTED, reasons, agreement

    # All known agree with base.
    if base_direction == "long":
        state = MtfAlignmentState.ALIGNED_LONG
    else:
        state = MtfAlignmentState.ALIGNED_SHORT
    reasons.append(f"base={base_direction} aligned with {len(known)} HTF tier(s)")
    return state, reasons, agreement