"""Multi-timeframe configuration.

All values are DOCUMENTED DEVELOPMENT DEFAULTS. The timeframe hierarchy is
explicitly configurable — never hard-coded to one combination. Multi-timeframe
context is analytical information, not a prediction guarantee.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MtfConfig:
    """Configuration for the multi-timeframe research engine."""

    # ── Timeframe hierarchy ────────────────────────────────────────────────────
    # The base (acting) timeframe the strategy runs on.
    base_timeframe: str = "15m"
    # Higher timeframes, ordered NEAREST → FARTHEST (e.g. H1, H4, D1).
    higher_timeframes: tuple = ("1h", "4h", "1d")

    # ── Alignment requirements ─────────────────────────────────────────────────
    # Minimum number of higher-timeframe contexts that must produce a known
    # (non-neutral) direction for a classification to be other than UNKNOWN.
    min_aligned: int = 1
    # When True, any higher-timeframe direction opposite to the base direction
    # forces CONFLICTED (otherwise CONFLICTED still triggers when all known).
    require_no_htf_conflict: bool = False
    # Optional hard requirements on the nearest higher timeframe.
    require_htf_bullish: bool = False
    require_htf_bearish: bool = False

    # ── Volatility quality gate ────────────────────────────────────────────────
    # Lowest acceptable volatility state among present HTF tiers
    # ("" = no restriction; "low"/"normal"/"high" allowed).
    min_volatility_quality: str = ""

    # ── News-risk gates (highest permitted state at the base moment) ───────────
    max_base_news_risk: str = "medium"
    max_htf_news_risk: str = "medium"

    # ── Missing-data behavior ──────────────────────────────────────────────────
    # How many HTF tiers may be entirely absent (no completed candle within the
    # bounded lookback) before the MTF context is considered unusable.
    max_missing_htf_allowed: int = 0
    # Bounded lookback (in candle periods) when stepping back past a gap.
    max_gap_lookback: int = 5

    # ── Custom timeframe periods (minutes) ─────────────────────────────────────
    # Optional override for non-standard timeframe strings. Defaults cover the
    # standard set used by the data layer.
    custom_timeframe_minutes: dict | None = None

    def to_dict(self) -> dict:
        return {
            "base_timeframe": self.base_timeframe,
            "higher_timeframes": list(self.higher_timeframes),
            "min_aligned": self.min_aligned,
            "require_no_htf_conflict": self.require_no_htf_conflict,
            "require_htf_bullish": self.require_htf_bullish,
            "require_htf_bearish": self.require_htf_bearish,
            "min_volatility_quality": self.min_volatility_quality,
            "max_base_news_risk": self.max_base_news_risk,
            "max_htf_news_risk": self.max_htf_news_risk,
            "max_missing_htf_allowed": self.max_missing_htf_allowed,
            "max_gap_lookback": self.max_gap_lookback,
            "custom_timeframe_minutes": self.custom_timeframe_minutes,
        }