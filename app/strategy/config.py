"""Strategy configuration.

All values are DOCUMENTED DEVELOPMENT DEFAULTS, not claimed optimal. Every
strategy parameter must be explicitly configurable and serializable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Shared strategy-layer configuration (development defaults)."""

    # ── Regime requirements ────────────────────────────────────────────────────
    min_regime_strength: float = 0.5  # 0..1 internal agreement; NOT probability
    allowed_market_states: tuple = (
        "trending",
        "transition",
    )

    # ── Structure requirements ─────────────────────────────────────────────────
    min_structure_points: int = 2
    min_consecutive_hh_hl: int = 2  # for bullish confirmation
    min_consecutive_lh_ll: int = 2  # for bearish confirmation

    # ── Displacement / entry confirmation ──────────────────────────────────────
    min_displacement_class: str = "large"  # large|extreme

    # ── Volatility acceptability ──────────────────────────────────────────────
    # Highest volatility state at which a signal may form (extreme disallowed).
    max_volatility_state: str = "high"

    # ── New risk ───────────────────────────────────────────────────────────────
    # highest permitted active news risk state (high disallowed).
    max_news_risk: str = "medium"

    # ── Liquidity ──────────────────────────────────────────────────────────────
    min_liquidity_zones: int = 1
    sweep_lookback_bars: int = 10

    # ── Risk geometry ──────────────────────────────────────────────────────────
    stop_distance_atr: float = 1.0
    reward_risk_target: float = 2.0

    # ── Cooldown / duplicate protection ────────────────────────────────────────
    cooldown_bars: int = 10

    # ── Session restrictions (leave empty = no restriction) ────────────────────
    allowed_hours_utc: tuple = ()  # e.g. (8, 9, 10)

    # ── Scoring thresholds (documented rule-agreement categories) ──────────────
    # Documented scheme: 0–2 WEAK, 3–4 MODERATE, 5+ STRONG (configurable).
    weak_score_threshold: float = 2.0
    moderate_score_threshold: float = 2.0
    strong_score_threshold: float = 5.0

    # ── Multi-timeframe (MTF) requirements ─────────────────────────────────────
    # When mtf_enabled=False (default) the strategy behaves exactly as before —
    # no MTF gate is applied and behavior is identical to prior versions.
    mtf_enabled: bool = False
    # Minimum number of aligning higher-timeframe tiers required.
    mtf_min_aligned: int = 0
    # Require no higher-timeframe conflict with the base direction.
    mtf_require_no_conflict: bool = False
    # Optional hard requirements on higher-timeframe direction.
    mtf_require_htf_bullish: bool = False
    mtf_require_htf_bearish: bool = False
    # Lowest acceptable HTF volatility quality ("" = unrestricted).
    mtf_min_volatility_quality: str = ""

    def to_dict(self) -> dict:
        return {
            "min_regime_strength": self.min_regime_strength,
            "allowed_market_states": list(self.allowed_market_states),
            "min_structure_points": self.min_structure_points,
            "min_consecutive_hh_hl": self.min_consecutive_hh_hl,
            "min_consecutive_lh_ll": self.min_consecutive_lh_ll,
            "min_displacement_class": self.min_displacement_class,
            "max_volatility_state": self.max_volatility_state,
            "max_news_risk": self.max_news_risk,
            "min_liquidity_zones": self.min_liquidity_zones,
            "sweep_lookback_bars": self.sweep_lookback_bars,
            "stop_distance_atr": self.stop_distance_atr,
            "reward_risk_target": self.reward_risk_target,
            "cooldown_bars": self.cooldown_bars,
            "allowed_hours_utc": list(self.allowed_hours_utc),
            "weak_score_threshold": self.weak_score_threshold,
            "moderate_score_threshold": self.moderate_score_threshold,
            "strong_score_threshold": self.strong_score_threshold,
            "mtf_enabled": self.mtf_enabled,
            "mtf_min_aligned": self.mtf_min_aligned,
            "mtf_require_no_conflict": self.mtf_require_no_conflict,
            "mtf_require_htf_bullish": self.mtf_require_htf_bullish,
            "mtf_require_htf_bearish": self.mtf_require_htf_bearish,
            "mtf_min_volatility_quality": self.mtf_min_volatility_quality,
        }
