"""Multi-Timeframe Research & Signal Context Engine.

Provides strictly causal higher-timeframe context for strategies. Timeframe
alignment follows the completed-candle rule, so a lower-timeframe observation
never sees an unfinished higher-timeframe candle.

Multi-timeframe context is ANALYTICAL INFORMATION — not a prediction guarantee.
"""

from app.mtf.alignment import (
    classify_alignment,
    compute_strength,
    tier_direction,
)
from app.mtf.availability import (
    completed_slot_close,
    completed_slot_open,
    latest_completed_candle_open,
    resolve_window,
    timeframe_to_minutes,
)
from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.mtf.engine import MtfAnalysis, MtfEngine
from app.mtf.models import (
    MtfAlignmentState,
    MtfContext,
    MtfWindow,
    TimeframeContext,
)

__all__ = [
    "MtfAlignmentState",
    "MtfAnalysis",
    "MtfConfig",
    "MtfContext",
    "MtfContextBuilder",
    "MtfEngine",
    "MtfWindow",
    "TimeframeContext",
    "classify_alignment",
    "completed_slot_close",
    "completed_slot_open",
    "compute_strength",
    "latest_completed_candle_open",
    "resolve_window",
    "tier_direction",
    "timeframe_to_minutes",
]