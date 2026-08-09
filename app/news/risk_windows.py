"""Event risk-window detection.

Default window sizes are DEVELOPMENT defaults only — not claimed to be optimal.
Pre/post windows are configurable per importance level, and the pair-aware
query answers "is EUR/USD currently inside a high-impact window?" by
considering both EUR and USD events.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.news.models import (
    EconomicEvent,
    EventImportance,
    PairRiskContext,
    RiskWindow,
    RiskWindowStatus,
)

__all__ = [
    "RiskWindowConfig",
    "build_risk_windows",
    "pair_risk_context",
    "window_status_at",
]


@dataclass(frozen=True)
class RiskWindowConfig:
    """Pre/post window sizes in minutes per importance level.

    These are DEFAULT DEVELOPMENT VALUES ONLY. They are not claimed to be
    optimal for any strategy.
    """

    low_pre: int = 5
    low_post: int = 5
    medium_pre: int = 15
    medium_post: int = 15
    high_pre: int = 30
    high_post: int = 30
    unknown_pre: int = 0
    unknown_post: int = 0

    def window_for(self, importance: EventImportance) -> tuple[int, int]:
        """Return (pre_minutes, post_minutes) for an importance level."""
        if importance == EventImportance.HIGH:
            return self.high_pre, self.high_post
        if importance == EventImportance.MEDIUM:
            return self.medium_pre, self.medium_post
        if importance == EventImportance.LOW:
            return self.low_pre, self.low_post
        return self.unknown_pre, self.unknown_post


def build_risk_windows(
    events: list[EconomicEvent],
    config: RiskWindowConfig,
) -> list[RiskWindow]:
    """Build risk windows for all events."""
    windows: list[RiskWindow] = []
    for event in events:
        pre, post = config.window_for(event.importance)
        if pre == 0 and post == 0:
            continue
        windows.append(
            RiskWindow(
                event=event,
                pre_window_minutes=pre,
                post_window_minutes=post,
                start=event.scheduled_at - timedelta(minutes=pre),
                end=event.scheduled_at + timedelta(minutes=post),
            )
        )
    return windows


def window_status_at(window: RiskWindow, timestamp: datetime) -> RiskWindowStatus:
    """Classify a timestamp relative to a risk window."""
    if timestamp < window.start:
        return RiskWindowStatus.BEFORE
    if timestamp <= window.end:
        return RiskWindowStatus.INSIDE
    return RiskWindowStatus.AFTER


def pair_risk_context(
    events: list[EconomicEvent],
    pair: str,
    timestamp: datetime,
    config: RiskWindowConfig,
) -> PairRiskContext:
    """Return structured risk context for a currency pair.

    For a pair like ``EUR/USD``, both EUR events and USD events are
    considered; unrelated currencies are ignored.
    """
    curr_a, curr_b = pair.upper().split("/")
    unique: list[EconomicEvent] = []
    seen: set[str] = set()
    for e in events:
        if e.event_id in seen:
            continue
        affected = [c.upper() for c in e.affected_currencies]
        if e.currency.upper() in (curr_a, curr_b) or curr_a in affected or curr_b in affected:
            seen.add(e.event_id)
            unique.append(e)

    windows = build_risk_windows(unique, config)
    active = [w for w in windows if window_status_at(w, timestamp) == RiskWindowStatus.INSIDE]
    upcoming = [
        e for e in unique
        if e.scheduled_at > timestamp
        and e.effective_available_from() > timestamp
    ]

    time_until = None
    if upcoming:
        time_until = (min(e.scheduled_at for e in upcoming) - timestamp).total_seconds()

    time_since = None
    past = [e for e in unique if e.scheduled_at <= timestamp]
    if past:
        time_since = (timestamp - max(e.scheduled_at for e in past)).total_seconds()

    highest = None
    if active:
        highest = max((w.event.importance for w in active), key=lambda i: _IMPORTANCE_RANK[i])

    if active:
        msg = f"Inside {len(active)} risk window(s) for {pair}."
    elif upcoming:
        mins = time_until / 60 if time_until is not None else 0
        msg = f"Next event for {pair} in ~{mins:.1f} minutes."
    else:
        msg = f"No active or upcoming events for {pair}."

    return PairRiskContext(
        pair=pair,
        timestamp=timestamp,
        active_events=active,
        upcoming_events=upcoming,
        time_until_next=time_until,
        time_since_last=time_since,
        highest_active_importance=highest,
        message=msg,
    )


_IMPORTANCE_RANK = {
    EventImportance.HIGH: 3,
    EventImportance.MEDIUM: 2,
    EventImportance.LOW: 1,
    EventImportance.UNKNOWN: 0,
}
