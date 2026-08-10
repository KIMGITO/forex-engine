"""Strict multi-timeframe candle alignment.

THE COMPLETED-CANDLE RULE
-------------------------
A lower-timeframe observation at time ``T`` may ONLY use a higher-timeframe
candle that was FULLY COMPLETED at or before ``T``.

For a timeframe with period ``P`` minutes:
- the candle currently open at ``T`` starts at ``floor(T / P) * P`` and is
  NOT available until ``floor(T / P) * P + P``.
- the LAST COMPLETED candle slot is therefore:
    open  = floor(T / P) * P - P
    close = floor(T / P) * P
    available_from = close

Example (P = 60, H1), T = 09:45:
  floor(585 / 60) = 9 → current open 09:00 (available 10:00)
  last completed = 08:00–09:00, available_from 09:00 ✓
  never 09:00–10:00 (incomplete).

Example (P = 240, H4), T = 12:00:
  floor(720 / 240) = 3 → current open 12:00 (available 16:00)
  last completed = 08:00–12:00, available_from 12:00 ✓
  at 12:00 the 08:00–12:00 H4 candle becomes available.

All timestamps are normalized to UTC before the floor math. Gaps / missing
candles are handled by a bounded lookback to the previous completed slot;
if none is found the window is ``present=False`` — data is NEVER fabricated.
"""

from datetime import datetime, timedelta, timezone

from app.mtf.config import MtfConfig
from app.mtf.models import MtfWindow

__all__ = [
    "completed_slot_close",
    "completed_slot_open",
    "latest_completed_candle_open",
    "resolve_window",
    "timeframe_to_minutes",
]

# Standard timeframe → minutes (development defaults; overridable).
_STD_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "45m": 45,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


def _as_utc_ts(value) -> datetime:
    """Return a tz-aware UTC datetime for a datetime or pandas Timestamp."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        # Treat naive as UTC (data layer guarantees tz-aware; defensive only).
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def timeframe_to_minutes(timeframe: str, config: MtfConfig) -> int:
    """Resolve a timeframe string to minutes (with custom overrides)."""
    if config.custom_timeframe_minutes:
        v = config.custom_timeframe_minutes.get(timeframe)
        if v is not None:
            return int(v)
    if timeframe in _STD_MINUTES:
        return _STD_MINUTES[timeframe]
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def completed_slot_open(observation: datetime, period_minutes: int) -> datetime:
    """Open (UTC) of the last fully-completed candle slot before ``observation``."""
    t = _as_utc_ts(observation)
    # Current open slot (may be incomplete).
    current_epoch = int(t.timestamp())
    period_epoch = period_minutes * 60
    current_open_epoch = (current_epoch // period_epoch) * period_epoch
    completed_open_epoch = current_open_epoch - period_epoch
    return datetime.fromtimestamp(completed_open_epoch, tz=timezone.utc)


def completed_slot_close(observation: datetime, period_minutes: int) -> datetime:
    """Close (UTC) = available_from of the last fully-completed candle slot."""
    open_ts = completed_slot_open(observation, period_minutes)
    return _as_utc_ts(open_ts) + timedelta(minutes=period_minutes)


def resolve_window(
    timeframe: str,
    observation: datetime,
    config: MtfConfig,
) -> MtfWindow:
    """Compute the completed-candle window for ``timeframe`` at ``observation``.

    Returns the window metadata only (open/close/available_from). The actual
    candle lookup happens against a frame via ``latest_completed_candle_open``.
    """
    period = timeframe_to_minutes(timeframe, config)
    open_ts = completed_slot_open(observation, period)
    close_ts = completed_slot_close(observation, period)
    return MtfWindow(
        timeframe=timeframe,
        open=open_ts,
        close=close_ts,
        available_from=close_ts,
        present=True,
    )


def latest_completed_candle_open(
    timeframe: str,
    observation: datetime,
    frame,
    config: MtfConfig,
) -> datetime | None:
    """Return the open (UTC) of the latest fully-completed candle at ``T``.

    ``frame`` is a DataFrame indexed by tz-aware candle-open timestamps for
    ``timeframe``. Walks backward through completed slots (bounded by
    ``max_gap_lookback``) to bridge gaps/weekend breaks. Returns ``None`` when
    no completed candle exists (never fabricated).
    """
    obs = _as_utc_ts(observation)
    period = timeframe_to_minutes(timeframe, config)
    slot_open = completed_slot_open(obs, period)

    for step in range(config.max_gap_lookback + 1):
        target_open = slot_open - timedelta(minutes=period * step)
        # Find the latest candle whose open <= target_open (i.e. completed
        # by the time target_open+period passed). Because the frame is sorted,
        # we take the largest open <= target_open.
        candidates = frame.index[frame.index <= target_open]
        if len(candidates) == 0:
            continue
        # The latest such candle's open is what it was (could be < target_open,
        # indicating a gap, but it is legally completed since its own period
        # closed before/at target_open).
        return _as_utc_ts(candidates[-1])
    return None