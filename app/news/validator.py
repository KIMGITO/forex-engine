"""Validation of economic events.

Covers timestamp sanity, required fields, numeric sanity, importance validity,
and duplicate detection. Suspicious records are reported via typed errors —
never silently discarded.
"""


from app.news.errors import ValidationError
from app.news.models import EconomicEvent, EventImportance

__all__ = ["detect_duplicates", "validate_event", "validate_events"]


def validate_event(event: EconomicEvent) -> None:
    """Validate a single event. Raises ValidationError on any problem."""
    if not event.event_id:
        raise ValidationError("event_id must not be empty.")
    if not event.event_name.strip():
        raise ValidationError(f"event_name must not be empty (event_id={event.event_id}).")
    if not event.country.strip():
        raise ValidationError(f"country must not be empty (event_id={event.event_id}).")
    if event.scheduled_at.tzinfo is None:
        raise ValidationError(
            f"scheduled_at must be timezone-aware (event_id={event.event_id})."
        )
    if event.scheduled_at.year < 1900 or event.scheduled_at.year > 2200:
        raise ValidationError(
            f"Impossible scheduled_at for event '{event.event_id}': {event.scheduled_at}"
        )
    if event.released_at is not None and event.released_at < event.scheduled_at:
        raise ValidationError(
            f"released_at before scheduled_at for event '{event.event_id}'."
        )
    if event.importance not in tuple(EventImportance):
        raise ValidationError(f"Invalid importance for event '{event.event_id}'.")


def detect_duplicates(events: list[EconomicEvent]) -> list[list[EconomicEvent]]:
    """Return groups of events sharing the same (provider, event_id).

    Only records that actually specify a provider are deduplicated; records
    without a provider are not considered duplicates of one another.
    """
    groups: dict = {}
    for event in events:
        if event.provider is None:
            continue
        key = (event.provider, event.event_id)
        groups.setdefault(key, []).append(event)
    return [g for g in groups.values() if len(g) > 1]


def validate_events(events: list[EconomicEvent]) -> None:
    """Validate a collection. Raises ValidationError on the first problem,
    or a ValidationError describing duplicate groups."""
    for event in events:
        validate_event(event)
    dupes = detect_duplicates(events)
    if dupes:
        ids = ", ".join(f"({e.provider}, {e.event_id})" for e in dupes[0])
        raise ValidationError(f"Duplicate economic events detected: {ids}")
