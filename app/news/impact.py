"""Economic surprise calculation.

Computes the *objective* difference between actual and forecast. No directional
market claim is made: a positive surprise is not asserted to be bullish and a
negative surprise is not asserted to be bearish. Market reaction depends on
many factors outside this module.
"""


from app.news.models import EconomicEvent, SurpriseResult

__all__ = ["calculate_surprise", "calculate_surprises"]


def calculate_surprise(event: EconomicEvent) -> SurpriseResult:
    """Compute the surprise for a single event.

    The surprise is ``actual - forecast``. When ``forecast`` is missing or
    zero, ``surprise_pct`` is ``None``. When ``actual`` is missing, a
    :class:`ValueError` is raised (you cannot compute a surprise without an
    actual release).
    """
    if event.actual is None:
        raise ValueError(
            f"Cannot compute surprise for event '{event.event_id}': actual is missing."
        )
    forecast = event.forecast

    surprise: float | None = None
    surprise_pct: float | None = None

    if forecast is not None:
        surprise = event.actual - forecast
        if forecast != 0:
            surprise_pct = (event.actual - forecast) / abs(forecast) * 100.0

    return SurpriseResult(
        event_id=event.event_id,
        actual=event.actual,
        forecast=forecast,
        surprise=surprise,
        surprise_pct=surprise_pct,
    )


def calculate_surprises(events: list[EconomicEvent]) -> list[SurpriseResult]:
    """Compute surprises for a batch of events (skipping missing actuals)."""
    results: list[SurpriseResult] = []
    for event in events:
        if event.actual is None:
            continue
        results.append(calculate_surprise(event))
    return results
