"""Tests that news risk is contextual only and never becomes directional bias."""

from datetime import datetime, timezone

from app.market_structure.models import MarketStructureResult
from app.regime import RegimeEngine
from app.regime.models import NewsRiskState


def _make_structure_result() -> MarketStructureResult:
    from datetime import datetime, timezone

    from app.market_structure.models import (
        StructurePoint,
        StructureType,
        Swing,
        SwingType,
    )

    ts = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)

    swing = Swing(
        symbol="EURUSD", timeframe="1h", timestamp=ts,
        swing_type=SwingType.HIGH, price=1.10,
        confirmation_timestamp=ts, available_from=ts, left=3, right=3,
    )
    point = StructurePoint(
        symbol="EURUSD", timeframe="1h", timestamp=ts,
        structure_type=StructureType.HIGHER_HIGH, price=1.11,
        prior_price=1.10, available_from=ts,
    )
    return MarketStructureResult(
        symbol="EURUSD", timeframe="1h",
        swings=[swing], structure=[point], breaks=[],
        liquidity_zones=[], sweeps=[], displacement=[], ranges=[],
    )


class TestNewsContextual:
    def test_news_does_not_flip_direction(self) -> None:
        engine = RegimeEngine()
        # No news context: no news flag.
        result = engine.analyze(
            _make_ohlc(), "EURUSD", "1h",
            market_structure=_make_structure_result(),
        )
        # With a high-impact news context, the direction must stay the same.
        # (We only assert here that the enum is honored; the news flag is
        # metadata. Use a non-high NEWS to ensure direction unchanged.)
        # Note: news_context is the PairRiskContext type; we construct from news layer.
        from app.news.models import (
            EconomicEvent,
            EventImportance,
            PairRiskContext,
        )
        from app.news.risk_windows import RiskWindowConfig, build_risk_windows

        ev = EconomicEvent(
            event_id="e1",
            scheduled_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
            country="US", currency="USD", event_name="CPI",
            importance=EventImportance.HIGH,
        )
        windows = build_risk_windows([ev], RiskWindowConfig())
        ctx = PairRiskContext(
            pair="EUR/USD",
            timestamp=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
            active_events=windows,
            highest_active_importance=EventImportance.HIGH,
            message="inside high impact",
        )
        result_news = engine.analyze(
            _make_ohlc(), "EURUSD", "1h",
            market_structure=_make_structure_result(),
            news_context=ctx,
        )
        # The news flag is ACTIVE_HIGH but direction states are unchanged.
        assert result_news[-1].news_risk == NewsRiskState.ACTIVE_HIGH
        assert result_news[-1].trend_state == result[-1].trend_state
        assert result_news[-1].volatility_state == result[-1].volatility_state


def _make_ohlc(n=200):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close},
        index=idx,
    )
