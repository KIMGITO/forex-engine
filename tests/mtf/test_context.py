"""Tests for the MtfContextBuilder per-timeframe context envelope."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.mtf.config import MtfConfig
from app.mtf.context import MtfContextBuilder
from app.mtf.models import TimeframeContext


def _frame(n=200, freq="1h"):
    idx = pd.date_range("2026-08-01", periods=n, freq=freq, tz="UTC")
    close = np.random.default_rng(7).normal(1.08, 0.01, n)
    return pd.DataFrame(
        {"open": close, "high": close + 0.002, "low": close - 0.002, "close": close},
        index=idx,
    )


class TestMtfContextBuilder:
    def test_build_returns_timeframe_context(self):
        cfg = MtfConfig()
        builder = MtfContextBuilder(cfg, "EURUSD")
        frame = _frame()
        ts = frame.index[30]
        tc = builder.build(
            "1h", ts, frame, features=None, structure=None, regimes=None, news_events=None
        )
        assert isinstance(tc, TimeframeContext)
        assert tc.timeframe == "1h"
        assert tc.present is True
        assert tc.available_from is not None

    def test_build_missing_returns_present_false(self):
        cfg = MtfConfig()
        builder = MtfContextBuilder(cfg, "EURUSD")
        # Observation before any data → no completed candle.
        frame = _frame()
        early = frame.index[0] - pd.Timedelta(days=1)
        tc = builder.build(
            "1h", early, frame, features=None, structure=None, regimes=None, news_events=None
        )
        assert tc.present is False

    def test_build_does_not_use_incomplete_candle(self):
        """An H1 observation at 09:45 must not see the 09:00 candle."""
        idx = pd.date_range("2026-08-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": [1.0] * 100, "high": [1.1] * 100, "low": [0.9] * 100, "close": [1.05] * 100},
            index=idx,
        )
        cfg = MtfConfig()
        builder = MtfContextBuilder(cfg, "EURUSD")
        obs_ts = datetime(2026, 8, 3, 9, 45, tzinfo=timezone.utc)
        tc = builder.build(
            "1h", obs_ts, df, features=None, structure=None, regimes=None, news_events=None
        )
        # The completed candle open must be 08:00, never 09:00.
        assert tc.candle_open is not None
        assert tc.candle_open == datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)

    def test_news_filtered_by_available_from(self):
        """News with available_from in the future must not appear."""
        from app.news.models import EconomicEvent, EventImportance

        now_event = EconomicEvent(
            event_id="now", scheduled_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            country="US", currency="USD", event_name="CPI", importance=EventImportance.HIGH,
            available_from=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        future_event = EconomicEvent(
            event_id="future", scheduled_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
            country="US", currency="USD", event_name="NFP", importance=EventImportance.HIGH,
            available_from=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )
        cfg = MtfConfig()
        builder = MtfContextBuilder(cfg, "EURUSD")
        frame = _frame()
        obs = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
        tc = builder.build(
            "1h", obs, frame, features=None, structure=None,
            regimes=None, news_events=[now_event, future_event],
        )
        # Only the now_event (available 08-01) is legal at obs (08-02).
        assert tc.news_risk_max == "high"  # from now_event; future_event excluded