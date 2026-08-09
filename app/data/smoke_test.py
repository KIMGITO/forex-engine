"""Optional real-data smoke test.

Runs ONLY when ``RUN_LIVE_DATA_TESTS=true`` and ``MARKET_DATA_API_KEY`` are
configured. Without credentials, normal unit tests still pass; this module is
never imported in that path.

Usage:
    python -m app.data.smoke_test --symbol EURUSD --timeframe 1h
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

from app.data.providers import create_provider
from app.data.validator import DataValidator

__all__ = ["run_smoke_test"]


def run_smoke_test(
    symbol: str = "EURUSD",
    timeframe: str = "1h",
    count: int = 24,
) -> None:
    """Execute a bounded smoke test against the configured real provider.

    - Authenticates
    - Requests a small, bounded dataset (``count`` candles)
    - Normalizes via the existing DataNormalizer path
    - Validates via the existing DataValidator path
    - Prints a summary
    - Optionally persists the result

    Does NOT print API keys or other secrets.
    """
    provider = create_provider()
    print(f"Provider: {getattr(provider, 'name', type(provider).__name__)}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=count)

    print(f"Requesting {symbol} {timeframe} candles from {start.isoformat()} to {end.isoformat()} ...")
    candles = provider.fetch_candles(symbol, timeframe, start, end)
    print(f"Fetched: {len(candles)} candle(s)")

    if not candles:
        print("No candles returned; smoke test inconclusive.")
        return

    # Validation via the existing pipeline

    from app.data.normalizer import DataNormalizer

    df = DataNormalizer.candles_to_df(candles)
    DataValidator.validate_dataframe(df)
    print("Validation: PASS")

    latest = max(c.timestamp for c in candles)
    oldest = min(c.timestamp for c in candles)
    print(f"Oldest: {oldest.isoformat()} | Latest: {latest.isoformat()}")
    print("Smoke test: PASS")


def _cli_main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-data smoke test (creds-gated)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--count", type=int, default=24)
    # parse_known_args so pytest's own args don't break the CLI when invoked
    # from within a test process.
    args, _ = parser.parse_known_args(argv)

    if not os.getenv("RUN_LIVE_DATA_TESTS") or not os.getenv("MARKET_DATA_API_KEY"):
        print("Smoke test skipped: RUN_LIVE_DATA_TESTS=true and MARKET_DATA_API_KEY required.")
        return 0

    try:
        run_smoke_test(args.symbol, args.timeframe, args.count)
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke test FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())