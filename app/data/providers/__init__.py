"""Provider-specific adapters for real market data.

The rest of the application depends only on
:class:`app.data.provider.BaseMarketDataProvider` and never imports a provider
SDK directly. ``create_provider`` selects an adapter from environment
configuration.
"""

import os

from app.data.provider import BaseMarketDataProvider, MockMarketDataProvider

__all__ = ["MockMarketDataProvider", "create_provider"]


def create_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseMarketDataProvider:
    """Instantiate a provider from environment configuration.

    - ``provider_name`` defaults to ``MARKET_DATA_PROVIDER`` (``mock`` default
      when unset, so unconfigured environments remain deterministic).
    - If the chosen provider's credentials are missing, an informative error
      is raised (never silently falls back to mock).
    """
    raw_name = provider_name or os.getenv("MARKET_DATA_PROVIDER")
    if raw_name is None:
        raw_name = "mock"
    name = raw_name.lower()

    if name == "mock":
        return MockMarketDataProvider()

    if name == "twelvedata":
        from app.data.providers.twelvedata import TwelveDataMarketDataProvider

        token = api_key or os.getenv("MARKET_DATA_API_KEY") or ""
        if not token:
            raise ValueError(
                "MARKET_DATA_API_KEY is required for the Twelve Data provider. "
                "Set it in your environment or .env file (see .env.example)."
            )
        base = base_url or os.getenv("MARKET_DATA_BASE_URL") or "https://api.twelvedata.com"
        return TwelveDataMarketDataProvider(
            api_key=token,
            base_url=base,
        )

    raise ValueError(f"Unknown market-data provider: {name!r}")
