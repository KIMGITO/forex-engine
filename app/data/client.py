"""Thin HTTP client for provider adapters."""

import logging
import time
from typing import Any

import requests

from app.data.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    ProviderServerError,
    RateLimitError,
    RetryExhaustedError,
    UnavailableSymbolError,
    UnavailableTimeframeError,
)

logger = logging.getLogger(__name__)

__all__ = ["HttpClient"]


class HttpClient:
    """Retrying, rate-limit-aware HTTP client."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 16.0,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.backoff_max = backoff_max_seconds
        self.timeout = timeout_seconds
        self.session = requests.Session()

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """GET + parse JSON with retries. Raises typed exceptions only."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        request_headers = self._auth_headers()
        if headers:
            request_headers.update(headers)

        attempt = 0
        while True:
            try:
                resp = self.session.get(
                    url, params=params, headers=request_headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise RetryExhaustedError(
                        f"Network failure after {attempt - 1} retries: {exc}"
                    ) from exc
                self._sleep_backoff(attempt)
                continue

            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise MalformedResponseError(
                        f"Provider returned invalid JSON (HTTP {resp.status_code})."
                    ) from exc

            if resp.status_code == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise RateLimitError("Rate limit exceeded; retries exhausted.")
                retry_after = self._retry_after_seconds(resp)
                logger.warning("Rate limited; retrying after %ss", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code in (401, 403):
                raise AuthenticationError(
                    f"Provider authentication failed (HTTP {resp.status_code})."
                )

            if resp.status_code >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise ProviderServerError(
                        f"Provider server error (HTTP {resp.status_code}); retries exhausted."
                    )
                self._sleep_backoff(attempt)
                continue

            body = ""
            try:
                body = str(resp.json())
            except ValueError:
                body = resp.text[:200]

            if resp.status_code == 404:
                raise UnavailableSymbolError(f"Provider returned 404: {body}")
            if resp.status_code in (400, 422):
                lower = body.lower()
                if any(k in lower for k in ("timeframe", "granularity", "interval")):
                    raise UnavailableTimeframeError(f"Provider rejected timeframe: {body}")
                raise UnavailableSymbolError(f"Provider rejected request: {body}")

            raise MalformedResponseError(
                f"Unexpected provider response (HTTP {resp.status_code}): {body}"
            )

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        time.sleep(delay)

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> float:
        value = resp.headers.get("Retry-After")
        if value is None:
            return 2.0
        try:
            return float(value)
        except ValueError:
            return 2.0
