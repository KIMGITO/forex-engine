"""Deterministic tests for the shared HTTP client (mocked responses)."""


import pytest
from requests import Session

from app.data.client import HttpClient
from app.data.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    ProviderServerError,
    RateLimitError,
    RetryExhaustedError,
    UnavailableSymbolError,
    UnavailableTimeframeError,
)


class FakeResp:
    def __init__(self, payload, status, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


def _patch_get(monkeypatch, payload, status=200, headers=None):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeResp(payload, status, headers)

    monkeypatch.setattr(Session, "get", fake_get)


class TestHttpClient:
    def test_success_returns_json(self, monkeypatch):
        _patch_get(monkeypatch, {"ok": True})
        client = HttpClient("https://example.com", max_retries=1)
        assert client.get_json("/path") == {"ok": True}

    def test_401_maps_to_auth_error(self, monkeypatch):
        _patch_get(monkeypatch, {"error": "nope"}, status=401)
        client = HttpClient("https://example.com", max_retries=1)
        with pytest.raises(AuthenticationError):
            client.get_json("/path")

    def test_403_maps_to_auth_error(self, monkeypatch):
        _patch_get(monkeypatch, {}, status=403)
        client = HttpClient("https://example.com", max_retries=1)
        with pytest.raises(AuthenticationError):
            client.get_json("/path")

    def test_429_retries_then_succeeds(self, monkeypatch):
        payloads = [
            FakeResp({"error": "rate"}, 429, {"Retry-After": "0.1"}),
            FakeResp({"ok": True}, 200),
        ]

        def fake_get(self, url, params=None, headers=None, timeout=None):
            return payloads.pop(0)

        monkeypatch.setattr(Session, "get", fake_get)
        client = HttpClient("https://example.com", max_retries=3)
        assert client.get_json("/path") == {"ok": True}

    def test_429_exhausted_raises_rate_limit(self, monkeypatch):
        _patch_get(monkeypatch, {"error": "rate"}, status=429)
        client = HttpClient("https://example.com", max_retries=2)
        with pytest.raises(RateLimitError):
            client.get_json("/path")

    def test_500_retries_then_succeeds(self, monkeypatch):
        payloads = [
            FakeResp({"error": "server"}, 500),
            FakeResp({"ok": True}, 200),
        ]

        def fake_get(self, url, params=None, headers=None, timeout=None):
            return payloads.pop(0)

        monkeypatch.setattr(Session, "get", fake_get)
        client = HttpClient("https://example.com", max_retries=3)
        assert client.get_json("/path") == {"ok": True}

    def test_500_exhausted_raises_provider_server_error(self, monkeypatch):
        _patch_get(monkeypatch, {"error": "server"}, status=500)
        client = HttpClient("https://example.com", max_retries=2)
        with pytest.raises(ProviderServerError):
            client.get_json("/path")

    def test_404_maps_to_unavailable_symbol(self, monkeypatch):
        _patch_get(monkeypatch, {"error": "not found"}, status=404)
        client = HttpClient("https://example.com", max_retries=1)
        with pytest.raises(UnavailableSymbolError):
            client.get_json("/path")

    def test_422_timeframe_maps_to_unavailable_timeframe(self, monkeypatch):
        _patch_get(
            monkeypatch, {"error": "invalid timeframe"}, status=400
        )
        client = HttpClient("https://example.com", max_retries=1)
        with pytest.raises(UnavailableTimeframeError):
            client.get_json("/path")

    def test_invalid_json_raises_malformed(self, monkeypatch):
        class BadResp:
            status_code = 200

            def __init__(self) -> None:
                self.headers: dict = {}

            def json(self):
                raise ValueError("not json")


        monkeypatch.setattr(Session, "get", lambda self, *a, **k: BadResp())
        client = HttpClient("https://example.com", max_retries=1)
        with pytest.raises(MalformedResponseError):
            client.get_json("/path")

    def test_network_retries_then_succeeds(self, monkeypatch):
        import requests

        payloads = [
            requests.RequestException("boom"),
            FakeResp({"ok": True}, 200),
        ]

        def fake_get(self, url, params=None, headers=None, timeout=None):
            item = payloads.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(Session, "get", fake_get)
        client = HttpClient("https://example.com", max_retries=3)
        assert client.get_json("/path") == {"ok": True}

    def test_network_retries_exhausted(self, monkeypatch):
        import requests

        def fake_get(self, url, params=None, headers=None, timeout=None):
            raise requests.RequestException("boom")

        monkeypatch.setattr(Session, "get", fake_get)
        client = HttpClient("https://example.com", max_retries=2)
        with pytest.raises(RetryExhaustedError):
            client.get_json("/path")