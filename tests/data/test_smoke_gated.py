"""Gated smoke-test invocation (no live network by default)."""


from app.data.smoke_test import _cli_main


def test_smoke_skipped_without_creds(monkeypatch, capsys):
    monkeypatch.delenv("RUN_LIVE_DATA_TESTS", raising=False)
    monkeypatch.delenv("MARKET_DATA_API_KEY", raising=False)
    rc = _cli_main()
    assert rc == 0
    captured = capsys.readouterr().out
    assert "skipped" in captured.lower()