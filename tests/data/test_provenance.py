"""Deterministic tests for provenance/sidecar metadata."""


from app.data.provenance import (
    ProviderMetadata,
    _meta_path,
    read_metadata,
    write_metadata,
)


class TestProvenance:
    def test_round_trip(self, tmp_path):
        data_path = tmp_path / "eurusd_1h.parquet"
        data_path.write_text("stub")

        meta = ProviderMetadata(
            symbol="EURUSD",
            timeframe="1h",
            source_type="historical",
            provider="oanda",
            data_type="mid",
            retrieved_at=None,
            timezone="UTC",
            api_version="v3",
            notes="smoke",
        )
        write_metadata(data_path, meta)
        loaded = read_metadata(data_path)
        assert loaded is not None
        assert loaded.symbol == "EURUSD"
        assert loaded.provider == "oanda"
        assert loaded.notes == "smoke"

    def test_missing_returns_none(self, tmp_path):
        assert read_metadata(tmp_path / "nonexistent.parquet") is None

    def test_bad_json_returns_none(self, tmp_path):
        data_path = tmp_path / "bad.parquet"
        data_path.write_text("stub")
        _meta_path(data_path).write_text("NOT_JSON")
        assert read_metadata(data_path) is None