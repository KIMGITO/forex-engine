"""Versioned research cache with deterministic, causal-safe artifact reuse.

Large historical research datasets are expensive to process. This module
provides a per-stage, per-symbol/timeframe on-disk cache so that:

* features, market structure, regime, MTF, and strategy signals are computed
  ONCE and reused across strategies, backtests, walk-forward, and optimization.
* cache hits are only trusted when the manifest matches the current inputs
  (source data hash, configuration hash, upstream artifact hashes, and engine
  version). Stale or incompatible caches are NEVER silently reused.
* every artifact is byte-deterministic: loading a cached artifact returns the
  exact same objects a fresh calculation would produce.

Causality is preserved because this cache only persists the OUTPUT of the
existing causal engines. The engines' ``available_from`` / completed-candle
rules are untouched. A cached artifact is only valid for the exact same source
data and configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.market_structure.models import MarketStructureResult
from app.mtf.models import MtfContext
from app.regime.models import MarketRegime, MarketState, NewsRiskState, TrendState, VolatilityState
from app.strategy.models import Signal

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ENGINE_VERSION",
    "CacheManifest",
    "ResearchCache",
    "StageTimer",
    "config_hash",
    "data_hash",
    "deser_features",
    "deser_mtf",
    "deser_regime",
    "deser_signals",
    "deser_structure",
    "ser_features",
    "ser_mtf",
    "ser_regime",
    "ser_signals",
    "ser_structure",
]

# Bump when the cache schema (file layout / manifest shape) changes.
CACHE_SCHEMA_VERSION = 1

# Bump when any analytical engine changes such that cached artifacts must be
# regenerated. This is a coarse safety net in addition to the input hashes.
ENGINE_VERSION = "1.1.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def data_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of the OHLC source data (index + OHLC columns)."""
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    sub = df[cols].sort_index()
    try:
        buf = sub.to_parquet(engine="pyarrow")
    except Exception:  # noqa: BLE001 - pragma: no cover - fallback for exotic environments
        buf = sub.to_csv().encode("utf-8")
    return sha256_hex(buf)


def config_hash(obj: Any) -> str:
    """Deterministic hash of a configuration object (dict/dataclass/str)."""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    elif hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return sha256_hex(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    )


def ser_features(df: pd.DataFrame) -> bytes:
    return df.sort_index().to_parquet(engine="pyarrow")


def deser_features(data: bytes) -> pd.DataFrame:
    import io

    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    return df.sort_index()


def ser_structure(result: MarketStructureResult) -> bytes:
    return json.dumps(result.model_dump(), default=str).encode("utf-8")


def deser_structure(data: bytes) -> MarketStructureResult:
    return MarketStructureResult.model_validate(json.loads(data.decode("utf-8")))


def ser_regime(regimes: list[MarketRegime] | None) -> bytes:
    if not regimes:
        return b""
    rows = [r.model_dump() for r in regimes]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["available_from"] = pd.to_datetime(df["available_from"], utc=True)
    df["metrics"] = df["metrics"].apply(json.dumps)
    df = df.set_index("timestamp").sort_index()
    return df.to_parquet(engine="pyarrow")


def deser_regime(data: bytes) -> list[MarketRegime]:
    """Deserialize regime observations from Parquet.

    ``ds`` is a dict hybrid (pandas Series is a dict-subclass); we access
    columns positionally to keep mypy happy.
    """
    if not data:
        return []
    import io

    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    out: list[MarketRegime] = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(str(idx))
        metrics = row["metrics"]
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        af = row["available_from"]
        if not isinstance(af, pd.Timestamp):
            af = pd.Timestamp(af)
        out.append(
            MarketRegime(
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                timestamp=ts.to_pydatetime(),
                trend_state=TrendState(str(row["trend_state"])),
                volatility_state=VolatilityState(str(row["volatility_state"])),
                market_state=MarketState(str(row["market_state"])),
                news_risk=NewsRiskState(str(row["news_risk"])),
                strength=float(row["strength"]),
                metrics=metrics or {},
                available_from=af.to_pydatetime(),
            )
        )
    return out


def ser_mtf(contexts: list[MtfContext] | None) -> bytes:
    """Serialize MTF contexts to compact JSON (chunked, memory-safe).

    The previous implementation materialized ``[c.model_dump() for c in
    contexts]`` followed by a single giant ``json.dumps`` — two full-size
    copies in RAM simultaneously. For production-scale runs (196K base bars x
    ~4 tiers) that amplification caused multi-GB peaks and OOM.

    Chunking serializes each batch to an independent JSON array and joins the
    encoded fragments, so peak memory is bounded by one batch (typically
    5_000 contexts) instead of the entire result set.
    """
    if not contexts:
        return b""
    batch = 5_000
    if len(contexts) <= batch:
        # Fast path for small results (identical output layout, compact).
        return json.dumps(
            [c.model_dump() for c in contexts],
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    parts: list[bytes] = []
    for i in range(0, len(contexts), batch):
        chunk = [c.model_dump() for c in contexts[i : i + batch]]
        encoded = json.dumps(chunk, default=str, separators=(",", ":"))
        # Strip the chunk's outer '[' ']' so joining produces ONE flat array.
        parts.append(encoded[1:-1].encode("utf-8"))
    return b"[" + b",".join(parts) + b"]"


def deser_mtf(data: bytes) -> list[MtfContext] | None:
    if not data:
        return None
    raw = json.loads(data.decode("utf-8"))
    return [MtfContext.model_validate(c) for c in raw]


def ser_signals(signals: list[Signal]) -> bytes:
    return json.dumps([s.model_dump() for s in signals], default=str).encode("utf-8")


def deser_signals(data: bytes) -> list[Signal]:
    if not data:
        return []
    raw = json.loads(data.decode("utf-8"))
    return [Signal.model_validate(s) for s in raw]


# ── Manifest ───────────────────────────────────────────────────────────────────


@dataclass
class CacheManifest:
    """Metadata describing one cached artifact and its validity inputs."""

    symbol: str
    timeframe: str
    stage: str
    schema_version: int
    engine_version: str
    source_data_hash: str
    first_timestamp: str
    last_timestamp: str
    row_count: int
    config_hash: str
    upstream_hashes: dict = field(default_factory=dict)
    artifact_hash: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "stage": self.stage,
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "source_data_hash": self.source_data_hash,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "row_count": self.row_count,
            "config_hash": self.config_hash,
            "upstream_hashes": self.upstream_hashes,
            "artifact_hash": self.artifact_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CacheManifest:
        return cls(
            symbol=str(d.get("symbol", "")),
            timeframe=str(d.get("timeframe", "")),
            stage=str(d.get("stage", "")),
            schema_version=int(d.get("schema_version", -1)),
            engine_version=str(d.get("engine_version", "")),
            source_data_hash=str(d.get("source_data_hash", "")),
            first_timestamp=str(d.get("first_timestamp", "")),
            last_timestamp=str(d.get("last_timestamp", "")),
            row_count=int(d.get("row_count", 0)),
            config_hash=str(d.get("config_hash", "")),
            upstream_hashes=dict(d.get("upstream_hashes", {})),
            artifact_hash=str(d.get("artifact_hash", "")),
            created_at=str(d.get("created_at", "")),
        )


# ── Stage timer ────────────────────────────────────────────────────────────────


@dataclass
class StageTimer:
    """Lightweight stage timing for profiling the research pipeline."""

    timings: dict[str, float] = field(default_factory=dict)
    _current: str | None = None
    _start: float = 0.0

    def begin(self, stage: str) -> None:
        if self._current is not None:
            self.end(self._current)
        self._current = stage
        self._start = time.monotonic()

    def end(self, stage: str | None = None) -> None:
        stage = stage or self._current
        if stage is None:
            return
        elapsed = time.monotonic() - self._start
        self.timings[stage] = self.timings.get(stage, 0.0) + elapsed
        self._current = None

    def add(self, stage: str, seconds: float) -> None:
        self.timings[stage] = self.timings.get(stage, 0.0) + seconds

    def summary(self) -> dict[str, float]:
        if self._current is not None:
            self.end(self._current)
        return dict(self.timings)

    def total(self) -> float:
        return sum(self.timings.values())


# ── Cache ──────────────────────────────────────────────────────────────────────


class ResearchCache:
    """Per-symbol/timeframe/stage on-disk cache with manifest validation.

    Layout::

        <root>/<SYMBOL>/<TIMEFRAME>/
            features.parquet             features._meta.json
            structure.json               structure._meta.json
            regime.parquet               regime._meta.json
            mtf.json                     mtf._meta.json
            signals_<strategy>_<h>.json  signals_<strategy>_<h>._meta.json
    """

    def __init__(self, root: str = "research/cache", use_cache: bool = True) -> None:
        self.root = Path(root)
        self.use_cache = use_cache
        self.hits = 0
        self.misses = 0

    def _dir(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol.upper() / timeframe.upper()

    def _artifact_path(self, symbol: str, timeframe: str, stage: str, key: str = "") -> Path:
        suffix = {"features": "parquet", "regime": "parquet"}.get(stage, "json")
        name = f"{stage}{key}.{suffix}"
        return self._dir(symbol, timeframe) / name

    def _meta_path(self, symbol: str, timeframe: str, stage: str, key: str = "") -> Path:
        name = f"{stage}{key}._meta.json"
        return self._dir(symbol, timeframe) / name

    def _load_meta(self, path: Path) -> CacheManifest | None:
        if not path.exists():
            return None
        try:
            return CacheManifest.from_dict(json.loads(path.read_text("utf-8")))
        except Exception:  # noqa: BLE001 - corrupt metadata -> treat as miss
            return None

    def _save_meta(self, path: Path, manifest: CacheManifest) -> None:
        """Atomically write metadata via temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="._meta_tmp_", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest.to_dict(), indent=2, default=str))
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _matches(
        self,
        stored: CacheManifest | None,
        expected: CacheManifest,
    ) -> bool:
        if stored is None:
            return False
        return (
            stored.schema_version == expected.schema_version
            and stored.engine_version == expected.engine_version
            and stored.source_data_hash == expected.source_data_hash
            and stored.config_hash == expected.config_hash
            and stored.upstream_hashes == expected.upstream_hashes
            and stored.row_count == expected.row_count
            and stored.first_timestamp == expected.first_timestamp
            and stored.last_timestamp == expected.last_timestamp
        )

    def get_or_compute(
        self,
        symbol: str,
        timeframe: str,
        stage: str,
        source_df: pd.DataFrame,
        config: Any,
        upstream: dict[str, str],
        compute: Callable[[], Any],
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Any],
        key: str = "",
    ) -> tuple[Any, bool]:
        """Return ``(artifact, was_hit)``.

        When ``self.use_cache`` and a valid artifact exists, load it (HIT).
        Otherwise compute, persist, and return it (MISS).
        """
        source_df = source_df.sort_index()
        src_hash = data_hash(source_df)
        cfg_hash = config_hash(config)
        first = (
            source_df.index[0].to_pydatetime().isoformat()
            if len(source_df)
            else ""
        )
        last = (
            source_df.index[-1].to_pydatetime().isoformat()
            if len(source_df)
            else ""
        )
        expected = CacheManifest(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            stage=stage,
            schema_version=CACHE_SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            source_data_hash=src_hash,
            first_timestamp=first,
            last_timestamp=last,
            row_count=len(source_df),
            config_hash=cfg_hash,
            upstream_hashes=dict(upstream),
            created_at=_utcnow(),
        )

        art_path = self._artifact_path(symbol, timeframe, stage, key)
        meta_path = self._meta_path(symbol, timeframe, stage, key)

        if self.use_cache:
            stored = self._load_meta(meta_path)
            if self._matches(stored, expected) and art_path.exists():
                payload = art_path.read_bytes()
                if stored is not None and sha256_hex(payload) == stored.artifact_hash:
                    self.hits += 1
                    return deser(payload), True

        # Cache miss (or cache disabled): compute + persist.
        self.misses += 1
        artifact = compute()
        payload = ser(artifact)
        expected.artifact_hash = sha256_hex(payload)
        art_path.parent.mkdir(parents=True, exist_ok=True)
        # Write artifact atomically via temp file + rename.
        suffix = art_path.suffix
        art_tmp_fd, art_tmp_path = tempfile.mkstemp(
            suffix=suffix, prefix=".artifact_tmp_", dir=str(art_path.parent)
        )
        try:
            os.write(art_tmp_fd, payload)
            os.fsync(art_tmp_fd)
            os.close(art_tmp_fd)
            os.replace(art_tmp_path, str(art_path))
        except Exception:
            try:
                Path(art_tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise
        # Metadata saved last so an artifact without valid meta is never trusted.
        self._save_meta(meta_path, expected)
        return artifact, False

    def invalidate(self, symbol: str, timeframe: str) -> None:
        """Delete all cached artifacts for a symbol/timeframe (safety tool)."""
        d = self._dir(symbol, timeframe)
        if d.exists():
            for p in d.iterdir():
                p.unlink()

    def describe(self) -> dict:
        """Count artifacts and total size on disk."""
        total_bytes = 0
        count = 0
        if self.root.exists():
            for p in self.root.rglob("*"):
                if p.is_file() and not p.name.endswith("._meta.json"):
                    total_bytes += p.stat().st_size
                    count += 1
        return {"artifact_count": count, "bytes": total_bytes}
