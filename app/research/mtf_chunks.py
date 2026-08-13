"""Chunked on-disk storage for large MTF artifacts (bounded-memory friendly).

Production MTF runs (e.g. EURUSD M15, 196K base bars x several tiers) cannot
be safely serialized as ONE giant JSON artifact: reading/writing it requires
holding every ``MtfContext`` in RAM simultaneously. This module stores MTF
output as a sequence of independently-validated chunks under::

    <cache_root>/<SYMBOL>/<TIMEFRAME>/mtf/
        manifest.json                 # identity + chunk count
        chunk_000000.json             # JSON array of MtfContext
        chunk_000000._meta.json       # index/start/end/count/hash
        ...

Writes are atomic (temp file + ``os.replace``); a chunk is only considered
complete when BOTH its artifact and its ``_meta.json`` exist and the artifact
hash matches. Interrupted runs therefore never lose completed chunks, and a
restart can skip validated chunks and resume from the first missing one.

Consumers can iterate contexts chunk-by-chunk with :meth:`MtfChunkStore.load`
/ :meth:`iter_contexts` without ever reconstructing the full 196K list.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import timezone
from pathlib import Path
from typing import Any

from app.mtf.models import MtfContext

__all__ = ["MTF_CHUNK_DIR", "MtfChunkStore", "MtfContextMap"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes atomically (temp file + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", prefix=".chunk_tmp_", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise


class MtfChunkStore:
    """Versioned, resumable chunk store for MTF batch output."""

    def __init__(self, symbol: str, timeframe: str, cache_root: str) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.root = Path(cache_root) / self.symbol / self.timeframe / MTF_CHUNK_DIR

    # ── Paths ────────────────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def chunk_path(self, index: int) -> Path:
        return self.root / f"chunk_{index:06d}.json"

    def chunk_meta_path(self, index: int) -> Path:
        return self.root / f"chunk_{index:06d}._meta.json"

    # ── Identity / manifest ──────────────────────────────────────────────────
    def write_manifest(
        self,
        *,
        source_data_hash: str,
        config_hash: str,
        upstream_hashes: dict,
        total_bars: int,
        chunk_size: int,
    ) -> None:
        manifest = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_bars": total_bars,
            "chunk_size": chunk_size,
            "chunks": 0,
            "completed_chunks": [],
            "source_data_hash": source_data_hash,
            "config_hash": config_hash,
            "upstream_hashes": upstream_hashes,
        }
        _atomic_write(
            self.manifest_path,
            json.dumps(manifest, indent=2, default=str).encode("utf-8"),
        )

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text("utf-8"))
        except Exception:  # noqa: BLE001 - corrupt manifest treated as miss
            return None

    # ── Chunk writes (atomic, validated) ─────────────────────────────────────
    def write_chunk(
        self,
        index: int,
        start_bar: int,
        end_bar: int,
        payload: bytes,
        *,
        source_data_hash: str,
        config_hash: str,
    ) -> None:
        """Atomically persist one chunk artifact + its meta.

        The ``_meta.json`` is written last so a chunk is never considered
        complete unless both artifact and meta exist and agree.
        """
        art = self.chunk_path(index)
        meta = self.chunk_meta_path(index)
        digest = _sha256(payload)

        _atomic_write(art, payload)
        meta_data = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "chunk_index": index,
            "start_bar": start_bar,
            "end_bar": end_bar,
            "row_count": end_bar - start_bar,
            "artifact_hash": digest,
            "source_data_hash": source_data_hash,
            "config_hash": config_hash,
            "complete": True,
        }
        _atomic_write(
            meta,
            json.dumps(meta_data, indent=2, default=str).encode("utf-8"),
        )

    # ── Validation / resume ──────────────────────────────────────────────────
    def is_chunk_complete(self, index: int) -> bool:
        """A chunk is valid only if artifact + meta exist and hashes match."""
        art = self.chunk_path(index)
        meta = self.chunk_meta_path(index)
        if not art.exists() or not meta.exists():
            return False
        try:
            m = json.loads(meta.read_text("utf-8"))
        except Exception:  # noqa: BLE001 - corrupt meta treated as invalid chunk
            return False
        if not m.get("complete"):
            return False
        return m.get("artifact_hash") == _sha256(art.read_bytes())

    def valid_chunk_indices(self) -> list[int]:
        """Sorted indices of complete, valid chunks (for resume)."""
        if not self.root.exists():
            return []
        out = []
        for p in self.root.glob("chunk_[0-9]*.json"):
            if p.name.endswith("._meta.json"):
                continue
            try:
                idx = int(p.stem.split("_")[1])
            except Exception:  # noqa: BLE001,S112 - non-chunk file skipped
                continue
            if self.is_chunk_complete(idx):
                out.append(idx)
        return sorted(out)

    def first_missing_index(self) -> int:
        """First index not valid/complete — the resume point."""
        valid = set(self.valid_chunk_indices())
        i = 0
        while i in valid:
            i += 1
        return i

    def chunk_set_hash(self) -> str:
        """Deterministic hash over the completed chunk set (streaming-safe).

        Reads only each chunk's ``_meta.json`` (small) and folds the
        ``artifact_hash`` values in index order. Does NOT load the chunk
        bodies, so this is safe for 196K-context artifacts.
        """
        h = hashlib.sha256()
        for idx in self.valid_chunk_indices():
            meta = json.loads(self.chunk_meta_path(idx).read_text("utf-8"))
            h.update(f"{idx}:{meta.get('artifact_hash', '')};".encode())
        return h.hexdigest()

    def load_chunk(self, index: int) -> list[dict]:
        """Load one chunk's contexts as raw dicts (bounded memory)."""
        if not self.is_chunk_complete(index):
            raise ValueError(f"chunk {index} is not complete/valid")
        return json.loads(self.chunk_path(index).read_text("utf-8"))

    # ── Streaming consumers ──────────────────────────────────────────────────
    def iter_contexts(self) -> Iterator[MtfContext]:
        """Yield every valid chunk's contexts (bounded memory).

        Does NOT reconstruct the full result list; each chunk is decoded and
        released before the next is read.
        """
        for idx in self.valid_chunk_indices():
            for raw in self.load_chunk(idx):
                yield MtfContext.model_validate(raw)


class MtfContextMap:
    """Memory-bounded timestamp→MtfContext lookup over chunked MTF output.

    Downstream consumers (signal scanner / backtester) previously built a
    dict of ALL 196K contexts in RAM just to do ``mtf_by_ts.get(ts)`` per bar.
    This class holds only ONE chunk decoded at a time and advances chunks
    lazily as the consumer's timestamp progresses (contexts are ordered by
    timestamp). Memory stays bounded at one chunk regardless of total bars.

    Both the consumer and this map must iterate in chronological order; a
    ``dict``-like ``get(ts)`` is provided for drop-in compatibility.
    """

    def __init__(self, store: MtfChunkStore) -> None:
        self._store = store
        self._valid = store.valid_chunk_indices()
        self._cursor = 0
        self._current_idx: int | None = None
        self._current_ctxs: list[MtfContext] | None = None
        self._ts_to_pos: dict[Any, int] | None = None

    def _load_next_chunk(self) -> None:
        if self._cursor >= len(self._valid):
            self._current_ctxs = None
            self._ts_to_pos = None
            return
        idx = self._valid[self._cursor]
        raw = self._store.load_chunk(idx)
        # Deserialize once per chunk into MtfContext objects so downstream
        # consumers (.alignment / .metadata / .hierarchy) work identically.
        ctxs = [MtfContext.model_validate(c) for c in raw]
        self._current_idx = idx
        self._current_ctxs = ctxs
        self._ts_to_pos = {
            _normalize_any_ts(c.timestamp): pos for pos, c in enumerate(ctxs)
        }
        self._cursor += 1

    def get(self, ts) -> MtfContext | None:
        """Return the MtfContext at ``ts`` or None (chronological streaming)."""
        key = _normalize_any_ts(ts)
        while True:
            if self._ts_to_pos is None:
                if self._cursor >= len(self._valid):
                    return None
                self._load_next_chunk()
            if key in self._ts_to_pos:
                return self._current_ctxs[self._ts_to_pos[key]]
            # Not in current chunk; advance to next chunk.
            if self._cursor >= len(self._valid):
                return None
            self._load_next_chunk()

    def chunk_set_hash(self) -> str:
        """Streaming-safe hash of the underlying chunk set (no full load)."""
        return self._store.chunk_set_hash()

    def __contains__(self, ts) -> bool:
        return self.get(ts) is not None


def _normalize_any_ts(value):
    """Normalize a datetime/pandas Timestamp/str to a hashable UTC key."""
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, str):
        from pandas import Timestamp

        value = Timestamp(value).to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0)


MTF_CHUNK_DIR = "mtf"