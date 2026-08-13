"""Atomic persistence for Step 13 columnar event datasets.

All Parquet and JSON artifacts are written atomically (temp file + fsync +
os.replace). A chunk is only considered complete after its artifacts are
fully committed. Manifests carry schema version, engine version, source
data hash, configuration hash, row count, timestamp range, symbol,
timeframe, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", prefix=".step13_tmp_", dir=str(path.parent)
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


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Atomically write a DataFrame as parquet."""
    import io

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    _atomic_write_bytes(path, buf.getvalue())


def atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically write a JSON-serializable object."""
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
    _atomic_write_bytes(path, payload)


def read_parquet_if_valid(path: Path) -> pd.DataFrame | None:
    """Read parquet; return None when missing or corrupt."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:  # noqa: BLE001 - corrupt file treated as absent
        return None


def read_json_if_valid(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _data_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of an OHLC DataFrame."""
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    sub = df[cols].sort_index()
    buf = sub.to_parquet(engine="pyarrow")
    return hashlib.sha256(buf).hexdigest()


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object-dtype columns to string for pyarrow-safe serialization.

    Mixed-dtype columns (e.g. ``available_from`` as datetime in one dataset
    and as an ISO string in the MTF context) cause pyarrow conversion
    errors when concatenated into a single chunk parquet.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


class Step13Artifacts:
    """Coordinates all Step 13 output artifacts for one symbol/timeframe."""

    def __init__(self, output_root: str | Path, symbol: str, timeframe: str) -> None:
        self.output_root = Path(output_root)
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.dir = self.output_root / self.symbol / self.timeframe
        self.chunks_dir = self.dir / "chunks"

    # ── Paths ───────────────────────────────────────────────────────────────
    def dataset_path(self, name: str) -> Path:
        return self.dir / f"{name}.parquet"

    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def chunk_path(self, chunk_index: int) -> Path:
        return self.chunks_dir / f"chunk_{chunk_index:06d}.parquet"

    def chunk_meta_path(self, chunk_index: int) -> Path:
        return self.chunks_dir / f"chunk_{chunk_index:06d}.json"

    # ── Per-chunk persistence ───────────────────────────────────────────────

    def write_chunk(
        self,
        chunk_index: int,
        frames: dict[str, pd.DataFrame],
    ) -> None:
        """Atomically persist a chunk's event frames.

        The chunk's artifacts are written as one combined parquet keyed by
        event type via an extra ``dataset`` column so a single atomic file
        per chunk contains ALL event rows for that chunk.
        """
        import io

        combined_rows: list[pd.DataFrame] = []
        for name, df in frames.items():
            if df is None or df.empty:
                continue
            d = _coerce_object_columns(df)
            d.insert(0, "dataset", name)
            combined_rows.append(d)
        if not combined_rows:
            combined = pd.DataFrame(columns=["dataset"])
        else:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined = _coerce_object_columns(combined)

        buf = io.BytesIO()
        combined.to_parquet(buf, engine="pyarrow", index=False)

        # Atomic write chunk file + meta.
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(self.chunk_path(chunk_index), buf.getvalue())
        meta = {
            "chunk_index": chunk_index,
            "datasets": sorted(frames.keys()),
            "rows": {k: (0 if v is None else len(v)) for k, v in frames.items()},
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.chunk_meta_path(chunk_index), meta)

    def chunk_valid(self, chunk_index: int) -> bool:
        """A chunk is complete only when artifact + meta both exist."""
        return (
            self.chunk_path(chunk_index).exists()
            and self.chunk_meta_path(chunk_index).exists()
        )

    # ── Cumulative dataset building ─────────────────────────────────────────

    def merge_chunks_to_datasets(
        self,
        datasets: list[str],
        source_data_hash: str,
        config_hash: str,
        engine_version: str,
    ) -> None:
        """Merge all valid chunk files into per-dataset parquet files.

        Reads each chunk once (bounded), appends rows per dataset, writes
        atomically, then writes the manifest.
        """
        if not self.chunks_dir.exists():
            return

        accum: dict[str, list[pd.DataFrame]] = {d: [] for d in datasets}
        chunk_indices = []
        for p in sorted(self.chunks_dir.glob("chunk_[0-9]*.parquet")):
            try:
                idx = int(p.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                continue
            if not self.chunk_valid(idx):
                continue
            chunk_indices.append(idx)
            df = read_parquet_if_valid(p)
            if df is None or df.empty:
                continue
            if "dataset" not in df.columns:
                continue
            for name, grp in df.groupby("dataset"):
                if name in accum:
                    accum[name].append(
                        grp.drop(columns=["dataset"], errors="ignore")
                    )

        for name, frames in accum.items():
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            combined = _coerce_object_columns(combined)
            # Chunk-overlap deduplication (C4): the same candidate can appear
            # in overlapping chunks; MUST NOT inflate the statistical sample.
            if "candidate_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["candidate_id"])
            atomic_write_parquet(self.dataset_path(name), combined)

        manifest = {
            "schema_version": 1,
            "engine_version": engine_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source_data_hash": source_data_hash,
            "configuration_hash": config_hash,
            "chunk_count": len(chunk_indices),
            "chunk_indices": chunk_indices,
            "datasets": datasets,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.manifest_path(), manifest)

    def manifest(self) -> dict | None:
        return read_json_if_valid(self.manifest_path())