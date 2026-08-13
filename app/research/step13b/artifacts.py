"""Atomic artifact writing for Step 13B.

All research artifacts (parquet + JSON) are written atomically (temp file +
fsync + os.replace) so an interrupted run never leaves partially-written
files. A window is only marked complete after its artifacts are fully
committed and validated.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically (temp file + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", prefix=".step13b_tmp_", dir=str(path.parent)
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


def atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically write a JSON-serializable object."""
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Atomically write a DataFrame as parquet."""
    import io

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    atomic_write_bytes(path, buf.getvalue())


def read_json_if_valid(path: Path) -> Any | None:
    """Read JSON; return None when missing or corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001 - corrupt file treated as absent
        return None


def read_parquet_if_valid(path: Path) -> pd.DataFrame | None:
    """Read parquet; return None when missing or corrupt."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:  # noqa: BLE001 - corrupt file treated as absent
        return None


class ArtifactManager:
    """Coordinates all research artifacts for one symbol/timeframe pair.

    Layout::

        <output_root>/<SYMBOL>/<TIMEFRAME>/
            strategy_validation.json
            window_metrics.parquet
            trade_log.parquet
            monthly_metrics.parquet
            regime_metrics.parquet
            research_summary.json
            windows/
                window_000.json
                window_000_trades.parquet
                ...
    """

    def __init__(self, output_root: str | Path, symbol: str, timeframe: str) -> None:
        self.output_root = Path(output_root)
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.dir = self.output_root / self.symbol / self.timeframe
        self.windows_dir = self.dir / "windows"

    # ── Paths ───────────────────────────────────────────────────────────────
    def strategy_validation_path(self) -> Path:
        return self.dir / "strategy_validation.json"

    def window_metrics_path(self) -> Path:
        return self.dir / "window_metrics.parquet"

    def trade_log_path(self) -> Path:
        return self.dir / "trade_log.parquet"

    def monthly_metrics_path(self) -> Path:
        return self.dir / "monthly_metrics.parquet"

    def regime_metrics_path(self) -> Path:
        return self.dir / "regime_metrics.parquet"

    def summary_path(self) -> Path:
        return self.dir / "research_summary.json"

    def window_json_path(self, index: int) -> Path:
        return self.windows_dir / f"window_{index:03d}.json"

    def window_trades_path(self, index: int) -> Path:
        return self.windows_dir / f"window_{index:03d}_trades.parquet"

    # ── Atomic writes ───────────────────────────────────────────────────────
    def write_strategy_validation(self, data: dict) -> None:
        atomic_write_json(self.strategy_validation_path(), data)

    def write_window_metrics(self, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.window_metrics_path(), df)

    def append_window_metrics(self, df: pd.DataFrame) -> None:
        """Append a window metrics frame to the cumulative file (atomic)."""
        existing = read_parquet_if_valid(self.window_metrics_path())
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        atomic_write_parquet(self.window_metrics_path(), combined)

    def write_trade_log(self, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.trade_log_path(), df)

    def append_trade_log(self, df: pd.DataFrame) -> None:
        """Append a trade snapshot frame to the cumulative log (atomic)."""
        existing = read_parquet_if_valid(self.trade_log_path())
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        atomic_write_parquet(self.trade_log_path(), combined)

    def write_monthly_metrics(self, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.monthly_metrics_path(), df)

    def write_regime_metrics(self, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.regime_metrics_path(), df)

    def write_summary(self, data: dict) -> None:
        atomic_write_json(self.summary_path(), data)

    def write_window_result(self, index: int, data: dict) -> None:
        atomic_write_json(self.window_json_path(index), data)

    def write_window_trades(self, index: int, df: pd.DataFrame) -> None:
        atomic_write_parquet(self.window_trades_path(index), df)

    # ── Validation helpers ──────────────────────────────────────────────────
    def window_artifacts_valid(self, index: int) -> bool:
        """A window is complete only when BOTH its JSON + trades parquet exist."""
        return self.window_json_path(index).exists() and (
            self.window_trades_path(index).exists()
            or read_parquet_if_valid(self.window_trades_path(index)) is not None
        )

    def strategy_validation_valid(self) -> bool:
        return read_json_if_valid(self.strategy_validation_path()) is not None