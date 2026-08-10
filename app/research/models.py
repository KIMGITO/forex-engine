"""Typed models for the research layer.

Chronicological splits, walk-forward windows, optimizer results, and research
reports are modelled here. Every result carries explicit provenance so it is
reproducible and never mixes in-sample/validation/out-of-sample data.
"""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class TimeframeDataset:
    """A cached dataset for one symbol/timeframe pair."""

    symbol: str
    timeframe: str
    provider: str
    start: datetime
    end: datetime
    row_count: int
    timezone: str
    gaps: int
    source_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TimeSplit:
    """A chronological train/validation/test split."""

    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward window: train / validation / test periods."""

    index: int
    split: TimeSplit
    selected_config: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "split": self.split.to_dict(),
            "selected_config": self.selected_config,
            "results": self.results,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class OptimizerCandidate:
    """One configuration candidate with its training performance."""

    params: dict[str, object]
    score: float
    metrics: dict
    trade_count: int

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "score": self.score,
            "metrics": self.metrics,
            "trade_count": self.trade_count,
        }


class ResearchReport(BaseModel):
    """A machine-readable research report."""

    model_config = ConfigDict(frozen=True)

    provider: str
    symbols: list[str]
    timeframes: list[str]
    strategy: str
    config_dump: dict
    date_range: dict
    training: dict
    validation: dict
    out_of_sample: dict
    cross_symbol: dict
    cross_window: dict
    cost_assumptions: dict
    warnings: list[str]
    limitations: list[str]