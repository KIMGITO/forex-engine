"""Structured representation of calculated quantitative features."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Feature(BaseModel):
    """A single feature computed from a market-data snapshot.

    ``values`` is a free-form mapping of feature name -> numerical value. This
    intentionally avoids a single rigid schema containing every possible
    future indicator: adding a new feature merely extends the ``values`` dict,
    so existing consumers do not break.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Standardized currency pair, e.g. EURUSD")
    timeframe: str = Field(..., description="Timeframe string, e.g. 1h, 1d")
    timestamp: datetime = Field(..., description="UTC timestamp this feature applies to")
    values: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of feature name -> computed numerical value",
    )


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata describing a calculable feature for registry/extensibility."""

    name: str
    category: str
    description: str
    default_params: dict[str, Any] = field(default_factory=dict)
    output_column: str | None = None

    def output(self, resolved: str | None = None) -> str:
        """Resolve the output column name for a feature instance."""
        return resolved or self.output_column or self.name