"""Quantitative feature engine for market-data analysis.

Provides causally computed features (returns, volatility, momentum, trend,
correlation) that are independent of trading strategies, signals, or AI models.
"""

from app.features.engine import FEATURE_REGISTRY, FeatureDefinition, FeatureEngine
from app.features.errors import FeatureError, InsufficientDataError, UnknownFeatureError
from app.features.models import Feature

__all__ = [
    "FEATURE_REGISTRY",
    "Feature",
    "FeatureDefinition",
    "FeatureEngine",
    "FeatureError",
    "InsufficientDataError",
    "UnknownFeatureError",
]