"""Central feature engine that orchestrates computation of selected features.

The engine maintains a registry of feature definitions and dispatches
calculation to the appropriate module. It is designed to be extensible:
new features can be registered without modifying the engine itself.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from app.features.correlation import (
    align_price_dfs,
    correlation_matrix,
    pairwise_correlation,
    rolling_correlation,
)
from app.features.errors import FeatureError, UnknownFeatureError
from app.features.models import FeatureDefinition
from app.features.momentum import price_momentum, rate_of_change, rsi
from app.features.returns import log_returns, pct_returns, simple_returns
from app.features.trend import distance_from_ma, ema, ma_slope, sma
from app.features.volatility import (
    annualized_volatility,
    atr,
    rolling_atr,
    rolling_std_init,
    volatility_percentile,
)

# ── Registry ──────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, FeatureDefinition] = {
    # Returns
    "simple_returns": FeatureDefinition(
        name="simple_returns",
        category="returns",
        description="Simple returns: (p_t - p_{t-1}) / p_{t-1}",
        default_params={"price_col": "close"},
    ),
    "pct_returns": FeatureDefinition(
        name="pct_returns",
        category="returns",
        description="Percentage returns: (p_t / p_{t-1} - 1) * 100",
        default_params={"price_col": "close"},
    ),
    "log_returns": FeatureDefinition(
        name="log_returns",
        category="returns",
        description="Logarithmic returns: log(p_t / p_{t-1})",
        default_params={"price_col": "close"},
    ),
    # Volatility
    "rolling_std": FeatureDefinition(
        name="rolling_std",
        category="volatility",
        description="Rolling standard deviation of price",
        default_params={"window": 20, "price_col": "close"},
    ),
    "annualized_volatility": FeatureDefinition(
        name="annualized_volatility",
        category="volatility",
        description="Annualized rolling volatility of returns",
        default_params={"window": 20, "price_col": "close", "timeframe": "1h"},
    ),
    "atr": FeatureDefinition(
        name="atr",
        category="volatility",
        description="Average True Range (Wilder smoothing)",
        default_params={"window": 14},
    ),
    "rolling_atr": FeatureDefinition(
        name="rolling_atr",
        category="volatility",
        description="Rolling (simple mean) ATR",
        default_params={"window": 14},
    ),
    "volatility_percentile": FeatureDefinition(
        name="volatility_percentile",
        category="volatility",
        description="Rolling percentile rank of volatility",
        default_params={"window": 20, "rank_window": 100},
    ),
    # Momentum
    "rate_of_change": FeatureDefinition(
        name="rate_of_change",
        category="momentum",
        description="Rate of change: (p_t / p_{t-period} - 1) * 100",
        default_params={"period": 10, "price_col": "close"},
    ),
    "price_momentum": FeatureDefinition(
        name="price_momentum",
        category="momentum",
        description="Price momentum: p_t - p_{t-period}",
        default_params={"period": 10, "price_col": "close"},
    ),
    "rsi": FeatureDefinition(
        name="rsi",
        category="momentum",
        description="Relative Strength Index (Wilder), [0, 100]",
        default_params={"period": 14, "price_col": "close"},
    ),
    # Trend
    "sma": FeatureDefinition(
        name="sma",
        category="trend",
        description="Simple moving average",
        default_params={"period": 20, "price_col": "close"},
    ),
    "ema": FeatureDefinition(
        name="ema",
        category="trend",
        description="Exponential moving average",
        default_params={"period": 20, "price_col": "close"},
    ),
    "distance_from_ma": FeatureDefinition(
        name="distance_from_ma",
        category="trend",
        description="Percentage distance from moving average",
        default_params={"period": 20, "price_col": "close", "ma_type": "sma"},
    ),
    "ma_slope": FeatureDefinition(
        name="ma_slope",
        category="trend",
        description="Moving average slope (ROC of MA)",
        default_params={"period": 20, "price_col": "close", "ma_type": "sma", "slope_periods": 5},
    ),
}

# Maps feature name to the callable that computes it.
# Each callable must accept (data: pd.DataFrame, **kwargs) -> pd.Series.
_FEATURE_FN: dict[str, Any] = {
    "simple_returns": simple_returns,
    "pct_returns": pct_returns,
    "log_returns": log_returns,
    "rolling_std": rolling_std_init,
    "annualized_volatility": annualized_volatility,
    "atr": atr,
    "rolling_atr": rolling_atr,
    "volatility_percentile": volatility_percentile,
    "rate_of_change": rate_of_change,
    "price_momentum": price_momentum,
    "rsi": rsi,
    "sma": sma,
    "ema": ema,
    "distance_from_ma": distance_from_ma,
    "ma_slope": ma_slope,
}


class FeatureEngine:
    """Central feature calculation engine.

    Usage::

        engine = FeatureEngine()
        features = engine.calculate(
            data=my_dataframe,
            features=["rsi", "atr", "sma"],
            params={"rsi": {"period": 14}, "sma": {"period": 50}},
        )
    """

    def __init__(self) -> None:
        self._registry = dict(FEATURE_REGISTRY)

    def register_feature(
        self,
        name: str,
        definition: FeatureDefinition,
        fn: Any,
    ) -> None:
        """Register a new custom feature at runtime.

        ``fn`` must accept ``(data: pd.DataFrame, **kwargs) -> pd.Series``.
        """
        self._registry[name] = definition
        _FEATURE_FN[name] = fn

    def list_features(self) -> dict[str, FeatureDefinition]:
        """Return a copy of the current feature registry."""
        return dict(self._registry)

    def calculate(
        self,
        data: pd.DataFrame,
        features: Sequence[str],
        params: dict[str, dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Compute a selection of features and return them as a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Input price data. Must contain the columns required by each feature
            (e.g. ``close``, ``high``, ``low``).
        features : list of str
            Feature names to compute (e.g. ``["rsi", "atr", "sma"]``).
        params : dict of str -> dict, optional
            Override default parameters per feature. For example::

                {"rsi": {"period": 14}, "sma": {"period": 50}}

        Returns
        -------
        pd.DataFrame
            Indexed by timestamp, with one column per computed feature.
        """
        params = params or {}
        result_columns: dict[str, pd.Series] = {}

        for name in features:
            if name not in self._registry:
                raise UnknownFeatureError(
                    f"Unknown feature '{name}'. "
                    f"Available: {sorted(self._registry.keys())}"
                )

            fn = _FEATURE_FN.get(name)
            if fn is None:
                raise FeatureError(f"Feature '{name}' has no registered compute function.")

            # Merge default params with user overrides
            defn = self._registry[name]
            kwargs = dict(defn.default_params)
            if name in params:
                kwargs.update(params[name])

            try:
                result = fn(data, **kwargs)
            except Exception as e:
                raise FeatureError(
                    f"Failed to compute feature '{name}': {e}"
                ) from e

            result_columns[name] = result

        result_df = pd.DataFrame(result_columns)
        result_df.index = data.sort_index().index
        return result_df

    def calculate_correlation(
        self,
        data_map: dict[str, pd.DataFrame],
        features: Sequence[str],
        params: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute correlation features across multiple instruments.

        ``features`` supports: ``"correlation_matrix"``, ``"pairwise_{a}_{b}"``,
        ``"rolling_{a}_{b}"``.

        Returns a dict keyed by feature name.
        """
        params = params or {}
        aligned = align_price_dfs(data_map)
        result: dict[str, Any] = {}

        for name in features:
            if name == "correlation_matrix":
                result[name] = correlation_matrix(
                    aligned, min_periods=params.get(name, {}).get("min_periods", 20)
                )
            elif name.startswith("pairwise_"):
                parts = name.split("_")
                if len(parts) < 3:
                    raise FeatureError(f"Invalid pairwise correlation name: '{name}'. Use 'pairwise_A_B'.")
                a, b = parts[1], "_".join(parts[2:])
                result[name] = pairwise_correlation(
                    aligned,
                    a,
                    b,
                    min_periods=params.get(name, {}).get("min_periods", 20),
                )
            elif name.startswith("rolling_"):
                parts = name.split("_")
                if len(parts) < 3:
                    raise FeatureError(f"Invalid rolling correlation name: '{name}'. Use 'rolling_A_B'.")
                a, b = parts[1], "_".join(parts[2:])
                p = params.get(name, {})
                result[name] = rolling_correlation(
                    aligned,
                    a,
                    b,
                    window=p.get("window", 20),
                    min_periods=p.get("min_periods"),
                )
            else:
                raise UnknownFeatureError(f"Unknown correlation feature '{name}'.")

        return result