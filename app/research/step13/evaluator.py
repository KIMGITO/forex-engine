"""Fast research evaluator for Step 13 Alpha Discovery.

Given a hypothesis and a compact candidate feature/label event table, this
evaluator computes per-outcome statistics WITHOUT running the full
EventBacktester per hypothesis. It:

1. Filters qualifying events by the hypothesis conditions.
2. Projects future bars for each qualifying event.
3. Computes R-outcome under the entry/stop/exit rules.
4. Aggregates: expectancy, win rate, profit factor, drawdown, streaks, etc.

The evaluator is deliberately VECTORIZED and memory-bounded: it uses compact
numpy operations over the candidate_events / candidate_labels parquet tables
produced by the extraction pipeline. No giant Pydantic object lists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13.execution_model import simulate_hypothesis_outcome
from app.research.step13.hypotheses import Hypothesis, conditions_pass


class ResearchEvaluatorResult:
    """Structured output of evaluating one hypothesis."""

    def __init__(
        self,
        hypothesis: Hypothesis,
        sample_count: int,
        r_values: list[float],
        holding_bars: list[int],
        win_count: int,
        loss_count: int,
        breakeven_count: int,
        groups: dict[str, list[float]] | None = None,
    ) -> None:
        self.hypothesis = hypothesis
        self.sample_count = sample_count
        self.r_values = r_values
        self.holding_bars = holding_bars
        self.win_count = win_count
        self.loss_count = loss_count
        self.breakeven_count = breakeven_count
        self.groups = groups or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis.hypothesis_id,
            "sample_count": self.sample_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "breakeven_count": self.breakeven_count,
            "mean_r": float(np.mean(self.r_values)) if self.r_values else None,
            "median_r": float(np.median(self.r_values)) if self.r_values else None,
            "std_r": float(np.std(self.r_values)) if len(self.r_values) > 1 else None,
            "win_rate": round(self.win_count / self.sample_count, 4) if self.sample_count else 0.0,
            "avg_holding_bars": float(np.mean(self.holding_bars)) if self.holding_bars else None,
        }


class FastResearchEvaluator:
    """Vectorized hypothesis evaluator over compact event tables."""

    def evaluate(
        self,
        hypothesis: Hypothesis,
        candidate_events: pd.DataFrame,
        candidate_labels: pd.DataFrame | None = None,
        candles: pd.DataFrame | None = None,
        *,
        costs: dict[str, float] | None = None,
    ) -> ResearchEvaluatorResult:
        """Evaluate a hypothesis.

        Parameters
        ----------
        hypothesis : the deterministic hypothesis to evaluate.
        candidate_events : DataFrame with candidate_id / timestamp / direction
            / feature_* columns (from ``candidate_events.parquet``).
        candidate_labels : optional DataFrame with candidate_id / label_*
            columns (from ``candidate_labels.parquet``).
        candles : optional base OHLC DataFrame (used to compute R outcomes
            when labels do not provide them).

        Returns a ResearchEvaluatorResult.
        """
        if candidate_events is None or candidate_events.empty:
            return ResearchEvaluatorResult(hypothesis, 0, [], [], 0, 0, 0)

        qualifying = self._filter_qualifying(hypothesis, candidate_events)
        if qualifying.empty:
            return ResearchEvaluatorResult(hypothesis, 0, [], [], 0, 0, 0)

        costs = costs or {}
        spread_pips = float(costs.get("spread_pips", 0.0))
        slippage_pips = float(costs.get("slippage_pips", 0.0))
        commission_per_lot = float(costs.get("commission_per_lot", 0.0))

        # Map candidate_id -> label row for fast lookup.
        label_map: dict[Any, pd.Series] = {}
        if candidate_labels is not None and not candidate_labels.empty:
            for _, lab_row in candidate_labels.iterrows():
                label_map[lab_row.get("candidate_id")] = lab_row

        r_values: list[float] = []
        holding_bars: list[int] = []
        win = loss = breakeven = 0
        groups: dict[str, list[float]] = {}

        for _, row in qualifying.iterrows():
            cand_id = row.get("candidate_id")
            direction = str(row.get("direction", ""))
            outcome = None
            if candles is not None and not candles.empty:
                outcome = simulate_hypothesis_outcome(
                    hypothesis, row.to_dict(), candles,
                    spread_pips=spread_pips,
                    slippage_pips=slippage_pips,
                    commission_per_lot=commission_per_lot,
                )
            r = (
                outcome["r"]
                if outcome is not None
                else self._r_for_candidate(row, label_map.get(cand_id), candles)
            )
            if r is None:
                continue
            r_values.append(float(r))
            bars = (
                outcome.get("holding_bars", 0)
                if outcome is not None
                else self._holding_for_candidate(label_map.get(cand_id))
            )
            holding_bars.append(bars)
            if r > 0:
                win += 1
            elif r < 0:
                loss += 1
            else:
                breakeven += 1

            session = str(row.get("feature_session", "") or "unknown")
            regime = str(row.get("regime", "") or "unknown")
            symbol = str(row.get("symbol", "") or "unknown")
            groups.setdefault(f"session={session}", []).append(float(r))
            groups.setdefault(f"regime={regime}", []).append(float(r))
            groups.setdefault(f"symbol={symbol}", []).append(float(r))

        return ResearchEvaluatorResult(
            hypothesis=hypothesis,
            sample_count=len(r_values),
            r_values=r_values,
            holding_bars=holding_bars,
            win_count=win,
            loss_count=loss,
            breakeven_count=breakeven,
            groups=groups,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _filter_qualifying(
        self, hypothesis: Hypothesis, candidate_events: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter candidate events by hypothesis event type + direction + conditions."""
        df = candidate_events.copy()

        # Event type filter from strategy_family.
        if "strategy_family" in df.columns:
            family = df["strategy_family"].fillna("")
            if hypothesis.event_type == "liquidity_sweep":
                df = df[family.str.contains("sweep", na=False)]
            elif hypothesis.event_type == "displacement":
                df = df[family.str.contains("displacement", na=False)]
            elif hypothesis.event_type == "structure_break":
                df = df[family.str.contains("structure", na=False)]

        # Direction filter.
        if "direction" in df.columns:
            df = df[df["direction"].fillna("") == hypothesis.direction]

        # Conditions filter.
        if hypothesis.conditions:
            mask = df.apply(
                lambda row: conditions_pass(hypothesis, row.to_dict()),
                axis=1,
            )
            df = df[mask]

        return df

    @staticmethod
    def _r_for_candidate(
        row: pd.Series,
        label_row: pd.Series | None,
        candles: pd.DataFrame | None,
    ) -> float | None:
        """Compute the R-outcome for one candidate.

        Priority: hypothesis-aware ``label_r`` from the label table, then
        candles fallback.
        """
        if label_row is not None:
            r = label_row.get("label_r")
            if r is not None:
                return float(r)
            # Backward-compat fallback: old label schema approximated from MFE/MAE.
            mfe = label_row.get("label_mfe")
            mae = label_row.get("label_mae")
            if mfe is not None and mae is not None:
                atr = float(row.get("feature_atr", 0.01) or 0.01)
                if atr > 0:
                    return round((float(mfe) - float(mae)) / (atr * 2.0), 4)
        if candles is not None and not candles.empty:
            ts = pd.Timestamp(row.get("timestamp"))
            if ts in candles.index:
                idx = candles.index.get_loc(ts)
                if idx + 1 < len(candles):
                    entry = float(candles.iloc[idx]["close"])
                    future = float(candles.iloc[min(idx + 20, len(candles) - 1)]["close"])
                    atr = float(row.get("feature_atr", 0.01) or 0.01)
                    direction = 1.0 if row.get("direction") == "long" else -1.0
                    if atr > 0:
                        return round(direction * (future - entry) / (atr * 2.0), 4)
        return None

    @staticmethod
    def _holding_for_candidate(label_row: pd.Series | None) -> int:
        if label_row is not None:
            return int(label_row.get("label_excursion_after_bars", 0) or 0)
        return 0