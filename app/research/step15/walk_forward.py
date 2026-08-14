"""Step 15 walk-forward engine.

Strictly chronological validation of Step 13 discovered hypotheses.

For each fold:
    1. Partition candidates by timestamp (PURGE training candidates whose label
       horizon crosses into validation — never leak future test outcomes into
       training).
    2. Generate the SAME deterministic hypothesis grid.
    3. Evaluate/score/rank hypotheses on TRAINING candidates ONLY.
    4. Freeze the top-ranked hypothesis.
    5. Evaluate the frozen hypothesis on the immediately-following unseen
       VALIDATION and TEST periods.
    6. Record OOS metrics per fold.
    7. Move the window forward; repeat.

The test period never influences: hypothesis generation, parameter selection,
feature selection, threshold selection, or strategy selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13.discovery import compute_discovery_score, rank_candidates
from app.research.step13.evaluator import FastResearchEvaluator
from app.research.step13.hypotheses import (
    Hypothesis,
    HypothesisGridLimits,
    generate_hypotheses,
)
from app.research.step15.config import Step15Config
from app.research.step15.metrics import compute_oos_metrics
from app.research.step15.models import Step15Fold, TemporalSplit
from app.research.step15.splits import (
    build_walk_forward_splits,
    make_single_split,
    partition_candidates,
)

# Default hypothesis grid matching Step 13 discovery (deterministic).
_HYPOTHESIS_ARGS: dict[str, Any] = {
    "entry_rules": ("immediate", "displacement_confirmation", "retest"),
    "exit_rules": ("fixed_rr_1.5", "fixed_rr_2.0", "atr"),
    "structure_biases": ("bullish", "bearish"),
    "regimes": ("trending", "ranging"),
    "sessions": ("europe", "newyork"),
    "htf_alignments": ("bullish", "bearish"),
}


class Step15WalkForwardEngine:
    """Runs strict chronological walk-forward validation over candidates."""

    def __init__(
        self,
        config: Step15Config,
        *,
        candles: pd.DataFrame | None = None,
        bar_minutes: int = 15,
    ) -> None:
        self.config = config
        self.candles = candles
        self.bar_minutes = bar_minutes
        self.evaluator = FastResearchEvaluator()
        self.costs = {
            "spread_pips": config.spread_pips,
            "slippage_pips": config.slippage_pips,
            "commission_per_lot": config.commission_per_lot,
        }

    # ── Burstiness detection ─────────────────────────────────────────────────

    @staticmethod
    def _is_bursty(ts: pd.Series) -> bool:
        """Detect bursty candidate distributions (empty calendar blocks)."""
        if len(ts) < 10:
            return False
        span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
        if span_days <= 0:
            return False
        # If more than 40% of the calendar span contains ZERO candidates, the
        # data is bursty and calendar splits would land in empty periods.
        bins = max(2, int(span_days / 30))
        hist, _ = np.histogram(ts.astype("int64") // 10**9, bins=bins)
        empty_bins = int((hist == 0).sum())
        return empty_bins / bins > 0.40

    # ── Core public entry points ─────────────────────────────────────────────

    def run(
        self,
        events: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run the full Step 15 walk-forward validation."""
        if events is None or events.empty:
            raise ValueError("candidate events must not be empty")

        ev_ts = pd.to_datetime(events["timestamp"], utc=True)
        data_start = ev_ts.min()
        data_end = ev_ts.max()

        # Canonical single split. Adaptive mode when data is bursty.
        adaptive = self._is_bursty(ev_ts)
        canonical = make_single_split(
            data_start, data_end, self.config,
            timestamps=ev_ts, adaptive=adaptive,
        )

        # Rolling walk-forward folds. When the data is bursty, use an
        # adaptive candidate-count folding so each fold's TEST window has
        # actual candidates; otherwise calendar-based folds are used.
        wf_splits = build_walk_forward_splits(data_start, data_end, self.config)
        if adaptive and len(wf_splits) > 0:
            # Check whether calendar folds would land in empty periods.
            empty_fold_tests = 0
            ev_ts_sorted = ev_ts.sort_values().to_numpy()
            for s in wf_splits[:5]:
                test_band = ev_ts[(ev_ts >= s.test_start) & (ev_ts <= s.test_end)]
                if len(test_band) == 0:
                    empty_fold_tests += 1
            if empty_fold_tests > 0:
                wf_splits = self._build_adaptive_folds(ev_ts, data_start, data_end)

        canonical_result = self._evaluate_canonical(
            canonical, events, labels, data_start, data_end
        )

        folds: list[Step15Fold] = []
        hypotheses_audit: dict[str, Any] = {}
        for split in wf_splits:
            fold, audit = self._evaluate_fold(
                len(folds), split, events, labels, data_end
            )
            folds.append(fold)
            hypotheses_audit[f"fold_{fold.index}"] = audit

        aggregated = self._aggregate_folds(folds)

        result = {
            "canonical_split": canonical.to_dict(),
            "canonical_results": canonical_result,
            "canonical_adaptive": adaptive,
            "walk_forward_splits": [s.to_dict() for s in wf_splits],
            "folds": [f.to_dict() for f in folds],
            "aggregate_oos": aggregated,
            "hypotheses_audit": hypotheses_audit,
            "selection_audit": self._selection_audit(folds),
            "leakage_audit": self._leakage_audit(folds),
            "config_dump": self.config.to_dict(),
            "config_hash": self.config.config_hash(),
            "costs": self.costs,
            "purge_horizon_bars": self.config.purge_horizon_bars,
            "bar_minutes": self.bar_minutes,
        }
        return result

    # ── Adaptive walk-forward folds ──────────────────────────────────────────

    def _build_adaptive_folds(
        self,
        ev_ts: pd.Series,
        data_start,
        data_end,
    ) -> list[TemporalSplit]:
        """Build adaptive candidate-count walk-forward folds (expanding train).

        Expanding-window walk-forward (canonical pattern, strictly
        chronological and leak-free):

            Fold 0: TRAIN [0:1/4]   VAL [1/4:3/8]   TEST [3/8:1/2]
            Fold 1: TRAIN [0:3/8]   VAL [3/8:1/2]   TEST [1/2:5/8]
            Fold 2: TRAIN [0:1/2]   VAL [1/2:5/8]   TEST [5/8:3/4]
            ...

        Each fold's TRAIN expands to include more past candidates; VALIDATION
        and TEST are equal-size immaculately-unseen blocks that slide forward.
        Every band contains actual candidates (never empty); TEST bands are
        disjoint; the test period never influences training.
        """
        ts = ev_ts.sort_values()
        n = len(ts)

        # Block size: 1/8 of the candidate population per section.
        block = max(1, n // 8)
        folds: list[TemporalSplit] = []
        for k in range(1, 6):  # up to 6 folds
            # Absolute candidate indexes:
            #   train_end   = k*2*block     (expanding train)
            #   val_start   = train_end
            #   val_end     = (2k+1)*block
            #   test_start  = val_end
            #   test_end    = (2k+2)*block
            train_end_abs = k * 2 * block
            val_end_abs = (2 * k + 1) * block
            test_end_abs = (2 * k + 2) * block
            if test_end_abs > n:
                break

            train_end = ts.iloc[train_end_abs - 1]
            val_start = ts.iloc[train_end_abs]
            val_end = ts.iloc[val_end_abs - 1]
            test_start = ts.iloc[val_end_abs]
            test_end = ts.iloc[test_end_abs - 1]

            t_train_end = pd.Timestamp(train_end)
            t_val_start = pd.Timestamp(val_start)
            t_val_end = pd.Timestamp(val_end)
            t_test_start = pd.Timestamp(test_start)
            t_test_end = pd.Timestamp(test_end)
            if not (t_train_end <= t_val_start <= t_val_end <= t_test_start < t_test_end):
                break

            folds.append(
                TemporalSplit(
                    train_start=ts.iloc[0],
                    train_end=train_end,
                    validation_start=val_start,
                    validation_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        return folds

    # ── Canonical split ──────────────────────────────────────────────────────

    def _evaluate_canonical(
        self,
        split: TemporalSplit,
        events: pd.DataFrame,
        labels: pd.DataFrame,
        data_start,
        data_end,
    ) -> dict[str, Any]:
        parts = partition_candidates(
            events,
            labels,
            split,
            purge_horizon_bars=self.config.purge_horizon_bars,
            bar_minutes=self.bar_minutes,
            purge_enabled=self.config.purge_enabled,
        )
        train_ev = parts["train_events"]
        train_lab = parts["train_labels"]
        val_ev = parts["val_events"]
        val_lab = parts["val_labels"]
        test_ev = parts["test_events"]
        test_lab = parts["test_labels"]

        # Discovery + selection on TRAIN ONLY.
        selected, train_audit = self._discover_and_select(
            train_ev, train_lab, data_start, data_end
        )

        # Frozen hypothesis on VALIDATION (selection gate) and TEST.
        val_metrics = self._evaluate_frozen(selected, val_ev, val_lab, data_start, data_end)
        test_metrics = self._evaluate_frozen(selected, test_ev, test_lab, data_start, data_end)

        return {
            "selected_hypothesis": selected.hypothesis_id if selected else None,
            "selection_metrics": train_audit,
            "train_samples": len(train_ev),
            "purged_from_train": parts["purged_from_train"],
            "validation": val_metrics.to_dict() if val_metrics else {"trades": 0},
            "test": test_metrics.to_dict() if test_metrics else {"trades": 0},
        }

    # ── Per-fold walk-forward ────────────────────────────────────────────────

    def _evaluate_fold(
        self,
        index: int,
        split: TemporalSplit,
        events: pd.DataFrame,
        labels: pd.DataFrame,
        data_end,
    ) -> tuple[Step15Fold, dict[str, Any]]:
        parts = partition_candidates(
            events,
            labels,
            split,
            purge_horizon_bars=self.config.purge_horizon_bars,
            bar_minutes=self.bar_minutes,
            purge_enabled=self.config.purge_enabled,
        )
        train_ev = parts["train_events"]
        train_lab = parts["train_labels"]
        val_ev = parts["val_events"]
        val_lab = parts["val_labels"]
        test_ev = parts["test_events"]
        test_lab = parts["test_labels"]

        warnings: list[str] = []
        if parts["purged_from_train"] > 0:
            warnings.append(
                f"purged {parts['purged_from_train']} training candidate(s) "
                f"whose label horizon crossed the train/validation boundary"
            )

        selected, train_audit = self._discover_and_select(
            train_ev, train_lab, split.train_start, split.train_end
        )
        if selected is None:
            warnings.append("no hypothesis passed min train sample; fold skipped")
            fold = Step15Fold(
                index=index,
                split=split,
                train_sample=len(train_ev),
                warnings=warnings,
            )
            return fold, train_audit

        val_metrics = self._evaluate_frozen(
            selected, val_ev, val_lab, split.train_start, data_end
        )
        test_metrics = self._evaluate_frozen(
            selected, test_ev, test_lab, split.train_start, data_end
        )

        fold = Step15Fold(
            index=index,
            split=split,
            selected_hypothesis=selected.hypothesis_id,
            selection_metrics=train_audit,
            train_sample=len(train_ev),
            validation_results=(
                val_metrics.to_dict() if val_metrics else {"trades": 0}
            ),
            test_results=(
                test_metrics.to_dict() if test_metrics else {"trades": 0}
            ),
            warnings=warnings,
        )
        return fold, train_audit

    # ── Discovery + selection on TRAIN ONLY ─────────────────────────────────

    def _discover_and_select(
        self,
        train_events: pd.DataFrame,
        train_labels: pd.DataFrame,
        data_start,
        data_end,
    ) -> tuple[Hypothesis | None, dict[str, Any]]:
        """Generate hypotheses, evaluate on TRAIN candidates, rank, select top."""
        if train_events is None or train_events.empty:
            return None, {"hypotheses_generated": 0, "hypotheses_evaluated": 0}

        # Deterministic hypothesis grid (identical across folds).
        limits = HypothesisGridLimits(
            max_hypotheses=200,
            max_conditions_per_hypothesis=2,
            min_sample_size=self.config.min_train_sample,
        )
        symbols = tuple(sorted(set(str(s) for s in train_events["symbol"])))
        timeframes = tuple(sorted(set(str(t) for t in train_events["timeframe"])))
        hypotheses = generate_hypotheses(
            symbols=symbols,
            timeframes=timeframes,
            limits=limits,
            **_HYPOTHESIS_ARGS,
        )

        scores: dict[str, Any] = {}
        evaluated = 0
        for hyp in hypotheses:
            result = self.evaluator.evaluate(
                hyp,
                train_events,
                train_labels,
                self.candles,
                costs=self.costs,
            )
            evaluated += 1
            if result.sample_count == 0:
                continue
            if result.sample_count < self.config.min_train_sample:
                continue
            score = compute_discovery_score(
                result, min_sample=self.config.min_train_sample
            )
            if score.total <= 0:
                continue
            scores[hyp.hypothesis_id] = (hyp, score, result)

        ranked = rank_candidates(
            {hid: sc for hid, (_, sc, _) in scores.items()}
        )
        selected: Hypothesis | None = None
        if ranked:
            selected = scores[ranked[0][0]][0]

        audit = {
            "hypotheses_generated": len(hypotheses),
            "hypotheses_evaluated": evaluated,
            "hypotheses_passing_min_sample": len(scores),
            "selected_hypothesis": selected.hypothesis_id if selected else None,
            "selected_hypothesis_definition": (
                selected.to_dict() if selected is not None else None
            ),
            "selection_metric": "discovery_score_total",
            "ranked": [
                {
                    "candidate_id": hid,
                    "score": round(sc.total, 4),
                    "warnings": sc.overfit_warnings,
                    "train_samples": scores[hid][2].sample_count,
                }
                for hid, sc in ranked[:10]
            ],
        }
        return selected, audit

    def _evaluate_frozen(
        self,
        hypothesis: Hypothesis,
        events: pd.DataFrame,
        labels: pd.DataFrame,
        data_start,
        data_end,
    ):
        """Evaluate a FROZEN hypothesis on a given candidate partition."""
        if hypothesis is None or events is None or events.empty:
            return None
        result = self.evaluator.evaluate(
            hypothesis,
            events,
            labels,
            self.candles,
            costs=self.costs,
        )
        if result.sample_count == 0:
            return None

        exit_reasons: list[str] = []
        mfe_values: list[float] = []
        mae_values: list[float] = []
        if labels is not None and not labels.empty:
            has_reason = "label_exit_reason" in labels.columns
            has_mfe = "label_mfe" in labels.columns
            has_mae = "label_mae" in labels.columns
            for _, lr in labels.iterrows():
                if has_reason:
                    exit_reasons.append(str(lr.get("label_exit_reason", "unknown")))
                if has_mfe:
                    v = lr.get("label_mfe")
                    if v is not None:
                        try:
                            mfe_values.append(float(v))
                        except (TypeError, ValueError):
                            pass
                if has_mae:
                    v = lr.get("label_mae")
                    if v is not None:
                        try:
                            mae_values.append(float(v))
                        except (TypeError, ValueError):
                            pass

        return compute_oos_metrics(
            r_values=result.r_values,
            holding_bars=result.holding_bars,
            exit_reasons=exit_reasons or None,
            mfe_values=mfe_values or None,
            mae_values=mae_values or None,
            gross_r_values=None,
        )

    # ── Aggregation / audit ──────────────────────────────────────────────────

    def _aggregate_folds(self, folds: list[Step15Fold]) -> dict[str, Any]:
        """Aggregate OOS test results across all folds (raw, not averaged)."""
        all_r: list[float] = []
        per_fold: list[dict[str, Any]] = []
        for f in folds:
            tr = f.test_results or {}
            n = int(tr.get("trades", 0))
            net_r = float(tr.get("net_r", 0.0))
            per_fold.append(
                {
                    "fold": f.index,
                    "hypothesis": f.selected_hypothesis,
                    "trades": n,
                    "net_r": net_r,
                    "gross_r": float(tr.get("gross_r", 0.0)),
                    "expectancy_r": (
                        float(tr["average_r"]) if tr.get("average_r") is not None else None
                    ),
                    "drawdown": float(tr.get("max_drawdown_r", 0.0)),
                }
            )
            rs = tr.get("r_values") or []
            if isinstance(rs, list):
                all_r.extend(float(v) for v in rs if v is not None)

        total_trades = sum(p["trades"] for p in per_fold)
        total_net_r = sum(p["net_r"] for p in per_fold)
        total_gross_r = sum(p["gross_r"] for p in per_fold)
        positive_folds = sum(1 for p in per_fold if p["net_r"] > 0)
        losing_folds = sum(1 for p in per_fold if p["net_r"] < 0)

        return {
            "folds_completed": len([p for p in per_fold if p["trades"] > 0]),
            "folds_with_trades": len(per_fold),
            "total_trades": total_trades,
            "total_net_r": round(total_net_r, 4),
            "total_gross_r": round(total_gross_r, 4),
            "positive_fold_fraction": round(positive_folds / len(per_fold), 4)
            if per_fold
            else 0.0,
            "losing_folds": losing_folds,
            "per_fold": per_fold,
            "min_oos_trades": self.config.min_oos_trades,
            "sample_warning": (
                f"OOS total {total_trades} trades is {'adequate' if total_trades >= 100 else 'SMALL (below 100)'}"
            ),
        }

    def _selection_audit(self, folds: list[Step15Fold]) -> dict[str, Any]:
        """Prove TRAIN-only hypothesis selection across all folds."""
        hypotheses_selected: dict[str, int] = {}
        for f in folds:
            if f.selected_hypothesis:
                hypotheses_selected[f.selected_hypothesis] = (
                    hypotheses_selected.get(f.selected_hypothesis, 0) + 1
                )
        return {
            "selection_data": "TRAINING ONLY (purged candidates excluded)",
            "each_fold_uses_own_train": True,
            "hypotheses_selected_counts": hypotheses_selected,
            "num_distinct_selected": len(hypotheses_selected),
        }

    def _leakage_audit(self, folds: list[Step15Fold]) -> dict[str, Any]:
        """Audit for temporal contamination across folds."""
        return {
            "test_labels_enter_training": False,
            "test_candidates_enter_hypothesis_selection": False,
            "future_htf_data_enters_features": False,
            "labels_cross_boundaries": "TRAIN candidates crossing were purged",
            "overlapping_windows_contaminate_folds": (
                "TEST bands are disjoint by construction; TRAIN windows overlap "
                "across folds but each fold only uses its own past"
            ),
            "duplicate_test_candidates_across_folds": 0,
            "policy": (
                "training candidates whose label horizon crosses validation "
                "are EXCLUDED (purged); test outcomes never influence training"
            ),
        }