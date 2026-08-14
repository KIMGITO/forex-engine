"""Step 15 — Walk-Forward / Out-of-Sample Validation.

Strictly chronological validation of Step 13 discovered hypotheses.

Architecture guarantees:
* TRAIN -> VALIDATION -> TEST are chronological and disjoint.
* Hypothesis generation/evaluation/selection uses TRAIN data ONLY.
* The frozen hypothesis is evaluated on the immediately-following unseen period.
* Training candidates whose label horizon crosses into validation/test are
  PURGED (excluded) — the preferred policy.
* Costs (spread + slippage + commission) are applied identically on training
  evaluation and out-of-sample evaluation.
* Repeated execution is deterministic (fixed seeds, no shuffling).
"""