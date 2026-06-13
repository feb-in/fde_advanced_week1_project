# Calibration & Operating-Threshold Decision

This note explains how we turned the tuned CatBoost model's raw scores into
**trustworthy probabilities** and how we chose the **risk cutoff** at which a
patient is flagged for follow-up. It is written to be readable by a care-team lead,
not just a data scientist.

## First, what calibration and the threshold are NOT

Calibration and the threshold are **deployment decisions**. They do **not** change
how well the model *ranks* patients — that is fixed by the model itself and measured
by AUPRC and ROC-AUC. Calibration only rescales the scores so a "0.30" actually
means "about a 30% chance of readmission," and the threshold only decides where we
draw the flag/no-flag line. We prove ranking is untouched below.

The model was trained and tuned in earlier gates; the held-out **test set
(13,998 patients) is the same one used throughout, scored exactly once here**.
Calibration was fit and the threshold chosen on a **separate internal validation
slice of the training data** (`tr_val`, 11,198 patients) — never on the test set.

## Calibration: isotonic vs Platt (sigmoid)

We fit both standard methods **without leakage** (`CalibratedClassifierCV` with
5-fold cross-validation on the training portion — the model is never calibrated on
its own training rows) and compared them on the held-out `tr_val` slice using the
**Brier score** (lower = better-calibrated; it rewards probabilities that match
reality).

| | Brier on `tr_val` |
|---|---:|
| Raw (uncalibrated) | 0.0994 |
| Isotonic | 0.0791 |
| **Platt / sigmoid (chosen)** | **0.0790** |

Both methods improve calibration dramatically over the raw scores (~0.099 → ~0.079).
The two are effectively **tied**, so we chose **Platt (sigmoid)**: it fits just two
parameters (a logistic rescaling), which is more robust and less prone to
overfitting the calibration folds than isotonic's flexible step function — the
sensible default when the data is moderately sized and the two scores are a
coin-flip apart. The reliability curves for raw / isotonic / sigmoid are logged to
MLflow (`reliability_val.png`).

### Proof that calibration did not touch ranking

On the held-out test set, before vs after calibration:

| metric | uncalibrated | calibrated | Δ |
|---|---:|---:|---:|
| AUPRC | 0.2056 | 0.2071 | +0.0015 |
| ROC-AUC | 0.6661 | 0.6676 | +0.0015 |
| **Brier** | **0.0982** | **0.0778** | **−0.0204** |

AUPRC and ROC-AUC are **statistically unchanged** (Δ ≈ 0.0015 is noise; Platt is a
strictly monotonic transform, so it provably cannot re-order patients — the tiny
delta comes from the calibrated model being a 5-fold ensemble, not from the
calibration map). Meanwhile **Brier drops ~21%** — exactly the intended effect:
same ranking, far more trustworthy probabilities.

## Operating threshold: lean toward recall

This is a **screening tool**. A **missed 30-day readmission is far costlier** — to
the patient and the hospital — than an extra follow-up phone call to someone who
would not have been readmitted. So we lean toward **recall** (catching true
readmissions), accepting lower precision, bounded by follow-up capacity.

We targeted **recall ≈ 0.50 — catch about half of all true 30-day readmissions** —
and read the threshold off the `tr_val` calibrated scores. The capacity trade-off
at nearby targets (on `tr_val`):

| recall target | threshold | precision | flag rate (share of discharges flagged) |
|---:|---:|---:|---:|
| 0.40 | 0.1075 | 0.163 | 22.0% |
| **0.50 (chosen)** | **0.0910** | **0.148** | **30.4%** |
| 0.60 | 0.0796 | 0.133 | 40.7% |

**Chosen threshold = 0.091** (calibrated probability). Reasoning: 0.50 recall is the
point where we still catch half of readmissions at a flag rate (~30%) that is
plausible for phone-call-level follow-up; pushing to 0.60 recall flags 40%+ of all
discharges for a small recall gain, and 0.40 is the fallback if follow-up capacity
is tighter. The lead can move this dial — the threshold is a stored, swappable
parameter, not baked into the model.

## Resulting operating point on the held-out test set (scored once)

At threshold **0.091** on the **test set (13,998 patients, 1,257 true readmits)**:

| metric | value |
|---|---:|
| Precision | 0.154 |
| Recall | 0.511 |
| Flag rate | 29.7% |

Confusion matrix:

| | predicted no-flag | predicted flag |
|---|---:|---:|
| **actually not readmitted** | TN 9,222 | FP 3,519 |
| **actually readmitted <30d** | FN 615 | TP 642 |

**Reading it:** of 1,257 patients who were truly readmitted within 30 days, the tool
flags **642 (51%)**. About **1 in 6.5 flagged patients** (precision 0.154) is a true
readmission — acceptable for a low-cost intervention (a call / medication review).

**Alert volume:** ~29.7% of discharges are flagged. At an illustrative **50 diabetic
discharges/day**, that is **~15 flagged patients/day** to work through — scale
linearly to your actual discharge volume.

## Registration & rollback

The final calibrated model (Platt-calibrated, refit on the full training portion) is
logged to the MLflow Model Registry as **`readmission-catboost-calibrated` version 1**
and transitioned to **Staging**. The chosen threshold (`0.091046`) and calibration
method (`sigmoid`) are attached as model-version tags so serving reads them from the
registry, not from code.

**The registry stage transition IS the rollback mechanism.** The API loads whatever
version is in `Production`; promotion is `Staging → Production`, and rollback is
moving the previous known-good version back to `Production` (and the regressed one
to `Archived`). MLflow 3.x is migrating from stages to aliases, so we also set the
`staging` **alias** to version 1 as the forward-compatible mirror — either handle
resolves the same artifact.

## Still deferred (next gates)
- Serving the model behind a FastAPI `/predict` that returns this calibrated score
  plus top SHAP factors, applying the stored 0.091 threshold for the flag.
- Promotion `Staging → Production` once the API + monitoring are in place.
- The fairness audit will re-examine this single global threshold across age / gender
  / race subgroups (a fixed cutoff can land differently per group).
