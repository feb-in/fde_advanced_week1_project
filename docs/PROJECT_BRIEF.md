# Project Brief — 30-Day Hospital Readmission Risk

---

## Problem Statement

You are the **data science team for a regional hospital network**. Across the
network, too many **diabetic patients are readmitted within 30 days** of
discharge. These early readmissions are costly for the hospital and harmful for
patients, and the care team has limited capacity for proactive follow-up.

**Objective:** At the moment of discharge, produce a calibrated probability that
a given patient will be readmitted within 30 days. The score lets the care team
rank patients by risk and direct extra follow-up (calls, home visits, medication
review) to the highest-risk individuals first.

**Prediction target:** Binary — readmitted within 30 days (`<30`) vs. not. The
raw label has three classes (`<30`, `>30`, `NO`); collapse `>30` and `NO` into
the negative class. *Why binary:* the care team triggers one action — 30-day
follow-up, yes/no. A readmission at day 45 is outside the actionable window, so
`>30` and `NO` lead to the same action. Three classes also do not reduce the
~11% imbalance; you would collapse back to P(`<30`) for the decision anyway.

**Unit of prediction:** One patient discharge encounter (one row = one hospital
stay). The decision point is discharge time — **no post-discharge information
may enter the features**.

**What success looks like:** Not a notebook model — a *running, monitored,
governed service*: an API returning a risk score plus top contributing factors,
experiment tracking, a reproducible and versioned data pipeline, dashboards,
drift detection, a fairness audit, audit logs, a model card, and a documented
rollback/retrain plan.

**Input / serving pattern:**
- Training: a single pass over the fixed historical dataset (re-run on retrain).
- Serving: a real-time request/response API (score one discharge on demand).
  End-of-day batch scoring (a nightly ranked worklist) is a natural thin wrapper.

**Grading weight:** ~20% modeling, ~80% everything around it.

---

## Graded Traps & Caveats

These are the traps that silently cost marks. The hard rules in `CLAUDE.md`
enforce them; this section explains the *why*. Data facts verified against the
actual dataset (101,766 raw rows).

### 1. `?` = missing, not a category
Missing values in the source CSV are the literal string `?`, **not** blanks.
Untreated, the model reads `?` as a real value.
→ Load with `pd.read_csv(..., na_values=["?"], keep_default_na=False)`.

### 2. Heavy-missing columns: decide drop vs. keep, with evidence

| Column | % missing | Decision |
|---|---:|---|
| `weight` | **96.9%** | Drop — almost entirely empty, no recovery possible |
| `max_glu_serum` | **94.7%** | "not measured" is signal — keep as category |
| `A1Cresult` | **83.3%** | "not measured" is signal — keep as category |
| `medical_specialty` | **49.1%** | Fill NaN → `"Unknown"`, keep as category |
| `payer_code` | **39.6%** | Fill NaN → `"Unknown"`, keep as category |
| `race` | **2.2%** | Fill NaN → `"Unknown"`, keep for fairness audit |

### 3. A missing A1c is NOT a healthy A1c
"Not measured" is information — whether the clinician ordered the test is itself
a signal. The string `"None"` in `A1Cresult` / `max_glu_serum` means "test not
ordered." **Preserve it as its own category. Never convert to NaN or impute.**

### 4. First-encounter dedup is the leakage guard
The same `patient_nbr` appears in multiple encounters. We keep only the
**first encounter** (smallest `encounter_id`) per patient, then assert
`df["patient_nbr"].is_unique`. This ensures a patient never appears in both
train and test when we do a plain `StratifiedKFold` split.

### 5. Expired / hospice discharges can't be readmitted
`discharge_disposition_id` includes codes meaning the patient died or went to
hospice. Those rows can never be positive. Filter them out and document the
filter. Codes: {11, 13, 14, 19, 20, 21}.

### 6. ID columns are codes, not quantities
`admission_type_id`, `discharge_disposition_id`, `admission_source_id` are
**categorical integer codes**, not numeric magnitudes. Treat as categorical.

### 7. Dead / constant medication columns
`examide` and `citoglipton` have a single value for all rows — zero variance,
zero information. Drop them.

### 8. High-cardinality ICD-9 diagnoses
`diag_1`/`diag_2`/`diag_3` have ~700–800 unique codes each. Do not one-hot raw.
Bucket into Strack-9 clinical groups (circulatory, respiratory, diabetes,
digestive, injury, neoplasms, musculoskeletal, genitourinary, other). This is a
Stage 2 (Featurization) deliverable.

### 9. Accuracy is a trap
Positive class ≈ **11.2%** (11,357 `<30` / 35,545 `>30` / 54,864 `NO`).
"Always predict no" ≈ 89% accuracy and is useless.
**Headline metrics: PR-AUC, recall at fixed precision, calibration (Brier /
reliability curve).** Never report accuracy as the primary metric.

### 10. SMOTE is rejected
SMOTE fabricates ~54k synthetic clinical records, has weak defensibility in a
regulated domain, and hurts AUC here. Handle imbalance via class weights
(`class_weight="balanced"` in LR; `scale_pos_weight` in CatBoost). Document
SMOTE as "considered and rejected" in the model card.

### 11. Reproducibility, not one-off cleaning
The test is not "did cleaning steps run once" — it's "can those same steps
re-run on production data whose missingness differs." Build a script, version it
with DVC, and make it re-runnable with one command (`dvc repro`).

---

## Class Context

This project is the practical application of an ML-lifecycle course. The
philosophy below explains *why* the brief weights infrastructure over modeling.

### Six truths that last

1. **It's a loop, not a line.** Build → ship → watch → and back.
2. **Data work dominates.** Most effort — and most bugs — live in the data,
   not the model. (~70% of effort on data; training ≤ 5–10%.)
3. **Models decay.** Even a perfect model worsens as the world drifts from its
   training data.
4. **Simple usually wins.** The model you can ship, explain, and maintain beats
   the clever one you can't. Don't start with CatBoost; start with LR.
5. **Monitoring is the job.** Deploying *starts* the work.
6. **Boring is reliable.** Gradual rollouts, rollbacks, reproducible builds.

### The lifecycle stages (and where each goes wrong)

**1. Frame the problem.** Turn a fuzzy goal into a precise prediction target.
*Goes wrong:* optimizing accuracy when the rare case is what matters.
> *IBM Watson for Oncology* was framed around a goal the data couldn't support;
> unsafe recommendations surfaced and the effort was wound down.

**2. Data engineering.** Ingest, clean, validate, version. Build a reproducible
pipeline, not a one-off notebook. *Goes wrong:* data leakage; a pipeline only
its author can re-run.
> *Amazon's hiring AI* learned bias from a decade of mostly-male résumés —
> baked into the data, not the algorithm. Scrapped in 2017.

**3. Modeling.** Baseline first, then stronger models; tune; handle imbalance.
Only ~20% of the grade.

**4. Evaluation.** Metrics tied to the costly rare event; calibration; a
justified operating threshold.

**5. Deploy.** Package, containerize, expose an API; plan rollback. Deploying
is the *start* of the work.

**6. Monitor.** Service metrics, prediction logging, drift detection, a concrete
retrain trigger. *Goes wrong:* monitoring treated as optional.

**7. Govern (continuous).** Fairness audits, SHAP explanations, model cards,
audit logs, human review of low-confidence cases.
> *Dutch childcare-benefits scandal:* a risk-scoring algorithm wrongly flagged
> thousands of families — disproportionately minorities — for fraud. The fallout
> contributed to the **entire Dutch government resigning in 2021**. Ungoverned
> models can do societal-scale harm.

### Three thresholds, set on day one
Define up front the thresholds for when you **keep** the model running, when you
**retrain**, and when you **retire** it.
