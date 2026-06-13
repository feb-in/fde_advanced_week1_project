# Model Card — 30-Day Diabetic Readmission Risk

A care-team decision-support model: at the moment of discharge, it produces a
**calibrated probability** that a diabetic patient will be **readmitted within 30
days**, plus the **top contributing factors** for that patient. It exists to help a
capacity-limited care team **rank** patients and direct follow-up (calls, home visits,
medication review) to the highest-risk first.

| | |
|---|---|
| **Model** | `readmission-catboost-calibrated` **v1 @ `staging`** (MLflow registry) |
| **Type** | CatBoost gradient-boosted trees, Optuna-tuned, **sigmoid (Platt) calibrated** |
| **Baseline retained** | Logistic regression (explainable floor / fallback) |
| **Operating threshold** | **0.091046** (recall-leaning; stored as a registry tag, swappable) |
| **Status** | Staging. Not cleared for autonomous clinical action — decision-support only |
| **Owners** | Readmission ML project team |

---

## Intended use

- **Use it for:** ranking just-discharged diabetic inpatients by 30-day readmission
  risk so finite follow-up capacity goes to the highest-risk patients first; surfacing
  per-patient contributing factors to inform (not replace) clinical judgement.
- **Decision point:** discharge time. **No post-discharge information enters the
  features** — anything knowable only after discharge would be leakage.
- **Do NOT use it for:** denying or rationing care; any automated/punitive action;
  patients outside the training population (non-diabetic, paediatric — see Limits);
  or as a diagnosis. A flag means "worth a follow-up contact," nothing more.
- **Users:** a clinical care team / discharge planners, with a human in the loop.

---

## Training data

- **Source:** UCI "Diabetes 130-US hospitals" dataset — **101,766 encounters,
  1999–2008**, 130 US hospitals.
- **Cleaning & leakage controls** (`src/data/clean.py`, logged in `docs/FEATURE_LOG.md`):
  - `?` loaded as missing (`na_values=["?"], keep_default_na=False`) — never a category.
  - Dropped expired/hospice discharges (`discharge_disposition_id ∈ {11,13,14,19,20,21}`,
    −2,423) — those patients cannot be readmitted.
  - **First-encounter dedup** (keep `min(encounter_id)` per `patient_nbr`, −~29k) — the
    leakage guard; guarantees one patient = one row, so a plain stratified split is safe.
  - `"None"` in `A1Cresult` / `max_glu_serum` **preserved as "test not ordered"** — real
    signal, not a null. `weight` (96.9% missing) and constant `examide`/`citoglipton`
    dropped.
  - **Final: 69,987 patients, positive rate 0.0898 (~9%).**
- **Target:** binary — `readmitted == "<30"` → 1; `>30` and `NO` → 0. The care action
  (30-day follow-up) is the same for `>30` and `NO`, so they collapse to the negative.
- **Features:** 54 model inputs (16 categorical), incl. Strack-9 ICD-9 diagnosis
  buckets, service-utilization counts, medication-change counts, A1c×med-change
  interaction, and demographic/administrative buckets. Full list: `docs/FEATURE_LOG.md`.
- **Split:** stratified 80/20, **seed 42** → train 55,989 / **test 13,998** (1,257 true
  readmits). The test set is touched once, at final evaluation.
- **Imbalance handling:** class weights (`auto_class_weights="SqrtBalanced"`).
  **SMOTE was considered and rejected** — it would fabricate ~54k synthetic clinical
  records (weak defensibility in a regulated domain) and hurt AUPRC here.

---

## Performance (held-out test set, scored once)

Headline metrics are PR-based and calibration — **not accuracy** (always-predict-"no"
scores ~91% here and is useless). No-skill AUPRC = prevalence ≈ **0.090**.

| metric | value | note |
|---|---:|---|
| **AUPRC** | **0.207** | ~2.3× the no-skill floor (0.090) |
| **ROC-AUC** | **0.668** | secondary (optimistic under imbalance) |
| **Brier** | **0.078** | after calibration (was 0.098 raw → −21%) |
| Recall @ thr 0.091 | **0.511** | catches ~half of true 30-day readmits |
| Precision @ thr 0.091 | **0.154** | ~1 in 6.5 flagged patients truly readmits |
| Flag rate @ thr 0.091 | **0.297** | ~30% of discharges flagged for follow-up |

Confusion matrix at 0.091 (test, 13,998 patients): TN 9,222 · FP 3,519 · FN 615 ·
TP 642. **Why this threshold:** a missed readmission costs far more than an extra
follow-up call, so the cut leans toward recall; a documented dial-down to recall ~0.40
(22% flagged) exists for tighter capacity. Full rationale: `docs/THRESHOLD_DECISION.md`.

**Calibration:** sigmoid vs isotonic were a Brier tie (0.0790 vs 0.0791) — sigmoid
chosen as the simpler, more robust map. Calibration is monotonic, so it left ranking
(AUPRC/ROC-AUC) unchanged while making probabilities trustworthy.

**Model choice:** CatBoost beat the tuned LR baseline by **+0.041 test AUPRC (+25%)** and
caught **2.6× as many** true readmits at fixed precision; LR is saturated (linear ceiling
on this data). LR is retained as the transparent fallback. See `docs/MODEL_COMPARISON.md`.

---

## Explainability

- **Local (per prediction):** the API returns the top signed SHAP factors with every
  `/predict` response (`src/app/model.py`) — e.g. for the golden encounter: primary
  diagnosis = Circulatory (↑ risk), admission via Transfer (↓), age 85 (↑).
- **Global (across patients):** mean-|SHAP| importance (`src/governance/explain.py`,
  plot in `reports/governance/shap_global_importance.png`, logged to MLflow run
  `shap_global`). **Top drivers:** `discharged_home`, `discharge_disposition_grp`,
  `diag_1_bucket` (primary diagnosis), `number_inpatient` (prior inpatient visits),
  `medical_specialty`, `time_in_hospital`, `age_midpoint`, `service_utilization`. These
  are clinically sensible — discharge destination, prior utilization, diagnosis, and
  acuity dominate.
- **⚠️ Magnitude caveat:** SHAP values are computed on the base CatBoost learner's
  **log-odds margin, before** the sigmoid calibration. **Directions are valid**
  (what raises vs lowers risk); **magnitudes are on the uncalibrated scale** and are not
  deltas in the calibrated probability. Global and local factors use the same base
  learner, so the caveat is identical for both.

---

## Fairness findings (audit on the test set — `docs/FAIRNESS_AUDIT.md`)

Subgroup performance was measured with Fairlearn at the **single global 0.091
threshold** across age, gender, and race. The audit **measures**; it does not yet
mitigate. The headline question — *does one global threshold land differently across
groups?* — is answered: **yes, decisively along age.**

- **Age — the material disparity.** Recall gap **0.69**, flag-rate gap **0.49**. The
  cut flags ~49% of patients in their 80s but only ~15% in their 40s, and catches ~69%
  of true readmits among 80-somethings vs **~28% among 40-somethings**. Two effects are
  tangled and must be kept separate:
  - *Legitimate:* readmission prevalence genuinely rises with age (0.07→0.10), and a
    calibrated score is supposed to reflect that — flagging more older patients is not
    itself a bug.
  - *Inequity:* the **recall** gap means a readmission-bound middle-aged patient is far
    more likely to be **missed**; the model also simply ranks worse for the 40–60 band
    (ROC-AUC ~0.63 vs ~0.66 at the age extremes). A program run on this score would
    systematically under-serve readmission-bound 40–50-year-olds. This is the part that
    needs mitigation (see Human-in-the-loop, below).
- **Gender — effectively fair.** Recall gap 0.047, flag-rate gap 0.034, identical AUPRC
  (0.208). No action beyond monitoring.
- **Race — no large well-supported disparity; small cells inconclusive.** Among
  well-supported groups recall spans only 0.50 (AfricanAmerican) → 0.52 (Caucasian), and
  AfricanAmerican patients are flagged *less* often (26% vs 31%) at equal prevalence —
  not a flag-them-more bias against the largest minority. The headline 0.184 recall gap
  is an artifact of the **n=96 Asian** cell (3 of 9 readmits). Honest verdict: fairness
  for Asian / Hispanic / Other / Unknown patients **cannot be certified** on this sample —
  collect more data, do not claim parity.

**Mitigation stance:** do not "fix" the age gap blindly (it is part legitimate signal,
part inequity). Candidate remedies for the next iteration: per-age-band thresholds
targeting equal recall; standing subgroup-recall monitoring; treating small race cells
as insufficient evidence rather than parity.

---

## Limitations

- **Performance ceiling is real and inherent to the data.** ROC-AUC ~0.67 / AUPRC ~0.21
  is the honest level; both a linear model and a tuned GBM converge there. The signal in
  these features is genuinely weak — this is a **data ceiling, not a modelling failure**.
  More data (socioeconomic context, post-discharge follow-up) would move it; more tuning
  will not.
- **Low precision at the operating point.** ~5 of every 6 flags are false positives —
  acceptable *only because* the intervention is cheap (a call, a med review). Wrong tool
  for gating anything expensive or invasive.
- **Era and population shift.** Trained on **1999–2008** US hospital practice; it
  predates current diabetes drugs, coding, and discharge workflows, and is diabetic-only,
  adult-skewed. It **needs revalidation — ideally retraining — on current, local data**
  before clinical use, and must not be applied to populations it wasn't trained on.
- **SHAP magnitudes are pre-calibration** (directions valid) — see Explainability.
- **Fairness is established only as above** — age-disparate, gender-fair, race
  inconclusive on small cells. Not a clean bill of health.

---

## Lineage & versioning (what produced a prediction)

Every prediction is reconstructable from three pinned coordinates:

- **Data:** `data/featurized/diabetes_features.parquet`, **DVC-tracked** (rebuild with
  `dvc repro`: `validate_raw → clean → featurize → validate_processed`).
- **Code:** the git commit of `src/` (clean → featurize → train → calibrate). Serving
  reuses the *exact* training feature path (`src/app/featurize.py` → `build_features.py`)
  — no train/serve skew (golden encounter 12522 = 0.074595 in training, API, container).
- **Model:** MLflow registry **`readmission-catboost-calibrated` v1 @ `staging`**; the
  operating threshold + calibration method travel as **model-version tags**, so they can
  never drift from the model. The container bakes this exact artifact in
  (`deploy/model_bundle/`).
- **Audit trail:** every scored request is logged (`src/app/audit.py`) with request_id,
  UTC timestamp, the model name/version/alias/threshold, the inputs, the score+factors,
  and latency — so any individual prediction is traceable back to these coordinates.
- **Rollback:** move the registry **stage/alias** back to the prior good version (and
  pin the matching image tag). The alias swap is the rollback mechanism — no code change.

---

## Human-in-the-loop

The score is **decision-support, never an autonomous decision** — a flag triggers a
human follow-up contact, not an automated action, and a clinician can always override.

Review is routed, not uniform, and is **tied directly to the fairness findings**:

- **Borderline scores route to a clinician.** Predictions in a band around the threshold
  (≈0.07–0.12) are "uncertain," not auto-flag/auto-clear, and are surfaced for human
  judgement rather than acted on mechanically.
- **Weakest-subgroup escalation.** Because the model under-performs on the **40–60 age
  band** (lower recall *and* lower ROC-AUC), a *no-flag* for a middle-aged patient is
  trusted less: borderline or clinically-concerning mid-age cases get explicit clinician
  review so the age recall gap is caught by a human, not baked into the worklist.
- **Low-support subgroups treated as lower-confidence.** For Asian / Hispanic / Other /
  Unknown patients (small, un-certifiable cells), scores carry less weight and lean on
  clinical judgement.
- **Overrides are logged.** Clinician accept/override decisions feed the audit trail and
  the Stage-6 feedback loop, so disparities and model decay surface over time.

---

*See also: `docs/REFLECTION.md` (trade-offs & limits), `docs/THRESHOLD_DECISION.md`
(threshold + rollback), `docs/FAIRNESS_AUDIT.md` (full subgroup tables),
`docs/MODEL_COMPARISON.md` (LR vs CatBoost + SMOTE rejection), `docs/DATA_VALIDATION.md`
(data contract).*
