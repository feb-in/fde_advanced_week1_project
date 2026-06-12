# Feature Log

Every cleaning decision and every engineered feature is recorded here.
Format: column/action | stage | transformation | reason.

---

## Stage 2 — Data Cleaning (`src/data/clean.py`)

### Row-level operations

| Action | Detail | Reason |
|---|---|---|
| Drop bad-gender rows | `gender` not in {Male, Female}: 3 rows removed | "Unknown/Invalid" is not a real sex category; too few rows to model as its own level |
| Drop expired/hospice discharges | `discharge_disposition_id` ∈ {11,13,14,19,20,21}: 2,423 rows removed | These patients structurally cannot be readmitted — keeping them would guarantee them as negatives and corrupt the label distribution |
| First-encounter dedup | Keep `min(encounter_id)` per `patient_nbr`; assert uniqueness: ~29k rows removed | The same patient appears in multiple encounter rows (repeat visits). Keeping all rows would allow the same patient to appear in both train and test, inflating all metrics. One patient = one row = leakage-free StratifiedKFold split |

### Column-level operations

| Column | Action | Reason |
|---|---|---|
| `weight` | Dropped | 96.9% missing; the pattern of missingness is not clinically consistent enough to recover from; no imputation strategy is defensible |
| `examide` | Dropped | Single value ("No") for all 101,766 rows — zero variance, zero information |
| `citoglipton` | Dropped | Single value ("No") for all 101,766 rows — zero variance, zero information |
| `payer_code` | NaN → `"Unknown"` (category) | 39.6% missing; missingness is informative (payment type is an SES/insurance proxy; blank = likely different care path) |
| `medical_specialty` | NaN → `"Unknown"` (category) | 49.1% missing; missingness is the strongest signal here — blank specialty often indicates a different admission pathway |
| `race` | NaN → `"Unknown"` (category) | 2.2% missing; kept for the fairness audit — dropping rows would silently remove a demographic group |
| `A1Cresult` | String `"None"` → `"NotMeasured"` (category) | `"None"` in this column means "test was not ordered" — a real clinical fact (not a blank). Whether the clinician ordered the test is itself predictive. Never convert to NaN |
| `max_glu_serum` | String `"None"` → `"NotMeasured"` (category) | Same logic as `A1Cresult` above |
| `readmitted` | Dropped after target creation | Replaced by binary `target` column |
| All object columns (except diag_*, keys) | Cast to `category` dtype | Reduces parquet file size and signals to downstream code that these are nominals, not free strings |

### Target

| Column | Transformation | Reason |
|---|---|---|
| `target` (new) | `readmitted == "<30"` → 1; `">30"` and `"NO"` → 0 | Binary: the care team triggers one action — 30-day follow-up, yes/no. `">30"` and `"NO"` lead to the same action (no urgent follow-up). See PROGRESS.md "Decision — Target column" for full rationale |

### Columns held back for featurization (Stage 3)

These columns are present in the prepared parquet but intentionally untransformed here:

| Column | Planned transformation |
|---|---|
| `diag_1`, `diag_2`, `diag_3` | Strack-9 ICD-9 bucket encoding |
| `A1Cresult`, `max_glu_serum` | `_measured` binary flags derived in Stage 3 |
| `age` | 3-bucket or midpoint encoding in Stage 3 |
| `admission_type_id`, `discharge_disposition_id`, `admission_source_id` | Coarse-bucket encoding in Stage 3 |

---

## Stage 3 — Featurization (`src/features/build_features.py`)

Input: `data/processed/diabetes_clean.parquet` (69,987 × 47). Output: one typed
parquet `data/featurized/diabetes_features.parquet` (69,987 × 58), DVC-tracked via
the `featurize` stage in `dvc.yaml`. Positive rate unchanged at 0.0898. All
transformations are deterministic; no rows added or removed.

### 1. Diagnosis buckets (Strack-9) — `diag_1`, `diag_2`, `diag_3`

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `diag_1_bucket`, `diag_2_bucket`, `diag_3_bucket` (category) | `diag_1/2/3` | Map each ICD-9 code to one of 8 Strack groups (Circulatory 390–459+785, Respiratory 460–519+786, Digestive 520–579+787, Diabetes 250.xx, Injury 800–999, Musculoskeletal 710–739, Genitourinary 580–629+788, Neoplasms 140–239), else `Other` (incl. E/V codes); NaN → `Missing` | ~700–800 unique ICD-9 codes per column would explode one-hot dimensionality and overfit; clinical grouping is the standard Strack scheme |
| `diabetes_primary` (int8 0/1) | `diag_1` | 1 if `diag_1` is 250.xx | Whether diabetes is the *primary* reason for admission is distinct risk signal |
| `n_diabetes_diag` (int8 0–3) | `diag_1/2/3` | Count of the three diagnosis slots that are 250.xx | Diabetes burden across all coded diagnoses, not just the primary |
| `diag_1`, `diag_2`, `diag_3` | — | **Dropped** after bucketing | Raw high-cardinality ICD-9 must never reach the model |

### 2. Medication features — the 21 surviving drug columns

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `n_med_changes` (int8) | all drug cols | Count of drugs with value `Up` or `Down` this visit | Number of active medication adjustments is a proxy for clinical instability |
| `n_meds_used` (int8) | all drug cols | Count of drugs with value ≠ `No` | Total diabetes-drug burden |
| each drug column (int8) | each drug col | Ordinal-encode in place: `No`=0, `Down`=1, `Steady`=2, `Up`=3 | Compact ordinal scale (None < change < on-and-steady < up) usable directly by LR and CatBoost |

### 3. Lab measurement flags

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `a1c_measured` (int8 0/1) | `A1Cresult` | 1 if in {`>7`,`>8`,`Norm`}, else 0 (`NotMeasured`→0) | Whether the clinician *ordered* the A1c test is itself predictive |
| `glu_measured` (int8 0/1) | `max_glu_serum` | 1 if in {`>200`,`>300`,`Norm`}, else 0 | Same logic for glucose-serum testing |

### 4. A1c × med-change interaction (Strack's key feature)

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `a1c_state` (category, 4 levels) | `A1Cresult`, `change` | `no_test` (A1c not measured) / `normal` (Norm) / `high_changed` (>7 or >8 AND `change`==Ch) / `high_not_changed` (>7 or >8 AND `change`!=Ch) | Strack found the high-but-meds-NOT-changed cell is the elevated-readmission group — the interaction carries signal neither column carries alone |

### 5. Service utilization

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `service_utilization` (int32) | `number_outpatient` + `number_emergency` + `number_inpatient` | Sum of prior-year visit counts | Aggregate prior healthcare contact is a strong readmission predictor |
| `inpatient_ge_2` (int8 0/1) | `number_inpatient` | 1 if `number_inpatient` ≥ 2 | Repeat recent inpatient stays flag the highest-risk patients |
| `number_inpatient` | — | **Kept** as its own column (not folded away into the sum) | Prior inpatient count is the single strongest utilization signal; keep it separable |

### 6. Demographic / administrative

| Feature | Source | Transformation | Why |
|---|---|---|---|
| `age_midpoint` (int16) | `age` | Midpoint of the 10-year band (`[60-70)`→65) | Numeric age preserving full original granularity for linear/tree models |
| `age_bucket` (category, 3 levels) | `age` | `[0-30)` / `[30-60)` / `[60-100)` | Coarse age band for interpretable subgroup/fairness slicing |
| `admission_type_grp` (category) | `admission_type_id` | Code→name bucket: Emergency{1}, Urgent{2}, Trauma{7}, Elective{3}, Newborn{4}, Unknown{5,6,8} | IDs are categorical codes, not magnitudes; coarse named groups avoid sparse rare codes |
| `admission_source_grp` (category) | `admission_source_id` | Referral{1,2,3}, Transfer{4,5,6,10,22,25}, EmergencyRoom{7}, Delivery{11,13,14}, Other{8}, Unknown{9,17,20} | Same rationale |
| `discharge_disposition_grp` (category) | `discharge_disposition_id` | Home{1,6,8}, Transfer{2,9,10,16,17}, Facility{3,4,5,15,22,23,24,27,28}, AMA{7}, Unknown{12,18,25} | Same rationale (expired/hospice already removed in cleaning) |
| `discharged_home` (int8 0/1) | `discharge_disposition_id` | 1 if disposition in {1,6,8} (any home destination) | Discharge-to-home vs. facility is a coarse acuity signal |
| `age`, `admission_type_id`, `admission_source_id`, `discharge_disposition_id` | — | **Dropped** after bucketing | Replaced by their engineered representations |

**Pass-through (untouched):** `encounter_id`, `patient_nbr` (audit keys), `target`,
and the cleaning-stage categoricals (`race`, `gender`, `payer_code`,
`medical_specialty`, `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`) plus the
numeric clinical counts (`time_in_hospital`, `num_lab_procedures`, `num_procedures`,
`num_medications`, `number_outpatient`, `number_emergency`, `number_diagnoses`).

---

## Stage 4 — Modeling

*No new features at this stage.*
