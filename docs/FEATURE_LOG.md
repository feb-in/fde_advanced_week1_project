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

*To be filled in when Stage 3 is implemented.*

---

## Stage 4 — Modeling

*No new features at this stage.*
