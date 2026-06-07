# Caveats & Known Traps (Graded — Do Not Miss)

These are the traps that silently cost marks. The hard rules in `CLAUDE.md`
enforce them; this file explains the *why*. Data facts below were verified
against the actual dataset (101,766 rows).

---

## The four headline traps

### 1. `?` = missing, not a category
Missing values in the source CSV are the literal string `?`, **not** blanks.
Untreated, the model reads `?` as a real value.
→ Load with `pd.read_csv(..., na_values="?")` before any encoding.

### 2. Heavy-missing columns: decide drop vs. impute, **with evidence**
Verified missingness:

| Column | % missing | Note |
|---|---:|---|
| `weight` | **96.9%** | almost entirely empty — likely drop; justify |
| `max_glu_serum` | **94.7%** | mostly "not measured" (see trap 3) |
| `A1Cresult` | **83.3%** | mostly "not measured" (see trap 3) |
| `medical_specialty` | **49.1%** | high-cardinality (72) — bucket + "Missing" |
| `payer_code` | **39.6%** | administrative — often dropped |
| `race` | 2.2% | small — keep, encode "Unknown" |

Do not just drop or just impute by reflex. **State the decision and back it with
evidence** (missingness %, whether the column is clinical vs. administrative,
whether missingness itself is informative).

### 3. A missing A1c is NOT a healthy A1c
"Not measured" is **information** — whether the clinician ordered the test is
itself a signal. Encode missing `A1Cresult` / `max_glu_serum` as its **own
category** (e.g. `"None"`/`"NotMeasured"`). **Never fill with `Norm`.**

### 4. Patients recur across rows → split by patient ID
The same `patient_nbr` appears in multiple encounters. A random row split puts
the same patient in train *and* test → **leakage** → inflated metrics.
→ Use `GroupShuffleSplit` / grouped CV on `patient_nbr`.
→ **Consequence:** use the **Kaggle CSV** (`diabetic_data.csv`), which keeps
`encounter_id` and `patient_nbr`. The `ucimlrepo` loader **drops** the patient ID
and therefore **cannot be used** for the leakage-safe split.

---

## Additional traps worth knowing

### 5. Expired / hospice discharges can't be readmitted
`discharge_disposition_id` includes codes meaning the patient died or went to
hospice. Those rows can never be positive. Common practice is to **filter them
out** (and document the filter). Map the integer codes via UCI's
`IDs_mapping.csv`.

### 6. ID columns are codes, not quantities
`admission_type_id`, `discharge_disposition_id`, `admission_source_id` are
**categorical integer codes**, not numeric magnitudes. Treat as categorical; map
to text where it aids interpretation.

### 7. Dead / near-constant medication columns
Verified: `examide` and `citoglipton` have a **single value** (drop them).
`troglitazone`, `acetohexamide`, `tolbutamide`, and several combination drugs are
near-constant (≤2 values, almost all `No`) → very low signal; consider dropping
or be aware they add noise.

### 8. High-cardinality ICD-9 diagnoses
`diag_1`/`diag_2`/`diag_3` have ~700–800 unique codes each. **Do not one-hot raw.**
Bucket into clinical groups (circulatory, respiratory, diabetes, digestive,
injury, neoplasms, …). This is a Stage-2 deliverable.

### 9. Accuracy is a trap
Positive class ≈ **11.2%** (verified: 11,357 `<30` / 35,545 `>30` / 54,864 `NO`).
"Always predict no" ≈ 89% accuracy and is useless. **Headline metrics: PR-AUC,
recall at fixed precision, calibration (Brier / reliability curve).**

### 10. Reproducibility, not one-off cleaning
The test is not "did 10 cleaning steps turn raw into train data once" — it's "can
those same steps re-run on fresh production data whose missingness differs." Build
a script/pipeline, version it (DVC), and make it re-runnable with one command.
