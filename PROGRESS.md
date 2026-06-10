# Project Progress — How I Tackled This

A plain-language log of the steps I took, in order. (Updated only when explicitly
instructed to do so.)

---

## Step 1 — Version the raw data with DVC
Goal: treat the 19 MB dataset like code, but without bloating git.

- Ran `dvc init`, then `dvc add` on the raw CSV.
- Git now stores a tiny **pointer file** (a hash), while the actual data lives in a
  local DVC cache. Anyone cloning the repo runs `dvc pull` to get the exact bytes.
- Why it matters: the data is now reproducible and versioned — if it ever changes,
  the hash changes and git shows it.

## Step 2 — First robust cleaning of the dataset
Goal: one trustworthy base table that later feature work builds from.

Key decisions (each guards against a known trap):

- **Missing values** are coded `?` in the file → converted to real NaN on load,
  then handled per column (never left as a fake category).
- **Lab tests** (`A1Cresult`, `max_glu_serum`): "not measured" is *signal*, not
  missing → kept as an explicit `NotMeasured` level. The string "None" in those
  columns means "test was not ordered" — a real clinical fact, not a blank.
- **Dropped dead columns:** `weight` (97% missing), `examide` + `citoglipton`
  (same value for every row → useless).
- **Dropped impossible rows:** patients discharged as expired/hospice
  (`discharge_disposition_id` in {11, 13, 14, 19, 20, 21}) can't be readmitted,
  so keeping them would corrupt the label.
- **`payer_code` / `medical_specialty` / `race`:** these columns have lots of
  missing values (up to ~50%). Rather than drop the column or the rows, I label
  the missing entries as their own `"Unknown"` group. The reason: *the fact that
  a value is missing can itself be a clue* (e.g. a blank specialty often means the
  patient came through a different care path), so "Unknown" is kept as a real,
  usable category.
- **First-encounter dedup:** the same `patient_nbr` can appear in multiple rows
  (repeat hospital visits). I keep only the row with the smallest `encounter_id`
  per patient and drop the rest. Immediately after, I assert that `patient_nbr`
  is unique. This is the leakage guard — it means one patient appears exactly once,
  so a plain stratified train/test split is safe without any grouped CV logic.

## Decision — Target column: binary, "<30" vs everything else
The raw `readmitted` column has three values: `<30`, `>30`, and `NO`. I
collapsed it to binary — `<30` = 1 (readmitted within 30 days), `>30` and
`NO` = 0.
Why:
- The care team's real decision at discharge is binary: does this patient get
  extra 30-day follow-up, or not? The `>30`-vs-`NO` boundary doesn't change
  that action.
- The brief is specifically about 30-day readmission, so `<30` is the event of
  interest; merging `<30` and `>30` would answer a different question.
- Three classes also worsens the imbalance and muddies the metrics.

Result: ~70k rows after discharge filter + first-encounter dedup, positive rate
~0.09–0.11. The whole step is one reproducible command (`dvc repro`).

## Up next
- Handle the diagnosis codes (`diag_1/2/3`) — group thousands of ICD-9 codes into
  Strack-9 clinical categories (featurization stage).
- Engineer service-utilization, medication, A1c/glucose measurement flags, and
  other fixed features (featurization stage).
- Train LR baseline → CatBoost + Optuna, all logged to MLflow (modeling stage).
