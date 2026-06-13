# Data Validation (Great Expectations)

Formal, automated checks that the data entering and leaving the pipeline matches a
written contract — so a broken upstream feed or a regression in our own cleaning
**stops a retrain loudly** instead of silently shipping a bad model.

## Why Great Expectations, and where it sits

Great Expectations (GX) is a **batch** validation tool: it validates a whole
dataframe at a pipeline checkpoint. It is **not** in the real-time `/predict` path —
there, Pydantic validates one request at a time (next gate). The two share one
contract so they can't drift apart (see "The contract, read three ways").

GX moving parts (so the setup isn't a black box):
- **Data Context** — the GX project state, persisted under `./gx/` (gitignored; it's
  rebuilt in code each run, so nothing to version).
- **Expectation Suite** — the list of assertions. *This is the contract*, built from
  `src/contracts/data_contract.py`.
- **Validation Definition** — binds a suite to a specific data batch.
- **Checkpoint** — runs the validation and fires actions; ours renders the **data
  docs** (an HTML report of every expectation and its result).

## The two suites (they catch different failures)

### RAW suite — guards the INPUT (`data/raw/diabetic_data.csv`, before cleaning)
Read with `keep_default_na=False` so the missing token `"?"` stays a literal string.
**45 expectations:**
- All 50 expected columns present (exact set).
- `readmitted` ∈ {`<30`, `>30`, `NO`}; `gender`, `race` (+`"?"`), `age` in their
  known sets; `A1Cresult` / `max_glu_serum` in their allowed levels **including the
  `"None"` level** (= test not ordered, a real signal — never a null).
- `gender` has no nulls — proves missingness is the `"?"` token, not blank/NaN.
- `admission_type_id` (1–8), `discharge_disposition_id` (1–30),
  `admission_source_id` (1–26) and the clinical counts within documented ranges.
- All 23 drug columns ∈ {`No`, `Down`, `Steady`, `Up`}.

### PROCESSED suite — guards the OUTPUT (`data/featurized/diabetes_features.parquet`)
**28 expectations:**
- Exactly 58 columns; ≥1 row.
- **`patient_nbr` is unique** — the first-encounter dedup (leakage guard) held.
- `target` ∈ {0, 1}; positive rate within **0.08–0.12** (≈ prevalence).
- Strack-9 diagnosis buckets take only their 10 known values (8 clinical groups +
  `Other` + a `Missing` sentinel for NaN diagnoses); `a1c_state` and `age_bucket`
  within their fixed levels.
- No nulls in any of the 18 engineered features.

## Where it runs in the pipeline & what happens on failure

Both suites run as **DVC stages** (`dvc.yaml`), so `dvc repro` enforces them:

```
data/raw/diabetic_data.csv → validate_raw → clean → featurize → validate_processed
```

`clean` depends on `validate_raw`'s success marker
(`reports/validation/raw_status.json`), so **cleaning cannot start until the raw
contract passes**. `validate_processed` runs after featurization on the final output.

**On any failed expectation, `validate.py` exits non-zero** → the DVC stage fails →
`dvc repro` halts and downstream stages never run. A broken contract stops the
retrain; it does not warn-and-continue.

Demonstrated: `uv run python src/data/validate.py --suite raw --inject-bad` corrupts
one row (`gender = "Martian"`) and the suite reports **44/45 passed, success=False**,
naming the failed `expect_column_values_to_be_in_set` on `gender`, and returns exit
code 1 — the same command the DVC stage runs, so the pipeline would halt identically.

## The data docs (HTML report)

Each run renders a human-readable report at
`gx/uncommitted/data_docs/local_site/index.html` — open it in a browser to see every
expectation, its result, and observed values. It is regenerated on every run
(gitignored, not committed).

## The contract, read three ways

`src/contracts/data_contract.py` is the **single source of truth**. From it:
1. **GX batch suites** (`src/data/validate.py`) — the checkpoints above.
2. **`src/contracts/input_contract.json`** — a machine-readable schema of the **44
   model-input fields** (raw minus identifiers, the label, and cleaning-dropped
   columns), with each field's type / allowed-values / range / nullability. The
   **FastAPI Pydantic schema (next gate) is generated from / kept in sync with this
   file**, so the API boundary and the pipeline validate inputs identically.
3. **Stage-6 Evidently drift** — the same field list as the training reference schema.

Run manually:
```bash
uv run python src/data/validate.py --suite raw         # validate raw input
uv run python src/data/validate.py --suite processed   # validate featurized output
uv run python src/contracts/data_contract.py           # regenerate input_contract.json
```
