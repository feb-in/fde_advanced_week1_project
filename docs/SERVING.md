# Serving — FastAPI `/predict` (Stage 5)

A real-time API that scores one discharge: calibrated 30-day readmission risk, a
follow-up flag, and the top contributing factors for *that* patient. Code in
`src/app/` (the serving layer — not `src/serving/`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | One patient's raw discharge-time fields → calibrated risk + flag + top SHAP factors |
| GET | `/health` | Liveness + loaded model name/version/alias/threshold/calibration |
| GET | `/metrics` | Prometheus hook **stub** — Stage 6 wires real instrumentation here |

### Run
```bash
uv run uvicorn src.app.app:app --host 0.0.0.0 --port 8000
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data-binary @tests/sample_request.json
```

### `/predict` response shape
```json
{
  "readmission_probability": 0.074595,
  "flag": false,
  "threshold": 0.091046,
  "model_name": "readmission-catboost-calibrated",
  "model_version": "1",
  "model_alias": "staging",
  "top_factors": [
    {"feature": "diag_1_bucket", "value": "Circulatory", "contribution": 0.1369, "direction": "increases"},
    {"feature": "age_midpoint",  "value": 85,            "contribution": 0.0916, "direction": "increases"}
  ]
}
```
`contribution` is the signed SHAP value (log-odds): positive raises this patient's
risk, negative lowers it.

## No train/serve skew (the hard rule)

Incoming raw data passes through the **exact same** feature engineering as training.
There is **one** source of truth for the transforms; serving reuses it:

```
raw record → replace "?"/null with NaN (mirrors pd.read_csv(na_values=["?"]))
           → clean_columns()      (src/data/clean.py — shared with batch cleaning)
           → engineer_features()  (src/features/build_features.py — shared with batch)
           → reindex to the model's own feature_names_  → categoricals as strings
```

`src/app/featurize.py` contains **no feature logic of its own** — that would *be* the
skew. The model defines its expected columns (`feature_names_`), so nothing is
hardcoded.

**Verified:** `tests/test_smoke.py::test_no_train_serve_skew` confirms the API score
equals the training-pipeline score for the same record — exact match (diff `0.0`)
across sampled encounters. Example: encounter 12522 scores `0.0745948` in both the
training pipeline and through `/predict`.

## Model loading & rollback

The model is loaded **by registry alias** — `models:/readmission-catboost-calibrated@staging`
— never a pickle path, and the operating **threshold is read from the model-version
tag** (`operating_threshold`), so model + threshold travel together. A registry
stage/alias swap changes what serving uses with **no redeploy** — that is the
rollback mechanism (see `docs/THRESHOLD_DECISION.md`). The SHAP `TreeExplainer` is
built once at startup on the lead base learner of the calibration ensemble and
reused per request.

## Input schema — generated from the contract

`src/app/schemas.py` builds the `PatientRecord` model **at import time** from
`src/contracts/input_contract.json` (the 44 model-input fields the GX RAW suite
exports). The API boundary therefore cannot drift from the pipeline.

## Serving rejection policy (decided on top of the raw-faithful contract)

The raw GX contract is permissive because it validates historical bulk data. A live
discharge-time request is stricter:

**Accept as real signal (not errors):**
- `A1Cresult` / `max_glu_serum` == `"None"` — "test not ordered" is predictive.
- `"?"` or `null` for `race` / `payer_code` / `medical_specialty` / `diag_*` —
  informative missingness, mapped to `Unknown` / `Missing` exactly as in training.

**Reject (HTTP 422):**
- Any type / range / allowed-value violation (enforced by the generated field types).
- **Unknown / extra fields** (`extra="forbid"`) — blocks accidental post-discharge
  leakage fields (e.g. `readmitted`) from entering.
- `gender == "Unknown/Invalid"` — training dropped these rows; not scoreable.
- `discharge_disposition_id ∈ {11,13,14,19,20,21}` (expired/hospice) — these patients
  structurally cannot be readmitted; training filtered them out.

## Deferred to later gates
- Containerize under Podman (Stage 5 packaging).
- Wire Prometheus/Grafana to `/metrics` + per-request audit logging (Stage 6).
