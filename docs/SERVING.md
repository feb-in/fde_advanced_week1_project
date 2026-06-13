# Serving — FastAPI `/predict` (Stage 5)

A real-time API that scores one discharge: calibrated 30-day readmission risk, a
follow-up flag, and the top contributing factors for *that* patient. Code in
`src/app/` (the serving layer — not `src/serving/`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | One patient's raw discharge-time fields → calibrated risk + flag + top SHAP factors |
| GET | `/health` | Liveness + loaded model name/version/alias/threshold/calibration |
| GET | `/metrics` | **Real Prometheus metrics** — request count, latency histogram, status codes (error rate) via `prometheus-fastapi-instrumentator` |

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

## Audit trail

Every scored request is logged append-only (JSONL) by `src/app/audit.py`: a
server-generated `request_id`, UTC timestamp, the model lineage
(name/version/alias/threshold/calibration/`load_source`), the 44 validated raw inputs,
the score + flag + top factors, and server latency. Best-effort — a logging failure is
surfaced to stderr but never drops a prediction. Path is `AUDIT_LOG_PATH` (default
`logs/audit/predictions.jsonl`, gitignored — entries hold clinical inputs). Production
should ship these to an access-controlled, append-only, retention-governed sink.

## Observability stack (Stage 6 — setup)

`compose.yaml` brings up the **instrumented API + Prometheus + Grafana** together
(Podman-compatible; bind mounts carry `:Z` for SELinux). Prometheus scrapes the API's
real `/metrics`; Grafana auto-provisions the Prometheus datasource on startup.

```bash
uv run python deploy/export_model.py        # ensure the baked bundle exists
podman compose up --build -d                 # start api + prometheus + grafana

# (podman compose talks to the Podman API socket; if it errors with
#  "podman.sock ... no such file", start it once:  systemctl --user start podman.socket)

podman compose ps                            # all three Up
podman compose down                          # tear the stack down
```

| Service | URL | Check |
|---|---|---|
| API | http://localhost:8000 | `/health`, `/predict`, `/metrics` |
| Prometheus | http://localhost:9090 | Status → Targets: `readmission-api` = **UP** |
| Grafana | http://localhost:3000 | login `admin` / `admin`; Prometheus datasource pre-wired |

What's **set up** (this gate): real metrics emitting, Prometheus scraping the target,
Grafana connected to Prometheus. What's **deferred** to the next gate: Grafana dashboard
panels (latency, req/s, error rate, score distribution), an Evidently drift report, ≥1
alert, and a concrete numeric retrain trigger.

## Deferred to later gates
- Grafana dashboards + Evidently drift + alerting + retrain trigger (Stage 6, next session).
