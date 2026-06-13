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

The calibrated model is **committed in `deploy/model_bundle/`** and baked into the image,
so `podman compose up --build` serves the golden model directly — **no export or training
step is needed**. (`deploy/export_model.py` only *refreshes* the bundle after registering a
new model, and itself requires a populated MLflow registry.)

```bash
podman compose up --build -d                 # start api + prometheus + grafana (serves the committed bundle)

# (podman compose talks to the Podman API socket; if it errors with
#  "podman.sock ... no such file", start it once:  systemctl --user start podman.socket)

podman compose ps                            # all three Up
podman compose down                          # tear the stack down
```

| Service | URL | Check |
|---|---|---|
| API | http://localhost:8000 | `/health`, `/predict`, `/metrics` |
| Prometheus | http://localhost:9090 | Status → Targets: `readmission-api` = **UP**; Status → Rules: 3 alerts loaded |
| Grafana | http://localhost:3000 | login `admin` / `admin`; **"Readmission API — Observability"** dashboard auto-loads |

**Dashboard** (`deploy/grafana/provisioning/dashboards/readmission.json`, auto-provisioned)
panels over the real metrics: request rate by handler, p50/p95 latency, 5xx error rate,
total predictions served, requests/sec by status class. A **score-distribution** panel is
a documented follow-up — it would need a new prediction-score histogram in the app, which
would touch the slim serving image, so it is intentionally not added.

**Alerts** (`deploy/alerts.yml`, loaded via `rule_files` in `deploy/prometheus.yml`) —
three firing-capable Prometheus rules, visible at Prometheus → Alerts (inactive when
healthy): **APIDown** (target unscrapeable >1m, critical), **HighErrorRate** (5xx share
>5% for 2m), **HighP95Latency** (p95 >1s for 5m). No Alertmanager/notification channel is
wired (not required this gate); the rules load, evaluate, and show state in the UI.

**Drift + retrain trigger** are the offline half of observability — see `docs/MONITORING.md`
(`src/monitoring/drift.py` validates the detector; `src/monitoring/retrain_trigger.py`
is the concrete keep/retrain rule).

## Streamlit UI (optional demo front-end)

A clinical decision-support front-end — `src/ui/app_streamlit.py`, a **thin HTTP client**
of `/predict`. It **computes nothing**: it collects raw patient fields, POSTs them to the
API, and renders the response (probability, the API's flag + threshold, and the top SHAP
factors as a diverging bar chart). All featurization/scoring stays server-side — doing it
in the UI would reintroduce train/serve skew. The form is **derived from
`src/contracts/input_contract.json`** so it matches the API schema exactly; the only HTTP
code lives in `src/ui/api_client.py`.

```bash
# the API must be running first (local or container); then, in a separate process:
# (the UI is dark by default — theme set in .streamlit/config.toml)
READMISSION_API_URL=http://localhost:8000 uv run --group ui \
    streamlit run src/ui/app_streamlit.py        # opens http://localhost:8501
```

- **API URL** from `READMISSION_API_URL` (default `http://localhost:8000`).
- **Deps** are a separate **`ui` dependency group** (`streamlit`, `requests`) — **not** in
  the slim serving image; the UI is its own process.
- **Main view** (high-signal fields): Demographics (race, sex, age), Prior utilization
  (inpatient/emergency/outpatient), This admission (type/source/discharge codes, length of
  stay, medications, lab/other procedures, diagnoses count), Labs & meds (A1C, max glucose,
  diabetesMed, change, insulin, metformin), Diagnoses (diag_1/2/3). **Advanced expander:**
  payer code, admitting specialty, and the other 19 diabetes-drug fields (default `No`).
- Pre-filled with a realistic sample patient → one click runs the demo. Errors degrade
  gracefully (422 → "check these fields"; unreachable API → "API not reachable at <url>").
- **Load random patient** (`src/ui/sample_data.py`) pulls a real row from the **seed-42
  held-out test split** (data the model never trained on; reproduced without importing the
  training stack), fills the form from its raw fields, **still scores it through the API**,
  and shows **ground truth vs prediction** with a ✓/✗ — false positives/negatives are shown
  honestly, not hidden. Reading the test data here is a **display-only UI convenience**, not
  a serving dependency; the UI never scores locally.
- A persistent **DEMONSTRATION / BETA** banner states this is not a deployed medical device
  and not for real clinical decisions, tying it to the human-in-the-loop stance.
- **Verified (no skew):** the displayed probability is the API's number unaltered — the
  UI's `/predict` call and a direct POST return the **same** score (sample patient
  `0.074595`; every random held-out patient matches exactly).

## Deferred to later gates
- A score-distribution Grafana panel (needs a prediction-score histogram in the app).
- Alertmanager + a real notification channel (SMTP/Slack) for the alert rules.
