# 30-Day Hospital Readmission Risk — Production ML System

Predicts, **at discharge**, the probability a diabetic patient is readmitted
within 30 days — and takes that prediction all the way to a **running, monitored,
governed service**, not a notebook model.

> **Start here:** `CLAUDE.md` is the build manual (read automatically by Claude
> Code). Then `docs/PROJECT_BRIEF.md` (problem + graded traps) and
> `docs/GOALS.md` (the authoritative staged plan and definition of done).

## Stack
Python 3.12 · **uv** · scikit-learn / **CatBoost** · MLflow · DVC · FastAPI ·
SHAP · Fairlearn · Evidently · Prometheus + Grafana · **Podman** (not Docker).
Deploy target: AWS or GCP.

## Layout (cookiecutter-data-science flavour)
```
docs/         brief, goals, feature log, model comparison/card, threshold, fairness,
              reflection, serving, monitoring, data validation, RESUME_HERE (checklist)
src/
  contracts/  data_contract.py · input_contract.json  — single source of truth for inputs
  data/       clean.py · validate.py    — reproducible cleaning + GX suites → data/processed/
  features/   build_features.py         — Strack-9 ICD-9 + engineered features → data/featurized/
  models/     train.py · tune.py · calibrate.py · evaluate.py · wrappers.py — MLflow-logged
  app/        FastAPI service: /predict (+ SHAP), /health, real /metrics; audit.py
  monitoring/ make_reference.py · drift.py (Evidently) · retrain_trigger.py
  governance/ fairness.py (Fairlearn) · explain.py (global SHAP)
  ui/         app_streamlit.py — thin /predict client (+ random held-out patient demo)
deploy/       Containerfile · export_model.py · model_bundle/ · prometheus.yml · alerts.yml · grafana/
compose.yaml  api + prometheus + grafana stack (Podman/Docker-compatible)
data/         raw/ processed/ featurized/ monitoring/   (DVC-tracked, gitignored)
EDA/          exploratory analysis (Streamlit dashboard + analysis engine)
tests/        smoke + 3-tier suite (schema / skew / behaviour / container)
dvc.yaml      validate_raw → clean → featurize → validate_processed (`dvc repro`)
.github/      workflows/ci.yml — test-gated build → push to Amazon ECR
```

## Quickstart
```bash
# 1. environment (uv is the source of truth; pyproject.toml + uv.lock are pinned)
uv sync

# 2. data — NOTE: no DVC remote is configured yet (see "Reproducibility" below).
#    On a fresh clone, place the raw Kaggle CSV at data/raw/diabetic_data.csv, then:
dvc repro                     # validate_raw → clean → featurize → validate_processed

# 3. train: LR baseline + CatBoost, logged to MLflow (sqlite:///mlflow.db by default)
uv run python src/models/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db        # → http://localhost:5000

# 4. serve + monitor: build the slim image, bring up api + prometheus + grafana
uv run python deploy/export_model.py
podman compose up --build -d  # API :8000 · Prometheus :9090 · Grafana :3000 (admin/admin)
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data-binary @tests/sample_request.json           # → 0.074595

# 5. demo UI (separate process; thin /predict client)
READMISSION_API_URL=http://localhost:8000 uv run --group ui \
     streamlit run src/ui/app_streamlit.py              # → http://localhost:8501
```
Governance + monitoring artifacts: `uv run python src/governance/fairness.py`,
`… explain.py`, `src/monitoring/drift.py`, `… retrain_trigger.py`.

## Status — all stages complete
- ✅ **Stages 1–4** — env/DVC, reproducible cleaning + GX validation, deterministic
  featurization, LR baseline + CatBoost (Optuna), calibration + threshold, registered.
- ✅ **Stage 5** — FastAPI + slim Podman container + compose stack; **CI/CD → Amazon ECR**
  (GitHub Actions, green).
- ✅ **Stage 6** — real `/metrics`, Grafana dashboard, 3 Prometheus alert rules, Evidently
  drift (validated), concrete retrain trigger.
- ✅ **Stage 7** — Fairlearn fairness audit, global+local SHAP, per-request audit log,
  model card, reflection.
- ✅ **Demo UI** — thin-client Streamlit with a load-random-held-out-patient
  truth-vs-prediction demo.
- ⬜ **Remaining** — verify clean-checkout reproducibility (no DVC remote yet) + push.

## Reproducibility
`dvc repro` rebuilds the dataset from the raw CSV. **No DVC remote is configured**, so
`dvc pull` will not work on a fresh clone — obtain the raw Kaggle CSV and place it at
`data/raw/diabetic_data.csv`, or configure a DVC remote and `dvc push`. See
`docs/RESUME_HERE.md` (submission checklist) for the full done-vs-left list.

## Headline metrics (not accuracy)
The positive class is ~9%, so accuracy is a trap. We report **PR-AUC**, **recall
at a fixed precision**, and **calibration (Brier)**. Healthy range on this dataset:
ROC-AUC ~0.66–0.70, AUPRC ~0.20–0.30. A leakage tripwire stops us if test
ROC-AUC > 0.75.
