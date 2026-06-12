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
docs/         project brief, goals, feature log, model comparison/card, threshold note
src/
  data/       clean.py            — reproducible cleaning  → data/processed/
  features/   build_features.py   — Strack-9 ICD-9 + engineered features → data/featurized/
  models/     train.py · evaluate.py — LR baseline + CatBoost, MLflow-logged
  app/        (later) FastAPI service: /predict (+ SHAP), /health, /metrics
  monitoring/ (later) Evidently drift + retrain trigger
  governance/ (later) Fairlearn audit + SHAP helpers
deploy/       (later) Containerfile, compose.yaml, prometheus.yml, grafana/
data/         raw/ interim/ processed/ featurized/   (DVC-tracked, gitignored)
EDA/          exploratory analysis (Streamlit dashboard + analysis engine)
notebooks/    exploration ONLY — never the source of truth
tests/        smoke test
dvc.yaml      clean → featurize pipeline (`dvc repro`)
```

## Quickstart
```bash
# 1. environment (uv is the source of truth; pyproject.toml + uv.lock are pinned)
uv sync                       # or, pip path: pip install -r requirements.txt

# 2. data: pull the DVC-tracked raw CSV, then rebuild the pipeline
dvc pull                      # restores data/raw/diabetic_data.csv from the .dvc pointer
dvc repro                     # clean → featurize  (data/processed, data/featurized)

# 3. experiment tracking (local server, SQLite backend)
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 4. train: LR baseline + CatBoost, both logged to MLflow
uv run python src/models/train.py
#    (writes to sqlite:///mlflow.db by default; set MLFLOW_TRACKING_URI to use the server)
```

## Status
- ✅ Stage 1–3 — env/DVC, reproducible cleaning, deterministic featurization.
- ✅ Stage 4 (modeling) — LR baseline + CatBoost, plain `StratifiedKFold` CV, a
  20% held-out test touched once, all runs in MLflow. See
  `docs/MODEL_COMPARISON.md` for the "earn your complexity" decision.
- ⏭️ Next — calibration, operating-threshold choice (`docs/THRESHOLD_DECISION.md`),
  register the best model to the MLflow registry, then package + serve under Podman.

## Headline metrics (not accuracy)
The positive class is ~9%, so accuracy is a trap. We report **PR-AUC**, **recall
at a fixed precision**, and **calibration (Brier)**. Healthy range on this dataset:
ROC-AUC ~0.66–0.70, AUPRC ~0.20–0.30. A leakage tripwire stops us if test
ROC-AUC > 0.75.
