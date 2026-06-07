# 30-Day Hospital Readmission Risk — Production ML System

Predicts, **at discharge**, the probability a diabetic patient is readmitted
within 30 days — and takes that prediction all the way to a **running, monitored,
governed service**.

> **Start here:** read `docs/PROBLEM_STATEMENT.md`, `docs/CAVEATS.md`, and
> `docs/GOALS.md`, then work `PLAN.md` in order. `CLAUDE.md` is the build manual
> Claude Code reads automatically.

## Stack
Python · scikit-learn / XGBoost · MLflow · DVC · FastAPI · SHAP · Fairlearn ·
Evidently · Prometheus + Grafana · **Podman** (not Docker).

## Layout
```
docs/        problem statement, caveats, goals, class context, model card, logs
src/         data/ features/ models/ serving/ monitoring/ governance/  (real logic)
deploy/      Containerfile, compose.yaml, prometheus.yml, grafana/
data/        raw/ interim/ processed/  (DVC-tracked, gitignored)
notebooks/   exploration ONLY
tests/       smoke test
```

## Quickstart (fill in as you build — Phase 0 in `PLAN.md`)
```bash
# env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# experiment tracking
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# data versioning
dvc init

# (later) rebuild data → train → serve
python -m src.data.download
python -m src.data.clean
python -m src.models.train
podman compose -f deploy/compose.yaml up   # api + mlflow + prometheus + grafana
```

## Key reminders (full list in `docs/CAVEATS.md`)
- Load with `na_values="?"` · **split by `patient_nbr`** (use the Kaggle CSV) ·
  A1c-missing is its own category · filter expired discharges · **headline metric
  is PR-AUC, not accuracy** · log every run to MLflow.

## Data
Diabetes 130-US Hospitals (1999–2008), 101,766 encounters, ~11.2% positive.
- Modeling source (keeps patient ID): `kaggle.com/datasets/brandao/diabetes`
- Canonical original: `archive.ics.uci.edu/dataset/296`
