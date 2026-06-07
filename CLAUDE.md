# CLAUDE.md — Project Context for Claude Code

> Claude Code reads this file automatically. It is the operating manual for this
> repo. Read it fully before writing code. The supporting docs in `docs/` and the
> work plan in `PLAN.md` are authoritative — follow them.

---

## 0. What this project is (one paragraph)

We are building an **end-to-end, production-style ML system** that predicts, **at
the moment of discharge**, the probability a diabetic patient will be **readmitted
within 30 days**. The output lets a hospital care team rank patients and direct
extra follow-up to the highest-risk ones. **This is not a notebook model** — the
deliverable is a *running, monitored, governed service*: a clean reproducible data
pipeline, an API returning a risk score + top contributing factors, experiment
tracking, containerized deployment, dashboards, drift detection, a fairness audit,
SHAP explainability, audit logs, a model card, and a documented rollback/retrain
plan.

**Grading reality: ~20% of the grade is the model, ~80% is everything around it.**
Optimize effort accordingly. See `docs/GOALS.md`.

Full problem statement → `docs/PROBLEM_STATEMENT.md`.
Graded traps you must not fall into → `docs/CAVEATS.md`.
Why we build it this way (course philosophy) → `docs/CLASS_CONTEXT.md`.

---

## 1. Hard rules (do not violate these)

These encode the graded traps. Breaking one silently costs marks. See
`docs/CAVEATS.md` for the why.

1. **Missing values are coded as `?`** in the source CSV — load with
   `na_values="?"`. Never let `?` become a real category.
2. **Split by patient ID, never randomly by row.** The same `patient_nbr`
   appears in multiple rows. A random split leaks the same patient into train
   and test. Use `GroupShuffleSplit` / grouped CV on `patient_nbr`.
   → **This means we use the Kaggle CSV, which keeps `patient_nbr`. The
   `ucimlrepo` loader DROPS the patient ID and cannot be used for the split.**
3. **A missing A1c is NOT a healthy A1c.** Encode "not measured" as its own
   category. Do not impute `Norm`. Same logic for `max_glu_serum`.
4. **No post-discharge information may enter the features.** The decision point
   is discharge time. Anything that could only be known after discharge is
   leakage.
5. **Filter expired / hospice discharges.** `discharge_disposition_id` includes
   codes for "expired" and "hospice" — those patients cannot be readmitted.
   Document the filter.
6. **Accuracy is a trap.** The positive class is ~11%, so "always predict no"
   scores ~89%. **Headline metrics are PR-AUC, recall at fixed precision, and
   calibration.** Never report accuracy as the primary metric.
7. **Collapse the target to binary:** raw `readmitted` has 3 classes
   (`<30`, `>30`, `NO`). Map `<30` → 1 (positive), `>30` and `NO` → 0. Document
   this decision in the model card.
8. **Log every experiment to MLflow from the first model.** Do not bolt
   tracking on at the end. Params, metrics, plots, and the model artifact all go
   to MLflow.
9. **Write docs as you go** — model card, feature log, README. Not on the last
   day.

---

## 2. Tech stack & tooling conventions

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Env | venv or conda | pin deps in `requirements.txt` |
| Data versioning | **DVC** | track raw + cleaned data and split |
| Experiment tracking | **MLflow** | local server, SQLite backend, model registry |
| Modeling | scikit-learn, **XGBoost** (or LightGBM) | LR baseline first |
| Calibration | sklearn `CalibratedClassifierCV` | isotonic or Platt |
| Explainability | **SHAP** | global + local; local feeds the API response |
| Fairness | **Fairlearn** | `MetricFrame` across age / gender / race |
| API | **FastAPI** + Pydantic | `/predict`, `/health`, `/metrics` |
| Serving | **uvicorn** | |
| Containers | **Podman** (NOT Docker) | see Podman rules below |
| Orchestration | `compose.yaml` via `podman compose` / `podman-compose` | |
| Metrics | **Prometheus** | scrape FastAPI `/metrics` |
| Dashboards | **Grafana** | latency, req/s, error rate, score dist |
| Instrumentation | `prometheus-fastapi-instrumentator` | |
| Drift | **Evidently** | data + prediction drift report |

### Podman-specific rules (we use Podman, not Docker)
- The Dockerfile is named **`Containerfile`**. Build with `podman build`.
- Podman runs **rootless** by default. Do not assume root.
- **SELinux volume mounts need `:Z`** (or `:z` for shared) suffix, e.g.
  `-v ./data:/app/data:Z`. Forgetting this causes permission-denied on mounts.
- Compose: prefer `podman compose` (Podman v4+) or `podman-compose`. Keep the
  `compose.yaml` Docker-compatible so it works under either.
- A user may `alias docker=podman`; do not hardcode the `docker` binary in
  scripts — call `podman` explicitly or detect it.
- For multi-service local stacks (api + prometheus + grafana + mlflow), use a
  Podman **pod** or the compose file; document the exact `podman` commands in the
  README.

### MLflow conventions
- Start the server with:
  `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000`
- One experiment for the project; one run per model attempt.
- Log: all hyperparameters, PR-AUC / recall@precision / Brier score, the
  calibration plot, PR curve, confusion matrix at the chosen threshold, SHAP
  summary plot, the chosen threshold value, and the model via
  `mlflow.sklearn.log_model` / `mlflow.xgboost.log_model`.
- **Register the best calibrated model** to the MLflow Model Registry. Use stages
  (`None` → `Staging` → `Production`). The API loads the `Production` model. **The
  registry stage transition IS our rollback mechanism** — document it.

---

## 3. Target repo structure (build toward this)

```
.
├── CLAUDE.md                 # this file
├── README.md                 # human front door + run commands
├── PLAN.md                   # the work plan (session-based)
├── requirements.txt
├── .gitignore
├── .dvc/                     # created by `dvc init`
├── dvc.yaml                  # pipeline stages (optional but encouraged)
├── docs/
│   ├── PROBLEM_STATEMENT.md
│   ├── CAVEATS.md
│   ├── GOALS.md
│   ├── CLASS_CONTEXT.md
│   ├── FEATURE_LOG.md        # YOU/Claude maintain this: every feature + why
│   ├── MODEL_CARD.md         # intended use, data, perf, limits, fairness
│   └── THRESHOLD_DECISION.md # the cost trade-off + chosen operating point
├── data/
│   ├── raw/                  # diabetic_data.csv (DVC-tracked, gitignored)
│   ├── interim/
│   └── processed/            # cleaned + train/val/test (DVC-tracked)
├── src/
│   ├── data/
│   │   ├── download.py       # fetch Kaggle CSV
│   │   └── clean.py          # reproducible cleaning pipeline (NOT a notebook)
│   ├── features/
│   │   └── build_features.py # ICD-9 buckets, prior-visit/med-churn features
│   ├── models/
│   │   ├── train.py          # baseline + XGBoost, calibrate, log to MLflow, register
│   │   └── evaluate.py       # PR-AUC, recall@precision, calibration, plots
│   ├── serving/
│   │   ├── app.py            # FastAPI: /predict (score + SHAP factors), /health
│   │   └── schemas.py        # Pydantic models
│   ├── monitoring/
│   │   ├── drift.py          # Evidently report
│   │   └── retrain_trigger.py# concrete numeric trigger
│   └── governance/
│       ├── fairness.py       # Fairlearn MetricFrame
│       └── explain.py        # SHAP global + local helpers
├── notebooks/                # EXPLORATION ONLY — never the source of truth
├── deploy/
│   ├── Containerfile         # Podman build file for the API
│   ├── compose.yaml          # api + mlflow + prometheus + grafana
│   ├── prometheus.yml        # scrape config
│   └── grafana/              # dashboard provisioning
└── tests/
    └── test_smoke.py         # end-to-end smoke test
```

> Notebooks are for exploration only. **All real logic lives in `src/` as
> importable, re-runnable scripts.** A pipeline only its author can re-run by
> hand is not reproducible.

---

## 4. How Claude Code should work in this repo

- **Read `PLAN.md` and work the current phase.** Don't jump ahead to later
  stages — the plan is ordered so data discipline comes before modeling and
  deployment before observability.
- **Build the simplest thing that ships, then iterate.** Logistic regression
  baseline before XGBoost. A working local API before a deployed one.
- **Every new feature → add a row to `docs/FEATURE_LOG.md`** (name, source,
  transformation, why it exists).
- **Keep functions small and importable.** No giant cells, no logic hidden in
  notebooks.
- **Commit in small, logical units** with clear messages. Suggest a commit after
  each meaningful step.
- **When a step is done, state the "definition of done" check** from
  `docs/GOALS.md` and confirm it's met.
- **Ask before installing heavy/unusual dependencies**; prefer the pinned stack.
- **The user is learning the pipeline.** When you set up MLflow, Podman,
  Prometheus, or Evidently for the first time, briefly explain what each moving
  part does — don't just generate config silently.

---

## 5. Definition of "done" for the whole project

A reviewer can, from a clean checkout, follow the README to: rebuild the dataset
with one command, retrain and see the run in MLflow, start the API + monitoring
stack under Podman, hit `/predict` and get a calibrated risk score plus top SHAP
factors, view live metrics in Grafana, generate an Evidently drift report, read a
fairness audit and a model card, and find an audit log entry for every scored
request — plus a written rollback and retrain-trigger plan. See `docs/GOALS.md`
for per-stage detail.
