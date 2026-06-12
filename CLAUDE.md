# CLAUDE.md — Project Context for Claude Code

> Claude Code reads this file automatically. It is the operating manual for this
> repo. Read it fully before writing code. The supporting docs in `docs/` and the
> staged plan in `docs/GOALS.md` are authoritative — follow them.

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

Full project context → `docs/PROJECT_BRIEF.md` (problem statement, graded traps,
class philosophy).

---

## 1. Hard rules (do not violate these)

These encode the graded traps. Breaking one silently costs marks. See
`docs/PROJECT_BRIEF.md` for the why behind each.

1. **Missing values are coded as `?`** in the source CSV — load with
   `na_values=["?"], keep_default_na=False`. Never let `?` become a real category.
2. **First-encounter dedup is the leakage guard.** The same `patient_nbr` appears
   in multiple rows. Keep only the **smallest `encounter_id`** per patient; drop
   the rest. Immediately after, `assert df["patient_nbr"].is_unique`. This ensures
   one patient appears exactly once, so a plain `StratifiedKFold` split is safe.
   → **This means we use the Kaggle CSV, which keeps `patient_nbr` and
   `encounter_id`. The `ucimlrepo` loader drops both and cannot be used.**
3. **A missing A1c is NOT a healthy A1c.** The string `"None"` in `A1Cresult` and
   `max_glu_serum` means "test not ordered" — real signal. Preserve it as a
   category. Do not convert to NaN. Do not impute `Norm`.
4. **No post-discharge information may enter the features.** The decision point
   is discharge time. Anything that could only be known after discharge is leakage.
5. **Filter expired / hospice discharges.** `discharge_disposition_id` codes
   {11, 13, 14, 19, 20, 21} — those patients cannot be readmitted. Document the
   filter.
6. **Accuracy is a trap.** The positive class is ~11%, so "always predict no"
   scores ~89%. **Headline metrics are PR-AUC, recall at fixed precision, and
   calibration.** Never report accuracy as the primary metric.
7. **Collapse the target to binary:** raw `readmitted` has 3 classes (`<30`,
   `>30`, `NO`). Map `<30` → 1 (positive), `>30` and `NO` → 0. Document this
   decision in the model card.
8. **Log every experiment to MLflow from the first model.** Do not bolt tracking
   on at the end. Params, metrics, plots, and the model artifact all go to MLflow.
9. **Write docs as you go** — model card, feature log, README. Not on the last day.
10. **SMOTE is rejected.** Fabricates ~54k synthetic clinical records; weak
    defensibility; hurts AUC here. Handle imbalance via class weights
    (`class_weight="balanced"` in LR; `scale_pos_weight` in CatBoost). Record
    "considered and rejected" in the model card. Never implement it.
11. **Single dataset, single preprocessing path.** No dataset variants, recipes,
    or data-level A/B arms. Every preprocessing decision is made up front by
    reasoning. Only MODEL-level experimentation (hyperparameters, architecture).

---

## 2. Tech stack & tooling conventions

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.12+ | |
| Env | **uv** | `uv add` / `uv run`; never `pip install` into base; deps in `pyproject.toml` |
| Data versioning | **DVC** | track raw CSV + prepared parquet |
| Experiment tracking | **MLflow** | local server, SQLite backend, model registry |
| Modeling | scikit-learn, **CatBoost** | LR baseline first, then CatBoost; Optuna for tuning |
| Calibration | sklearn `CalibratedClassifierCV` | isotonic or Platt |
| Explainability | **SHAP** | global + local; local feeds the API response |
| Fairness | **Fairlearn** | `MetricFrame` across age / gender / race |
| API | **FastAPI** + Pydantic | `/predict`, `/health`, `/metrics` |
| Serving | **uvicorn** | |
| Containers | **Podman** (NOT Docker) | see Podman rules below |
| Orchestration | `compose.yaml` via `podman compose` | |
| Metrics | **Prometheus** | scrape FastAPI `/metrics` |
| Dashboards | **Grafana** | latency, req/s, error rate, score dist |
| Instrumentation | `prometheus-fastapi-instrumentator` | |
| Drift | **Evidently** | data + prediction drift report |
| Tuning | **Optuna** | integrated with MLflow; DVC for pipeline |

### Podman-specific rules (we use Podman, not Docker)
- The Dockerfile is named **`Containerfile`**. Build with `podman build`.
- Podman runs **rootless** by default. Do not assume root.
- **SELinux volume mounts need `:Z`** (or `:z` for shared) suffix, e.g.
  `-v ./data:/app/data:Z`. Forgetting this causes permission-denied on mounts.
- Compose: prefer `podman compose` (Podman v4+). Keep `compose.yaml`
  Docker-compatible so it works under either.
- Call `podman` explicitly in scripts — do not hardcode `docker`.
- For multi-service local stacks (api + prometheus + grafana + mlflow), use a
  Podman **pod** or the compose file; document the exact commands in the README.
- **Deployment targets: AWS or GCP only.** Not Hugging Face Spaces, not AWS App
  Runner, not any other platform.

### MLflow conventions
- Start the server with:
  `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000`
- One experiment for the project; one run per model attempt.
- Log: all hyperparameters, PR-AUC / recall@precision / Brier score, calibration
  plot, PR curve, confusion matrix at the chosen threshold, SHAP summary plot,
  chosen threshold value, and the model via `mlflow.sklearn.log_model` /
  `mlflow.catboost.log_model`.
- **Register the best calibrated model** to the MLflow Model Registry. Use stages
  (`None` → `Staging` → `Production`). The API loads the `Production` model.
  **The registry stage transition IS our rollback mechanism** — document it.

### CV / split conventions
- **Carve a held-out test set** (stratified, ~20%) before any tuning. Touch it
  **once**, at final evaluation only.
- Use **`StratifiedKFold`** (plain, not grouped) for cross-validation — the
  first-encounter dedup guarantees one row per patient, so there is no
  within-patient leakage to guard against.
- No `GroupShuffleSplit`, no `StratifiedGroupKFold`, no Ray Tune,
  no multi-dataset tuning arms.

---

## 3. Repo structure (create directories as you need them)

**Do not pre-scaffold empty folders.** Create a directory the moment the first
real file that belongs in it is written.

**What exists now:**

```
.
├── CLAUDE.md · requirements.txt · pyproject.toml · .gitignore
├── docs/        # PROJECT_BRIEF.md · GOALS.md · FEATURE_LOG.md
├── data/raw/    # diabetic_data.csv (DVC-tracked, gitignored)
├── data/prepared/ # diabetes_clean.parquet (DVC-tracked, written by clean stage)
├── dataset/     # original Kaggle download (CSV + data-dictionary PDF)
└── EDA/         # exploratory data analysis: Streamlit dashboard + analysis engine
```

**Where new code lands as you build it (create each on first use):**

```
src/data/        clean.py                            # reproducible cleaning → data/prepared/
src/features/    build_features.py                  # Strack-9 ICD-9, engineered features
src/models/      train.py · evaluate.py             # LR baseline + CatBoost + Optuna, MLflow
src/serving/     app.py · schemas.py                # FastAPI /predict (score + SHAP), /health
src/monitoring/  drift.py · retrain_trigger.py      # Evidently report + numeric trigger
src/governance/  fairness.py · explain.py           # Fairlearn MetricFrame + SHAP helpers
deploy/          Containerfile · compose.yaml · prometheus.yml · grafana/
tests/           test_smoke.py
docs/            FEATURE_LOG.md · MODEL_CARD.md · THRESHOLD_DECISION.md
notebooks/       exploration only — never the source of truth
.dvc/ · dvc.yaml                                    # created by `dvc init`
```

> **All real logic lives in importable, re-runnable scripts** under `src/` or
> `EDA/` — never hidden in a notebook.

---

## 4. How Claude Code should work in this repo

- **Read `docs/GOALS.md` and work the current stage.** Don't jump ahead — the
  plan is ordered so data discipline comes before modeling.
- **Build the simplest thing that ships, then iterate.** LR baseline before
  CatBoost. A working local API before a deployed one.
- **Every new feature → add a row to `docs/FEATURE_LOG.md`** (name, source,
  transformation, why it exists).
- **Keep functions small and importable.** No giant cells, no logic hidden in
  notebooks.
- **Commit in small, logical units** with clear messages.
- **When a step is done, state the "definition of done" check** from
  `docs/GOALS.md` and confirm it's met.
- **Manage deps with `uv add`**, never raw pip. Ask before adding heavy/unusual
  packages.
- **The user is learning the pipeline.** When you set up DVC, MLflow, Podman,
  Prometheus, or Evidently for the first time, briefly explain what each moving
  part does — don't just generate config silently.

---

## 5. Definition of "done" for the whole project

A reviewer can, from a clean checkout, follow the README to: rebuild the dataset
with one command (`dvc repro`), retrain and see the run in MLflow, start the API
+ monitoring stack under Podman, hit `/predict` and get a calibrated risk score
plus top SHAP factors, view live metrics in Grafana, generate an Evidently drift
report, read a fairness audit and a model card, and find an audit log entry for
every scored request — plus a written rollback and retrain-trigger plan. See
`docs/GOALS.md` for per-stage detail.

---

## 6. Standing instruction — end every response with a SYNC block

At the END of every response, append a section titled `── SYNC FOR CHAT ──`
written FOR a separate ML architect who sees ONLY this block, not the full output.
Concise and scannable — no narration. State:
- **DONE:** what changed this turn (files created/edited/deleted, one line each)
- **NUMBERS:** sanity figures (row/col counts, rates, metric values) — bullets
- **DECISIONS:** any choice made that wasn't explicitly specified (or "none")
- **FLAGS:** anything surprising, off, or needing an architecture decision (or "none")
- **NEXT:** the single next step, phrased as a question to confirm

Max ~12 lines. This block is the bridge between Claude Code and chat.
