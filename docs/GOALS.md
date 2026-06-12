# Goals — Staged Plan & Definition of Done

The deliverable is a **production-style ML system you can defend**, not a model
in a notebook. This file is the authoritative staged plan. Work one stage at a
time; don't jump ahead.

**Grading shape: ~20% modeling · ~80% everything around it.** The 80% = data
discipline, reproducibility, packaging, deployment, observability, governance.

---

## North-star qualities

- **Reproducible** — `dvc repro` rebuilds the dataset; one command retrains.
  Nothing depends on hidden notebook state.
- **Honest about imbalance** — PR-AUC, recall@fixed-precision, calibration;
  never accuracy as the headline.
- **Leakage-free** — first-encounter dedup; no post-discharge features; test
  set touched once.
- **Explainable** — global + local SHAP; the API returns top contributing
  factors per prediction.
- **Fair & auditable** — subgroup metrics across age/gender/race; every scored
  request traceable.
- **Operable** — containerized under Podman, monitored, with a written rollback
  and a concrete numeric retrain trigger.
- **Simple that ships > clever that doesn't.** LR baseline before CatBoost;
  local before deployed.

---

## Stage 1 — Setup & Tooling

### What we build
- Environment: `uv` + `pyproject.toml`; all deps installed via `uv add`.
- DVC initialized; raw CSV versioned (`dvc add data/raw/diabetic_data.csv`).
- Repo structure matching `CLAUDE.md` section 3.

### Definition of done
- [ ] `uv run python -c "import pandas, sklearn, catboost, mlflow, shap"` passes.
- [ ] `data/raw/diabetic_data.csv.dvc` committed; actual CSV gitignored.
- [ ] `dvc pull` on a fresh clone would restore the raw file.

---

## Stage 2 — Data Cleaning

### What we build
`src/data/clean.py` — a single, re-runnable cleaning script that produces
`data/processed/diabetes_clean.parquet`, DVC-tracked via `dvc.yaml`.

### Key decisions (all locked — do not vary)
- Load with `na_values=["?"], keep_default_na=False`.
- Drop discharge codes {11, 13, 14, 19, 20, 21} (expired/hospice).
- **First-encounter dedup:** keep `min(encounter_id)` per `patient_nbr`; assert
  uniqueness immediately after.
- Drop `weight`, `examide`, `citoglipton`.
- Fill `payer_code`, `medical_specialty`, `race` NaN → `"Unknown"`.
- Preserve `"None"` in `A1Cresult`/`max_glu_serum` as a real category level.
- Target: `<30` → 1, everything else → 0.
- Output: one typed parquet to `data/processed/`.

### Definition of done
- [ ] `dvc repro` runs `clean.py` end-to-end with no errors.
- [ ] Sanity report printed: ~70k rows, positive rate 0.09–0.11.
- [ ] `df["patient_nbr"].is_unique` assertion passes (confirmed in script).
- [ ] `data/processed/diabetes_clean.parquet.dvc` (or equivalent DVC out) committed.
- [ ] Cleaning decisions logged in `docs/FEATURE_LOG.md`.

---

## Stage 3 — Featurization

### What we build
`src/features/build_features.py` — reads the prepared parquet, adds engineered
features, writes `data/features/diabetes_features.parquet`, DVC-tracked.

### Key decisions (all locked — do not vary)
- **ICD-9 bucketing:** Strack-9 scheme on `diag_1`/`diag_2`/`diag_3` only.
  No Charlson, no Elixhauser, no CCS, no LLM/MedGemma resolution.
- **Engineered features (fixed list):**
  - Service-utilization sum WITH `number_inpatient` kept separately +
    `inpatient_ge_2` flag.
  - Med-change count; meds-used count.
  - `A1c_measured` + `glu_measured` binary flags.
  - A1c × med-change interaction term.
  - Age 3-bucket (or midpoint).
  - `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
    bucketed to coarse categories.
  - `diabetes_primary` flag (diag_1 in 250.xx).
  - Count of `diag_1`/`diag_2`/`diag_3` in 250.xx family.
- Every new feature → row in `docs/FEATURE_LOG.md`.

### Definition of done
- [ ] `dvc repro` runs the feature stage end-to-end.
- [ ] `docs/FEATURE_LOG.md` has an entry for every engineered feature.
- [ ] No feature uses post-discharge information.

---

## Stage 4 — Modeling

### What we build
`src/models/train.py` + `evaluate.py` — baseline LR → CatBoost + Optuna;
all runs tracked in MLflow; best calibrated model registered.

### Key decisions (all locked — do not vary)
- **Split:** carve a stratified held-out test set (~20%) before any tuning;
  touch it exactly once at final evaluation.
- **CV:** plain `StratifiedKFold` (first-encounter dedup makes grouped CV
  unnecessary).
- **Imbalance:** class weights (`class_weight="balanced"` in LR;
  `scale_pos_weight` in CatBoost). No SMOTE.
- **Models in order:** LR baseline → CatBoost + Optuna tuning.
- **Calibration:** `CalibratedClassifierCV` (isotonic or Platt).
- **Leakage tripwire:** if test ROC-AUC > 0.75 on first run, STOP and hunt the
  leak before continuing.
- **Tuning stack:** DVC + MLflow + Optuna. No Ray Tune.

### Definition of done
- [ ] MLflow experiment has at least one LR run and one CatBoost run.
- [ ] Each run logs: all hyperparams, PR-AUC, ROC-AUC, recall@precision,
      Brier score, calibration plot, PR curve, SHAP summary, chosen threshold.
- [ ] Best calibrated model registered in MLflow Model Registry at `Staging`.
- [ ] `docs/THRESHOLD_DECISION.md` written (threshold choice justified on
      cost trade-off).
- [ ] SMOTE documented as "considered and rejected" in the model card.
- [ ] Test set touched exactly once; final metrics reported.

---

## Stage 5 — Package & Deploy

### What we build
FastAPI service + Podman container + compose stack on AWS or GCP.

### Key decisions
- API routes: `/predict` (risk score + top SHAP factors), `/health`, `/metrics`.
- Containerfile + compose.yaml for api + prometheus + grafana + mlflow.
- Deployment: AWS (ECS/EC2) or GCP (Cloud Run/GCE). Not Hugging Face Spaces,
  not AWS App Runner.
- Rollback: MLflow registry stage swap (`Production` ↔ `Staging`) + pinned
  image tag. Document exact commands.

### Definition of done
- [ ] `podman build -f Containerfile -t readmission-api .` succeeds.
- [ ] `podman compose up` starts all services.
- [ ] `curl .../predict` with a sample payload returns a calibrated probability
      + top SHAP factors.
- [ ] `/health` returns 200. `/metrics` is scrapeable by Prometheus.
- [ ] Rollback plan documented (MLflow stage swap + image tag).

---

## Stage 6 — Observability

### What we build
Prometheus → Grafana dashboard + Evidently drift reports + concrete retrain
trigger.

### Definition of done
- [ ] Grafana dashboard shows latency, request rate, error rate, score
      distribution.
- [ ] Every prediction logged (request, response, model version, latency).
- [ ] Evidently report generated; demonstrated firing on a shifted batch.
- [ ] Retrain trigger documented with a concrete numeric threshold (e.g. PSI >
      0.2 on top features, or PR-AUC on labeled feedback < X) — not a vibe.

---

## Stage 7 — Governance

### What we build
Fairlearn fairness audit + SHAP explanations + audit logging + model card +
reflection.

### Definition of done
- [ ] Fairlearn `MetricFrame` computed across age, gender, race; subgroup
      recall/PR-AUC gaps reported; mitigation stance stated.
- [ ] Global SHAP summary plot logged to MLflow; local SHAP returned by `/predict`.
- [ ] Audit log entry per scored request (request, latency, response, model
      version), traceable.
- [ ] `docs/MODEL_CARD.md` complete: intended use, data, performance, limits,
      fairness findings, SMOTE rejection note.
- [ ] Written rollback + retrain plan.

---

## Whole-project done

From a clean checkout, a reviewer follows the README to: `dvc repro` rebuilds
data → `uv run python src/models/train.py` retrains and logs to MLflow → `podman
compose up` starts the stack → `curl /predict` returns a calibrated score + top
SHAP factors → Grafana shows live metrics → Evidently report generated → fairness
audit and model card are readable → every scored request has an audit log entry —
and a written rollback + retrain plan exists.
