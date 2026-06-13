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

> **PROGRESS (current position):** Stages 1–4 ✅ complete. Stage 5 🔄 in progress —
> local FastAPI + containerized API done and skew-verified; **AWS deploy + full
> compose stack still pending.** Next: ECR push + GitHub Actions CI/CD → AWS Fargate,
> then Stages 6–7. See `docs/RESUME_HERE.md`.

## Stage 1 — Setup & Tooling ✅

### What we build
- Environment: `uv` + `pyproject.toml`; all deps installed via `uv add`.
- DVC initialized; raw CSV versioned (`dvc add data/raw/diabetic_data.csv`).
- Repo structure matching `CLAUDE.md` section 3.

### Definition of done
- [x] `uv run python -c "import pandas, sklearn, catboost, mlflow, shap"` passes.
- [x] `data/raw/diabetic_data.csv.dvc` committed; actual CSV gitignored.
- [x] `dvc pull` on a fresh clone would restore the raw file.

---

## Stage 2 — Data Cleaning ✅

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
- [x] `dvc repro` runs `clean.py` end-to-end with no errors.
- [x] Sanity report printed: ~70k rows, positive rate 0.09–0.11.
- [x] `df["patient_nbr"].is_unique` assertion passes (confirmed in script).
- [x] `data/processed/diabetes_clean.parquet.dvc` (or equivalent DVC out) committed.
- [x] Cleaning decisions logged in `docs/FEATURE_LOG.md`.

---

## Stage 3 — Featurization ✅

### What we build
`src/features/build_features.py` — reads the prepared parquet, adds engineered
features, writes `data/featurized/diabetes_features.parquet`, DVC-tracked.

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
- [x] `dvc repro` runs the feature stage end-to-end.
- [x] `docs/FEATURE_LOG.md` has an entry for every engineered feature.
- [x] No feature uses post-discharge information.

---

## Stage 4 — Modeling ✅

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
- [x] MLflow experiment has at least one LR run and one CatBoost run.
- [x] Each run logs: all hyperparams, PR-AUC, ROC-AUC, recall@precision,
      Brier score, calibration plot, PR curve, chosen threshold.
      *(Global SHAP summary plot → deferred to Stage 7 governance.)*
- [x] Best calibrated model registered in MLflow Model Registry at `Staging`.
- [x] `docs/THRESHOLD_DECISION.md` written (threshold choice justified on
      cost trade-off).
- [~] SMOTE documented as "considered and rejected" — in `MODEL_COMPARISON.md`;
      restate in `docs/MODEL_CARD.md` at Stage 7.
- [x] Test set touched exactly once; final metrics reported.

---

## Stage 5 — Package & Deploy ✅ (CI/CD → ECR done; live Fargate deploy optional)

> **Build + ship arc COMPLETE.** Image is built, test-gated, and pushed to **Amazon
> ECR** by GitHub Actions on push to `main`. The remaining live **Fargate** deploy is
> now OPTIONAL (the ECR image satisfies the deployable-artifact deliverable).
> **Next graded work: Reflection → Governance (Stage 7) → Observability (Stage 6).**

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
- [x] `podman build -f deploy/Containerfile -t readmission-api .` succeeds (rootless).
- [ ] `podman compose up` starts all services. *(api builds; prometheus/grafana/mlflow
      scaffolded as commented placeholders → wired in Stage 6.)*
- [x] `curl .../predict` returns a calibrated probability + top SHAP factors
      (container == local, exact: encounter 12522 → 0.074595).
- [~] `/health` returns 200 ✅. `/metrics` scrapeable by Prometheus → Stage 6.
- [x] Rollback plan documented (registry alias swap + pinned image tag) —
      `docs/THRESHOLD_DECISION.md` + `docs/SERVING.md`.
- [x] **CI/CD → Amazon ECR** — GitHub Actions (`.github/workflows/ci.yml`) runs the
      test suite as a merge gate, builds the image, tests the running container, and
      pushes to ECR (git SHA + `latest`) on green. **Pipeline is GREEN.**
- [ ] **Live deploy to AWS Fargate** — OPTIONAL / not done. The ECR image is the
      deployable artifact; a Fargate service is the optional "reachable URL" step.

---

## Stage 6 — Observability  ⏳ REMAINING (do AFTER governance — see priority order)

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

## Stage 7 — Governance  ◀ NEXT GRADED WORK (after the Reflection doc)

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
