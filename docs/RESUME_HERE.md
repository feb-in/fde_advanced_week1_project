# RESUME HERE — Submission Checklist

*Single source for the final stretch. Everything graded is built and verified; what
remains is **verification + push**, not building. Read this + `CLAUDE.md` §0.5.*

## Status: all stages complete — final-submission state

Stages 1–7 ✅ (data → model → calibration/threshold → serving → CI/CD→ECR →
observability → governance), plus a demo UI; pushed to GitHub and a **DagsHub DVC remote**
is configured (data pushed). One open pre-submission item: **verify clean-checkout
reproducibility end-to-end** (below). Optional: AWS Fargate live deploy.

---

## ✅ DONE — and how to view each artifact

Bring the API up first (most demos need it): either
`uv run uvicorn src.app.app:app --port 8000` (loads model from the MLflow registry),
or the full stack `uv run python deploy/export_model.py && podman compose up --build -d`
(api + prometheus + grafana; baked model).

| Deliverable | Status | View it |
|---|---|---|
| **Reproducible data pipeline** | ✅ | `dvc repro` (validate_raw → clean → featurize → validate_processed) — *needs raw CSV, see caveat* |
| **Experiment tracking (MLflow)** | ✅ | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 (runs, calibration, registry `readmission-catboost-calibrated` v1 @ staging) |
| **Calibrated model + threshold** | ✅ | `docs/MODEL_COMPARISON.md`, `docs/THRESHOLD_DECISION.md` (thr 0.091046, recall ~0.51 / precision 0.154) |
| **API `/predict` + SHAP + `/health` + `/metrics`** | ✅ | `curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' --data-binary @tests/sample_request.json` → `0.074595` |
| **Container + CI/CD → Amazon ECR** | ✅ | `.github/workflows/ci.yml` (green on push to `main`); `podman build -f deploy/Containerfile -t readmission-api .` |
| **Audit log (per request)** | ✅ | hit `/predict`, then `tail -1 logs/audit/predictions.jsonl` (request_id, model lineage, inputs, score, latency) |
| **Fairness audit** | ✅ | `uv run python src/governance/fairness.py`; `docs/FAIRNESS_AUDIT.md` (age recall gap 0.69; gender fair; race inconclusive) |
| **Global + local SHAP** | ✅ | `uv run python src/governance/explain.py` → `reports/governance/shap_*.png` + MLflow run `shap_global`; local factors in every `/predict` |
| **Model card** | ✅ | `docs/MODEL_CARD.md` (intended use, 1999–2008 data, real metrics, limits, fairness, lineage, human-in-the-loop, SMOTE rejection) |
| **Reflection** | ✅ | `docs/REFLECTION.md` |
| **Prometheus + Grafana stack** | ✅ | `podman compose up -d` → Grafana http://localhost:3000 (admin/admin), dashboard *"Readmission API — Observability"*; Prometheus http://localhost:9090 (Targets UP, Status→Rules = 3 alerts) |
| **Drift detection (Evidently)** | ✅ | `uv run python src/monitoring/drift.py` → `reports/monitoring/drift_{control,shifted}.html` + `drift_summary.json` (control silent, shifted fires) |
| **Retrain trigger** | ✅ | `uv run python src/monitoring/retrain_trigger.py` (control→keep, shifted→RETRAIN; share>0.10 OR PSI>0.20 OR PR-AUC<0.15) |
| **≥1 alert rule** | ✅ | `deploy/alerts.yml` — APIDown / HighErrorRate / HighP95Latency (`curl localhost:9090/api/v1/rules`) |
| **Demo UI** | ✅ | `READMISSION_API_URL=http://localhost:8000 uv run --group ui streamlit run src/ui/app_streamlit.py` → http://localhost:8501 (form + **load random held-out patient** truth-vs-prediction + BETA banner) |
| **Test suite** | ✅ | `uv run pytest tests/ -q` (28 pass incl. the train/serve-skew invariant) |

---

## ⬜ LEFT before submission

1. ✅ **Pushed to GitHub** — `main` is in sync with `origin/main`
   (`https://github.com/feb-in/fde_advanced_week1_project.git`); CI runs on each push.
2. **Verify clean-checkout reproducibility** — actually run the README path on a fresh
   clone (see below); fix anything that doesn't run end-to-end.
3. **(Optional)** AWS Fargate live deploy for a reachable URL — the ECR image already
   satisfies the deployable-artifact deliverable, so this is a nice-to-have.

## Reproducibility — how a clean clone gets the data

A fresh clone can get the data two ways:

- **(a) rebuild from source (primary, no credentials):** obtain the raw Kaggle CSV → place
  at `data/raw/diabetic_data.csv` → `dvc repro` runs the full pipeline
  (`validate_raw → clean → featurize → validate_processed → make_reference`) and
  regenerates the processed/featurized parquets and the drift reference. **or**
- **(b) pull from DagsHub (optional, needs auth):** a **DagsHub DVC remote** (`origin`) is
  configured and the data is pushed, so `dvc pull -r origin` fetches it instead of
  rebuilding. Needs DagsHub credentials (token lives in `.dvc/config.local`, gitignored) —
  a reviewer without access uses path (a).

The README documents both, with (a) as the default. **Still to do:** actually run path (a)
end-to-end on a clean clone to confirm nothing breaks — that dry-run hasn't been done yet.

---

## Invariants to re-verify before any change

```bash
uv run pytest tests/ -q     # 28 pass; the skew test is the must-not-break invariant
```
Golden: container/local `/predict` for encounter 12522 must return **0.074595**
(`@ staging`, threshold 0.091046). If either breaks, stop and fix first.

## Key references
- `CLAUDE.md` §0.5 — current state. `docs/GOALS.md` — staged plan (all ticked).
- `docs/SERVING.md` — API + stack + UI. `docs/MONITORING.md` — drift + retrain trigger.
- `docs/MODEL_CARD.md`, `docs/FAIRNESS_AUDIT.md`, `docs/REFLECTION.md` — governance.
- `.github/workflows/ci.yml` — ECR pipeline (4 GitHub Secrets: `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`).
