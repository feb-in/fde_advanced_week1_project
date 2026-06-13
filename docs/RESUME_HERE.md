# RESUME HERE — Submission Checklist

*Single source for the final stretch. Everything graded is built and verified; what
remains is **verification + push**, not building.*

## Status: all stages complete — final-submission state

Stages 1–7 ✅ (data → model → calibration/threshold → serving → CI/CD→ECR →
observability → governance), plus a demo UI; pushed to GitHub. The clean-checkout
reproducibility dry-run is **done** — it found build-order gaps, now fixed; the documented
build order below is the corrected, verified path. Optional: AWS Fargate live deploy.

---

## ✅ DONE — and how to view each artifact

Bring the API up first (most demos need it): the container stack
`podman compose up --build -d` (api + prometheus + grafana) — the calibrated model is
committed in `deploy/model_bundle/` and baked into the image, so **no export or training is
needed**. (Local `uv run uvicorn src.app.app:app --port 8000` also works but loads from the
MLflow registry, so it needs the model registered `@staging` first — see the build order.)

| Deliverable | Status | View it |
|---|---|---|
| **Reproducible data pipeline** | ✅ | `dvc repro validate_processed` (validate_raw → clean → featurize → validate_processed) — *needs raw CSV; see Reproducibility below* |
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

## Status of the final items

1. ✅ **Pushed to GitHub** — `main` is in sync with `origin/main`
   (`https://github.com/feb-in/fde_advanced_week1_project.git`); CI runs on each push.
2. ✅ **Clean-checkout reproducibility dry-run** — done. It surfaced build-order gaps
   (bare `dvc repro` hit `make_reference` before the model existed; `tune.py`/`calibrate.py`
   were undocumented; `dvc pull` isn't anonymous; a redundant `export_model.py` step) —
   all fixed in the README + this file.
3. **(Optional)** AWS Fargate live deploy for a reachable URL — the ECR image already
   satisfies the deployable-artifact deliverable, so this is a nice-to-have.

## Reproducibility — clean-clone build order

**Primary, supported path** (no credentials), in order:

```bash
# 1. data — stops before make_reference (which needs the model):
dvc repro validate_processed          # validate_raw → clean → featurize → validate_processed
# 2. model — IN ORDER (tune.py is a PREREQUISITE of calibrate.py, which reads its run):
uv run python src/models/train.py        # LR baseline + CatBoost → MLflow
uv run python src/models/tune.py          # Optuna search (REQUIRED before calibrate)
uv run python src/models/calibrate.py     # calibrates + registers v1 @staging
# 3. drift baseline — now the registry is populated:
dvc repro make_reference
```

- A *bare* `dvc repro` also runs `make_reference`, which fails (exit 255) until step 2 has
  registered `@staging` — so on a fresh clone use `dvc repro validate_processed` for data.
- **Just running the system** needs none of the above: the calibrated model is committed in
  `deploy/model_bundle/`, so `podman compose up --build` serves the golden model directly.
- **DagsHub pull (optional):** a DVC remote (`origin`) is configured and the data was pushed,
  but **`dvc pull -r origin` requires DagsHub auth and is not guaranteed to work
  anonymously** — the raw-CSV rebuild above is THE supported path.

> **Golden score note:** the committed bundle reproduces `0.074595` exactly (serving + CI
> are deterministic); a *from-scratch retrain* matches it only with the full default Optuna
> trial budget (~30+ min) — fewer trials give a slightly different score, which is expected.

---

## Invariants to re-verify before any change

```bash
uv run pytest tests/ -q     # 28 pass; the skew test is the must-not-break invariant
```
Golden: container/local `/predict` for encounter 12522 must return **0.074595**
(`@ staging`, threshold 0.091046). If either breaks, stop and fix first.

## Key references
- `docs/GOALS.md` — the authoritative staged plan + definition of done (all ticked).
- `docs/SERVING.md` — API + stack + UI. `docs/MONITORING.md` — drift + retrain trigger.
- `docs/MODEL_CARD.md`, `docs/FAIRNESS_AUDIT.md`, `docs/REFLECTION.md` — governance.
- `.github/workflows/ci.yml` — ECR pipeline (4 GitHub Secrets: `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`).
