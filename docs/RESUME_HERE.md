# RESUME HERE — start-of-session brief

*Single source for picking this project back up with zero context loss. Read this +
`CLAUDE.md` §0.5 and you're current.*

## Where we are (one paragraph)

The system is built and verified **through containerization (Stages 1–5)**: a
reproducible DVC pipeline with Great Expectations validation gates, a calibrated
**CatBoost** model (registered `readmission-catboost-calibrated` **v1 @ `staging`**,
operating threshold **0.091**, sigmoid calibration; logistic regression kept as the
explainable baseline), a **FastAPI** service (`src/app/`) exposing `/predict`
(calibrated risk + flag + top SHAP factors), `/health`, and a `/metrics` stub, and a
**rootless Podman container** that runs the API. The **train/serve skew check is
exact** — encounter 12522 scores **0.074595** identically in the training pipeline,
the local API, and the container. The model is **baked into the image**
(`deploy/export_model.py` → `deploy/model_bundle/`) because the MLflow registry
stores absolute host paths that don't resolve in-container; the registry **alias
remains the logical rollback handle** (the dual-load code in `src/app/model.py` uses
the alias locally and the baked bundle in-container).

## Exact next action

**ECR push + GitHub Actions CI/CD → AWS Fargate deploy.** Then, in order:
1. **Stage 6 — Observability:** Prometheus scraping `/metrics`, Grafana dashboards,
   Evidently drift report + a concrete numeric retrain trigger, per-request audit log.
2. **Stage 7 — Governance:** Fairlearn subgroup audit (age/gender/race), global SHAP
   summary, `docs/MODEL_CARD.md`, audit-log traceability, written rollback+retrain
   plan, reflection.

Do **not** skip ahead — finish CI/CD + deploy before monitoring.

## CI/CD intent (so it isn't re-derived)

- The pipeline **runs `tests/test_smoke.py` as a MERGE GATE** — push the image to ECR
  **only on green**. The **skew test is the must-not-break invariant**.
- **Bake-in is the correct model approach for CI**: a GitHub Actions runner is
  **stateless** (no `mlflow.db` / `mlruns`), exactly why the runtime mount fails and
  baking works. CI should run `deploy/export_model.py` (needs registry access — decide
  how the runner reaches the registry, e.g. a checked-in/restored store or an S3
  artifact backend) then `podman build`. Note this dependency when wiring CI.
- Target runtime: **AWS Fargate** (per CLAUDE.md: AWS or GCP only).

## Open items to resolve first

1. **SHAP magnitudes are pre-calibration log-odds.** The `/predict` `top_factors`
   contributions come from the base CatBoost learner's margin, not the calibrated
   probability — directions are valid, magnitudes are pre-calibration. **Note this in
   `docs/MODEL_CARD.md`** (Stage 7).
2. **`requirements.txt` vs `pyproject.toml` drift.** `pyproject.toml` + `uv.lock` are
   authoritative (the container uses `uv sync --frozen`). `requirements.txt` is a
   human-readable mirror that has **drifted** (missing `optuna`, `great-expectations`
   added later via `uv add`). Decide: regenerate it from the lock (`uv export`) or
   drop it. Not blocking — the container doesn't use it.
3. **Confirm the baked-in model approach** is acceptable as the deploy path (it is the
   chosen approach; the alias indirection is preserved). If a live registry is wanted
   in-cluster instead, that's a Stage-6+ decision.
4. **Image size ≈ 7.4 GB** (full pinned dep set). A serving-only dependency group
   would slim it materially before pushing to ECR — worth doing as part of CI/CD.

## The invariant to re-verify BEFORE building anything

```bash
uv run pytest tests/test_smoke.py -q     # 5 tests; test_no_train_serve_skew is the gate
```
Expect green, and the skew test confirms encounter 12522 → **0.074595** (API ==
training pipeline). If this breaks, stop and fix before any deploy work.

## Get running again (exact commands)

```bash
# 1. re-verify the invariant
uv run pytest tests/test_smoke.py -q

# 2. (re)bake the @staging model, then build the image (rootless Podman)
uv run python deploy/export_model.py
podman build -f deploy/Containerfile -t readmission-api:latest .

# 3. run (NO mounts — the model is baked in) and hit it
podman rm -f readmission-api 2>/dev/null
podman run -d --name readmission-api -p 8000:8000 readmission-api:latest
curl -s localhost:8000/health        # → v1 @ staging, threshold 0.091, load_source=baked-bundle
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data-binary @tests/sample_request.json    # → readmission_probability 0.074595
```

Local dev (no container) loads from the registry by alias automatically:
`uv run uvicorn src.app.app:app --port 8000`.

## Key references
- `CLAUDE.md` §0.5 — current state & hard rules. `docs/GOALS.md` — staged plan (ticked).
- `docs/SERVING.md` — API + rejection policy. `docs/THRESHOLD_DECISION.md` — 0.091 + rollback.
- `docs/DATA_VALIDATION.md` — GX suites + the contract read three ways.
