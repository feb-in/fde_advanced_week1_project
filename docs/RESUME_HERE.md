# RESUME HERE — start-of-session brief

*Single source for picking this project back up with zero context loss. Read this +
`CLAUDE.md` §0.5 and you're current.*

## Where we are (one paragraph)

The **build + ship arc is complete**. A reproducible DVC pipeline cleans the data and
gates it with Great Expectations; **CatBoost** is tuned, **calibrated (sigmoid)**, and
registered as `readmission-catboost-calibrated` **@ `staging`** with operating
threshold **0.091046** in the version tag (logistic regression kept as the explainable
baseline). A **skew-free FastAPI** service (`src/app/`) serves `/predict` (calibrated
risk + flag + top SHAP factors), `/health`, and a `/metrics` stub — the golden
encounter **12522 scores 0.074595** identically in training, the local API, and the
container. The service is packaged as a **slim 943 MB container** with the model
**baked in** (`deploy/model_bundle/`, tracked in git), and **CI/CD via GitHub Actions
is GREEN**: a test-gated pipeline that, on push to `main`, runs the schema tests,
builds the image, tests the running container over HTTP, and **pushes to Amazon ECR**
(git SHA + `latest`) only when everything passes.

## Exact next action (grade-priority order)

1. **Reflection doc first** — `docs/REFLECTION.md`. Highest grade-value, low effort:
   the lifecycle story (framing → data discipline → modeling → calibration/threshold →
   serving/skew → containerize → CI/CD), what went wrong and what you'd do differently.
2. **Governance (Stage 7)** — Fairlearn `MetricFrame` across **age / gender / race**
   (subgroup recall / PR-AUC gaps + mitigation stance); SHAP global summary + the local
   per-prediction factors the API already returns; **audit logging** per scored request
   (request, response, model version, latency); `docs/MODEL_CARD.md`; MLflow
   lineage/versioning; a **human-in-the-loop** policy for low-confidence cases.
3. **Observability (Stage 6)** — Prometheus scraping `/metrics` + Grafana dashboards;
   prediction logging; an Evidently drift report demonstrated firing on a shifted
   batch; **≥1 alert**; a **concrete numeric retrain trigger** (e.g. PSI > 0.2 on top
   features, or labelled-feedback PR-AUC < X).
4. **OPTIONAL** — Fargate live deploy (the ECR image already satisfies the
   deployable-artifact deliverable); a Streamlit UI over `/predict`.

Do them in this order — Reflection + Governance are where the remaining grade lives.

## Open items for the model card / reflection to capture

- **SHAP magnitudes are pre-calibration log-odds.** The `/predict` `top_factors`
  contributions come from the base CatBoost learner's margin, not the calibrated
  probability — **directions are valid**, magnitudes are pre-calibration. State this.
- **Threshold 0.091046 flags ~30% of discharges at recall ~0.5.** A documented
  dial-down exists (recall ~0.40 → ~22% flagged) in `docs/THRESHOLD_DECISION.md`.
- **The single GLOBAL threshold must be revisited per-subgroup** in the fairness audit —
  one cutoff can land very differently across age / gender / race.
- **CI IAM uses a managed ECR PowerUser policy.** Note **least-privilege + OIDC
  (keyless)** as the production hardening (vs the static access-key the workflow uses).
- **`requirements.txt`** is a human-readable mirror that has drifted from `pyproject.toml`
  + `uv.lock` (the authoritative source). Regenerate or drop it — not blocking.

## The invariant to re-verify BEFORE building anything

```bash
uv run pytest tests/ -q          # full suite (container tier skips without a server)
```
And the golden artifact check — container `/predict` for encounter 12522 must return
**0.074595** with the model loaded `@ staging`. If either breaks, stop and fix first.

## Get running again (exact commands)

```bash
# 0. (governance/observability work needs the registry) start MLflow locally
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000   # or just use sqlite:/// directly

# 1. re-verify the invariant
uv run pytest tests/ -q

# 2. (re)bake the @staging model + build the slim image (rootless Podman)
uv run python deploy/export_model.py
podman build -f deploy/Containerfile -t readmission-api:latest .

# 3. run the container (model baked in, NO mounts) and hit it
podman rm -f rapi 2>/dev/null
podman run -d --name rapi -p 8000:8000 readmission-api:latest
curl -s localhost:8000/health        # → v1 @ staging, threshold 0.091046, load_source=baked-bundle
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data-binary @tests/sample_request.json    # → readmission_probability 0.074595

# 4. trigger CI: commit + push to main → GitHub Actions tests, builds, pushes to ECR on green
git push origin main
```

Local dev API (no container) loads from the registry by alias automatically:
`uv run uvicorn src.app.app:app --port 8000`.

## Key references
- `CLAUDE.md` §0.5 — current state + hard rules. `docs/GOALS.md` — staged plan (ticked).
- `.github/workflows/ci.yml` — the test-gated ECR pipeline (4 GitHub Secrets required:
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`).
- `docs/SERVING.md` — API + rejection policy. `docs/THRESHOLD_DECISION.md` — 0.091 + rollback.
- `docs/DATA_VALIDATION.md` — GX suites. `docs/MODEL_COMPARISON.md` — LR vs CatBoost verdict.
