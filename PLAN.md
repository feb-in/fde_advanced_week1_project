# PLAN — Build Plan (session-based, part-time)

You're fitting this around office work, so this plan is organized as **6 phases by
effort**, not as 6 whole days. Each phase has an hour budget and is broken into
**evening-sized chunks** (~2–3h). A suggested calendar maps them onto evenings
plus one weekend block — but **work the phases in order; the calendar flexes.**

**Total effort: ~28–34 focused hours.** Modeling is deliberately compressed (it's
20%); the new tooling (MLflow, Podman, Prometheus, Evidently) is front-loaded
because that's where the grade *and* your learning live.

**Golden rules:** log to MLflow from the first model · write docs as you go ·
commit small and often · simple that ships beats clever that doesn't · read what
Claude Code generates, don't just paste.

---

## Suggested calendar (adjust freely)

| Slot | Phase | Hours |
|---|---|---|
| Evening 1 | Phase 0 — Setup & tooling (part A) | ~2.5 |
| Evening 2 | Phase 0 (part B) + start Phase 1 | ~2.5 |
| Evening 3 | Phase 1 — Data engineering | ~2.5 |
| **Weekend block** | Phase 2 (model) + start Phase 3 (deploy) | ~5–6 |
| Evening 4 | Phase 3 — finish deploy | ~2.5 |
| Evening 5 | Phase 4 — observability | ~3 |
| Evening 6 | Phase 5 — governance + polish | ~3 |
| Spare evening / 2nd weekend | Buffer + demo | ~2–3 |

> The two heaviest, highest-friction phases are **Phase 3 (deploy)** and
> **Phase 4 (observability)** — each may spill into a second evening. If anything
> slips, let it slip *there* and into the buffer — **never into Phase 5
> (governance)**, which is graded heavily.

---

## Phase 0 — Setup & Tooling  (~3–4h) — *the conceptual investment*

**Part A — environment & containers (~2h)**
- [ ] `git init`, create repo from this starter, push.
- [ ] Python env (venv/conda); `pip install -r requirements.txt`.
- [ ] **Podman working:** `podman --version`; run a rootless hello-world; note the
      `:Z` volume-mount rule (see `CLAUDE.md` Podman section).
- [ ] `dvc init`; understand `dvc add` + the git pointer model (keep minimal).

**Part B — learn MLflow (~1.5h)** *(do this before any modeling)*
- [ ] Start the server (command in `CLAUDE.md`).
- [ ] Throwaway run: log a param, a metric, an artifact, and a model. Open the UI.
- [ ] Understand: experiments → runs → artifacts → **model registry** stages
      (`None`/`Staging`/`Production`) — this is your future rollback switch.

**Done when:** repo pushed, env reproducible, Podman runs rootless, MLflow UI up
with one real logged run.

---

## Phase 1 — Data Engineering (Stage 1)  (~4–5h)

- [ ] `src/data/download.py`: fetch the **Kaggle CSV** (`diabetic_data.csv` — keeps
      `patient_nbr`). Load with `na_values="?"`.
- [ ] Quick profiling report (`ydata-profiling`) → eyeball missingness + balance.
- [ ] `src/data/clean.py` (a script, not cells): handle missing; **drop dead cols**
      (`examide`, `citoglipton`); **filter expired discharges**; **A1c-missing as
      its own category**; collapse target → binary; map ID codes via
      `IDs_mapping.csv`. Document drop-vs-impute decisions.
- [ ] **Patient-ID-grouped split** (`GroupShuffleSplit` on `patient_nbr`) →
      train/val/test; save to `data/processed/`.
- [ ] `dvc add` raw + processed; commit.

**Done when:** `clean.py` runs end-to-end, grouped split saved + DVC-tracked,
decisions written down. *(See `docs/CAVEATS.md` for every trap this phase must
clear.)*

---

## Phase 2 — Features + Model + Eval (Stages 2–3)  (~5–6h) — *the 20%, move fast*

**Features (~2h)**
- [ ] `src/features/build_features.py`: **ICD-9 → clinical buckets**;
      `total_prior_visits`; med-churn count (Up/Down); num active meds; ordinal
      `age`; encode categoricals.
- [ ] Log each feature in `docs/FEATURE_LOG.md`.

**Model + eval (~3–4h) — log everything to MLflow**
- [ ] Baseline: **logistic regression + `class_weight='balanced'`** → MLflow.
- [ ] Stronger: **XGBoost** (`scale_pos_weight`), CV-tuned → MLflow.
- [ ] Evaluate: **PR-AUC, recall@fixed-precision, calibration curve, Brier** — log
      all plots. **Not accuracy.**
- [ ] **Calibrate** (`CalibratedClassifierCV`).
- [ ] **Choose + justify the operating threshold** on the cost trade-off →
      `docs/THRESHOLD_DECISION.md`.
- [ ] **Register** the best calibrated model → MLflow registry → `Staging`.

**Done when:** registered calibrated model, eval plots in MLflow, threshold
justified. **Do not gold-plate the model — stop and move on.**

---

## Phase 3 — Package & Deploy (Stage 4)  (~5–6h) — *heavy; may span 2 evenings*

- [ ] `src/serving/app.py`: **FastAPI**, loads model from MLflow registry;
      `/predict` → score + **top SHAP factors**; Pydantic input validation;
      `/health`. Test with uvicorn.
- [ ] `deploy/Containerfile` (**Podman**); `podman build` + `podman run` (remember
      `:Z` on mounts).
- [ ] `deploy/compose.yaml`; bring up with `podman compose` / `podman-compose`.
- [ ] **Expose:** deploy API to a free host for a public URL **or** document the
      local-API path (local is allowed here). *Verify current free-tier terms —
      Render / Railway / Fly.io / HF Spaces shift often.*
- [ ] **Rollback plan**: MLflow stage swap (`Production` ← prior version) + pinned
      image tag → write it down.

**Done when:** containerized API reachable, returns score + factors, rollback
documented.

---

## Phase 4 — Observability (Stage 5)  (~5–6h) — *most new infra; may span 2 evenings*

- [ ] Instrument FastAPI with `prometheus-fastapi-instrumentator` (latency,
      requests, errors → `/metrics`).
- [ ] Add **Prometheus** + **Grafana** to `compose.yaml` (under Podman);
      `deploy/prometheus.yml` scrape config; one Grafana dashboard (p95 latency,
      req/s, error rate).
- [ ] **Log every prediction** (input hash, score, model version, timestamp);
      track score distribution over time.
- [ ] **Evidently** drift report: training vs. a **shifted** batch (shift a few
      features so drift visibly fires).
- [ ] **Retrain trigger**: a concrete numeric threshold (e.g. PSI > 0.2 on top
      features) → `src/monitoring/retrain_trigger.py`.

**Done when:** Grafana shows live metrics, Evidently report generated, retrain
trigger coded + documented.

---

## Phase 5 — Governance + Polish (Stage 6)  (~4–5h) — *protect this time*

- [ ] **Fairness audit** across **age/gender/race** (Fairlearn `MetricFrame`;
      subgroup recall/PR-AUC gaps) → note disparities + stance.
- [ ] **SHAP** global summary + a couple of local explanations; document.
- [ ] **Audit logging**: every scored request fully traceable (request, latency,
      response, model version).
- [ ] **Model card** → `docs/MODEL_CARD.md` (intended use, data, performance,
      limits, fairness).
- [ ] **README**: one-command reproduce (clean → train → serve under Podman).
- [ ] End-to-end **smoke test** (`tests/test_smoke.py`); short demo walkthrough.

**Done when:** fairness + SHAP + audit log + model card done, repo reproducible
from the README.

---

## Buffer
Spillover from Phases 3–4, a real `retrain.py` the trigger would invoke, deploy
hardening, a short demo video, doc cleanup.
