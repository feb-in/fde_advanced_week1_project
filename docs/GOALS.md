# Goals — What We Strive For

The deliverable is a **production-style ML system you can defend**, not a model in
a notebook. Below: the grading shape, what "good" means, and a concrete
**definition of done** per stage.

---

## Grading shape

**~20% modeling · ~80% everything around it.** The 80% = data discipline,
reproducibility, packaging, deployment, observability, governance. Spend effort
where the marks (and the learning) are.

## North-star qualities

- **Reproducible** — one command rebuilds the dataset; one command retrains.
  Nothing depends on hidden notebook state.
- **Honest about imbalance** — PR-AUC, recall@fixed-precision, calibration; never
  accuracy as the headline.
- **Leakage-free** — patient-ID split; no post-discharge features.
- **Explainable** — global + local SHAP; the API returns top contributing factors.
- **Fair & auditable** — subgroup metrics across age/gender/race; every scored
  request traceable.
- **Operable** — containerized, monitored, with a written rollback and a concrete
  retrain trigger.
- **Simple that ships > clever that doesn't.** Baseline before boosting; local
  before deployed.

---

## Definition of done — per stage

### Stage 1 — Data Engineering & Exploration
- [ ] Raw data profiled: dtypes, **missingness per column**, class balance.
- [ ] **`clean.py`** runs end-to-end (script, not cells): handles `?`, resolves
      heavy-missing columns (decision documented), filters expired discharges,
      encodes A1c-missing as its own level, collapses target → binary.
- [ ] **Prediction target defined explicitly** and documented.
- [ ] **Patient-ID-grouped** train/val/test split saved.
- [ ] Data + cleaning steps **versioned with DVC** (or a documented re-runnable
      script).

### Stage 2 — Feature Engineering
- [ ] ICD-9 diagnoses **bucketed into clinical groups**.
- [ ] Categoricals encoded; engineered **total_prior_visits**, **med-change
      count**, **num active meds**, ordinal **age**.
- [ ] **`docs/FEATURE_LOG.md`** records every feature and why it exists.

### Stage 3 — Modeling & Evaluation
- [ ] Split by patient ID; **test set touched once, at the end**.
- [ ] **Baseline:** logistic regression + class weighting (the bar to beat).
- [ ] **Stronger:** XGBoost/LightGBM; CV-tuned; imbalance handled.
- [ ] Evaluated with **PR-AUC, recall@fixed-precision, calibration** — not
      accuracy.
- [ ] Model **calibrated**; **operating threshold chosen & justified** on the cost
      trade-off (missed readmission vs. wasted follow-up) →
      `docs/THRESHOLD_DECISION.md`.
- [ ] **Every experiment tracked in MLflow**; best calibrated model **registered**.

### Stage 4 — Package & Deploy
- [ ] **FastAPI** service: `/predict` → risk score + **top SHAP factors**;
      Pydantic validation; `/health`.
- [ ] **Containerfile** + **compose.yaml** build & run under **Podman**;
      reproducible.
- [ ] Deployed to a free-tier host with a **URL**, or a clearly documented
      local-API path (local is acceptable for this problem).
- [ ] **Rollback plan documented** (MLflow registry stage swap + pinned image tag).

### Stage 5 — Observability & Monitoring
- [ ] Service metrics (**latency, requests, errors**) via **Prometheus** →
      **Grafana** dashboard.
- [ ] **Every prediction logged**; score/confidence distribution tracked over time.
- [ ] **Data & prediction drift** via **Evidently** (demonstrate it firing on a
      shifted batch).
- [ ] **Concrete retrain trigger** — a numeric threshold (e.g. PSI > 0.2 on top
      features, or PR-AUC on labeled feedback < X), not a vibe.

### Stage 6 — Governance & Re-evaluation
- [ ] **Fairness audit** across **age, gender, race** (Fairlearn `MetricFrame`;
      subgroup recall/PR-AUC gaps, mitigation stance).
- [ ] **SHAP** explanations — global (summary) + local (per-prediction).
- [ ] **Audit logging** — request, latency, response, model version traceable for
      every scored request.
- [ ] **Model card** — intended use, data, performance, limits, fairness findings
      → `docs/MODEL_CARD.md`.

---

## Whole-project done

From a clean checkout, a reviewer follows the README to: rebuild data (1 cmd) →
retrain & see the run in MLflow → start API + monitoring under Podman → hit
`/predict` for a calibrated score + top factors → view live Grafana metrics →
generate an Evidently drift report → read the fairness audit + model card → find
an audit-log entry per scored request — and finds a written rollback + retrain
plan.
