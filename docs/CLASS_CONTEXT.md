# Class Context — The ML Lifecycle (Why We Build It This Way)

This project is the practical application of an ML-lifecycle course. The
philosophy below explains *why* the brief weights infrastructure over modeling.
Keep it in mind when making design calls.

---

## Six truths that last

1. **It's a loop, not a line.** Build → ship → watch → and back. The shape is a
   circle for a reason.
2. **Data work dominates.** Most effort — and most bugs — live in the data, not
   the model. (~**70%** of effort on data; training ≤ **5–10%**.)
3. **Models decay.** Even a perfect model worsens as the world drifts from its
   training data.
4. **Simple usually wins.** The model you can ship, explain, and maintain beats
   the clever one you can't. Don't start with XGBoost; start simple.
5. **Monitoring is the job.** Deploying *starts* the work. The long game is
   watching and re-evaluating.
6. **Boring is reliable.** Gradual rollouts, rollbacks, reproducible builds —
   unglamorous, and what survives.

---

## The lifecycle stages (and where each goes wrong)

**1. Frame the problem.** Turn a fuzzy goal into a precise prediction target;
choose metrics tied to real business cost; define who acts on the output. *Goes
wrong:* optimizing accuracy when the rare case is what matters.
> *IBM Watson for Oncology* was framed around a goal the data couldn't support;
> unsafe recommendations surfaced and the effort was wound down. Solving the wrong
> problem impressively is still solving the wrong problem.

**2. Data engineering.** Ingest, clean, validate, version. Build a **reproducible
pipeline, not a one-off notebook**. *Goes wrong:* **data leakage**; a pipeline
only its author can re-run.
> *Amazon's hiring AI* learned bias from a decade of mostly-male résumés —
> baked into the **data, not the algorithm**. Scrapped in 2017.
> *Reproducibility test:* not "did 10 steps clean the data once" but "can those
> steps re-run on production data whose missingness differs."

**3. Modeling.** Baseline first, then stronger models; tune; handle imbalance.
Only ~20% of the grade.

**4. Evaluation.** Metrics tied to the costly rare event; calibration; a
justified operating threshold. *Goes wrong:* a 95%-accurate churn model that
"wins" by predicting nobody ever leaves.

**5. Deploy.** Package, containerize, expose an API; plan rollback. Deploying is
the *start* of the work.

**6. Monitor.** Service metrics, prediction logging, drift detection, a concrete
retrain trigger. *Goes wrong:* monitoring treated as optional; the model quietly
rots.

**7. Govern (continuous).** Fairness audits, SHAP explanations, model cards,
audit logs, human review of low-confidence cases; decide keep / retrain / retire.
*Goes wrong:* governance treated as one-time paperwork.
> *Dutch childcare-benefits scandal:* a risk-scoring algorithm wrongly flagged
> thousands of families — disproportionately minorities — for fraud. The fallout
> contributed to the **entire Dutch government resigning in 2021**. Ungoverned
> models can do societal-scale harm. Regulators — and denied patients — will not
> accept "the model decided" as an explanation.

---

## Three thresholds, set on day one
Define, up front, the thresholds for when you **keep** the model running, when you
**retrain**, and when you **retire** it. The lifecycle loops from training through
governance and keeps running while the model is in production.
