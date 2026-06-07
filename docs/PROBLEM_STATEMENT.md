# Problem Statement — 30-Day Hospital Readmission Risk

*Supervised ML · end-to-end production system · Problem 1*

---

## Context

You are the **data science team for a regional hospital network**. Across the
network, too many **diabetic patients are readmitted within 30 days** of
discharge. These early readmissions are costly for the hospital and harmful for
patients, and the care team has limited capacity for proactive follow-up.

## Objective

**At the moment of discharge, produce a calibrated probability that a given
patient will be readmitted within 30 days.** The score lets the care team **rank
patients by risk** and direct extra follow-up (calls, home visits, medication
review) to the **highest-risk individuals first**.

## Prediction target

Binary: **readmitted within 30 days (`<30`) vs. not**. The raw label has three
classes (`<30`, `>30`, `NO`); collapse `>30` and `NO` into the negative class and
**document this decision**.

*Why binary, not 3-class:* the care team triggers one action — 30-day follow-up,
yes/no. A readmission at day 45 (`>30`) is outside the actionable window, so
`>30` and `NO` lead to the same action. Three classes also do **not** reduce the
~11% imbalance — they just split the majority — and you would collapse back to
P(`<30`) for the decision anyway. (A 3-class or ordinal model is fine as an
optional ablation, but the production target is binary.)

## Unit of prediction

One **patient discharge encounter** (one row = one hospital stay). The decision
point is **discharge time**, so **no post-discharge information may enter the
features**.

## What success looks like

Not a notebook model — a **running, monitored, governed service**:
- an **API** that returns a risk score plus the **top contributing factors**,
- experiment tracking, a reproducible & versioned data pipeline,
- dashboards, drift detection, a fairness audit,
- audit logs, a model card, and a documented **rollback/retrain plan**.

## Input / serving pattern

- **Training:** a single pass over the fixed historical dataset (re-run on
  retrain). Not streaming.
- **Serving:** the brief requires a **real-time request/response API** (score one
  discharge on demand). This is "online single-record inference," **not** a
  streaming-infra job — do not build Kafka/Flink. **End-of-day batch scoring** (a
  nightly ranked worklist for the care team) is the more natural operational
  pattern and a good thin wrapper to add on top of the same model.

## Grading weight

**~20% modeling, ~80% everything around it** (data discipline, packaging,
deployment, observability, governance). Plan effort accordingly.

---

## Why it's hard (design these in from the start)

1. **Severe class imbalance (~11% positive).** A "never readmitted" model scores
   ~89% accuracy and is useless. **Accuracy is a trap** — use PR-AUC, recall at
   fixed precision, and calibration.
2. **Non-obvious signals.** Number of **prior inpatient visits** often outweighs a
   dramatic-looking lab value.
3. **Messy data.** Missing values are coded as **`?`** (not blanks); some columns
   are nearly empty; medical codes are **high-cardinality**.
4. **Regulated, human domain.** **Fairness, explainability, and auditability are
   deliverables, not extras.**
