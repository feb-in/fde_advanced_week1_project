# Monitoring — Drift Detection (Stage 6)

How we detect that incoming data has drifted away from what the model was trained on —
the signal that a model may be going stale. This gate stands up the **drift detector and
validates it**; the dashboards, alerts, and the automated retrain trigger are the next
gate.

## ⚠️ This SIMULATES drift — there is no production traffic yet

The service has no live request stream, so there is no real "current" batch to compare
against the baseline. To prove the detector actually works (and is pointed at the right
reference), `src/monitoring/drift.py` **synthesizes** a current batch and **deliberately
shifts a few features**, then checks two things:

1. an **unshifted control** batch does **not** trip the detector (no crying wolf), and
2. a **shifted** batch **does** — and the specific features we moved are the ones flagged.

Everything here is a detector test, not a finding about real-world data.

## The reference (drift baseline)

`data/monitoring/reference.parquet` (**DVC-tracked**) is the model's **training feature
distribution**: the seed-42 **train** split of the exact featurized data the model
trained on — **55,989 rows × 54 features**, plus the calibrated `@staging` `prediction`
per row (for prediction drift) and `target`. Built by `src/monitoring/make_reference.py`.
Baseline mean calibrated score **0.0899** ≈ prevalence 0.0898. Labels are **not** present
at serving time, so target drift is only meaningful in this offline simulation.

## The detector

[Evidently](https://www.evidentlyai.com/) **0.7.21**, `DataDriftPreset`, comparing
current vs reference. Per column it auto-selects a test (K-S / chi-square / Z-test
p-value for smaller cardinalities, Wasserstein/PSI distance for larger) and flags the
column as drifted; `DriftedColumnsCount` aggregates to a **share of drifted columns**.
Dataset drift is declared when that share exceeds **`drift_share = 0.10`** — a
*documented demonstration threshold* (a targeted shift moves only a handful of 54
features), **not** the production trigger (that is the next gate). Evidently is an
**offline analysis** dependency — it is deliberately **excluded from the slim serving
image** (`deploy/requirements-serve.txt`); `/predict` does not import it.

## The intentional shifts (configurable)

Applied to a copy of an in-distribution sample, each clearly commented INTENTIONAL and
tunable by CLI flag:

| Shift | Feature(s) touched | CLI flag (default) |
|---|---|---|
| Population ages older | `age_midpoint`, `age_bucket` | `--age-shift 20` |
| Diagnosis-coding shift → Circulatory | `diag_1_bucket` | `--diag1-circulatory-frac 0.40` |
| Sicker case-mix (more prior inpatient) | `number_inpatient`, `service_utilization`, `inpatient_ge_2` | `--inpatient-bump 2` |
| More emergency admissions | `admission_type_grp` | `--emergency-frac 0.30` |

The shifted batch is **re-scored** through the model so **prediction drift** reflects the
input shift.

## Result (default settings, `--current-n 3000`)

| batch | drifted columns | share | dataset verdict | prediction drift | mean score |
|---|---:|---:|---|---|---:|
| baseline (reference) | — | — | — | — | 0.0899 |
| **control** (unshifted) | **0 / 54** | 0.000 | **no drift** ✓ | no (Wasserstein 0.026) | 0.0912 |
| **shifted** (intentional) | **8** | 0.145 | **DATASET DRIFT DETECTED** ✓ | **YES** (Wasserstein 1.66) | 0.1831 |

The shifted run flags **all 7 deliberately-moved features (7/7)** plus `prediction`;
mean calibrated risk roughly doubles (0.090 → 0.183) as the cohort ages and gets sicker.
The control stays silent. **Detector validation: PASS.** Machine-readable record:
`reports/monitoring/drift_summary.json` (committed); full interactive reports
`reports/monitoring/drift_{control,shifted}.html` (regenerable, gitignored — ~6 MB each).

## Retrain trigger (implemented — `src/monitoring/retrain_trigger.py`)

The drift signals feed a **concrete, deterministic retrain rule** — a pure function over
a drift summary, so it is testable and is exactly what a scheduled job (or a
Prometheus/Grafana alert) would evaluate against real scored-traffic batches.

**Retrain if ANY of:**

| # | Condition | Threshold | Why this number |
|---|---|---|---|
| 1 | dataset-drift share | **> 0.10** | mirrors the detector's `drift_share` — one consistent dataset-drift line |
| 2 | PSI on **any top-SHAP feature** | **> 0.20** | standard "significant population shift" line (<0.1 stable, 0.1–0.2 moderate, >0.2 act); applied to the model's own top drivers |
| 3 | PR-AUC on freshly-labelled data | **< 0.15** | ground-truth backstop ≈ 75% of the 0.207 test AUPRC; only evaluable once outcomes exist |

These map to the project's three day-one thresholds — **keep / retrain / retire**. Drift
(rules 1–2) is the leading indicator; the labelled PR-AUC floor (rule 3) is the
definitive backstop.

**Validated against the two batches from the drift run:**

| batch | decision | tripped rules |
|---|---|---|
| control (unshifted) | **keep model** | none |
| shifted (intentional) | **RETRAIN** | share 0.145 > 0.10; PSI>0.2 on `number_inpatient` (13.98), `service_utilization` (13.44), `age_midpoint` (1.91), `diag_1_bucket` (0.35) |

Silent on control, fires on shifted — `[trigger validation] PASS`.
Run: `uv run python src/monitoring/retrain_trigger.py`.

**Wired to alerting/dashboards:** the live API metrics drive a Grafana dashboard and a
Prometheus alert rule (see `docs/SERVING.md` → Observability stack). A score-distribution
panel would need a new prediction-score histogram in the app (noted as a follow-up, not
built — it would touch the serving image).

## Run

```bash
uv run python src/monitoring/make_reference.py   # (re)build the baseline; then: dvc add ...
uv run python src/monitoring/drift.py            # control + shifted; writes HTML + JSON
uv run python src/monitoring/drift.py --age-shift 25 --emergency-frac 0.5 --current-n 4000
```
