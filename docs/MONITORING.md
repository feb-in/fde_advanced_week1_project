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

## What this feeds next (NOT implemented this gate)

The drift signals above are the inputs to the **retrain trigger** (next gate). Candidate
numeric triggers, to be finalized then:

- **dataset-drift share** sustained above a threshold over a rolling window of scored
  batches, and/or
- **PSI > 0.2** on the **top SHAP features** (`discharged_home`, `diag_1_bucket`,
  `number_inpatient`, `medical_specialty`, `age_midpoint` — see `docs/MODEL_CARD.md`),
  and/or
- **prediction drift** (score-distribution shift) beyond a bound, and/or
- **labelled-feedback PR-AUC** falling below a floor once outcomes are observed.

These map to the project's three day-one thresholds — **keep / retrain / retire**. The
alert wiring (Prometheus/Grafana) and the concrete trigger numbers are the next session.

## Run

```bash
uv run python src/monitoring/make_reference.py   # (re)build the baseline; then: dvc add ...
uv run python src/monitoring/drift.py            # control + shifted; writes HTML + JSON
uv run python src/monitoring/drift.py --age-shift 25 --emergency-frac 0.5 --current-n 4000
```
