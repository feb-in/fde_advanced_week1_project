# Fairness Audit — Subgroup Performance of the Lead Model

**Model audited:** calibrated CatBoost **v1 @ `staging`**, single global operating
threshold **0.091046** (sigmoid-calibrated probability).
**Data:** the held-out **test set (13,998 patients, 1,257 true 30-day readmits,
prevalence 0.0898)** — the same seed-42 split used throughout, scored once here.
**Tool:** Fairlearn `MetricFrame` across **age, gender, race**. Reproduce with
`uv run python src/governance/fairness.py`.

This audit **measures and reports**. It does not mitigate — no per-group thresholds,
no reweighting. Findings here drive the model card and the mitigation stance.

**Overall operating point (all groups):** recall **0.511**, flag rate **0.297**.
Every subgroup below is evaluated at the *same* 0.091 cut, so differences are what
one global threshold actually produces — the open question flagged in
`docs/REFLECTION.md`.

> **Reading the columns.** `prevalence` = true readmission rate in the group;
> `recall` = share of true readmits the tool flags; `precision` = share of flags that
> are true readmits; `flag_rate` = share of the group flagged for follow-up; `FPR` =
> share of non-readmits wrongly flagged; `AUPRC`/`ROC_AUC` = ranking quality within
> the group. The two disparity numbers are Fairlearn's **demographic-parity
> difference** (max flag-rate gap) and **equalized-odds difference** (max TPR/FPR gap).

---

## 1. Age — the dominant disparity

| age band | support | prevalence | recall | precision | flag_rate | FPR | AUPRC | ROC_AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [0-10)\* | 29 | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | n/a |
| [10-20) | 109 | 0.046 | 0.400 | 0.200 | 0.092 | 0.077 | 0.351 | 0.798 |
| [20-30) | 204 | 0.074 | 0.533 | 0.267 | 0.147 | 0.116 | 0.358 | 0.782 |
| [30-40) | 537 | 0.065 | 0.371 | 0.200 | 0.121 | 0.104 | 0.237 | 0.743 |
| [40-50) | 1,352 | 0.070 | **0.277** | 0.132 | 0.146 | 0.136 | 0.160 | 0.632 |
| [50-60) | 2,466 | 0.079 | 0.390 | 0.191 | 0.161 | 0.141 | 0.196 | 0.685 |
| [60-70) | 3,125 | 0.088 | 0.431 | 0.138 | 0.274 | 0.259 | 0.185 | 0.636 |
| [70-80) | 3,606 | 0.105 | 0.593 | 0.164 | 0.379 | 0.354 | 0.229 | 0.659 |
| [80-90) | 2,239 | 0.102 | **0.693** | 0.144 | **0.491** | 0.468 | 0.231 | 0.664 |
| [90-100) | 331 | 0.100 | 0.515 | 0.123 | 0.417 | 0.406 | 0.211 | 0.630 |

\* low support (n<100) — interpret with caution.

**Demographic-parity diff 0.491 · equalized-odds diff 0.693 · recall gap 0.693 ·
flag-rate gap 0.491.** Collapsed to the model's 3-level `age_bucket`, the pattern
holds: [30-60) recall **0.355** / flag 0.151 vs [60-100) recall **0.566** / flag
**0.372** (DP diff 0.255, recall gap 0.211).

**Reading it.** This is the audit's biggest finding. The single 0.091 threshold flags
**~49% of patients in their 80s but only ~15% of those in their 40s** (~3.4×), and it
catches **~69% of true readmits among 80-somethings versus ~28% among 40-somethings**
(~2.5×). Two things are tangled here, and honesty requires separating them:

- **Part of this is correct risk stratification.** Readmission prevalence genuinely
  rises with age (0.07 → 0.10), and a *calibrated* score is supposed to reflect that.
  Flagging more older patients is not in itself a bug.
- **Part of it is a real equal-opportunity gap.** The **recall** difference means a
  middle-aged patient who *will* be readmitted is far more likely to be **missed**
  than an identical-outcome older patient. The model also simply *ranks worse* for the
  40-60 band (ROC-AUC ~0.63-0.69, AUPRC ~0.16-0.20) than at the age extremes. A
  follow-up program run on this score would systematically under-serve readmission-bound
  patients in their 40s-50s. That is the disparity a global threshold bakes in.

---

## 2. Gender — negligible disparity

| gender | support | prevalence | recall | precision | flag_rate | FPR | AUPRC | ROC_AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Female | 7,507 | 0.088 | 0.533 | 0.151 | 0.313 | 0.292 | 0.208 | 0.673 |
| Male | 6,491 | 0.091 | 0.486 | 0.159 | 0.279 | 0.258 | 0.208 | 0.661 |

**Demographic-parity diff 0.034 · equalized-odds diff 0.047 · recall gap 0.047.**

**Reading it.** Effectively fair. Women are flagged ~3 points more often and caught
~5 points more often; ranking quality (AUPRC 0.208 for both) is identical. The gaps
are small and within the range expected from the modest prevalence difference — no
action indicated beyond continued monitoring.

---

## 3. Race — modest disparity, partly small-sample noise

| race | support | prevalence | recall | precision | flag_rate | FPR | AUPRC | ROC_AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Caucasian | 10,531 | 0.091 | 0.517 | 0.150 | 0.313 | 0.293 | 0.203 | 0.657 |
| AfricanAmerican | 2,508 | 0.093 | 0.496 | 0.175 | 0.263 | 0.239 | 0.223 | 0.691 |
| Hispanic | 287 | 0.108 | 0.516 | 0.232 | 0.240 | 0.207 | 0.331 | 0.702 |
| Other | 229 | 0.048 | 0.455 | 0.122 | 0.179 | 0.165 | 0.240 | 0.774 |
| Unknown | 347 | 0.049 | 0.471 | 0.103 | 0.225 | 0.212 | 0.140 | 0.710 |
| Asian\* | 96 | 0.094 | 0.333 | 0.214 | 0.146 | 0.126 | 0.292 | 0.773 |

\* low support (n<100) — the 0.333 recall is 3 of 9 true readmits; do not over-read.

**Demographic-parity diff 0.168 · equalized-odds diff 0.184 · recall gap 0.184.**
Both disparity numbers are driven by the **n=96 Asian** group; excluding it, recall
spans only **0.455 (Other) → 0.517 (Caucasian)** — a ~6-point gap.

**Reading it.** Among the well-supported groups (Caucasian, AfricanAmerican), recall
is close (0.52 vs 0.50) and AfricanAmerican patients are actually flagged *less* often
(26% vs 31%) at near-identical prevalence — so this is not a flag-them-more bias
against the largest minority group. The headline 0.184 gap is a small-sample artifact
of the 96-patient Asian and 287-patient Hispanic cells, where a handful of patients
swing the rate. The honest statement: **no large, well-supported racial disparity is
evident, but the small-group estimates are too noisy to certify fairness for Asian,
Hispanic, Other, or Unknown patients.**

---

## 4. The answer to the open question, and the stance

**Does one global 0.091 threshold land differently across groups? Yes — decisively
along age, trivially along gender, and modestly (mostly small-sample) along race.**
Age is the axis that matters: a single cut buys high recall on the elderly at the cost
of missing roughly **two-thirds of readmission-bound patients in their 40s-50s**.

**Mitigation stance (measured here; decision deferred):**
- **Do not deploy a fairness fix blindly.** The age gap is part legitimate risk
  signal, part genuine recall inequity — they must be separated before acting.
- **Candidate remedies for the next gate / model card:** (a) **per-age-band
  thresholds** targeting equal recall, which would lift the 40-60 catch rate at the
  cost of more flags there; (b) report and monitor subgroup recall as a standing
  fairness metric, not a one-off; (c) treat the small race cells as **insufficient
  evidence** and collect more data rather than claim parity.
- **Human-in-the-loop matters most exactly where the model is weakest** — the
  middle-aged, lower-ROC-AUC band — which is where borderline scores should route to
  clinician review (see the model card's human-in-the-loop policy).

These findings are carried verbatim into `docs/MODEL_CARD.md` (Fairness section).
