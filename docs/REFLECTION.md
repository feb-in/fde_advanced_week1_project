# Reflection — Trade-offs, Production Gaps, and Limits

This is the analytical companion to `PROGRESS.md`. `PROGRESS.md` records *what* was
done; this records the decisions that were genuinely under tension, what I would
change before this ran on real patients, and where the model's ceiling actually is.
Numbers are from the held-out test set (13,998 patients), touched once.

## 1. Key trade-offs I made, and what each cost

**Binary target — clarity bought with lost granularity.** The raw label has three
classes (`<30`, `>30`, `NO`); I collapsed it to `<30` vs everything else. The care
team triggers exactly one action — schedule 30-day follow-up or don't — so a binary
score maps cleanly onto the decision and onto PR-AUC. The cost is real: I threw away
the `>30` vs `NO` distinction, so the model cannot tell a patient who returns on day
40 from one who never returns. A later "any readmission" or ordinal model is a
defensible extension; for *this* decision, the collapse is correct.

**First-encounter dedup — ~30% of rows spent to buy honest metrics.** Keeping only
`min(encounter_id)` per patient dropped ~29k of ~99k post-filter rows (down to
69,987). That is the single largest data sacrifice in the project, and it was a
deliberate data-vs-rigor trade. Had I kept repeat encounters, the same patient could
sit in both train and test and every metric would be inflated by patient-identity
leakage. Losing a third of the data hurts a model already starved for signal — but a
leaked 0.75 AUPRC is worth nothing and a real 0.21 is worth something. Rigor won, and
it should.

**Class weights over SMOTE — defensibility over a synthetic crutch.** At ~9%
prevalence the obvious move is resampling. I rejected SMOTE: it would have fabricated
~54k synthetic "patients" by interpolating between real clinical records — indefensible
in a regulated setting, and it hurt AUPRC in practice. Class weights
(`class_weight="balanced"`, `auto_class_weights="SqrtBalanced"`) handle the imbalance
without inventing data. The cost is inflated raw probabilities, which is exactly what
calibration is for — so the "cost" was already on the roadmap.

**CatBoost over the LR I'd hoped to ship — performance kept, explainability
recovered.** I wanted logistic regression to win: signed coefficients are explainable
for free. It didn't. Tuned CatBoost beats tuned LR by **+0.041 test AUPRC (+25%
relative)**, and on the metric that actually governs the product — recall at fixed
precision — it catches **17.1%** of true readmits at precision 0.30 versus LR's
**6.6%**, ~2.6× the actionable patients per unit of follow-up capacity. Tuning bought
LR nothing (0.1702 → 0.1646); it is saturated, a linear model at its representational
ceiling, and the gap is structural, not a tuning artifact. The cost — losing native
coefficients — is largely *recoverable*: TreeSHAP gives exact per-prediction
attributions, which the `/predict` API returns. So I didn't trade explainability for
accuracy; I moved explainability from coefficients to SHAP and kept LR as the
always-defensible baseline.

**Calibration: a tie broken toward the simpler method.** Isotonic and Platt were a
coin-flip on Brier (0.0791 vs 0.0790 on validation). I chose **Platt/sigmoid** — two
parameters, more robust on moderate data, less prone to overfitting calibration folds
than isotonic's flexible step function. Calibration cut test Brier ~21% (0.098 →
0.078) while leaving ranking provably untouched (AUPRC/ROC-AUC moved <0.002, within
noise). Choosing the simpler model on a tie is the conservative, defensible call.

**A recall-leaning threshold — a capacity cost taken on purpose.** I set the operating
cut at **0.091**, targeting recall ~0.50. This is a screening tool, and a missed
30-day readmission is far costlier than an extra follow-up call, so I lean toward
catching readmits. The price is precision **0.154** and a **~30% flag rate**: of every
~6.5 flagged patients, ~1 truly returns, and the team works through ~30% of all
discharges. The threshold is a stored, swappable tag, not baked into the model — a
documented dial-down to recall 0.40 / 22% flagged exists for tighter capacity.

## 2. What I'd change in production, or with more time

- **Harden the CI identity.** The pipeline authenticates to ECR with a static
  access-key pair on a managed ECR-PowerUser policy. In production I'd scope the IAM
  policy to the single repository ARN and move to **GitHub OIDC** — short-lived,
  keyless credentials with no long-term secret stored in GitHub at all.
- **Revisit the single global threshold per subgroup.** One 0.091 cutoff almost
  certainly does not land fairly across age / gender / race — a fixed threshold can
  mean very different recall and flag rates per group. The pending Fairlearn audit
  should drive either per-subgroup thresholds or an explicit, documented decision to
  accept one global cut.
- **Spend effort on data, not tuning.** Tuning is exhausted — it moved CatBoost
  +0.004 AUPRC. The ceiling moves with *information*, not hyperparameters: socioeconomic
  context, post-discharge follow-up, social determinants. That, plus the
  comorbidity-index and GenAI-assisted diagnosis-resolution feature experiments I
  deliberately deferred, is where real headroom is — not another Optuna sweep.
- **Richer monitoring before live traffic** — the Evidently drift report, prediction
  logging, and a concrete numeric retrain trigger (Stage 6) are scaffolded but not
  wired; I would not run this on live patients without them firing.

## 3. The model's limits — stated plainly

- **The ceiling is real, and it's the data, not the model.** Test ROC-AUC ~0.67 and
  AUPRC ~0.21 are the honest numbers, and they sit squarely in the known range for
  this dataset. Both LR and a tuned gradient-boosted ensemble converge there; the
  signal in these features is genuinely weak. This is a data ceiling, not a modelling
  failure — and recognising that is the point, not an excuse.
- **Most flagged patients will not return.** At the operating point precision is
  0.154 — ~5 of every 6 flags are false positives. That is acceptable *because* the
  intervention is cheap (a phone call, a medication review), but it would be the wrong
  tool to gate anything expensive or invasive.
- **SHAP magnitudes are on the uncalibrated scale.** The `top_factors` the API returns
  come from the base CatBoost learner's log-odds margin, before the Platt rescaling —
  so the **direction** of each factor is trustworthy, but the **magnitude** is not the
  contribution to the calibrated probability. Read them as "what pushed this patient
  up or down," not as exact probability deltas.
- **It reflects 1999–2008 US hospital practice.** The data predates modern diabetes
  drugs, coding practice, and discharge workflows. The model encodes how those
  hospitals behaved, and would need revalidation — ideally retraining — on
  current, local data before clinical use.
- **Fairness is not yet established.** No subgroup metrics have been computed. Until
  the Fairlearn audit reports recall/PR-AUC gaps across age, gender, and race, the
  honest statement is that this model's fairness is **unknown**, and it should not be
  deployed against any protected group on the assumption that one global threshold is
  equitable.
