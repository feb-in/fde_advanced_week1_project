"""tune.py — Stage 4 (cont.): Optuna hyperparameter search for LR and CatBoost.

We tune BOTH models side by side — LR is NOT dropped; it gets a real chance to
close the gap now that the baseline ran it at defaults only.

Discipline carried over from the baseline gate (src/models/train.py):
  * The held-out test set and the train/fold split come from prepare_data(), the
    single shared definition — SAME seed, SAME 20% stratified test set. The test
    set is touched exactly ONCE per model, at the very end, NEVER inside the search.
  * Search uses plain StratifiedKFold(k=5) on the TRAIN portion only.
  * Objective = MEAN CV AUPRC across the 5 folds (average precision). This is the
    chased metric. ROC-AUC is computed and logged every trial but is NOT the
    objective or the selection criterion — at ~9% positives AUPRC is honest and
    ROC-AUC flatters.
  * Imbalance via class weights / auto_class_weights. NO SMOTE.
  * Leakage tripwire stays live: final held-out test ROC-AUC > 0.75 => STOP.

Search stack: Optuna (TPE), seeded. NO Ray. Every trial is a nested MLflow run
under a per-model parent run, tagged with the model name. After the search each
model is refit on the full train portion and scored once on the held-out test.
We do NOT calibrate, choose an operating threshold, or register a model here.

Run:
    uv run python src/models/tune.py                      # default 50 LR / 40 CB
    uv run python src/models/tune.py --lr-trials 60 --cb-trials 50
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import train as T  # noqa: E402  — reuse prepare_data, builders, constants, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Per-fold CV scoring — objective is the MEAN of per-fold AUPRC.
# ---------------------------------------------------------------------------

def cv_scores(make_estimator, X, y, skf):
    """Return (mean_auprc, mean_roc_auc) averaged over the 5 folds.

    A fresh estimator per fold, fit on that fold's train rows only — preprocessing
    (LR ColumnTransformer) and category handling (CatBoost) never see the held-out
    fold. We average per-fold AP (not pooled-OOF AP) to match the stated objective.
    """
    aps, rocs = [], []
    for tr_idx, va_idx in skf.split(X, y):
        est = make_estimator()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = est.predict_proba(X.iloc[va_idx])[:, 1]
        aps.append(average_precision_score(y.iloc[va_idx], p))
        rocs.append(roc_auc_score(y.iloc[va_idx], p))
    return float(np.mean(aps)), float(np.mean(rocs))


# ---------------------------------------------------------------------------
# Estimator builders from a sampled param dict
# ---------------------------------------------------------------------------

# Map the searched penalty family to the l1_ratio knob (sklearn ≥1.8 API: the
# `penalty` arg is deprecated; l1_ratio expresses the whole l2↔elasticnet↔l1 span).
_PENALTY_TO_L1_RATIO = {"l2": 0.0, "l1": 1.0}  # elasticnet uses the sampled float


def build_lr_tuned(cat_cols, num_cols, p):
    """LR pipeline (one-hot + scale INSIDE, so encoding refits per fold) with
    searched hyperparameters. saga covers l1 / l2 / elasticnet via l1_ratio."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    l1_ratio = _PENALTY_TO_L1_RATIO.get(p["penalty"], p.get("l1_ratio"))
    clf = LogisticRegression(
        C=p["C"],
        l1_ratio=l1_ratio,
        class_weight=p["class_weight"],
        solver="saga",
        max_iter=1000,   # ranking metric (AUPRC) tolerates loose convergence
        tol=1e-2,
        random_state=T.SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_catboost_tuned(cat_cols, p):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        cat_features=cat_cols,
        iterations=p["iterations"],
        depth=p["depth"],
        learning_rate=p["learning_rate"],
        l2_leaf_reg=p["l2_leaf_reg"],
        random_strength=p["random_strength"],
        bagging_temperature=p["bagging_temperature"],
        border_count=p["border_count"],
        auto_class_weights=p["auto_class_weights"],
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=T.SEED,
        verbose=False,
        allow_writing_files=False,
    )


# ---------------------------------------------------------------------------
# Param samplers
# ---------------------------------------------------------------------------

def sample_lr(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    p = {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "penalty": penalty,
        # class_weight searched as required; "balanced" is the imbalance lever,
        # None is the unweighted control. (No SMOTE — resampling is off the table.)
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    if penalty == "elasticnet":
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return p


def sample_catboost(trial, iterations):
    return {
        "iterations": iterations,  # fixed budget (not searched), for fair trials
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        # both are class-weighting strategies — imbalance stays handled either way.
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced"]
        ),
    }


# ---------------------------------------------------------------------------
# Tune one model: parent run + nested trial runs + refit + single test score
# ---------------------------------------------------------------------------

def tune_model(name, sample_fn, build_fn, X_train, y_train, X_test, y_test, skf,
               n_trials, baseline_test_auprc):
    """Run the Optuna study for one model and evaluate the winner once on test."""
    print(f"\n════════ tuning {name}  ({n_trials} trials) ════════")
    with mlflow.start_run(run_name=f"{name}_tuning") as parent:
        mlflow.set_tag("model", name)
        mlflow.set_tag("phase", "tuning")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", T.N_SPLITS)
        mlflow.log_param("seed", T.SEED)
        mlflow.log_param("test_size", T.TEST_SIZE)
        mlflow.log_param("objective", "mean_cv_auprc")

        def objective(trial):
            p = sample_fn(trial)
            with mlflow.start_run(nested=True, run_name=f"{name}_trial_{trial.number}"):
                mlflow.set_tag("model", name)
                mlflow.set_tag("phase", "trial")
                mlflow.log_params(p)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    ap, roc = cv_scores(lambda: build_fn(p), X_train, y_train, skf)
                mlflow.log_metric("cv_auprc", ap)
                mlflow.log_metric("cv_roc_auc", roc)
            trial.set_user_attr("cv_roc_auc", roc)
            return ap

        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=T.SEED)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_cv_auprc = study.best_value
        best_cv_roc = study.best_trial.user_attrs.get("cv_roc_auc")
        print(f"  best CV AUPRC={best_cv_auprc:.4f}  (CV ROC-AUC={best_cv_roc:.4f})")
        print(f"  best params: {best_params}")

        # Rebuild the FULL param dict the builder expects (samplers add fixed keys).
        full_best = _full_params(name, best_params)
        mlflow.log_params({f"best_{k}": v for k, v in full_best.items()})
        mlflow.log_metric("best_cv_auprc", best_cv_auprc)
        mlflow.log_metric("best_cv_roc_auc", best_cv_roc)

        # ---- Refit best config on full train; touch the held-out test ONCE.
        print("  refitting best config on full train, scoring held-out test (once)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est = build_fn(full_best)
            est.fit(X_train, y_train)
        test_prob = est.predict_proba(X_test)[:, 1]
        test = ev.compute_metrics(y_test, test_prob)
        for k, v in test.items():
            mlflow.log_metric(f"test_{k}", v)
        print(f"  TEST AUPRC={test['auprc']:.4f}  ROC-AUC={test['roc_auc']:.4f}  "
              f"recall@p{ev.TARGET_PRECISION:.2f}={test['recall_at_precision']:.4f}  "
              f"Brier={test['brier']:.4f}")

        mlflow.log_figure(ev.pr_curve_fig(y_test, test_prob, f"{name} (tuned) — PR (test)"),
                          "pr_curve_test.png")
        mlflow.log_figure(ev.roc_curve_fig(y_test, test_prob, f"{name} (tuned) — ROC (test)"),
                          "roc_curve_test.png")

        tripwire_ok = test["roc_auc"] <= T.TRIPWIRE_ROC_AUC
        mlflow.set_tag("leakage_tripwire_ok", str(tripwire_ok))
        if not tripwire_ok:
            print(f"  !! TRIPWIRE: test ROC-AUC {test['roc_auc']:.4f} > {T.TRIPWIRE_ROC_AUC}")

        delta = test["auprc"] - baseline_test_auprc
        print(f"  untuned→tuned test AUPRC: {baseline_test_auprc:.4f} → {test['auprc']:.4f} "
              f"({delta:+.4f})")

    return {
        "name": name, "best_params": full_best,
        "cv_auprc": best_cv_auprc, "cv_roc_auc": best_cv_roc,
        "test": test, "tripwire_ok": tripwire_ok,
        "baseline_test_auprc": baseline_test_auprc, "delta_test_auprc": delta,
    }


def _full_params(name, best_params):
    """Optuna's best_params omits keys the sampler fixed/conditional. Restore them
    so the builder gets a complete dict."""
    p = dict(best_params)
    if name == "catboost":
        p.setdefault("iterations", CB_ITERATIONS)
    if name.startswith("logreg"):
        p.setdefault("l1_ratio", None)  # absent unless penalty was elasticnet
    return p


# Baseline (untuned) held-out test AUPRC from the prior gate, for the delta.
# Source: docs/MODEL_COMPARISON.md (logreg 0.1702, catboost 0.2015).
BASELINE_TEST_AUPRC = {"logreg": 0.1702, "catboost": 0.2015}
CB_ITERATIONS = 500


def main(features_path, experiment, lr_trials, cb_trials):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] tracking_uri={tracking_uri}  experiment={experiment!r}")

    # SAME split as the baseline — identical seed/test set (prepare_data is shared).
    X_train, X_test, y_train, y_test, cat_cols, num_cols, dropped = T.prepare_data(features_path)
    print(f"[split] train={len(X_train):,}  test={len(X_test):,}  "
          f"test_pos_rate={y_test.mean():.4f}  dropped={dropped or 'none'}")

    skf = StratifiedKFold(n_splits=T.N_SPLITS, shuffle=True, random_state=T.SEED)

    results = []

    # ---- LR (one-hot/scaled). Real chance to close the gap.
    results.append(tune_model(
        "logreg", sample_lr,
        lambda p: build_lr_tuned(cat_cols, num_cols, p),
        X_train, y_train, X_test, y_test, skf, lr_trials,
        BASELINE_TEST_AUPRC["logreg"],
    ))

    # ---- CatBoost (native categoricals). String-typed cat cols.
    Xtr_cb = T.as_catboost_frame(X_train, cat_cols)
    Xte_cb = T.as_catboost_frame(X_test, cat_cols)
    results.append(tune_model(
        "catboost",
        lambda trial: sample_catboost(trial, CB_ITERATIONS),
        lambda p: build_catboost_tuned(cat_cols, p),
        Xtr_cb, y_train, Xte_cb, y_test, skf, cb_trials,
        BASELINE_TEST_AUPRC["catboost"],
    ))

    _print_summary(results)

    if not all(r["tripwire_ok"] for r in results):
        print("\nSTOP: leakage tripwire fired — investigate before proceeding.")
        sys.exit(1)


def _print_summary(results):
    print("\n========= Stage-4 tuned comparison (gate) =========")
    hdr = (f"{'model':<10} {'CV AUPRC':>9} {'CV ROC':>8} {'test AUPRC':>11} "
           f"{'test ROC':>9} {'rec@P.30':>9} {'ΔAUPRC':>8} {'trip':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<10} {r['cv_auprc']:>9.4f} {r['cv_roc_auc']:>8.4f} "
              f"{r['test']['auprc']:>11.4f} {r['test']['roc_auc']:>9.4f} "
              f"{r['test']['recall_at_precision']:>9.4f} {r['delta_test_auprc']:>+8.4f} "
              f"{'OK' if r['tripwire_ok'] else 'FIRE':>5}")
    print("===================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/featurized/diabetes_features.parquet")
    ap.add_argument("--experiment", default="readmission-30d")
    ap.add_argument("--lr-trials", type=int, default=50)
    ap.add_argument("--cb-trials", type=int, default=40)
    args = ap.parse_args()
    main(args.features, args.experiment, args.lr_trials, args.cb_trials)
