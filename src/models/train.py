"""train.py — Stage 4 modeling: LR baseline + CatBoost, CV-evaluated, MLflow-logged.

Scope of THIS gate (locked by docs/GOALS.md Stage 4 and the task brief):
  * Two models only — LogisticRegression baseline, then CatBoost. NO Optuna.
  * Imbalance via class weights (no SMOTE — fabricates clinical records).
  * Plain StratifiedKFold CV. NO grouped CV: first-encounter dedup upstream makes
    every patient_nbr unique, so a patient can never straddle train and test.
  * A stratified ~20% held-out test set carved ONCE, scored ONCE, at the very end.
  * Log everything to MLflow. Do NOT calibrate, tune, choose a threshold, or
    register a model yet — those are later gates.

Leakage tripwire: if held-out test ROC-AUC > 0.75, something leaked. We STOP and
investigate rather than celebrate (healthy range here is ~0.66-0.70).

Run:
    uv run python src/models/train.py
    # optional: point at a running MLflow server instead of the local sqlite store
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python src/models/train.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import the metric/plot helpers as a sibling module. Running this file puts
# src/models/ on sys.path[0]; the insert also supports `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — every choice here is fixed and documented, not searched.
# ---------------------------------------------------------------------------
KEY_COLS = ["encounter_id", "patient_nbr"]  # identifiers / audit keys, NEVER features
TARGET_COL = "target"
TEST_SIZE = 0.20
SEED = 42
N_SPLITS = 5
TRIPWIRE_ROC_AUC = 0.75  # test ROC-AUC above this => suspect leakage, stop

# A sane fixed CatBoost config (NOT tuned — Optuna is the next gate).
CATBOOST_PARAMS = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    auto_class_weights="Balanced",  # the imbalance lever (mirror of class_weight)
    eval_metric="PRAUC",
    loss_function="Logloss",
    verbose=False,
    allow_writing_files=False,  # don't litter the repo with catboost_info/
)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_xy(path: str):
    """Load featurized parquet -> (X, y, cat_cols, num_cols). Drops identifiers."""
    df = pd.read_parquet(path)
    X = df.drop(columns=KEY_COLS + [TARGET_COL])
    y = df[TARGET_COL].astype(int)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols


def drop_zero_variance(X_train, X_test, cat_cols, num_cols):
    """Drop columns constant on the TRAIN split (e.g. glimepiride-pioglitazone).

    Decided on train only — a column carrying no information there carries none
    for the model. Same columns are removed from test. Returns updated frames and
    column lists plus the dropped names for logging.
    """
    dropped = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    if dropped:
        X_train = X_train.drop(columns=dropped)
        X_test = X_test.drop(columns=dropped)
    cat_cols = [c for c in cat_cols if c not in dropped]
    num_cols = [c for c in num_cols if c not in dropped]
    return X_train, X_test, cat_cols, num_cols, dropped


def build_lr(cat_cols, num_cols) -> Pipeline:
    """LR baseline: one-hot the categoricals, standardize the numerics, then a
    class-balanced logistic regression. The ColumnTransformer is INSIDE the
    pipeline so it refits per CV fold — no preprocessing leakage across folds."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_catboost(cat_cols) -> CatBoostClassifier:
    """CatBoost consumes categoricals natively (no one-hot). cat_features is set
    by NAME; train.py hands it string-typed categorical columns."""
    return CatBoostClassifier(cat_features=cat_cols, **CATBOOST_PARAMS)


def as_catboost_frame(X, cat_cols):
    """CatBoost wants categorical columns as plain strings (not pandas category)."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").astype("object")
    return X


# ---------------------------------------------------------------------------
# Train / evaluate one model, logging a single MLflow run
# ---------------------------------------------------------------------------

def oof_predict(make_estimator, X, y, skf):
    """Manual out-of-fold predicted probabilities.

    A fresh estimator is built per fold via the factory and fit only on that
    fold's training rows — so preprocessing (LR) and category handling (CatBoost)
    never see the held-out rows. We roll our own instead of cross_val_predict
    because sklearn's clone() rejects a CatBoostClassifier that carries
    cat_features in its constructor.
    """
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        est = make_estimator()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = est.predict_proba(X.iloc[va_idx])[:, 1]
    return oof


def run_model(name, make_estimator, X_train, y_train, X_test, y_test, params, skf):
    """CV (out-of-fold) + fit + single held-out test score, all logged to MLflow.

    `make_estimator` is a zero-arg factory returning a fresh, unfit estimator.
    Returns a results dict for the cross-model comparison table.
    """
    print(f"\n──────── {name} ────────")
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model", name)
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", N_SPLITS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("recall_at_precision_target", ev.TARGET_PRECISION)
        mlflow.log_param("n_features", X_train.shape[1])

        no_skill = float(y_train.mean())
        mlflow.log_metric("no_skill_auprc", no_skill)

        # ---- Cross-validation: honest out-of-fold probabilities on TRAIN only.
        print("  cross-validating (out-of-fold predictions)...")
        oof = oof_predict(make_estimator, X_train, y_train, skf)
        cv = ev.compute_metrics(y_train.to_numpy(), oof)
        for k, v in cv.items():
            mlflow.log_metric(f"cv_{k}", v)
        print(f"  CV   AUPRC={cv['auprc']:.4f}  ROC-AUC={cv['roc_auc']:.4f}  "
              f"recall@p{ev.TARGET_PRECISION:.2f}={cv['recall_at_precision']:.4f}  "
              f"Brier={cv['brier']:.4f}")

        # ---- Fit on the FULL train split, then touch the test set exactly ONCE.
        print("  fitting on full train, scoring held-out test (once)...")
        estimator = make_estimator()
        estimator.fit(X_train, y_train)
        test_prob = estimator.predict_proba(X_test)[:, 1]
        test = ev.compute_metrics(y_test, test_prob)
        for k, v in test.items():
            mlflow.log_metric(f"test_{k}", v)
        print(f"  TEST AUPRC={test['auprc']:.4f}  ROC-AUC={test['roc_auc']:.4f}  "
              f"recall@p{ev.TARGET_PRECISION:.2f}={test['recall_at_precision']:.4f}  "
              f"Brier={test['brier']:.4f}")

        # ---- Figures (built from the final held-out test predictions).
        mlflow.log_figure(ev.pr_curve_fig(y_test, test_prob, f"{name} — PR (test)"),
                          "pr_curve_test.png")
        mlflow.log_figure(ev.roc_curve_fig(y_test, test_prob, f"{name} — ROC (test)"),
                          "roc_curve_test.png")
        mlflow.log_figure(ev.calibration_fig(y_test, test_prob, f"{name} — calibration (test)"),
                          "calibration_test.png")
        mlflow.log_figure(ev.confusion_fig(y_test, test_prob, f"{name} — confusion (test)"),
                          "confusion_test.png")

        # ---- Leakage tripwire.
        tripwire_ok = test["roc_auc"] <= TRIPWIRE_ROC_AUC
        mlflow.set_tag("leakage_tripwire_ok", str(tripwire_ok))
        if not tripwire_ok:
            print(f"  !! TRIPWIRE: test ROC-AUC {test['roc_auc']:.4f} > {TRIPWIRE_ROC_AUC} "
                  "— suspected leakage.")

    return {"name": name, "cv": cv, "test": test, "no_skill": no_skill,
            "tripwire_ok": tripwire_ok}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(features_path: str, experiment: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] tracking_uri={tracking_uri}  experiment={experiment!r}")

    X, y, cat_cols, num_cols = load_xy(features_path)
    print(f"[data] {X.shape[0]:,} rows x {X.shape[1]} feature cols  "
          f"(prevalence={y.mean():.4f}); {len(cat_cols)} categorical, {len(num_cols)} numeric")

    # Held-out test set: carved ONCE, before any CV/fitting, stratified on target.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_test, cat_cols, num_cols, dropped = drop_zero_variance(
        X_train, X_test, cat_cols, num_cols
    )
    print(f"[split] train={len(X_train):,}  test={len(X_test):,}  "
          f"dropped zero-variance: {dropped or 'none'}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = []

    # 1) Baseline — the bar CatBoost must clear to earn its complexity.
    lr_params = {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
                 "preprocessing": "onehot+standardize", "dropped_zero_variance": str(dropped)}
    results.append(run_model(
        "logreg_baseline", lambda: build_lr(cat_cols, num_cols),
        X_train, y_train, X_test, y_test, lr_params, skf,
    ))

    # 2) CatBoost — native categoricals, class-balanced, fixed (untuned) config.
    cb_params = {**CATBOOST_PARAMS, "dropped_zero_variance": str(dropped)}
    Xtr_cb, Xte_cb = as_catboost_frame(X_train, cat_cols), as_catboost_frame(X_test, cat_cols)
    results.append(run_model(
        "catboost", lambda: build_catboost(cat_cols),
        Xtr_cb, y_train, Xte_cb, y_test, cb_params, skf,
    ))

    _print_comparison(results)

    if not all(r["tripwire_ok"] for r in results):
        print("\nSTOP: leakage tripwire fired on at least one model — investigate "
              "before proceeding to tuning/calibration.")
        sys.exit(1)


def _print_comparison(results) -> None:
    print("\n================ Stage-4 comparison (gate) ================")
    print(f"no-skill AUPRC (prevalence) ≈ {results[0]['no_skill']:.4f}\n")
    hdr = f"{'model':<18} {'CV AUPRC':>9} {'CV ROC':>8} {'test AUPRC':>11} {'test ROC':>9} {'tripwire':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<18} {r['cv']['auprc']:>9.4f} {r['cv']['roc_auc']:>8.4f} "
              f"{r['test']['auprc']:>11.4f} {r['test']['roc_auc']:>9.4f} "
              f"{'OK' if r['tripwire_ok'] else 'FIRED':>9}")
    print("===========================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/featurized/diabetes_features.parquet")
    ap.add_argument("--experiment", default="readmission-30d")
    args = ap.parse_args()
    main(args.features, args.experiment)
