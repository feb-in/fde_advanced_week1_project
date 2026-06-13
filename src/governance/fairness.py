"""fairness.py — Stage 7 governance: subgroup fairness audit of the lead model.

WHAT THIS DOES (measure, do NOT mitigate):
  * Loads the SAME held-out test set (seed 42, shared T.prepare_data) the model was
    evaluated on — scored exactly once here, never used in training/calibration.
  * Loads the lead model — calibrated CatBoost v1 @ `staging` — from the MLflow
    registry, and reads its operating threshold from the model-version tag (0.091…),
    so the audit reflects exactly what serving applies.
  * Computes Fairlearn MetricFrames across age, gender, race: per-subgroup recall,
    precision, selection (flag) rate, false-positive rate, AUPRC, support, prevalence.
  * Surfaces the disparities (demographic-parity difference, equalized-odds
    difference, recall gap) and answers the open question from docs/REFLECTION.md:
    does ONE global threshold land differently across groups?

It only MEASURES. No reweighting, no per-group thresholds, no thresholding tricks.

Run:
    uv run python src/governance/fairness.py
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python src/governance/fairness.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# src/ on path so we can reuse the exact training split + catboost framing, and so
# the pickled CatBoostWrapper resolves when the registry model is unpickled.
_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "models"))
import models.train as T  # noqa: E402

from fairlearn.metrics import (  # noqa: E402
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_score,
    roc_auc_score,
)

MODEL_NAME = "readmission-catboost-calibrated"
MODEL_ALIAS = "staging"
FEATURES = "data/featurized/diabetes_features.parquet"
MIN_SUPPORT = 100  # subgroups smaller than this are flagged as low-confidence


# ---------------------------------------------------------------------------
# Load model + threshold from the registry (single source of truth)
# ---------------------------------------------------------------------------
def load_lead_model():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    thr = float(mv.tags["operating_threshold"])
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    print(f"[model] {MODEL_NAME} v{mv.version} @ {MODEL_ALIAS}  threshold={thr:.6f}  "
          f"calibration={mv.tags.get('calibration_method', '?')}")
    return model, thr, mv.version


# ---------------------------------------------------------------------------
# Sensitive features for the test rows (positional alignment with the scores)
# ---------------------------------------------------------------------------
def age_band_10yr(midpoint: int) -> str:
    """Recover the original 10-year band from age_midpoint (65 -> '[60-70)')."""
    low = int(midpoint) - 5
    return f"[{low}-{low + 10})"


def sensitive_frame(X_test: pd.DataFrame) -> pd.DataFrame:
    """Build the demographic slicing columns in the SAME row order as X_test."""
    return pd.DataFrame(
        {
            "age (10yr)": X_test["age_midpoint"].map(age_band_10yr).to_numpy(),
            "age (coarse)": X_test["age_bucket"].astype(str).to_numpy(),
            "gender": X_test["gender"].astype(str).to_numpy(),
            "race": X_test["race"].astype(str).to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# Safe group metrics (a subgroup may have one class only)
# ---------------------------------------------------------------------------
def _precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def _auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return average_precision_score(y_true, y_score)


def _roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return roc_auc_score(y_true, y_score)


def _base_rate(y_true, y_pred):
    return float(np.mean(y_true))


def audit_attribute(name, y_true, y_pred, y_score, sf_col):
    """Return a per-subgroup metrics DataFrame + a disparities dict for one attribute."""
    mf = MetricFrame(
        metrics={
            "support": count,
            "prevalence": _base_rate,
            "recall": true_positive_rate,
            "precision": _precision,
            "flag_rate": selection_rate,
            "FPR": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf_col,
    )
    # AUPRC / ROC-AUC need scores, not the thresholded prediction.
    mf_score = MetricFrame(
        metrics={"AUPRC": _auprc, "ROC_AUC": _roc_auc},
        y_true=y_true,
        y_pred=y_score,
        sensitive_features=sf_col,
    )
    table = mf.by_group.join(mf_score.by_group)
    table.index.name = name

    disparities = {
        "demographic_parity_diff": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sf_col
        ),
        "equalized_odds_diff": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sf_col
        ),
        "recall_gap": float(mf.by_group["recall"].max() - mf.by_group["recall"].min()),
        "flag_rate_gap": float(
            mf.by_group["flag_rate"].max() - mf.by_group["flag_rate"].min()
        ),
        "overall_recall": float(mf.overall["recall"]),
        "overall_flag_rate": float(mf.overall["flag_rate"]),
    }
    return table, disparities


# ---------------------------------------------------------------------------
def _fmt(table: pd.DataFrame) -> str:
    t = table.copy()
    t["support"] = t["support"].astype(int)
    for c in ("prevalence", "recall", "precision", "flag_rate", "FPR", "AUPRC", "ROC_AUC"):
        if c in t:
            t[c] = t[c].map(lambda v: "  n/a" if pd.isna(v) else f"{v:.3f}")
    return t.to_string()


def main():
    print("=" * 78)
    print("FAIRNESS AUDIT — calibrated CatBoost @ staging, single global threshold")
    print("=" * 78)

    model, thr, version = load_lead_model()

    X_train, X_test, y_train, y_test, cat_cols, num_cols, dropped = T.prepare_data(FEATURES)
    Xte = T.as_catboost_frame(X_test, cat_cols)
    y = y_test.to_numpy().astype(int)
    scores = model.predict_proba(Xte)[:, 1]
    y_pred = (scores >= thr).astype(int)

    sf = sensitive_frame(X_test)
    overall_recall = (y_pred[y == 1].sum()) / max(int(y.sum()), 1)
    print(f"[data] test n={len(y):,}  positives={int(y.sum()):,}  "
          f"prevalence={y.mean():.4f}")
    print(f"[overall @0.091] recall={overall_recall:.3f}  "
          f"flag_rate={y_pred.mean():.3f}\n")

    for attr in ["age (10yr)", "age (coarse)", "gender", "race"]:
        table, disp = audit_attribute(attr, y, y_pred, scores, sf[attr])
        print("-" * 78)
        print(f"### {attr}")
        print(_fmt(table))
        small = table.index[table["support"] < MIN_SUPPORT].tolist()
        print(f"\n  demographic-parity diff (max flag-rate gap): {disp['demographic_parity_diff']:.3f}")
        print(f"  equalized-odds diff (max TPR/FPR gap):        {disp['equalized_odds_diff']:.3f}")
        print(f"  recall gap (max-min):                          {disp['recall_gap']:.3f}")
        print(f"  flag-rate gap (max-min):                       {disp['flag_rate_gap']:.3f}")
        if small:
            print(f"  low-support (<{MIN_SUPPORT}, interpret with caution): {small}")
        print()


if __name__ == "__main__":
    main()
