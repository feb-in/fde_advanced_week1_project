"""explain.py — Stage 7 governance: GLOBAL SHAP explainability for the lead model.

Local (per-patient) SHAP already ships in the API: src/app/model.py builds ONE
shap.TreeExplainer on the lead base learner and returns the top signed factors with
every /predict. This script produces the GLOBAL view — which features drive the model
across the whole test set — and logs it to MLflow.

CAVEAT (stated everywhere these numbers appear): SHAP values are computed on the base
CatBoost learner's **log-odds margin**, BEFORE the sigmoid calibration map. So the
**direction** of every factor is valid (what pushes risk up vs down), but the
**magnitude** is on the uncalibrated scale — it is not a delta in the calibrated
probability. Identical caveat to the API's local factors, by construction: the global
plot and the served local factors use the SAME base learner.

Run:
    uv run python src/governance/explain.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "models"))
import models.train as T  # noqa: E402

MODEL_NAME = "readmission-catboost-calibrated"
MODEL_ALIAS = "staging"
FEATURES = "data/featurized/diabetes_features.parquet"
OUT_DIR = Path("reports/governance")
TOP_N = 20


def load_lead_base_learner():
    """Load v1 @ staging and return (calibrated_model, wrapper, base CatBoost, version).

    The wrapper is calibrated_classifiers_[0]'s estimator — EXACTLY what the serving
    Predictor uses (src/app/model.py); its .model_ is the trained CatBoost the API's
    TreeExplainer is built on, and its .cat_features names the categorical columns. So
    the global plot and the API's local factors explain the same learner."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    cc0 = model.calibrated_classifiers_[0]
    wrapper = getattr(cc0, "estimator", None) or cc0.base_estimator
    return model, wrapper, wrapper.model_, mv.version


def main():
    import shap

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, wrapper, cb, version = load_lead_base_learner()
    feature_names = list(cb.feature_names_)
    cat_features = list(wrapper.cat_features) if wrapper.cat_features else []
    print(f"[model] {MODEL_NAME} v{version} @ {MODEL_ALIAS} — "
          f"{len(feature_names)} features ({len(cat_features)} categorical)")

    # Same seed-42 test split, CatBoost-framed (cats as strings) exactly like serving.
    _, X_test, _, _, cat_cols, _, _ = T.prepare_data(FEATURES)
    Xte = T.as_catboost_frame(X_test, cat_cols).reindex(columns=feature_names)
    print(f"[data] explaining {len(Xte):,} held-out test rows")

    # SHAP on the base learner's margin (pre-calibration log-odds).
    explainer = shap.TreeExplainer(cb)
    sv = np.asarray(explainer.shap_values(Xte))
    if sv.ndim == 3:           # (n, features, classes) → positive class
        sv = sv[:, :, -1]

    # ---- Global importance = mean |SHAP| per feature (ranking is unambiguous).
    mean_abs = np.abs(sv).mean(axis=0)
    rank = (pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    rank_path = OUT_DIR / "shap_global_importance.csv"
    rank.to_csv(rank_path, index=False)

    print("\n  rank  feature                         mean|SHAP| (log-odds)")
    for i, row in rank.head(15).iterrows():
        print(f"  {i + 1:>4}  {row['feature']:<30}  {row['mean_abs_shap']:.4f}")

    # ---- Bar chart (top-N mean |SHAP|).
    top = rank.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b6ea5")
    ax.set_xlabel("mean |SHAP|  (log-odds margin, pre-calibration)")
    ax.set_title(f"Global feature importance — CatBoost v{version} @ staging\n"
                 f"(top {TOP_N}; magnitudes uncalibrated, directions valid)")
    fig.tight_layout()
    bar_path = OUT_DIR / "shap_global_importance.png"
    fig.savefig(bar_path, dpi=110)
    plt.close(fig)

    # ---- Beeswarm summary. Categoricals factorized to codes so the colour gradient
    #      renders; numeric features keep their real values.
    X_disp = Xte.copy()
    for c in X_disp.columns:
        if c in cat_features or X_disp[c].dtype == object:
            X_disp[c] = pd.factorize(X_disp[c])[0]
    X_disp = X_disp.astype(float)
    plt.figure()
    shap.summary_plot(sv, X_disp, feature_names=feature_names, max_display=TOP_N,
                      show=False)
    plt.title("SHAP summary (test) — pre-calibration log-odds; categoricals code-coloured")
    plt.tight_layout()
    bee_path = OUT_DIR / "shap_summary_global.png"
    plt.savefig(bee_path, dpi=110)
    plt.close()

    # ---- Log to MLflow under a governance run.
    import mlflow
    mlflow.set_experiment("readmission-30d")
    with mlflow.start_run(run_name="shap_global"):
        mlflow.set_tag("phase", "governance")
        mlflow.set_tag("model", "catboost")
        mlflow.set_tag("explained_model_version", str(version))
        mlflow.log_param("shap_scale", "base_learner_log_odds_pre_calibration")
        mlflow.log_param("n_rows_explained", len(Xte))
        for i, row in rank.head(TOP_N).iterrows():
            mlflow.log_metric(f"mean_abs_shap__{row['feature']}", float(row["mean_abs_shap"]))
        mlflow.log_artifact(str(bar_path), artifact_path="shap")
        mlflow.log_artifact(str(bee_path), artifact_path="shap")
        mlflow.log_artifact(str(rank_path), artifact_path="shap")

    print(f"\n[ok] global SHAP logged to MLflow (run 'shap_global') + saved under {OUT_DIR}/")
    print(f"     top driver: {rank.iloc[0]['feature']}  |  caveat: magnitudes pre-calibration")


if __name__ == "__main__":
    main()
