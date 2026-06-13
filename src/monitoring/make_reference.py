"""make_reference.py — Stage 6 observability: build the DRIFT BASELINE snapshot.

Evidently compares a "current" batch against a fixed "reference". The reference here is
the model's own TRAINING feature distribution — the seed-42 TRAIN split of the exact
featurized data the model learned from (`src/models/train.py::prepare_data`). That is
the distribution the model is calibrated for; any production batch that drifts away from
it is what we want to catch.

The snapshot stores, per training row:
  * the 54 model-input features (categoricals as strings, exactly as served),
  * `prediction` — the calibrated @staging score for that row (enables PREDICTION drift),
  * `target` — kept for reference/target-drift (NOTE: labels do NOT exist at serving
    time; target drift is only computable in this offline simulation).

Output: `data/monitoring/reference.parquet`, DVC-tracked (it is a data artifact, not code).

Run:
    uv run python src/monitoring/make_reference.py
    dvc add data/monitoring/reference.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "models"))
import models.train as T  # noqa: E402

FEATURES = "data/featurized/diabetes_features.parquet"
OUT = Path("data/monitoring/reference.parquet")
MODEL_NAME = "readmission-catboost-calibrated"
MODEL_ALIAS = "staging"


def load_model():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The SAME seed-42 train split the model trained on — this is the baseline.
    X_train, _, y_train, _, cat_cols, _, _ = T.prepare_data(FEATURES)
    model = load_model()
    feature_names = list(model.calibrated_classifiers_[0].estimator.model_.feature_names_)

    # Frame exactly like serving (cats as strings) and order by the model's features.
    Xtr = T.as_catboost_frame(X_train, cat_cols).reindex(columns=feature_names).reset_index(drop=True)
    ref = Xtr.copy()
    ref["prediction"] = model.predict_proba(Xtr)[:, 1]
    ref["target"] = y_train.reset_index(drop=True).astype(int).to_numpy()

    ref.to_parquet(OUT, index=False)
    print(f"[reference] wrote {OUT}  rows={len(ref):,}  cols={ref.shape[1]} "
          f"({len(feature_names)} features + prediction + target)")
    print(f"[reference] mean calibrated score (baseline)={ref['prediction'].mean():.4f}  "
          f"positive rate={ref['target'].mean():.4f}")
    print("[next] dvc add data/monitoring/reference.parquet")


if __name__ == "__main__":
    main()
