"""model.py — load the registered calibrated model and score one record.

Model + threshold travel together from the MLflow Model Registry:
  * the model is loaded BY ALIAS — models:/<name>@staging — never a pickle path, so
    a registry promotion/rollback changes what serving uses with no code change;
  * the operating threshold is read from the model-version TAG, not hardcoded, so it
    can never drift from the model it was chosen for.

A single shap.TreeExplainer is built ONCE at startup (on the lead base learner of
the calibration ensemble) and reused per request for the top contributing factors.
Scoring reuses src/app/featurize.py → the exact training feature path (no skew).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))           # repo/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))  # 'wrappers' unpickle

from app.featurize import featurize_record  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "readmission-catboost-calibrated")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "staging")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
TOP_K_FACTORS = int(os.environ.get("TOP_K_FACTORS", "6"))


class Predictor:
    """Holds the loaded model, threshold, feature contract, and SHAP explainer."""

    def __init__(self):
        import mlflow
        import shap

        mlflow.set_tracking_uri(TRACKING_URI)
        self.model_name = MODEL_NAME
        self.model_alias = MODEL_ALIAS
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.version = mv.version
        self.threshold = float(mv.tags["operating_threshold"])
        self.calibration_method = mv.tags.get("calibration_method", "unknown")

        self.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # The model defines its own inputs — feature order + which are categorical.
        base = getattr(self.model.calibrated_classifiers_[0], "estimator", None) \
            or self.model.calibrated_classifiers_[0].base_estimator
        self._cb = base.model_
        self.feature_names = list(self._cb.feature_names_)
        self.cat_features = list(base.cat_features)
        self.explainer = shap.TreeExplainer(self._cb)

    def _to_model_frame(self, record: dict) -> pd.DataFrame:
        """Raw record → exact model feature row (engineered, reindexed, cats as str)."""
        X = featurize_record(record).reindex(columns=self.feature_names)
        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            raise ValueError(f"featurization produced nulls in {bad} — unscoreable input")
        for c in self.cat_features:
            X[c] = X[c].astype("string").astype("object")
        return X

    def predict(self, record: dict) -> dict:
        X = self._to_model_frame(record)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        shap_row = np.asarray(self.explainer.shap_values(X)).reshape(len(self.feature_names))

        order = np.argsort(np.abs(shap_row))[::-1][:TOP_K_FACTORS]
        factors = [
            {
                "feature": self.feature_names[i],
                "value": _native(X.iloc[0][self.feature_names[i]]),
                "contribution": round(float(shap_row[i]), 4),
                "direction": "increases" if shap_row[i] > 0 else "decreases",
            }
            for i in order
        ]
        return {
            "readmission_probability": round(prob, 6),
            "flag": bool(prob >= self.threshold),
            "threshold": self.threshold,
            "model_name": MODEL_NAME,
            "model_version": str(self.version),
            "model_alias": MODEL_ALIAS,
            "top_factors": factors,
        }


def _native(v):
    """JSON-friendly scalar (numpy/pandas → python)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v)


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    """Lazy singleton — built once on first use (app startup)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR
