"""export_model.py — bake the @staging model into a self-contained bundle.

The container has no MLflow registry (the sqlite db + mlruns store absolute host
paths that don't resolve inside the image). This downloads whatever model the
registry ALIAS currently points to — plus its threshold/version tags — into
deploy/model_bundle/, which the Containerfile COPYs into the image.

The registry alias stays the LOGICAL rollback handle: to ship a different model,
re-point readmission-catboost-calibrated@staging, re-run this export, and rebuild
(or roll the image tag back). Model + threshold are captured together here so they
can never drift.

Run on the host (which has the registry):
    uv run python deploy/export_model.py
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow

NAME = os.environ.get("MODEL_NAME", "readmission-catboost-calibrated")
ALIAS = os.environ.get("MODEL_ALIAS", "staging")
OUT = Path(__file__).resolve().parent / "model_bundle"


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mv = mlflow.tracking.MlflowClient().get_model_version_by_alias(NAME, ALIAS)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{NAME}@{ALIAS}", dst_path=str(OUT / "model"))
    meta = {
        "model_name": NAME,
        "alias": ALIAS,
        "version": mv.version,
        "threshold": float(mv.tags["operating_threshold"]),
        "calibration_method": mv.tags.get("calibration_method", "unknown"),
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export] baked {NAME}@{ALIAS} v{mv.version} "
          f"(threshold {meta['threshold']}) → {OUT}")


if __name__ == "__main__":
    main()
