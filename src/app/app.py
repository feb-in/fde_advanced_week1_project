"""app.py — FastAPI service for 30-day readmission risk.

Endpoints:
  * POST /predict — one patient's raw discharge-time fields (validated against the
    contract-generated schema). Returns calibrated risk, the flag (prob >= the
    registry threshold), and the top signed SHAP contributing factors.
  * GET  /health  — liveness + the loaded model version/alias/threshold.
  * GET  /metrics — Prometheus hook stub (Stage 6 wires real instrumentation).

The model is loaded once at startup from the MLflow registry by alias; rollback is
a registry stage/alias swap, no redeploy.

Run:
    uv run uvicorn src.app.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src
from app.model import get_predictor  # noqa: E402
from app.schemas import (  # noqa: E402
    HealthResponse,
    PatientRecord,
    PredictResponse,
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the predictor (load model + threshold + SHAP explainer) ONCE at startup.
    _state["predictor"] = get_predictor()
    yield
    _state.clear()


app = FastAPI(
    title="30-Day Readmission Risk API",
    version="1.0",
    description="Calibrated readmission risk + top SHAP factors for one discharge.",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict(record: PatientRecord) -> PredictResponse:
    predictor = _state["predictor"]
    # by_alias=True restores hyphenated raw column names for featurization.
    result = predictor.predict(record.model_dump(by_alias=True))
    return PredictResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(
        status="ok",
        model_name=p.model_name,
        model_version=str(p.version),
        model_alias=p.model_alias,
        threshold=p.threshold,
        calibration_method=p.calibration_method,
        load_source=p.load_source,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    # Prometheus hook — Stage 6 (observability) wires the instrumentator here.
    return ("# HELP readmission_api Prometheus metrics are wired in Stage 6.\n"
            "# Placeholder endpoint so the scrape target exists.\n")
