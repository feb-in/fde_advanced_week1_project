"""app.py — FastAPI service for 30-day readmission risk.

Endpoints:
  * POST /predict — one patient's raw discharge-time fields (validated against the
    contract-generated schema). Returns calibrated risk, the flag (prob >= the
    registry threshold), and the top signed SHAP contributing factors.
  * GET  /health  — liveness + the loaded model version/alias/threshold.
  * GET  /metrics — real Prometheus metrics (request count, latency histogram,
    status codes → error rate) via prometheus-fastapi-instrumentator.

The model is loaded once at startup from the MLflow registry by alias; rollback is
a registry stage/alias swap, no redeploy.

Run:
    uv run uvicorn src.app.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src
from app.audit import log_prediction  # noqa: E402
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

# Observability (Stage 6): instrument the app and expose real Prometheus metrics at
# /metrics — default HTTP request count, latency histogram, and status-code labels
# (error rate is derived from 5xx vs total). This replaces the Stage-5 stub and does
# NOT touch the /predict request/response contract.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.post("/predict", response_model=PredictResponse)
def predict(record: PatientRecord) -> PredictResponse:
    predictor = _state["predictor"]
    # by_alias=True restores hyphenated raw column names for featurization.
    inputs = record.model_dump(by_alias=True)
    t0 = time.perf_counter()
    result = predictor.predict(inputs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # Audit trail: every scored request is logged (best-effort, never blocks scoring).
    log_prediction(inputs=inputs, result=result, latency_ms=latency_ms, predictor=predictor)
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
