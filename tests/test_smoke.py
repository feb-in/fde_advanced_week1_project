"""Smoke + skew tests for the FastAPI serving layer.

The critical one is test_no_train_serve_skew: a record pushed through /predict must
produce the SAME calibrated score the training pipeline produced for that record.
Run headlessly via FastAPI's TestClient (no live uvicorn server needed).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "models"))

INT_FIELDS = {"admission_type_id", "discharge_disposition_id", "admission_source_id",
              "time_in_hospital", "num_lab_procedures", "num_procedures",
              "num_medications", "number_outpatient", "number_emergency",
              "number_inpatient", "number_diagnoses"}

from app.app import app  # noqa: E402
from app.featurize import featurize_record  # noqa: E402
from app.model import get_predictor  # noqa: E402
from contracts.data_contract import model_input_fields  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = json.loads((REPO / "tests" / "sample_request.json").read_text())
INPUT_FIELDS = list(model_input_fields().keys())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # triggers lifespan → loads model once
        yield c


def _raw_record(encounter_id: int) -> dict:
    """Build an API request dict from the raw CSV row, read exactly as training."""
    raw = pd.read_csv(REPO / "data/raw/diabetic_data.csv",
                      na_values=["?"], keep_default_na=False, low_memory=False)
    r = raw[raw["encounter_id"] == encounter_id].iloc[0]
    rec = {}
    for f in INPUT_FIELDS:
        v = r[f]
        if v == "" or (isinstance(v, float) and pd.isna(v)):
            rec[f] = None
        elif f in INT_FIELDS:
            rec[f] = int(v)
        else:
            rec[f] = str(v)
    return rec


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_alias"] == "staging"
    assert 0.0 < body["threshold"] < 1.0


def test_predict_shape(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert isinstance(body["flag"], bool)
    # flag must be exactly prob >= threshold
    assert body["flag"] == (body["readmission_probability"] >= body["threshold"])
    assert len(body["top_factors"]) >= 1
    for f in body["top_factors"]:
        assert f["direction"] in ("increases", "decreases")
        assert set(f) == {"feature", "value", "contribution", "direction"}


def test_no_train_serve_skew(client):
    """The skew gate: API score == training-pipeline score for the same record."""
    predictor = get_predictor()
    feat = pd.read_parquet(REPO / "data/featurized/diabetes_features.parquet")
    for i in range(5):
        enc = int(feat.iloc[i]["encounter_id"])
        # training-pipeline score: model on the featurized parquet row
        Xt = feat.iloc[[i]][predictor.feature_names].copy()
        for c in predictor.cat_features:
            Xt[c] = Xt[c].astype("string").astype("object")
        train_score = float(predictor.model.predict_proba(Xt)[:, 1][0])
        # serving score: same record through the API
        resp = client.post("/predict", json=_raw_record(enc))
        assert resp.status_code == 200, resp.text
        api_score = resp.json()["readmission_probability"]
        assert abs(api_score - round(train_score, 6)) < 1e-6, (
            f"SKEW on encounter {enc}: api={api_score} train={train_score}")


def test_rejection_policy(client):
    # extra/unknown field → 422 (blocks post-discharge leakage fields)
    assert client.post("/predict", json={**SAMPLE, "readmitted": "<30"}).status_code == 422
    # gender Unknown/Invalid → 422 (excluded in training)
    assert client.post("/predict", json={**SAMPLE, "gender": "Unknown/Invalid"}).status_code == 422
    # expired/hospice discharge → 422 (cannot be readmitted)
    assert client.post("/predict", json={**SAMPLE, "discharge_disposition_id": 11}).status_code == 422
    # out-of-range integer → 422
    assert client.post("/predict", json={**SAMPLE, "time_in_hospital": 99}).status_code == 422
    # invalid categorical → 422
    assert client.post("/predict", json={**SAMPLE, "metformin": "Maybe"}).status_code == 422


def test_accepts_not_measured_labs(client):
    # "None" (test not ordered) is a REAL value and must be accepted.
    rec = {**SAMPLE, "A1Cresult": "None", "max_glu_serum": "None", "race": None}
    assert client.post("/predict", json=rec).status_code == 200
