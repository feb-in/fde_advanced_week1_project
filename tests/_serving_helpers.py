"""Shared constants + helpers for the tiered serving tests (plain importable module).

Sets MODEL_BUNDLE_DIR so every model-loading test uses the BAKED bundle — offline,
deterministic, and the exact path the container/CI use — not the local MLflow
registry. No network anywhere in the suite.
"""
import json
import os
import sys
from pathlib import Path

# NOTE: pandas is imported lazily inside the helpers that need it, so the Tier-1
# schema tests (and CI's model-free Tier-1 step) can run with only pydantic.

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "src" / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("MODEL_BUNDLE_DIR", str(REPO / "deploy" / "model_bundle"))
os.environ.setdefault("TQDM_DISABLE", "1")

from contracts.data_contract import model_input_fields  # noqa: E402

THRESHOLD = 0.091046
GOLDEN_ENCOUNTER = 12522
GOLDEN_SCORE = 0.074595
SKEW_ENCOUNTERS = [12522, 15738, 16680]   # ≥3 known encounters; 12522 is golden

INPUT_FIELDS = list(model_input_fields().keys())
INT_FIELDS = {
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
}
SAMPLE = json.loads((REPO / "tests" / "sample_request.json").read_text())

# Schema-valid synthetic extremes for behavioral / monotonic sanity. NOT a hard
# threshold — just "obviously high-risk must outrank obviously low-risk".
HIGH_RISK = {**SAMPLE, "number_inpatient": 10, "number_emergency": 5,
             "number_outpatient": 5, "number_diagnoses": 16, "num_medications": 40,
             "time_in_hospital": 14, "discharge_disposition_id": 2,
             "admission_type_id": 1, "insulin": "Up", "change": "Ch",
             "age": "[70-80)", "diabetesMed": "Yes"}
LOW_RISK = {**SAMPLE, "number_inpatient": 0, "number_emergency": 0,
            "number_outpatient": 0, "number_diagnoses": 1, "num_medications": 1,
            "time_in_hospital": 1, "discharge_disposition_id": 1,
            "admission_type_id": 3, "insulin": "No", "change": "No",
            "age": "[20-30)", "diabetesMed": "No"}


def raw_record(encounter_id: int) -> dict:
    """Build an API request dict from the raw CSV row, read exactly as training."""
    import pandas as pd
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


def training_score(predictor, encounter_id: int) -> float:
    """The training-pipeline score = the model applied to the featurized parquet row."""
    import pandas as pd
    feat = pd.read_parquet(REPO / "data/featurized/diabetes_features.parquet")
    row = feat[feat["encounter_id"] == encounter_id].iloc[[0]][predictor.feature_names].copy()
    for c in predictor.cat_features:
        row[c] = row[c].astype("string").astype("object")
    return float(predictor.model.predict_proba(row)[:, 1][0])
