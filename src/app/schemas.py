"""schemas.py — request/response models for the API.

The request model (PatientRecord) is GENERATED at import time from
src/contracts/input_contract.json — the same 44-field contract the Great
Expectations RAW suite enforces. The API boundary therefore cannot drift from the
pipeline: change the contract, both move together.

Serving rejection policy (decided ON TOP of the raw-faithful contract — the raw
contract is permissive because it validates historical bulk data; a live
discharge-time request is stricter):

  ACCEPT as real signal (NOT errors):
    * A1Cresult / max_glu_serum == "None"  — "test not ordered" is predictive.
    * "?" or null for race / payer_code / medical_specialty / diag_*  — informative
      missingness; mapped to "Unknown"/"Missing" exactly as in training.
  REJECT (422):
    * any type / range / allowed-value violation (enforced by the field types);
    * unknown / extra fields (extra="forbid") — blocks accidental post-discharge
      leakage fields from entering;
    * gender == "Unknown/Invalid" — training dropped these rows; not scoreable;
    * discharge_disposition_id in {11,13,14,19,20,21} (expired/hospice) — these
      patients structurally cannot be readmitted; training filtered them out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "input_contract.json"

# Semantic rejections that mirror training's row filters (see clean.py).
EXPIRED_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}
REJECTED_GENDER = "Unknown/Invalid"


def _field_for(spec: dict, alias: str | None):
    """Map one contract field spec → (type, FieldInfo) for create_model."""
    t = spec["type"]
    nullable = spec.get("nullable", False)
    common = {"alias": alias} if alias else {}

    if t == "integer":
        return (int, Field(ge=spec["min"], le=spec["max"], **common))
    if t == "categorical":
        values = list(spec["allowed_values"])
        if nullable:
            values = values + [spec.get("missing_token", "?")]
        lit = Literal[tuple(values)]
        if nullable:
            return (Optional[lit], Field(default=None, **common))
        return (lit, Field(**common))
    # free-form string (high-cardinality: payer codes, specialties, ICD-9 codes)
    if nullable:
        return (Optional[str], Field(default=None, **common))
    return (str, Field(**common))


def _build_patient_model():
    contract = json.loads(CONTRACT_PATH.read_text())
    fields = {}
    for name, spec in contract["fields"].items():
        pyname = name.replace("-", "_")
        alias = name if pyname != name else None
        fields[pyname] = _field_for(spec, alias)

    def _reject_gender(cls, v):
        if v == REJECTED_GENDER:
            raise ValueError(
                "gender 'Unknown/Invalid' is not scoreable (excluded during training)")
        return v

    def _reject_expired(cls, v):
        if v in EXPIRED_HOSPICE_IDS:
            raise ValueError(
                f"discharge_disposition_id {v} is expired/hospice — patient cannot be "
                "readmitted; not scoreable")
        return v

    validators = {
        "_reject_gender": field_validator("gender")(_reject_gender),
        "_reject_expired": field_validator("discharge_disposition_id")(_reject_expired),
    }
    model = create_model(
        "PatientRecord",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        __validators__=validators,
        **fields,
    )
    model.__doc__ = ("One patient's raw discharge-time fields (44, per "
                     "input_contract.json). Missing → '?'/null where allowed.")
    return model


PatientRecord = _build_patient_model()
N_INPUT_FIELDS = len(json.loads(CONTRACT_PATH.read_text())["fields"])


class Factor(BaseModel):
    feature: str
    value: object
    contribution: float = Field(description="signed SHAP value (log-odds); >0 raises risk")
    direction: Literal["increases", "decreases"]


class PredictResponse(BaseModel):
    readmission_probability: float = Field(ge=0.0, le=1.0)
    flag: bool = Field(description="True if probability >= operating threshold")
    threshold: float
    model_name: str
    model_version: str
    model_alias: str
    top_factors: list[Factor]
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_alias: str
    threshold: float
    calibration_method: str
    model_config = ConfigDict(protected_namespaces=())
