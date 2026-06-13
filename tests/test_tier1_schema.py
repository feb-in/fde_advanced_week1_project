"""Tier 1 — schema/contract validation. FAST: no model load, no server.

Validates the contract-generated PatientRecord directly (pure Pydantic), so these
run in milliseconds and catch boundary/policy regressions without touching the model.
"""
import pytest
from pydantic import ValidationError

from _serving_helpers import SAMPLE
from app.schemas import PatientRecord


def test_valid_record_accepted():
    PatientRecord(**SAMPLE)  # must not raise


def test_out_of_range_rejected():
    with pytest.raises(ValidationError):
        PatientRecord(**{**SAMPLE, "time_in_hospital": 99})  # contract max is 14


def test_unknown_extra_field_rejected():
    # extra="forbid" blocks accidental post-discharge leakage fields (e.g. the label)
    with pytest.raises(ValidationError):
        PatientRecord(**{**SAMPLE, "readmitted": "<30"})


def test_not_measured_lab_accepted():
    # "None" = test not ordered = real signal, NOT an error.
    m = PatientRecord(**{**SAMPLE, "A1Cresult": "None", "max_glu_serum": "None"})
    assert m.A1Cresult == "None" and m.max_glu_serum == "None"


def test_missing_categorical_accepted_as_signal():
    # informative missingness for race is accepted (mapped to Unknown downstream)
    PatientRecord(**{**SAMPLE, "race": None})


@pytest.mark.parametrize("code", [11, 13, 14, 19, 20, 21])
def test_expired_hospice_discharge_rejected(code):
    with pytest.raises(ValidationError):
        PatientRecord(**{**SAMPLE, "discharge_disposition_id": code})


def test_unknown_invalid_gender_rejected():
    with pytest.raises(ValidationError):
        PatientRecord(**{**SAMPLE, "gender": "Unknown/Invalid"})


def test_invalid_categorical_value_rejected():
    with pytest.raises(ValidationError):
        PatientRecord(**{**SAMPLE, "metformin": "Maybe"})  # not in {No,Down,Steady,Up}
