"""data_contract.py — the single source of truth for the dataset's input rules.

Everything downstream reads FROM here:
  * src/data/validate.py builds the RAW + PROCESSED Great Expectations suites from
    these constants.
  * export_input_contract() emits input_contract.json — the machine-readable schema
    the FastAPI Pydantic model will be generated from / kept in sync with, so the
    API boundary and the pipeline validate inputs identically.
  * Stage 6 drift uses the same field list as its reference schema.

Values were verified against the actual raw CSV (101,766 rows) and the featurized
parquet — not guessed. Missing values in the raw data are the literal string "?".
"""
from __future__ import annotations

import json
from pathlib import Path

MISSING_TOKEN = "?"  # raw missingness is coded as "?", never blank / NaN

# ---------------------------------------------------------------------------
# RAW layer — the 50 columns of data/raw/diabetic_data.csv (validated as-is)
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted",
]

# Label (validated in the raw suite, not a model input).
TARGET_RAW = "readmitted"
READMITTED_VALUES = ["<30", ">30", "NO"]

# Identifiers + columns dropped in cleaning — NOT model inputs.
IDENTIFIERS = ["encounter_id", "patient_nbr"]
DROPPED_IN_CLEANING = ["weight", "examide", "citoglipton"]

# All 23 drug-dose columns present in raw (examide/citoglipton are dropped later).
ALL_RAW_DRUGS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
DRUG_VALUES = ["No", "Down", "Steady", "Up"]
# The 21 drugs that survive cleaning and ARE model inputs.
MODEL_DRUGS = [d for d in ALL_RAW_DRUGS if d not in DROPPED_IN_CLEANING]

# Bounded categoricals (small, enumerable value sets). "?" handled via NULLABLE.
CATEGORICAL_VALUES = {
    "race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"],
    "gender": ["Male", "Female", "Unknown/Invalid"],
    "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
            "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
    # "None" here is a REAL level meaning "test not ordered" — never a null.
    "max_glu_serum": [">200", ">300", "None", "Norm"],
    "A1Cresult": [">7", ">8", "None", "Norm"],
    "change": ["Ch", "No"],
    "diabetesMed": ["No", "Yes"],
}

# High-cardinality string fields: type-checked + nullable, NO enumerated set
# (payer codes / specialties / ~700 ICD-9 codes change over time).
STRING_FIELDS = ["payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]

# Integer fields: clinical/admin counts with documented or generously-bounded
# ranges (catch garbage without false alarms on plausible future values).
INT_RANGES = {
    "admission_type_id": (1, 8),
    "discharge_disposition_id": (1, 30),
    "admission_source_id": (1, 26),
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 200),
    "num_procedures": (0, 12),
    "num_medications": (0, 150),
    "number_outpatient": (0, 100),
    "number_emergency": (0, 100),
    "number_inpatient": (0, 50),
    "number_diagnoses": (1, 16),
}

# Fields where "?" (missing token) is an allowed value in raw.
NULLABLE = {"race", "weight", "payer_code", "medical_specialty",
            "diag_1", "diag_2", "diag_3"}

# ---------------------------------------------------------------------------
# PROCESSED layer — the 58 columns of the featurized parquet
# ---------------------------------------------------------------------------

N_FEATURIZED_COLUMNS = 58
TARGET = "target"
POS_RATE_RANGE = (0.08, 0.12)  # sane band for the binary positive rate

# Strack-9 = 8 clinical groups + "Other" (= 9), plus a "Missing" sentinel for NaN
# diagnoses → 10 allowed bucket values.
STRACK_BUCKETS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]

# Features created in Stage 3 — must never be null in the pipeline output.
ENGINEERED_FEATURES = [
    "diag_1_bucket", "diag_2_bucket", "diag_3_bucket", "diabetes_primary",
    "n_diabetes_diag", "n_med_changes", "n_meds_used", "a1c_measured",
    "glu_measured", "a1c_state", "service_utilization", "inpatient_ge_2",
    "age_midpoint", "age_bucket", "admission_type_grp", "admission_source_grp",
    "discharge_disposition_grp", "discharged_home",
]
# A couple of engineered categoricals with small, fixed value sets.
A1C_STATE_VALUES = ["no_test", "normal", "high_changed", "high_not_changed"]
AGE_BUCKET_VALUES = ["[0-30)", "[30-60)", "[60-100)"]

# ---------------------------------------------------------------------------
# The exported input contract — model input fields only (44)
# ---------------------------------------------------------------------------

def model_input_fields() -> dict:
    """Per-field rules for the 44 fields the model/API consumes (raw minus
    identifiers, the label, and cleaning-dropped columns)."""
    fields: dict = {}
    for col, values in CATEGORICAL_VALUES.items():
        fields[col] = {
            "type": "categorical",
            "allowed_values": list(values),
            "nullable": col in NULLABLE,
            **({"missing_token": MISSING_TOKEN} if col in NULLABLE else {}),
        }
    for col in MODEL_DRUGS:
        fields[col] = {"type": "categorical", "allowed_values": list(DRUG_VALUES),
                       "nullable": False}
    for col in STRING_FIELDS:
        fields[col] = {
            "type": "string",
            "nullable": col in NULLABLE,
            **({"missing_token": MISSING_TOKEN} if col in NULLABLE else {}),
            **({"note": "ICD-9 diagnosis code"} if col.startswith("diag_") else {}),
        }
    for col, (lo, hi) in INT_RANGES.items():
        fields[col] = {"type": "integer", "min": lo, "max": hi, "nullable": False}
    return fields


def export_input_contract(path: str | Path) -> dict:
    """Write input_contract.json — the source of truth for the serving schema."""
    fields = model_input_fields()
    contract = {
        "$schema_version": 1,
        "description": ("Input contract for the 30-day readmission model. Derived "
                        "from the RAW Great Expectations suite. The FastAPI Pydantic "
                        "schema is generated from / kept in sync with this file so "
                        "the API and the pipeline validate inputs identically."),
        "missing_token": MISSING_TOKEN,
        "n_fields": len(fields),
        "excluded_from_input": {
            "identifiers": IDENTIFIERS,
            "label": [TARGET_RAW],
            "dropped_in_cleaning": DROPPED_IN_CLEANING,
        },
        "fields": dict(sorted(fields.items())),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


CONTRACT_JSON_PATH = Path(__file__).resolve().parent / "input_contract.json"


if __name__ == "__main__":
    c = export_input_contract(CONTRACT_JSON_PATH)
    print(f"wrote {CONTRACT_JSON_PATH}  ({c['n_fields']} model-input fields)")
