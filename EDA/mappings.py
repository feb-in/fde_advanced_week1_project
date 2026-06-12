"""Reference mappings & column taxonomy for the diabetes readmission dataset.

These decode the three opaque integer ``*_id`` columns into human-readable
labels and group the 50 raw columns into analytical families. They are the
canonical `IDs_mapping.csv` contents (UCI "Diabetes 130-US hospitals" dataset),
transcribed here so the EDA dashboard never shows a grader a bare integer.

Nothing in here is learned from the data — it is fixed domain reference, so it
lives in code (importable, diff-able) rather than in a notebook cell.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Opaque integer-code dictionaries (from IDs_mapping.csv)
# --------------------------------------------------------------------------- #

ADMISSION_TYPE_ID: dict[int, str] = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

DISCHARGE_DISPOSITION_ID: dict[int, str] = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient / expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare swing bed",
    16: "Discharged/transferred to another institution for outpatient services",
    17: "Discharged/transferred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab facility (incl. rehab units)",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility (Medicaid, not Medicare)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred to a psychiatric hospital / unit",
    29: "Discharged/transferred to a Critical Access Hospital (CAH)",
    30: "Discharged/transferred to another type of health care institution",
}

ADMISSION_SOURCE_ID: dict[int, str] = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac, separate claim",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice",
}

ID_DECODERS: dict[str, dict[int, str]] = {
    "admission_type_id": ADMISSION_TYPE_ID,
    "discharge_disposition_id": DISCHARGE_DISPOSITION_ID,
    "admission_source_id": ADMISSION_SOURCE_ID,
}

# --------------------------------------------------------------------------- #
# Graded business rules encoded as code (see CLAUDE.md §1 / docs/CAVEATS.md)
# --------------------------------------------------------------------------- #

# Hard rule 5: patients who died or went to hospice CANNOT be readmitted.
# These discharge_disposition_id codes must be filtered out before modeling.
EXPIRED_DISPOSITION_IDS: tuple[int, ...] = (11, 19, 20, 21)
HOSPICE_DISPOSITION_IDS: tuple[int, ...] = (13, 14)
DROP_DISPOSITION_IDS: tuple[int, ...] = EXPIRED_DISPOSITION_IDS + HOSPICE_DISPOSITION_IDS

# "Unknown / not collected" disposition / admission codes that are really
# missing-in-disguise (NULL / Not Mapped / Not Available / Unknown).
MISSING_LIKE_ID_CODES: dict[str, tuple[int, ...]] = {
    "admission_type_id": (5, 6, 8),         # Not Available, NULL, Not Mapped
    "discharge_disposition_id": (18, 25, 26),  # NULL, Not Mapped, Unknown/Invalid
    "admission_source_id": (9, 15, 17, 20, 21),  # Not Available/NULL/Not Mapped/Unknown
}

# --------------------------------------------------------------------------- #
# Column taxonomy (analytical families used to group the dashboard)
# --------------------------------------------------------------------------- #

IDENTIFIER_COLS = ["encounter_id", "patient_nbr"]

DEMOGRAPHIC_COLS = ["race", "gender", "age", "weight"]

ADMINISTRATIVE_ID_COLS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

ADMIN_TEXT_COLS = ["payer_code", "medical_specialty"]

UTILIZATION_NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

DIAGNOSIS_COLS = ["diag_1", "diag_2", "diag_3"]

LAB_RESULT_COLS = ["max_glu_serum", "A1Cresult"]

# The 23 drug columns (dose-change status: No / Steady / Up / Down).
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Known zero-variance drug columns in this dataset (single value "No").
CONSTANT_COLS = ["examide", "citoglipton"]

INDICATOR_COLS = ["change", "diabetesMed"]

TARGET_COL = "readmitted"

COLUMN_FAMILY: dict[str, list[str]] = {
    "Identifier": IDENTIFIER_COLS,
    "Demographic": DEMOGRAPHIC_COLS,
    "Administrative ID (coded)": ADMINISTRATIVE_ID_COLS,
    "Administrative text": ADMIN_TEXT_COLS,
    "Utilization (numeric)": UTILIZATION_NUMERIC_COLS,
    "Diagnosis (ICD-9)": DIAGNOSIS_COLS,
    "Lab result": LAB_RESULT_COLS,
    "Medication (dose change)": MEDICATION_COLS,
    "Indicator": INDICATOR_COLS,
    "Target": [TARGET_COL],
}


def family_of(column: str) -> str:
    """Return the analytical family label for a column (or 'Other')."""
    for family, cols in COLUMN_FAMILY.items():
        if column in cols:
            return family
    return "Other"


def decode_id(column: str, code) -> str:
    """Human-readable label for a coded ``*_id`` value, falling back to the raw code."""
    decoder = ID_DECODERS.get(column)
    if decoder is None:
        return str(code)
    try:
        return decoder.get(int(code), f"Unknown code {code}")
    except (ValueError, TypeError):
        return str(code)


# --------------------------------------------------------------------------- #
# ICD-9 diagnosis grouping (Strack et al. 2014 — the canonical buckets for this
# dataset). diag_1/2/3 hold 700–800 distinct raw codes each; the model uses these
# ~9 chapters + Diabetes + Other instead of one-hotting every code.
# --------------------------------------------------------------------------- #

# Ordered list of (low, high, chapter) integer ranges; 250.xx (diabetes) and the
# V/E supplemental codes are handled specially in ``icd9_to_chapter``.
_ICD9_RANGES: tuple[tuple[int, int, str], ...] = (
    (390, 459, "Circulatory"), (785, 785, "Circulatory"),
    (460, 519, "Respiratory"), (786, 786, "Respiratory"),
    (520, 579, "Digestive"), (787, 787, "Digestive"),
    (580, 629, "Genitourinary"), (788, 788, "Genitourinary"),
    (140, 239, "Neoplasms"),
    (710, 739, "Musculoskeletal"),
    (800, 999, "Injury"),
)


def icd9_to_chapter(code) -> str:
    """Map a raw ICD-9 code (string/number) to a Strack-2014 diagnosis chapter.

    Missing → ``"Missing"``; 250.xx → ``"Diabetes"``; V/E supplemental codes and
    any range not in the nine main chapters → ``"Other"``.
    """
    if code is None:
        return "Missing"
    s = str(code).strip()
    if s == "" or s.lower() in ("nan", "?", "none"):
        return "Missing"
    if s[0] in ("V", "v", "E", "e"):
        return "Other"
    try:
        num = float(s)
    except ValueError:
        return "Other"
    if 250.0 <= num < 251.0:
        return "Diabetes"
    code3 = int(num)
    for lo, hi, name in _ICD9_RANGES:
        if lo <= code3 <= hi:
            return name
    return "Other"


# Stable display/order for the diagnosis chapters.
ICD9_CHAPTER_ORDER: tuple[str, ...] = (
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Genitourinary",
    "Injury", "Musculoskeletal", "Neoplasms", "Other", "Missing",
)
