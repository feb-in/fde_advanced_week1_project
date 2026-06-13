"""build_features.py — deterministic feature engineering for readmission risk.

Reads the frozen clean parquet (data/processed/diabetes_clean.parquet) and writes
ONE typed parquet to data/featurized/. Every transformation here is a fixed,
justified rule — there is NO experimentation and NO modeling in this stage. The
feature list is locked in docs/GOALS.md (Stage 3) and each feature is logged in
docs/FEATURE_LOG.md.

Design notes
------------
* Categorical outputs are pandas 'category' dtype so CatBoost can consume them
  natively and parquet stays compact.
* All bucketers use an explicit code→name dict with an "Other"/"Unknown" fallback,
  so an unseen code on a future (retrain) batch is handled deterministically
  rather than crashing — this is production-safety, not just one-off cleaning.
* patient_nbr, encounter_id (audit keys) and target pass through untouched.

Run directly:
    uv run python src/features/build_features.py \
        --in data/processed/diabetes_clean.parquet \
        --out data/featurized/diabetes_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column groups (single source of truth)
# ---------------------------------------------------------------------------

KEY_COLS = ["encounter_id", "patient_nbr"]
TARGET_COL = "target"
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# The 21 medication columns that survived cleaning (examide/citoglipton dropped
# upstream). Values are one of {No, Down, Steady, Up}. Listed explicitly for
# determinism — we never infer the drug set from dtype guessing.
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Ordinal scale for medication dosage change. No=baseline, Up/Down=an active
# change this visit, Steady=on the drug but unchanged.
DRUG_ORDINAL = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

# A1c / glucose levels that mean "the test was actually run".
A1C_MEASURED_LEVELS = {">7", ">8", "Norm"}
GLU_MEASURED_LEVELS = {">200", ">300", "Norm"}
A1C_HIGH_LEVELS = {">7", ">8"}

# --- Admin ID → coarse bucket maps (codes are categorical, never numeric) -----
# Sourced from the dataset's IDs_mapping. Any code absent from a map falls back
# to the group default via .map(...).fillna(default).

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 7: "Trauma",
    3: "Elective",
    4: "Newborn",
    5: "Unknown", 6: "Unknown", 8: "Unknown",  # Not Available / NULL / Not Mapped
}

ADMISSION_SOURCE_MAP = {
    1: "Referral", 2: "Referral", 3: "Referral",            # physician/clinic/HMO
    4: "Transfer", 5: "Transfer", 6: "Transfer",            # hospital/SNF/other facility
    10: "Transfer", 22: "Transfer", 25: "Transfer",
    7: "EmergencyRoom",
    8: "Other",                                             # court/law enforcement
    11: "Delivery", 13: "Delivery", 14: "Delivery",
    9: "Unknown", 17: "Unknown", 20: "Unknown",             # Not Available / NULL / Not Mapped
}

# discharge_disposition_id — expired/hospice codes {11,13,14,19,20,21} were
# already removed in cleaning, so they do not appear here.
DISCHARGE_DISPOSITION_MAP = {
    1: "Home", 6: "Home", 8: "Home",                        # home / home health / home IV
    2: "Transfer", 9: "Transfer", 10: "Transfer",
    16: "Transfer", 17: "Transfer",
    3: "Facility", 4: "Facility", 5: "Facility", 15: "Facility",
    22: "Facility", 23: "Facility", 24: "Facility",
    27: "Facility", 28: "Facility",
    7: "AMA",                                               # left against medical advice
    12: "Unknown", 18: "Unknown", 25: "Unknown",
}
# "Discharged home" flag = any home destination (home, home-health, home-IV).
DISCHARGED_HOME_CODES = {1, 6, 8}


# ---------------------------------------------------------------------------
# 1. ICD-9 diagnosis bucketing (Strack-9)
# ---------------------------------------------------------------------------

def bucket_icd9(code) -> str:
    """Map a single ICD-9 code string to one of the 8 Strack groups, else 'Other'.

    NaN → 'Missing'. E/V codes and anything not in a numeric range → 'Other'.
    Diabetes (250.xx) is checked first by prefix so 250.x never falls into the
    Genitourinary/other numeric ranges.
    """
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s.startswith("250"):
        return "Diabetes"
    if s[:1] in ("E", "V"):
        return "Other"
    try:
        n = int(float(s))
    except ValueError:
        return "Other"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """diag_{1,2,3}_bucket (category) + diabetes_primary + n_diabetes_diag.

    Raw diag_1/diag_2/diag_3 are dropped after bucketing (high-cardinality ICD-9
    codes must never reach the model raw).
    """
    buckets = {}
    for c in DIAG_COLS:
        buckets[c] = df[c].map(bucket_icd9)

    df["diag_1_bucket"] = buckets["diag_1"].astype("category")
    df["diag_2_bucket"] = buckets["diag_2"].astype("category")
    df["diag_3_bucket"] = buckets["diag_3"].astype("category")

    is_diab = pd.DataFrame({c: buckets[c] == "Diabetes" for c in DIAG_COLS})
    df["diabetes_primary"] = is_diab["diag_1"].astype("int8")
    df["n_diabetes_diag"] = is_diab.sum(axis=1).astype("int8")

    return df.drop(columns=DIAG_COLS)


# ---------------------------------------------------------------------------
# 2. Medication features
# ---------------------------------------------------------------------------

def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """n_med_changes, n_meds_used, then ordinal-encode each drug column in place."""
    drugs = df[DRUG_COLS].astype("string")  # work on raw labels before encoding

    df["n_med_changes"] = (drugs.isin(["Up", "Down"])).sum(axis=1).astype("int8")
    df["n_meds_used"] = (drugs != "No").sum(axis=1).astype("int8")

    for c in DRUG_COLS:
        df[c] = df[c].map(DRUG_ORDINAL).astype("int8")

    return df


# ---------------------------------------------------------------------------
# 3 & 4. A1c / glucose measurement flags + A1c × med-change interaction
# ---------------------------------------------------------------------------

def add_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    """a1c_measured, glu_measured binary flags + a1c_state 4-level interaction.

    a1c_state (Strack's key engineered feature) crosses the A1c result with
    whether medications were changed this visit (the `change` column = Ch/No):
        no_test          — A1c not measured
        normal           — A1c measured, normal
        high_changed     — A1c high (>7/>8) AND meds changed
        high_not_changed — A1c high (>7/>8) AND meds NOT changed  (the risky cell)
    """
    a1c = df["A1Cresult"].astype("string")
    glu = df["max_glu_serum"].astype("string")
    changed = df["change"].astype("string") == "Ch"

    df["a1c_measured"] = a1c.isin(A1C_MEASURED_LEVELS).astype("int8")
    df["glu_measured"] = glu.isin(GLU_MEASURED_LEVELS).astype("int8")

    high = a1c.isin(A1C_HIGH_LEVELS)
    state = pd.Series("no_test", index=df.index, dtype="object")
    state[a1c == "Norm"] = "normal"
    state[high & changed] = "high_changed"
    state[high & ~changed] = "high_not_changed"
    df["a1c_state"] = state.astype("category")

    return df


# ---------------------------------------------------------------------------
# 5. Service utilization
# ---------------------------------------------------------------------------

def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """service_utilization sum (keeps number_inpatient separately) + inpatient_ge_2."""
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    ).astype("int32")
    df["inpatient_ge_2"] = (df["number_inpatient"] >= 2).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 6. Demographic / administrative
# ---------------------------------------------------------------------------

def _age_midpoint(bucket: str) -> int:
    """'[60-70)' → 65. Midpoint of the original 10-year band (full granularity)."""
    lo = int(bucket.strip("[)").split("-")[0])
    return lo + 5


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """age 3-bucket + numeric midpoint; coarse buckets for the 3 admin ID cols
    + discharged_home flag. Raw age and raw ID columns are dropped after."""
    age_str = df["age"].astype("string")
    df["age_midpoint"] = age_str.map(_age_midpoint).astype("int16")

    def to_3bucket(mid: int) -> str:
        if mid < 30:
            return "[0-30)"
        if mid < 60:
            return "[30-60)"
        return "[60-100)"

    df["age_bucket"] = df["age_midpoint"].map(to_3bucket).astype("category")

    df["admission_type_grp"] = (
        df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown").astype("category")
    )
    df["admission_source_grp"] = (
        df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown").astype("category")
    )
    df["discharge_disposition_grp"] = (
        df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP).fillna("Unknown").astype("category")
    )
    df["discharged_home"] = (
        df["discharge_disposition_id"].isin(DISCHARGED_HOME_CODES).astype("int8")
    )

    return df.drop(columns=[
        "age",
        "admission_type_id", "admission_source_id", "discharge_disposition_id",
    ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """The deterministic feature transforms, in order — shared by batch and serving.

    This is the SINGLE source of truth for Strack-9 bucketing, the A1c×med-change
    interaction, service-utilization, drug ordinal-encoding, age, and admission/
    discharge buckets. src/app/featurize.py calls this exact function on a one-row
    frame so a served record is engineered identically to training (no skew).
    Key/target columns (if present) pass through untouched.
    """
    df = add_diagnosis_features(df)
    df = add_medication_features(df)
    df = add_lab_features(df)
    df = add_utilization_features(df)
    df = add_demographic_features(df)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply every feature group in order. Returns (features_df, report)."""
    report: dict = {"rows_in": len(df), "cols_in": df.shape[1]}

    df = engineer_features(df)

    # Column order: keys first, target last, everything else in between.
    feats = [c for c in df.columns if c not in KEY_COLS + [TARGET_COL]]
    df = df[KEY_COLS + feats + [TARGET_COL]]

    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    num_cols = [
        c for c in df.columns
        if c not in cat_cols and c not in KEY_COLS + [TARGET_COL]
    ]
    report.update(
        rows_out=len(df),
        cols_out=df.shape[1],
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        pos_rate=round(float(df[TARGET_COL].mean()), 4),
        patient_nbr_present="patient_nbr" in df.columns,
        target_present=TARGET_COL in df.columns,
        patient_nbr_unique=bool(df["patient_nbr"].is_unique),
    )
    return df, report


def print_report(report: dict) -> None:
    print("\n============== build_features.py report ==============")
    print(f"in:   {report['rows_in']:,} rows x {report['cols_in']} cols")
    print(f"out:  {report['rows_out']:,} rows x {report['cols_out']} cols")
    print(f"positive rate:        {report['pos_rate']}  (expect ~0.0898)")
    print(f"patient_nbr present:  {report['patient_nbr_present']}  unique: {report['patient_nbr_unique']}")
    print(f"target present:       {report['target_present']}")
    print(f"\ncategorical ({len(report['categorical_cols'])}): {report['categorical_cols']}")
    print(f"\nnumeric ({len(report['numeric_cols'])}): {report['numeric_cols']}")
    print("======================================================\n")


def main(in_path: str, out_path: str) -> None:
    df = pd.read_parquet(in_path)
    df, report = build_features(df)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print_report(report)
    print(f"[features] wrote {out_path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="data/processed/diabetes_clean.parquet")
    ap.add_argument("--out", dest="out_path", default="data/featurized/diabetes_features.parquet")
    args = ap.parse_args()
    main(args.in_path, args.out_path)
