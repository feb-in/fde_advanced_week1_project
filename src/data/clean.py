"""clean.py — reproducible cleaning pass for the diabetes readmission dataset.

Reads the raw Kaggle CSV and writes ONE clean, typed parquet to data/processed/.
This script is the single source of truth for every cleaning decision. It is
tracked by DVC; `dvc repro` re-runs it whenever src/data/clean.py or the raw CSV
changes.

Deliberately out of scope here (handled in the featurization stage):
  * diag_1/diag_2/diag_3  ICD-9 → Strack-9 bucket encoding
  * A1c_measured / glu_measured binary flags
  * age bucketing / midpoint
  * service utilization engineering, med-change counts, etc.

Run directly:
    uv run python src/data/clean.py --raw data/raw/diabetic_data.csv \
                                    --out data/processed/diabetes_clean.parquet

Every decision below maps to a hard rule in CLAUDE.md / docs/PROJECT_BRIEF.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (single source of truth for all cleaning decisions)
# ---------------------------------------------------------------------------

# Columns where "None" is a real, predictive value meaning "test not ordered".
# pandas' default na list includes "None", so we use keep_default_na=False in
# load_raw() to prevent silent destruction of this signal.
LAB_COLS = ["A1Cresult", "max_glu_serum"]

# discharge_disposition_id codes for expired / hospice patients.
# These patients cannot be readmitted, so their rows corrupt the label.
# expired={11,19,20,21}, hospice={13,14}.  Code 21 has 0 rows but is listed
# for completeness.
EXPIRED_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}

# Columns with a single value across all 101,766 rows → zero information.
ZERO_VARIANCE_COLS = ["examide", "citoglipton"]

# Categoricals where missingness is informative (MNAR): "was not recorded" is
# itself a clue about the patient's care path or socioeconomic situation.
# Fill NaN → "Unknown" and keep as a real category level.
MISSING_AS_UNKNOWN = ["payer_code", "medical_specialty", "race"]

# Key columns: retained in the output but never used as model inputs.
#   patient_nbr  — needed for the uniqueness assertion + audit trail
#   encounter_id — per-request audit key; links predictions back to a row
KEY_COLS = ["encounter_id", "patient_nbr"]

POSITIVE_LABEL = "<30"  # ">30" and "NO" are both mapped to 0 (negative).


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV with the only correct missing-value contract for this dataset.

    keep_default_na=False — prevent pandas from converting the string "None"
                            (a real lab value) into NaN.
    na_values=["?"]       — the dataset's actual missing token is "?", and only
                            "?", so that is the one string that becomes NaN.
    low_memory=False      — read each column in full to get consistent dtype
                            inference (avoids mixed-type warnings on sparse cols).
    """
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the full cleaning pass. Returns (clean_df, report).

    Steps are ordered so that row-dropping happens before column-dropping and
    dedup, making the row-count sequence in the report easy to follow.
    """
    report: dict = {"rows_raw": len(df)}

    # 1. Drop rows: gender "Unknown/Invalid" (3 rows, not a real sex category).
    bad_gender = ~df["gender"].isin(["Male", "Female"])
    df = df[~bad_gender].copy()
    report["rows_dropped_bad_gender"] = int(bad_gender.sum())

    # 2. Drop rows: expired / hospice discharges cannot be readmitted.
    mask_dead = df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)
    report["rows_dropped_expired_hospice"] = int(mask_dead.sum())
    df = df[~mask_dead].copy()
    report["rows_after_discharge_filter"] = len(df)

    # 3. First-encounter dedup — LEAKAGE GUARD.
    #    The same patient_nbr appears across multiple encounter rows (repeat
    #    visits).  Keep only the encounter with the smallest encounter_id (the
    #    patient's earliest recorded stay), then assert uniqueness so a plain
    #    StratifiedKFold split cannot put the same patient in both train and test.
    df = (
        df.sort_values("encounter_id")
          .groupby("patient_nbr", sort=False)
          .first()
          .reset_index()
    )
    assert df["patient_nbr"].is_unique, (
        "BUG: patient_nbr is not unique after first-encounter dedup — "
        "check the groupby logic."
    )
    report["rows_after_first_encounter_dedup"] = len(df)

    # 4. Drop columns: dead weight (96.9% missing) and zero-variance drugs.
    drop_cols = ["weight"] + [c for c in ZERO_VARIANCE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    report["cols_dropped"] = drop_cols

    # 5. Lab columns: preserve "None" as an explicit "NotMeasured" category.
    #    "None" in these columns is a real clinical fact (test was not ordered),
    #    not a missing value.  It is already kept by keep_default_na=False;
    #    we rename it here to be unambiguous.
    for c in LAB_COLS:
        df[c] = df[c].where(df[c] != "None", "NotMeasured")

    # 6. High-missing categoricals: NaN → "Unknown" (missingness is informative).
    for c in MISSING_AS_UNKNOWN:
        df[c] = df[c].fillna("Unknown")

    # 7. Target: binary.  "<30" = readmitted within 30 days (positive).
    #    ">30" and "NO" both map to 0 — same care-team action (no urgent follow-up).
    df["target"] = (df["readmitted"] == POSITIVE_LABEL).astype("int8")
    df = df.drop(columns=["readmitted"])

    # 8. Compact dtypes: object → category for non-key, non-diag string columns.
    #    diag_1/2/3 stay as strings for the ICD-9 bucketing stage.
    diag_cols = {"diag_1", "diag_2", "diag_3"}
    for c in df.columns:
        if c in KEY_COLS or c in diag_cols:
            continue
        if df[c].dtype == object:
            df[c] = df[c].astype("category")

    # 9. Column order: keys first, target last, features in between.
    feats = [c for c in df.columns if c not in KEY_COLS + ["target"]]
    df = df[KEY_COLS + feats + ["target"]]

    report["rows_final"] = len(df)
    report["cols_final"] = df.shape[1]
    report["pos_rate"] = round(float(df["target"].mean()), 4)
    report["unique_patients"] = int(df["patient_nbr"].nunique())
    return df, report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: dict, df: pd.DataFrame) -> None:
    print("\n================ clean.py sanity report ================")
    print(f"raw rows:                 {report['rows_raw']:>8,}")
    print(f"after discharge filter:   {report['rows_after_discharge_filter']:>8,}  "
          f"(-{report['rows_dropped_expired_hospice']:,} expired/hospice, "
          f"-{report['rows_dropped_bad_gender']} bad-gender)")
    print(f"after first-encounter:    {report['rows_after_first_encounter_dedup']:>8,}  "
          f"(keep min encounter_id per patient_nbr)")
    print(f"final shape:              {report['rows_final']:,} rows x {report['cols_final']} cols")
    print(f"positive rate:            {report['pos_rate']}  (expect 0.09–0.11)")
    print(f"patient_nbr unique:       YES  (assertion passed)")
    print(f"cols dropped:             {report['cols_dropped']}")
    for c in LAB_COLS:
        print(f"  {c:<16} levels: {sorted(df[c].cat.categories.tolist())}")
    print("=========================================================\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(raw: str, out: str) -> None:
    df = load_raw(raw)
    df, report = clean(df)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print_report(report, df)
    print(f"[clean] wrote {out}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/diabetic_data.csv")
    ap.add_argument("--out", default="data/processed/diabetes_clean.parquet")
    args = ap.parse_args()
    main(args.raw, args.out)
