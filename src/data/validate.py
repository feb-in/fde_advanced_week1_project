"""validate.py — Great Expectations batch validation checkpoints.

Great Expectations (GX) is a BATCH validation tool: it checks a whole dataframe at
a pipeline checkpoint. It is NOT in the real-time /predict path (Pydantic guards
single requests there). The reusable artifact is the Expectation Suite — the data
contract — which here is built from src/contracts/data_contract.py.

GX moving parts used below:
  * Data Context   — the GX project state (persisted under ./gx).
  * Expectation Suite — the set of assertions (our contract, in code).
  * Validation Definition — binds a suite to a data batch.
  * Checkpoint     — runs the validation(s) and fires actions (here: render the
                     HTML "data docs" report).

Two suites, two failure modes:
  * RAW       — validates data/raw/diabetic_data.csv BEFORE cleaning (guards INPUT).
  * PROCESSED — validates the featurized parquet AFTER featurization (guards OUTPUT).

A failed expectation makes this script exit NON-ZERO, which fails the DVC stage and
halts `dvc repro` — a broken data contract must stop a retrain, not warn-and-continue.

Run:
    uv run python src/data/validate.py --suite raw
    uv run python src/data/validate.py --suite processed
    uv run python src/data/validate.py --suite raw --inject-bad   # demo a failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Quiet + offline: no tqdm progress bars, no analytics calls inside the pipeline.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402
from great_expectations import expectations as gxe  # noqa: E402
from great_expectations.checkpoint import UpdateDataDocsAction  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src on path
from contracts import data_contract as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GX_ROOT = REPO_ROOT  # GX creates ./gx here


# ---------------------------------------------------------------------------
# Suite builders — translate the contract into GX expectations
# ---------------------------------------------------------------------------

def raw_expectations() -> list:
    """~45 assertions guarding the raw CSV (read as-is, '?' kept literal)."""
    exp = [
        gxe.ExpectTableColumnsToMatchSet(column_set=C.RAW_COLUMNS, exact_match=True),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        # The label has exactly its 3 known classes.
        gxe.ExpectColumnValuesToBeInSet(column=C.TARGET_RAW, value_set=C.READMITTED_VALUES),
        # Missingness convention: gender is never blank/NaN — missing is the "?" token,
        # which only appears in NULLABLE columns (and stays a literal string).
        gxe.ExpectColumnValuesToNotBeNull(column="gender"),
    ]
    for col, values in C.CATEGORICAL_VALUES.items():
        allowed = list(values) + ([C.MISSING_TOKEN] if col in C.NULLABLE else [])
        exp.append(gxe.ExpectColumnValuesToBeInSet(column=col, value_set=allowed))
    for col, (lo, hi) in C.INT_RANGES.items():
        exp.append(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi))
    for drug in C.ALL_RAW_DRUGS:
        exp.append(gxe.ExpectColumnValuesToBeInSet(column=drug, value_set=C.DRUG_VALUES))
    return exp


def processed_expectations() -> list:
    """~28 assertions guarding the featurized parquet (the pipeline output)."""
    exp = [
        gxe.ExpectTableColumnCountToEqual(value=C.N_FEATURIZED_COLUMNS),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        # The dedup guarantee: one row per patient.
        gxe.ExpectColumnValuesToBeUnique(column="patient_nbr"),
        # Target is binary and the positive rate sits in a sane band.
        gxe.ExpectColumnValuesToBeInSet(column=C.TARGET, value_set=[0, 1]),
        gxe.ExpectColumnMeanToBeBetween(column=C.TARGET,
                                        min_value=C.POS_RATE_RANGE[0],
                                        max_value=C.POS_RATE_RANGE[1]),
        # Engineered categoricals take only their known values.
        gxe.ExpectColumnValuesToBeInSet(column="a1c_state", value_set=C.A1C_STATE_VALUES),
        gxe.ExpectColumnValuesToBeInSet(column="age_bucket", value_set=C.AGE_BUCKET_VALUES),
    ]
    for c in ("diag_1_bucket", "diag_2_bucket", "diag_3_bucket"):
        exp.append(gxe.ExpectColumnValuesToBeInSet(column=c, value_set=C.STRACK_BUCKETS))
    for c in C.ENGINEERED_FEATURES:  # no unexpected nulls in engineered features
        exp.append(gxe.ExpectColumnValuesToNotBeNull(column=c))
    return exp


SUITES = {
    "raw": {
        "expectations": raw_expectations,
        "load": lambda path: pd.read_csv(path, keep_default_na=False),  # keep "?" literal
    },
    "processed": {
        "expectations": processed_expectations,
        "load": lambda path: pd.read_parquet(path),
    },
}


def inject_bad_row(suite: str, df: pd.DataFrame) -> pd.DataFrame:
    """Corrupt one row to prove the suite fails loudly on a broken contract."""
    df = df.copy()
    if suite == "raw":
        df.loc[df.index[0], "gender"] = "Martian"          # not in {Male,Female,Unknown/Invalid}
    else:
        df.loc[df.index[0], "target"] = 7                  # target must be {0,1}
    return df


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(suite: str, data_path: str, inject_bad: bool, status_path: str | None) -> bool:
    spec = SUITES[suite]
    df = spec["load"](data_path)
    if inject_bad:
        df = inject_bad_row(suite, df)
        print(f"[validate] --inject-bad: corrupted one row to demonstrate failure")
    print(f"[validate] suite={suite!r}  rows={len(df):,}  cols={df.shape[1]}")

    ctx = gx.get_context(mode="file", project_root_dir=str(GX_ROOT))
    ds = ctx.data_sources.add_or_update_pandas(name=f"{suite}_source")
    asset = ds.add_dataframe_asset(name=f"{suite}_asset")
    batch_def = asset.add_batch_definition_whole_dataframe(name=f"{suite}_batch")

    gx_suite = ctx.suites.add_or_update(gx.ExpectationSuite(name=f"{suite}_suite"))
    expectations = spec["expectations"]()
    for e in expectations:
        gx_suite.add_expectation(e)

    vd = ctx.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{suite}_validation", data=batch_def, suite=gx_suite))
    cp = ctx.checkpoints.add_or_update(gx.Checkpoint(
        name=f"{suite}_checkpoint", validation_definitions=[vd],
        actions=[UpdateDataDocsAction(name="update_data_docs")]))

    result = cp.run(batch_parameters={"dataframe": df})
    ctx.build_data_docs()

    n_exp = len(expectations)
    vr = list(result.run_results.values())[0]
    stats = getattr(vr, "statistics", {}) or {}
    n_ok = stats.get("successful_expectations", n_exp if result.success else None)
    print(f"[validate] {suite}: {n_ok}/{n_exp} expectations passed  "
          f"→ success={result.success}")

    if not result.success:
        for r in vr.results:
            if not r.success:
                cfg = r.expectation_config
                print(f"  FAILED: {cfg.type}  {cfg.kwargs.get('column', cfg.kwargs)}")
        docs = GX_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html"
        print(f"[validate] data docs: {docs}")
        return False

    if status_path:
        Path(status_path).parent.mkdir(parents=True, exist_ok=True)
        Path(status_path).write_text(json.dumps(
            {"suite": suite, "success": True,
             "evaluated_expectations": n_exp, "successful_expectations": n_exp},
            indent=2, sort_keys=True) + "\n")
    docs = GX_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html"
    print(f"[validate] PASSED. data docs: {docs}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["raw", "processed"], required=True)
    ap.add_argument("--data", help="path to the data file (defaults per suite)")
    ap.add_argument("--status", help="write a success-marker JSON here (DVC out)")
    ap.add_argument("--inject-bad", action="store_true",
                    help="corrupt one row to demonstrate a loud failure")
    args = ap.parse_args()

    default_data = {
        "raw": "data/raw/diabetic_data.csv",
        "processed": "data/featurized/diabetes_features.parquet",
    }
    data_path = args.data or default_data[args.suite]
    ok = run(args.suite, data_path, args.inject_bad, args.status)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
