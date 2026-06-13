"""drift.py — Stage 6 observability: Evidently drift report on a SIMULATED batch.

WHY SIMULATED: this project has no live production traffic yet, so there is no real
"current" batch to compare against the training baseline. To VALIDATE that the drift
detector works (and is wired to the right reference), we synthesize a current batch and
deliberately, visibly shift a few features — then confirm Evidently fires on the shifted
batch and does NOT fire on an unshifted control (so the detector isn't crying wolf).

  reference  = data/monitoring/reference.parquet  (training distribution, seed-42 train)
  control    = a fresh in-distribution sample (held-out test split), unshifted → expect NO drift
  shifted    = the SAME sample with INTENTIONAL shifts applied + re-scored → expect DRIFT

The shifts are configurable (CLI flags) and every one is commented INTENTIONAL. This is a
detector test, not a finding about real data.

Outputs: HTML reports to reports/monitoring/ + a printed per-feature drift summary and the
overall verdict for both batches.

Run:
    uv run python src/monitoring/drift.py
    uv run python src/monitoring/drift.py --age-shift 25 --emergency-frac 0.5 --current-n 4000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "models"))
import models.train as T  # noqa: E402

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

FEATURES = "data/featurized/diabetes_features.parquet"
REFERENCE = Path("data/monitoring/reference.parquet")
REPORT_DIR = Path("reports/monitoring")
MODEL_NAME = "readmission-catboost-calibrated"
MODEL_ALIAS = "staging"

# Dataset-drift is declared when the SHARE of drifted columns exceeds this. 0.1 is a
# documented DEMONSTRATION threshold (a targeted shift moves only a handful of 54
# features). The retrain trigger (src/monitoring/retrain_trigger.py) consumes the
# share + the per-feature PSI below.
DRIFT_SHARE = 0.10

# Top global-SHAP drivers (docs/MODEL_CARD.md). We additionally report PSI for these so
# the retrain trigger can apply a "PSI > 0.2 on a key feature" rule, not just the
# dataset-level share. PSI (Population Stability Index): <0.1 stable, 0.1-0.2 moderate,
# >0.2 significant shift — the industry-standard drift trigger metric.
TOP_SHAP_FEATURES = [
    "discharged_home", "discharge_disposition_grp", "diag_1_bucket", "number_inpatient",
    "medical_specialty", "time_in_hospital", "age_midpoint", "service_utilization",
]


def psi(ref, cur, bins=10, eps=1e-6):
    """Population Stability Index between a reference and current series.

    Numeric features with enough distinct values are binned into reference deciles;
    categorical / low-cardinality features use category shares. PSI = Σ (c−r)·ln(c/r)."""
    ref = pd.Series(ref).dropna()
    cur = pd.Series(cur).dropna()
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > bins:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, bins=edges)[0] / len(ref)
        c = np.histogram(cur, bins=edges)[0] / len(cur)
    else:
        cats = pd.Index(ref.astype(str).unique()).union(cur.astype(str).unique())
        r = (ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()) / len(ref)
        c = (cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()) / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def load_model():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


def model_columns(model):
    base = model.calibrated_classifiers_[0].estimator
    return list(base.model_.feature_names_), list(base.cat_features)


# ---------------------------------------------------------------------------
# The INTENTIONAL shift — synthesize population/coding drift on a few features.
# ---------------------------------------------------------------------------
def apply_shifts(df, rng, *, age_shift, diag1_frac, inpatient_bump, emergency_frac):
    """Return (shifted_df, list_of_shifted_features). Every change is DELIBERATE."""
    d = df.copy()
    shifted = []

    # 1) Population ages UP: push age_midpoint older and recompute the age bucket so the
    #    two stay consistent (an older catchment area / ward-mix change).
    if age_shift:
        d["age_midpoint"] = np.clip(d["age_midpoint"] + age_shift, 5, 95)
        d["age_bucket"] = np.where(d["age_midpoint"] >= 60, "[60-100)",
                          np.where(d["age_midpoint"] >= 30, "[30-60)", "[0-30)"))
        shifted += ["age_midpoint", "age_bucket"]

    # 2) Diagnosis-CODING shift: over-represent Circulatory as the primary-diagnosis bucket
    #    (e.g. a new cardiology service or a coding-policy change).
    if diag1_frac:
        idx = rng.random(len(d)) < diag1_frac
        d.loc[idx, "diag_1_bucket"] = "Circulatory"
        shifted += ["diag_1_bucket"]

    # 3) Sicker case-mix: more prior inpatient stays. Keep the derived utilization features
    #    coherent (service_utilization sums prior visits; inpatient_ge_2 is its flag).
    if inpatient_bump:
        d["number_inpatient"] = d["number_inpatient"] + inpatient_bump
        d["service_utilization"] = d["service_utilization"] + inpatient_bump
        d["inpatient_ge_2"] = (d["number_inpatient"] >= 2).astype("int8")
        shifted += ["number_inpatient", "service_utilization", "inpatient_ge_2"]

    # 4) Admission-mix shift: more emergency admissions.
    if emergency_frac:
        idx = rng.random(len(d)) < emergency_frac
        d.loc[idx, "admission_type_grp"] = "Emergency"
        shifted += ["admission_type_grp"]

    return d, sorted(set(shifted))


# ---------------------------------------------------------------------------
def build_current(model, feature_names, cat_cols, n, rng):
    """A fresh in-distribution batch from the held-out test split, model-framed + scored."""
    _, X_test, _, _, cats, _, _ = T.prepare_data(FEATURES)
    Xte = T.as_catboost_frame(X_test, cats).reindex(columns=feature_names).reset_index(drop=True)
    take = min(n, len(Xte))
    sample = Xte.iloc[rng.permutation(len(Xte))[:take]].reset_index(drop=True)
    return sample


def score(model, df, feature_names):
    out = df.copy()
    out["prediction"] = model.predict_proba(df[feature_names])[:, 1]
    return out


def run_report(reference, current, feature_names, cat_features, html_path):
    """Run Evidently DataDriftPreset (features + prediction) and save HTML."""
    cols = feature_names + ["prediction"]
    num = [c for c in feature_names if c not in cat_features] + ["prediction"]
    ref = reference[cols].copy()
    cur = current[cols].copy()
    for c in cat_features:                      # categoricals as plain strings in both
        ref[c] = ref[c].astype(str)
        cur[c] = cur[c].astype(str)

    dd = DataDefinition(numerical_columns=num, categorical_columns=list(cat_features))
    rds = Dataset.from_pandas(ref, data_definition=dd)
    cds = Dataset.from_pandas(cur, data_definition=dd)
    snap = Report([DataDriftPreset(drift_share=DRIFT_SHARE)]).run(reference_data=rds, current_data=cds)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(html_path))
    res = summarize(snap)
    # PSI on the top-SHAP features (feeds the retrain trigger's key-feature rule).
    res["top_feature_psi"] = {
        f: round(psi(ref[f], cur[f]), 4) for f in TOP_SHAP_FEATURES if f in cols
    }
    return res


def summarize(snap):
    """Parse snap.dict() → overall {count, share, dataset_drift} + per-column drift flags."""
    d = snap.dict()
    overall, cols = {}, {}
    for m in d["metrics"]:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            v = m["value"]
            overall = {"drifted": int(v["count"]), "share": float(v["share"])}
        elif name.startswith("ValueDrift"):
            col = m["config"]["column"]
            thr = m["config"].get("threshold", 0.05)
            val = float(m["value"])
            # p-value methods: drift when value < threshold; distance methods: value >= threshold.
            is_p = "p_value" in name
            drifted = (val < thr) if is_p else (val >= thr)
            cols[col] = {"method": name.split("method=")[-1].rstrip(")"), "score": val,
                         "threshold": thr, "drifted": bool(drifted)}
    overall["dataset_drift"] = overall.get("share", 0.0) >= DRIFT_SHARE
    return {"overall": overall, "columns": cols}


def _print_summary(title, res, expected, shifted_features=None):
    o = res["overall"]
    verdict = "DATASET DRIFT DETECTED" if o["dataset_drift"] else "no dataset drift"
    print(f"\n──────── {title} ────────")
    print(f"  drifted columns: {o['drifted']}  share={o['share']:.3f}  "
          f"(threshold {DRIFT_SHARE}) → {verdict}   [expected: {expected}]")
    drifted_cols = sorted(c for c, v in res["columns"].items() if v["drifted"])
    if "prediction" in res["columns"]:
        p = res["columns"]["prediction"]
        print(f"  PREDICTION drift: {'YES' if p['drifted'] else 'no'} "
              f"(score={p['score']:.3g}, method={p['method']})")
    if shifted_features:
        caught = [f for f in shifted_features if res["columns"].get(f, {}).get("drifted")]
        print(f"  intentionally-shifted features: {shifted_features}")
        print(f"    → flagged by detector: {caught}  ({len(caught)}/{len(shifted_features)})")
    psis = res.get("top_feature_psi", {})
    if psis:
        hot = {f: v for f, v in psis.items() if v > 0.2}
        print(f"  top-SHAP-feature PSI (max={max(psis.values()):.3f}); PSI>0.2: "
              f"{ {f: v for f, v in sorted(hot.items(), key=lambda x: -x[1])} or 'none'}")
    print(f"  all drifted columns ({len(drifted_cols)}): {drifted_cols}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--age-shift", type=int, default=20, help="years added to age_midpoint")
    ap.add_argument("--diag1-circulatory-frac", type=float, default=0.40)
    ap.add_argument("--inpatient-bump", type=int, default=2)
    ap.add_argument("--emergency-frac", type=float, default=0.30)
    ap.add_argument("--current-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not REFERENCE.exists():
        sys.exit(f"reference missing: {REFERENCE} — run make_reference.py (and `dvc pull`) first")

    rng = np.random.default_rng(args.seed)
    reference = pd.read_parquet(REFERENCE)
    model = load_model()
    feature_names, cat_features = model_columns(model)
    print(f"[setup] reference rows={len(reference):,}  features={len(feature_names)} "
          f"({len(cat_features)} categorical)  drift_share threshold={DRIFT_SHARE}")

    base = build_current(model, feature_names, cat_features, args.current_n, rng)

    # CONTROL — unshifted, in-distribution. Must NOT fire (proves no false alarms).
    control = score(model, base, feature_names)
    res_control = run_report(reference, control, feature_names, cat_features,
                             REPORT_DIR / "drift_control.html")

    # SHIFTED — same rows, INTENTIONAL shifts, re-scored. Must fire.
    shifted_df, shifted_features = apply_shifts(
        base, rng, age_shift=args.age_shift, diag1_frac=args.diag1_circulatory_frac,
        inpatient_bump=args.inpatient_bump, emergency_frac=args.emergency_frac)
    shifted = score(model, shifted_df, feature_names)
    res_shift = run_report(reference, shifted, feature_names, cat_features,
                           REPORT_DIR / "drift_shifted.html")

    print(f"\n[scores] baseline mean={reference['prediction'].mean():.4f}  "
          f"control mean={control['prediction'].mean():.4f}  "
          f"shifted mean={shifted['prediction'].mean():.4f}")
    _print_summary("CONTROL (unshifted)", res_control, expected="no drift")
    _print_summary("SHIFTED (intentional)", res_shift, expected="DRIFT",
                   shifted_features=shifted_features)

    ok = (not res_control["overall"]["dataset_drift"]) and res_shift["overall"]["dataset_drift"]

    # Small, committable JSON summary (the heavy HTML is regenerable / gitignored).
    import json
    summary = {
        "reference": str(REFERENCE),
        "drift_share_threshold": DRIFT_SHARE,
        "current_n": int(len(base)),
        "score_means": {
            "baseline": round(float(reference["prediction"].mean()), 4),
            "control": round(float(control["prediction"].mean()), 4),
            "shifted": round(float(shifted["prediction"].mean()), 4),
        },
        "intentional_shifts": {
            "age_shift_years": args.age_shift,
            "diag1_circulatory_frac": args.diag1_circulatory_frac,
            "inpatient_bump": args.inpatient_bump,
            "emergency_frac": args.emergency_frac,
            "features_touched": shifted_features,
        },
        "control": res_control,
        "shifted": res_shift,
        "detector_validation_pass": bool(ok),
    }
    (REPORT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[detector validation] control silent AND shifted fired: "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"[reports] {REPORT_DIR}/drift_control.html  +  drift_shifted.html  "
          f"(+ committed drift_summary.json)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
