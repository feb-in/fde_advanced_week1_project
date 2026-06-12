"""Cross-cutting analyses for the deeper dashboard tabs.

`profile.py` answers "what is each column?"; this module answers questions that
span columns — correlation, feature↔target association, the ICD-9 diagnosis
landscape, the medication landscape, demographic subgroup risk, and the
patient-grouping / leakage structure. Pure functions over a DataFrame that
already carries the binary target (``profile.add_binary_target``); the dashboard
caches them. Every statistic is computed here, never in the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappings  # noqa: E402
import profile  # noqa: E402

TARGET = profile.TARGET_BINARY
NUMERIC_COLS = list(mappings.UTILIZATION_NUMERIC_COLS)

AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
_AGE_INDEX = {a: i for i, a in enumerate(AGE_ORDER)}


def age_to_ordinal(s: pd.Series) -> pd.Series:
    """Map the decade-bucket age string to an ordinal 0–9 (NaN if unknown)."""
    return s.map(_AGE_INDEX)


# --------------------------------------------------------------------------- #
# Correlation & association
# --------------------------------------------------------------------------- #

def numeric_correlation(df: pd.DataFrame, method: str = "pearson") -> dict:
    """Correlation matrix across the 8 utilization numerics + ordinal age + target."""
    work = df[NUMERIC_COLS].copy()
    work["age_ordinal"] = age_to_ordinal(df["age"])
    work[TARGET] = df[TARGET]
    corr = work.corr(method=method)
    return {
        "columns": list(corr.columns),
        "matrix": np.round(corr.values, 3).tolist(),
        "method": method,
    }


def top_correlated_pairs(corr: dict, k: int = 8) -> list[dict]:
    """Strongest off-diagonal absolute correlations from a numeric_correlation dict."""
    cols, m = corr["columns"], np.array(corr["matrix"])
    seen, out = set(), []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out.append({"a": cols[i], "b": cols[j], "corr": float(m[i, j])})
    out.sort(key=lambda r: -abs(r["corr"]))
    return out[:k]


def cramers_v(ct: np.ndarray | pd.DataFrame) -> float:
    """Bias-corrected Cramér's V for a contingency table (0 = none, 1 = perfect)."""
    arr = ct.values if hasattr(ct, "values") else np.asarray(ct)
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(arr, correction=False)[0]
    n = arr.sum()
    if n == 0:
        return 0.0
    r, k = arr.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def feature_target_association(df: pd.DataFrame, max_levels: int = 30) -> list[dict]:
    """Rank every feature by its association with the binary target.

    Numeric columns → |point-biserial r| (Pearson against the 0/1 target).
    Categorical / coded-id columns → bias-corrected Cramér's V (high-cardinality
    columns are first collapsed to their top ``max_levels`` levels + «other»).
    The two metrics share the 0–1 scale, so they rank comparably.
    """
    y = df[TARGET]
    rows: list[dict] = []
    for col in df.columns:
        if col in (profile.TARGET_RAW, TARGET) or col in profile.KEY_COLS:
            continue
        s = df[col]
        if s.nunique(dropna=True) <= 1:
            continue
        numeric = pd.api.types.is_numeric_dtype(s) and col not in mappings.ADMINISTRATIVE_ID_COLS
        if numeric:
            r = df[[col]].assign(_y=y).corr().iloc[0, 1]
            rows.append({"feature": col, "assoc": abs(float(r)),
                         "metric": "|point-biserial r|", "kind": "numeric"})
        else:
            ss = s.astype("object").where(s.notna(), "«missing»").astype(str)
            if ss.nunique() > max_levels:
                top = ss.value_counts().index[:max_levels]
                ss = ss.where(ss.isin(top), "«other»")
            v = cramers_v(pd.crosstab(ss, y))
            rows.append({"feature": col, "assoc": float(v),
                         "metric": "Cramér's V", "kind": "categorical"})
    rows.sort(key=lambda r: -r["assoc"])
    return rows


# --------------------------------------------------------------------------- #
# Diagnoses (ICD-9)
# --------------------------------------------------------------------------- #

def diagnosis_chapter_table(df: pd.DataFrame, which: str = "diag_1") -> list[dict]:
    """Frequency + readmit rate of each ICD-9 chapter for one diagnosis position."""
    ch = df[which].map(mappings.icd9_to_chapter)
    y = df[TARGET]
    base = float(y.mean())
    n = len(df)
    rows = []
    for chap in mappings.ICD9_CHAPTER_ORDER:
        mask = (ch == chap).values
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        rate = float(y[mask].mean())
        rows.append({
            "chapter": chap, "count": cnt, "pct": round(100 * cnt / n, 2),
            "positive_rate": round(rate, 5),
            "lift": round(rate / base, 3) if base else None,
        })
    return rows


def diabetes_anywhere(df: pd.DataFrame) -> dict:
    """Readmit rate split by whether a diabetes (250.xx) code appears in any diag."""
    chs = pd.concat([df[c].map(mappings.icd9_to_chapter) for c in mappings.DIAGNOSIS_COLS], axis=1)
    has = chs.eq("Diabetes").any(axis=1)
    y = df[TARGET]
    return {
        "base": round(float(y.mean()), 5),
        "with": {"count": int(has.sum()), "rate": round(float(y[has].mean()), 5)},
        "without": {"count": int((~has).sum()), "rate": round(float(y[~has].mean()), 5)},
    }


def comorbidity_vs_target(df: pd.DataFrame) -> list[dict]:
    """Readmit rate by number_diagnoses (comorbidity load)."""
    y = df[TARGET]
    out = []
    for k, sub in y.groupby(df["number_diagnoses"]):
        if sub.shape[0] < 30:
            continue
        out.append({"n_diagnoses": int(k), "count": int(sub.shape[0]),
                    "positive_rate": round(float(sub.mean()), 5)})
    return out


# --------------------------------------------------------------------------- #
# Medications
# --------------------------------------------------------------------------- #

def medication_landscape(df: pd.DataFrame) -> list[dict]:
    """Per-drug dose-change mix, active %, near-constant flag, and active readmit rate."""
    y = df[TARGET]
    base = float(y.mean())
    n = len(df)
    rows = []
    for drug in mappings.MEDICATION_COLS:
        s = df[drug]
        vc = s.value_counts()
        no = int(vc.get("No", 0))
        active = n - no
        changed = (s != "No")
        rows.append({
            "drug": drug,
            "pct_active": round(100 * active / n, 3),
            "No": no, "Down": int(vc.get("Down", 0)),
            "Steady": int(vc.get("Steady", 0)), "Up": int(vc.get("Up", 0)),
            "n_levels": int(s.nunique()),
            "near_constant": bool(active / n < 0.001),
            "rate_active": round(float(y[changed].mean()), 5) if active else None,
            "base": round(base, 5),
        })
    rows.sort(key=lambda r: -r["pct_active"])
    return rows


def medication_signal(df: pd.DataFrame) -> dict:
    """Readmit rate by the high-signal medication indicators + active-med count."""
    y = df[TARGET]
    out: dict = {}
    for col in ["change", "diabetesMed", "insulin"]:
        out[col] = [
            {"level": str(k), "count": int(v.shape[0]),
             "positive_rate": round(float(v.mean()), 5)}
            for k, v in y.groupby(df[col])
        ]
    active_count = (df[mappings.MEDICATION_COLS] != "No").sum(axis=1).clip(upper=6)
    out["n_active_meds"] = [
        {"n": int(k), "count": int(v.shape[0]), "positive_rate": round(float(v.mean()), 5)}
        for k, v in y.groupby(active_count) if v.shape[0] >= 30
    ]
    out["base"] = round(float(y.mean()), 5)
    return out


# --------------------------------------------------------------------------- #
# Demographic subgroups (fairness preview)
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def subgroup_rates(df: pd.DataFrame, col: str) -> dict:
    """Readmit rate + Wilson CI per level of a demographic column."""
    y = df[TARGET]
    s = df[col].astype("object").where(df[col].notna(), "«missing»").astype(str)
    rows = []
    for level, sub in y.groupby(s):
        n = int(sub.shape[0])
        k = int(sub.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"level": str(level), "count": n,
                     "positive_rate": round(k / n, 5) if n else 0.0,
                     "ci_low": round(lo, 5), "ci_high": round(hi, 5)})
    if col == "age":
        rows.sort(key=lambda r: _AGE_INDEX.get(r["level"], 99))
    else:
        rows.sort(key=lambda r: -r["count"])
    return {"base": round(float(y.mean()), 5), "rows": rows}


# --------------------------------------------------------------------------- #
# Patient grouping & leakage
# --------------------------------------------------------------------------- #

def patient_encounter_distribution(df: pd.DataFrame) -> dict:
    """How many patients have 1, 2, 3, 4, 5+ encounters."""
    vc = df["patient_nbr"].value_counts()
    capped = vc.clip(upper=5)
    dist = capped.value_counts().sort_index()
    rows = []
    for k in sorted(dist.index):
        label = "5+" if k == 5 else str(int(k))
        rows.append({"encounters": label, "n_patients": int(dist[k])})
    return {"rows": rows, "n_patients": int(vc.shape[0]), "n_rows": int(len(df))}


def leakage_overlap(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> dict:
    """Quantify patient leakage from a *random row* split vs a grouped split.

    Demonstrates rule 2: under a naive random split, a large share of test rows
    belong to patients also present in train — the model sees them twice.
    """
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    train_pat = set(df["patient_nbr"].iloc[idx[:cut]])
    test = df["patient_nbr"].iloc[idx[cut:]]
    test_pat = set(test)
    leaked_rows = float(test.isin(train_pat).mean())
    leaked_pat = len(test_pat & train_pat)
    return {
        "test_frac": test_frac,
        "n_test_rows": int(test.shape[0]),
        "n_test_patients": len(test_pat),
        "n_leaked_patients": leaked_pat,
        "pct_test_rows_leaked": round(100 * leaked_rows, 2),
        "pct_test_patients_leaked": round(100 * leaked_pat / max(1, len(test_pat)), 2),
    }


def prior_visit_signal(df: pd.DataFrame) -> dict:
    """Readmit rate by prior inpatient / emergency / outpatient visit counts."""
    y = df[TARGET]
    out = {}
    for col in ["number_inpatient", "number_emergency", "number_outpatient"]:
        capped = df[col].clip(upper=3)
        out[col] = [
            {"bucket": ("3+" if k == 3 else str(int(k))),
             "count": int(v.shape[0]), "positive_rate": round(float(v.mean()), 5)}
            for k, v in y.groupby(capped)
        ]
    out["base"] = round(float(y.mean()), 5)
    return out
