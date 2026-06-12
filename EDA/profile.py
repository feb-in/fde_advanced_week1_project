"""Analytical EDA engine for the diabetes readmission dataset.

Every number the dashboard shows comes from here — there is no hand-typed stat
anywhere in the UI. The engine is importable and re-runnable (CLAUDE.md §3:
"all real logic lives in src/ as importable, re-runnable scripts"), and it is
deliberately framework-agnostic: Streamlit only *renders* what these functions
return, so the same facts can be dumped to JSON for docs or other consumers.

Key modelling-aware choices baked in:
  * load with ``na_values="?"`` so the sentinel never becomes a real category
    (hard rule 1);
  * the target is collapsed to binary ``readmitted_30d`` = (readmitted == "<30")
    (hard rule 7), and every categorical level reports its *positive rate*
    against that binary target so you can see signal, not just counts;
  * coded ``*_id`` columns are decoded to labels;
  * missingness is treated as a first-class signal (per-column %, and a
    co-missing correlation matrix to reveal structural / MNAR patterns).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Flat, self-contained package: make this folder importable whether run as
# `python EDA/profile.py` or imported by `streamlit run EDA/dashboard.py`.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappings  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent           # EDA/
PROJECT_ROOT = HERE.parent                        # repo root
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
ARTIFACTS = HERE / "artifacts"
FACTS_JSON = ARTIFACTS / "eda_facts.json"

TARGET_RAW = "readmitted"
TARGET_BINARY = "readmitted_30d"
POSITIVE_LABEL = "<30"

# Columns we never profile as features (pure row/patient keys).
KEY_COLS = ("encounter_id", "patient_nbr")

# Above this many distinct values a column is treated as "high cardinality":
# we summarise the top-K levels instead of every level.
HIGH_CARD_THRESHOLD = 30


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV with the dataset's sentinel decoded to NaN.

    ``na_values="?"`` is non-negotiable (hard rule 1): the source encodes missing
    values as a literal ``?`` and we must not let it survive as a category.
    """
    df = pd.read_csv(path, na_values="?", low_memory=False)
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary 30-day-readmission target (hard rule 7) without mutating input."""
    out = df.copy()
    out[TARGET_BINARY] = (out[TARGET_RAW] == POSITIVE_LABEL).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Profile data structures
# --------------------------------------------------------------------------- #

@dataclass
class LevelStat:
    """One categorical level: how common it is and how risky it is."""
    label: str
    count: int
    pct: float
    positives: int
    positive_rate: float  # P(readmit<30 | level)


@dataclass
class ColumnProfile:
    name: str
    family: str
    dtype: str
    inferred_role: str            # numeric | categorical | binary_flag | high_card_cat | coded_id | constant | identifier
    n: int
    n_missing: int
    pct_missing: float
    n_unique: int
    is_constant: bool
    # categorical view
    levels: list[LevelStat] = field(default_factory=list)
    n_levels_shown: int = 0
    n_levels_total: int = 0
    # numeric view
    numeric_summary: dict[str, float] | None = None
    numeric_bins: list[dict[str, Any]] | None = None  # readmit rate by quantile bin
    # association with target
    target_lift: float | None = None  # max positive_rate / base_rate across levels
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Role inference
# --------------------------------------------------------------------------- #

def infer_role(df: pd.DataFrame, col: str) -> str:
    if col in KEY_COLS:
        return "identifier"
    if col in mappings.ADMINISTRATIVE_ID_COLS:
        return "coded_id"
    s = df[col]
    nun = s.nunique(dropna=True)
    if nun <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if nun == 2:
        return "binary_flag"
    if nun > HIGH_CARD_THRESHOLD:
        return "high_card_cat"
    return "categorical"


# --------------------------------------------------------------------------- #
# Per-column profiling
# --------------------------------------------------------------------------- #

def _level_stats(
    series: pd.Series,
    target: pd.Series,
    base_rate: float,
    decode: dict[int, str] | None = None,
    top_k: int = HIGH_CARD_THRESHOLD,
) -> tuple[list[LevelStat], int]:
    """Per-level count + positive rate, sorted by frequency, capped at top_k."""
    counts = series.value_counts(dropna=True)
    total_total = len(series)
    n_total_levels = counts.shape[0]
    grp = target.groupby(series, observed=True)
    pos = grp.sum()
    levels: list[LevelStat] = []
    for raw_val, cnt in counts.head(top_k).items():
        label = decode.get(int(raw_val), str(raw_val)) if decode else str(raw_val)
        p = int(pos.get(raw_val, 0))
        levels.append(
            LevelStat(
                label=label,
                count=int(cnt),
                pct=round(100 * cnt / total_total, 3),
                positives=p,
                positive_rate=round(p / cnt, 5) if cnt else 0.0,
            )
        )
    return levels, n_total_levels


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(s.min()),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.50]),
        "mean": float(s.mean()),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(s.max()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "zeros_pct": round(100 * float((s == 0).mean()), 2),
    }


def _numeric_bins(series: pd.Series, target: pd.Series, n_bins: int = 10) -> list[dict[str, Any]]:
    """Readmit positive-rate across (up to) decile bins of a numeric column."""
    s = series
    mask = s.notna()
    s, t = s[mask], target[mask]
    try:
        binned = pd.qcut(s, q=n_bins, duplicates="drop")
    except (ValueError, IndexError):
        return []
    out = []
    grp = t.groupby(binned, observed=True)
    for interval, sub in grp:
        cnt = int(sub.shape[0])
        if cnt == 0:
            continue
        out.append(
            {
                "bin": str(interval),
                "left": float(interval.left),
                "right": float(interval.right),
                "count": cnt,
                "positive_rate": round(float(sub.mean()), 5),
            }
        )
    return out


def profile_column(df: pd.DataFrame, col: str, base_rate: float) -> ColumnProfile:
    s = df[col]
    target = df[TARGET_BINARY]
    role = infer_role(df, col)
    n = len(s)
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))

    prof = ColumnProfile(
        name=col,
        family=mappings.family_of(col),
        dtype=str(s.dtype),
        inferred_role=role,
        n=n,
        n_missing=n_missing,
        pct_missing=round(100 * n_missing / n, 3),
        n_unique=n_unique,
        is_constant=(n_unique <= 1),
    )

    if role == "constant":
        prof.notes.append("Zero variance — single value. Drop before modelling.")
        only = s.dropna().unique()
        prof.levels = [LevelStat(str(only[0]) if len(only) else "<all-missing>", n - n_missing,
                                 round(100 * (n - n_missing) / n, 3), 0, 0.0)]
        return prof

    if role == "identifier":
        prof.notes.append("Row/patient key — not a feature. patient_nbr drives the GROUP split.")
        return prof

    if role == "numeric":
        prof.numeric_summary = _numeric_summary(s)
        prof.numeric_bins = _numeric_bins(s, target)
        if prof.numeric_bins:
            rates = [b["positive_rate"] for b in prof.numeric_bins]
            prof.target_lift = round(max(rates) / base_rate, 3) if base_rate else None
        return prof

    # categorical / binary / high-card / coded_id
    decode = mappings.ID_DECODERS.get(col) if role == "coded_id" else None
    levels, n_total = _level_stats(s, target, base_rate, decode=decode)
    prof.levels = levels
    prof.n_levels_shown = len(levels)
    prof.n_levels_total = n_total
    if levels:
        # lift = how much the riskiest reasonably-populated level beats base rate
        eligible = [lv for lv in levels if lv.count >= 30]
        if eligible:
            prof.target_lift = round(max(lv.positive_rate for lv in eligible) / base_rate, 3) if base_rate else None
    if role == "high_card_cat":
        prof.notes.append(f"High cardinality ({n_total} levels); showing top {len(levels)}.")
    return prof


# --------------------------------------------------------------------------- #
# Dataset-level facts
# --------------------------------------------------------------------------- #

def target_facts(df: pd.DataFrame) -> dict[str, Any]:
    raw = df[TARGET_RAW].value_counts(dropna=False)
    n = len(df)
    base_rate = float((df[TARGET_RAW] == POSITIVE_LABEL).mean())
    return {
        "n_rows": int(n),
        "raw_distribution": {str(k): int(v) for k, v in raw.items()},
        "binary_positive_label": POSITIVE_LABEL,
        "binary_positives": int((df[TARGET_RAW] == POSITIVE_LABEL).sum()),
        "base_rate": round(base_rate, 5),
        "majority_baseline_accuracy": round(1 - base_rate, 5),
    }


def grouping_facts(df: pd.DataFrame) -> dict[str, Any]:
    vc = df["patient_nbr"].value_counts()
    return {
        "n_rows": int(len(df)),
        "n_unique_patients": int(df["patient_nbr"].nunique()),
        "n_patients_multi_encounter": int((vc > 1).sum()),
        "max_encounters_per_patient": int(vc.max()),
        "pct_rows_from_repeat_patients": round(100 * float((vc[vc > 1].sum()) / len(df)), 2),
        "leakage_note": (
            "A random row split would put the SAME patient in train and test "
            "(hard rule 2). Split with GroupShuffleSplit on patient_nbr."
        ),
    }


def discharge_filter_facts(df: pd.DataFrame) -> dict[str, Any]:
    col = "discharge_disposition_id"
    n = len(df)
    expired = df[col].isin(mappings.EXPIRED_DISPOSITION_IDS)
    hospice = df[col].isin(mappings.HOSPICE_DISPOSITION_IDS)
    drop = df[col].isin(mappings.DROP_DISPOSITION_IDS)
    breakdown = []
    for code in mappings.DROP_DISPOSITION_IDS:
        cnt = int((df[col] == code).sum())
        breakdown.append({
            "code": int(code),
            "label": mappings.DISCHARGE_DISPOSITION_ID.get(code, str(code)),
            "kind": "expired" if code in mappings.EXPIRED_DISPOSITION_IDS else "hospice",
            "count": cnt,
            "pct": round(100 * cnt / n, 3),
        })
    return {
        "n_expired_rows": int(expired.sum()),
        "n_hospice_rows": int(hospice.sum()),
        "n_drop_rows": int(drop.sum()),
        "pct_drop": round(100 * float(drop.mean()), 3),
        "rows_after_filter": int(n - drop.sum()),
        "breakdown": breakdown,
        "rule": (
            "Hard rule 5: expired/hospice patients cannot be readmitted — drop "
            "them so the target is well-defined and the model is not trained on "
            "structurally-impossible negatives."
        ),
    }


def co_missing_matrix(df: pd.DataFrame, min_pct: float = 1.0) -> dict[str, Any]:
    """Correlation of missingness indicators across columns with >= min_pct missing.

    A high positive correlation between two columns' missingness suggests a
    *structural* (not random) cause — e.g. fields collected by the same form
    section — which is evidence for MAR/MNAR rather than MCAR.
    """
    miss = df.isna()
    cols = [c for c in df.columns if 100 * miss[c].mean() >= min_pct]
    if len(cols) < 2:
        return {"columns": cols, "matrix": []}
    corr = miss[cols].astype(int).corr().round(3)
    return {
        "columns": cols,
        "missing_pct": {c: round(100 * float(miss[c].mean()), 3) for c in cols},
        "matrix": corr.values.tolist(),
    }


def compute_facts(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Full EDA fact pack: target, grouping, discharge filter, per-column profiles."""
    if df is None:
        df = load_raw()
    df = add_binary_target(df)
    base_rate = float(df[TARGET_BINARY].mean())

    profiles: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if col in (TARGET_RAW, TARGET_BINARY):
            continue
        profiles[col] = profile_column(df, col, base_rate).to_dict()

    return {
        "schema": {
            "n_rows": int(len(df)),
            "n_columns_raw": int(len([c for c in df.columns if c != TARGET_BINARY])),
            "base_rate": round(base_rate, 5),
        },
        "target": target_facts(df),
        "grouping": grouping_facts(df),
        "discharge_filter": discharge_filter_facts(df),
        "co_missing": co_missing_matrix(df),
        "columns": profiles,
    }


def dump_facts(path: str | Path = FACTS_JSON) -> Path:
    """Compute facts from the raw CSV and write them to JSON (for docs / agents)."""
    facts = compute_facts()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)
    return path


if __name__ == "__main__":
    out = dump_facts()
    print(f"Wrote EDA facts -> {out}")
