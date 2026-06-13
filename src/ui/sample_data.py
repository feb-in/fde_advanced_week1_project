"""sample_data.py — UI convenience: pull a random REAL patient from the HELD-OUT test set.

READ-ONLY, DISPLAY-ONLY. This reads local data files to (a) fill the form with a real
patient's raw fields and (b) surface that patient's TRUE outcome for a right-vs-wrong
demo. It NEVER scores — scoring still goes through the API's /predict. Keeping this out
of the scoring path preserves the "thin client, no local model" rule.

The patients come from the **seed-42 held-out test split** — rows the model never trained
on (the same split as src/models/train.py::prepare_data, reproduced here without importing
the training stack). The raw field values come from the raw CSV (joined by encounter_id);
the true label is `readmitted` mapped to the binary target (<30 → 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FEATURIZED = ROOT / "data/featurized/diabetes_features.parquet"
RAW = ROOT / "data/raw/diabetic_data.csv"
CONTRACT = json.loads((ROOT / "src/contracts/input_contract.json").read_text())
FIELD_NAMES = list(CONTRACT["fields"].keys())

TEST_SIZE = 0.20
SEED = 42  # MUST match src/models/train.py — identical held-out test split

_cache: dict = {}


def _native(v):
    """numpy/pandas scalar → JSON-friendly python; missing (NaN) → None."""
    try:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def test_encounter_ids() -> set:
    """Reproduce the seed-42 stratified test split → the set of test encounter_ids."""
    feat = pd.read_parquet(FEATURIZED, columns=["encounter_id", "target"])
    y = feat["target"].astype(int).to_numpy()
    _, idx_test = train_test_split(
        np.arange(len(feat)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    return set(feat.iloc[idx_test]["encounter_id"].tolist())


def _test_rows() -> pd.DataFrame:
    """Raw CSV rows for the held-out test patients (cached for the session)."""
    if "rows" not in _cache:
        eids = test_encounter_ids()
        raw = pd.read_csv(RAW, na_values=["?"], keep_default_na=False, low_memory=False)
        _cache["rows"] = raw[raw["encounter_id"].isin(eids)].reset_index(drop=True)
    return _cache["rows"]


def random_test_patient(random_state=None):
    """Return (record, truth) for one random held-out patient.

    record: dict of the 44 raw contract fields (missing → None).
    truth:  {encounter_id, readmitted_raw, label (<30→1), was_readmitted_30d}.
    """
    rows = _test_rows()
    row = rows.sample(n=1, random_state=random_state).iloc[0]
    record = {name: _native(row[name]) for name in FIELD_NAMES}
    readmitted = row["readmitted"]
    return record, {
        "encounter_id": int(row["encounter_id"]),
        "readmitted_raw": readmitted,
        "label": 1 if readmitted == "<30" else 0,
        "was_readmitted_30d": readmitted == "<30",
    }
