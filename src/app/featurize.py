"""featurize.py — turn ONE raw discharge-time record into the model feature row.

THE anti-skew module. It does not reimplement any feature logic; it replays the
exact training path on a single-row frame:

    raw record  →  replace "?" with NaN   (mirrors pd.read_csv(na_values=["?"]))
                →  clean_columns(...)      (reused from src/data/clean.py)
                →  engineer_features(...)  (reused from src/features/build_features.py)

The result is reindexed to the model's own feature_names_ in model.py, so the
served record is engineered byte-for-byte the way training was. If this file ever
needed feature logic of its own, that would BE the skew — hence it has none.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put repo/src on path
from data.clean import clean_columns  # noqa: E402
from features.build_features import engineer_features  # noqa: E402

MISSING_TOKEN = "?"


def featurize_record(record: dict) -> pd.DataFrame:
    """One raw record (dict of the 44 input fields) → 1-row engineered DataFrame.

    Missing values may arrive as the "?" token or as None/null; both are normalized
    to NaN so clean_columns fills them exactly as the training load did. Genuine
    "None" lab values are NOT touched here (they are a real level, not missing).
    """
    df = pd.DataFrame([record])
    df = df.replace({MISSING_TOKEN: np.nan, None: np.nan})
    df = clean_columns(df)
    df = engineer_features(df)
    return df
