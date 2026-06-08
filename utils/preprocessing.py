"""
utils/preprocessing.py
======================
Data loading + preprocessing pipeline.

Responsibilities
----------------
1. Load the raw survey spreadsheet.
2. Construct the 5-band CGPA target from the current-CGPA column.
3. Remove duplicate and missing-target rows.
4. Select predictor features (FEATURE_COLS) — leakage/proxy columns are
   excluded by omission, since they are not part of FEATURE_COLS.
5. Build a deterministic sklearn ColumnTransformer that:
     - Ordinally encodes ordered survey responses (preserves their order).
     - One-hot encodes nominal categorical columns.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from config import (
    CGPA_COL, CGPA_TO_RISK, DATASET_PATH, FEATURE_COLS, LEAKAGE_COLS,
    NOMINAL_COLS, ORDINAL_ORDERINGS, TARGET_COL,
)


# ===========================================================================
# Data loading
# ===========================================================================
def load_raw(path=DATASET_PATH) -> pd.DataFrame:
    """Read the raw spreadsheet exactly as collected."""
    df = pd.read_excel(path)
    # Light normalisation for one ambiguous column observed in the data
    if "What devices do you use for studying?" in df.columns:
        df["What devices do you use for studying?"] = (
            df["What devices do you use for studying?"].astype(str).str.strip()
        )
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5-band CGPA target derived from the current-CGPA bucket.

    The mapping is defined in config.CGPA_TO_RISK and uses the exact
    labels: 'Below 1.50', '1.50 – 2.49', '2.50 – 3.49', '3.50 – 4.49',
    '4.50 – 5.00'."""
    df = df.copy()
    df[TARGET_COL] = df[CGPA_COL].map(CGPA_TO_RISK)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and any rows that lack a target."""
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns documented as target-leakage / proxies for CGPA.

    Not part of the default load_and_prepare flow — feature selection via
    FEATURE_COLS already excludes these columns. Provided for callers that
    want to drop them explicitly from a full dataframe."""
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns and c != CGPA_COL]
    # CGPA itself is kept until target is generated, then dropped here as well
    cols_to_drop = list(set(cols_to_drop + [CGPA_COL]))
    return df.drop(columns=cols_to_drop, errors="ignore")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ===========================================================================
# Pipeline construction
# ===========================================================================
def build_preprocessor() -> ColumnTransformer:
    """Build the deterministic sklearn ColumnTransformer."""
    ordinal_cols, ordinal_categories = [], []
    for col in FEATURE_COLS:
        if col in ORDINAL_ORDERINGS:
            ordinal_cols.append(col)
            ordinal_categories.append(ORDINAL_ORDERINGS[col])

    nominal_cols = [c for c in NOMINAL_COLS if c in FEATURE_COLS]

    transformers = []
    if ordinal_cols:
        transformers.append((
            "ordinal",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ordinal_cols,
        ))
    if nominal_cols:
        transformers.append((
            "nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            nominal_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable names for the transformed columns."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "ordinal":
                names.extend(cols)
            elif name == "nominal":
                try:
                    names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    names.extend(cols)
        return names


# ===========================================================================
# Convenience entry point
# ===========================================================================
def load_and_prepare(path=DATASET_PATH):
    """End-to-end: returns (X_df, y_series, preprocessor)."""
    df = load_raw(path)
    df = add_target(df)
    df = clean(df)
    X, y = split_features_target(df)
    preproc = build_preprocessor()
    return X, y, preproc