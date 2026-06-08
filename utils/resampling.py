"""
utils/resampling.py
===================
Class-imbalance handling.

IMPORTANT: All resamplers are applied to the TRAINING set only, AFTER the
train-test split, so the test set distribution remains untouched and there
is no leakage from the synthetic samples.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE


def _safe_k_neighbors(y) -> int:
    """SMOTE requires k <= (smallest class count - 1). Pick a safe value."""
    _, counts = np.unique(y, return_counts=True)
    return max(1, min(5, int(counts.min()) - 1))


def apply_resampling(X_train, y_train, strategy: str
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a resampling strategy to the training set only.

    Parameters
    ----------
    X_train, y_train : training features and labels (already preprocessed)
    strategy : one of {"None (Original)", "SMOTE", "SMOTE-ENN",
                       "Borderline-SMOTE"}

    Returns
    -------
    (X_resampled, y_resampled)
    """
    strategy = (strategy or "None (Original)").strip()

    if strategy == "None (Original)":
        return np.asarray(X_train), np.asarray(y_train)

    k = _safe_k_neighbors(y_train)

    if strategy == "SMOTE":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    elif strategy == "SMOTE+Tomek":
        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=k),
        )
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def class_distribution(y) -> dict:
    """Return a {class: count} dict for quick reporting."""
    values, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}
