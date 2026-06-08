"""
utils/evaluation.py
===================
Centralised metric computations.

All averaged metrics use MACRO averaging (every class weighted equally),
which is the appropriate choice for this imbalanced 5-class problem: it
prevents the populous middle CGPA bands from masking performance on the
small but important minority bands (e.g. "Below 1.50").
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import CV_FOLDS, RISK_CLASSES


def safe_roc_auc(y_true, y_proba, classes=None) -> float:
    """Multi-class ROC-AUC (OvR, macro) with safe fallback for tiny test sets."""
    try:
        if classes is None:
            classes = np.unique(y_true)
        return roc_auc_score(y_true, y_proba, multi_class="ovr",
                             labels=classes, average="macro")
    except Exception:
        return float("nan")


def compute_metrics(y_true, y_pred, y_proba=None,
                    classes=None) -> Dict[str, float]:
    """Standard classification metrics (macro-averaged — see module docstring)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc":   safe_roc_auc(y_true, y_proba, classes) if y_proba is not None
                     else float("nan"),
    }


def make_confusion_matrix(y_true, y_pred, labels=None) -> pd.DataFrame:
    if labels is None:
        labels = RISK_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"actual: {l}" for l in labels],
                        columns=[f"pred: {l}" for l in labels])


def make_classification_report(y_true, y_pred, labels=None) -> pd.DataFrame:
    if labels is None:
        labels = RISK_CLASSES
    rep = classification_report(y_true, y_pred, labels=labels,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def cross_validate_model(model, X, y, scoring="accuracy", n_splits=CV_FOLDS):
    """Simple stratified k-fold cross-validation."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "n_splits": n_splits, "scores": scores.tolist()}