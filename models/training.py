"""
models/training.py
==================
End-to-end training orchestration for the Meta-Ensemble Pipeline.

Steps:
  1. Split data into train/test.
  2. Fit the preprocessor on training data only.
  3. Train the Meta-Ensemble Pipeline on the (un-resampled) training set —
     the ensemble performs its own leakage-free, per-fold SMOTE+Tomek
     resampling internally.
  4. Evaluate on the held-out test set.
  5. Persist the model, preprocessor, label encoder, and reports to disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    ENSEMBLE_STRATEGIES, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
    RESAMPLING_STRATEGIES, RISK_CLASSES, TEST_SIZE,
)
from models.base_models import make_rf, make_xgb
from models.ensembles import MetaEnsemblePipeline, build_ensemble
from utils.evaluation import (
    compute_metrics, cross_validate_model, make_confusion_matrix,
)
from utils.preprocessing import (
    build_preprocessor, get_feature_names, load_and_prepare,
)
from utils.resampling import apply_resampling, class_distribution


# ===========================================================================
# Helper
# ===========================================================================
def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


# ===========================================================================
# Main training routine
# ===========================================================================
def train_all(resampling: str = "SMOTE+Tomek",
              progress_cb=None) -> Dict:
    """
    Train the Meta-Ensemble Pipeline using the chosen resampling strategy
    and persist everything to ``saved_models/``.

    Parameters
    ----------
    resampling : str
        One of ``config.RESAMPLING_STRATEGIES``.
    progress_cb : callable, optional
        A callback ``(percent: int, message: str)`` invoked at each step.

    Returns
    -------
    dict
        A summary dictionary containing metrics and class distributions.
    """
    def tick(p, msg):
        if progress_cb:
            try:
                progress_cb(p, msg)
            except Exception:
                pass

    started = time.time()
    tick(2, "Loading data and target …")
    X, y, preproc = load_and_prepare()

    tick(8, "Encoding labels …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    tick(12, "Splitting train/test …")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE,
        stratify=y_enc, random_state=RANDOM_STATE,
    )

    tick(18, "Fitting preprocessing pipeline on training data …")
    X_train = preproc.fit_transform(X_train_raw)
    X_test  = preproc.transform(X_test_raw)
    feature_names = get_feature_names(preproc)

    if resampling not in ("None (Original)", "SMOTE+Tomek"):
        raise ValueError(
            "Resampling must be 'None (Original)' or 'SMOTE+Tomek' for this project."
        )

    # Distribution numbers for the UI report only. The ensemble does its own
    # leakage-free, per-fold resampling internally, so we DO NOT feed resampled
    # data into it — that would resample twice.
    before_dist = class_distribution([class_names[i] for i in y_train])
    tick(25, f"Resampling strategy: {resampling} (applied inside the ensemble) …")
    _, y_report_res = apply_resampling(X_train, y_train, resampling)
    after_dist = class_distribution([class_names[i] for i in y_report_res])

    results: List[Dict] = []
    trained_models: Dict = {}

    # ---------------------- Meta-Ensemble only ------------------------------
    tick(45, "Training Meta-Ensemble Pipeline …")
    meta_model = MetaEnsemblePipeline(resampling=resampling)
    meta_model.fit(X_train, y_train)   # raw train set; ensemble balances internally
    trained_models["Meta-Ensemble Pipeline"] = meta_model

    # ---------------------- evaluate every model ---------------------------
    tick(92, "Evaluating on the held-out test set …")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
        m = compute_metrics(y_test, y_pred, y_proba,
                            classes=np.arange(len(class_names)))
        m["model"] = name
        m["category"] = "Base" if name in ("Random Forest", "XGBoost") else "Ensemble"
        results.append(m)

    # ---------------------- persist artefacts ------------------------------
    tick(96, "Saving models, pipeline, and reports …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preproc, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    for name, mdl in trained_models.items():
        joblib.dump(mdl, MODELS_DIR / f"{_slug(name)}.joblib")

    # Save training meta-info
    meta_info = {
        "resampling": resampling,
        "feature_names": feature_names,
        "class_names": class_names,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 2),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_resampled": int(len(y_report_res)),
        "before_resampling": before_dist,
        "after_resampling": after_dist,
    }
    with open(MODELS_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    # Save comparison report
    comp_df = pd.DataFrame(results)[
        ["model", "category", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].sort_values("f1", ascending=False).reset_index(drop=True)
    comp_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Save per-model confusion matrices (so the UI doesn't have to rerun)
    cms = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        cm = make_confusion_matrix(
            [class_names[i] for i in y_test],
            [class_names[i] for i in y_pred],
            labels=RISK_CLASSES,
        )
        cms[name] = cm.values.tolist()
    with open(REPORTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)

    tick(100, "Done.")

    return {
        "comparison": comp_df,
        "meta": meta_info,
        "confusion_matrices": cms,
    }


def list_trained_models() -> List[str]:
    """Names of models present on disk."""
    if not MODELS_DIR.exists():
        return []
    out = []
    for p in MODELS_DIR.glob("*.joblib"):
        n = p.stem
        if n in ("preprocessor", "label_encoder"):
            continue
        out.append(n)
    return out