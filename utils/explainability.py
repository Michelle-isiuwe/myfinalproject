"""
utils/explainability.py
=======================
SHAP and LIME helpers for explaining predictions of any trained model in
this project. TreeExplainer is used for tree-based models (RF, XGBoost)
for speed; KernelExplainer is the fallback for other models.

For the Meta-Ensemble Pipeline (a stacked blend), SHAP explanations are
computed on its inner Random Forest base learner — exposed via the model's
explainer_model() method — which is fast (TreeExplainer) and a faithful
proxy for feature direction on the final prediction. LIME always runs
against the full model's predict_proba (it is model-agnostic).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# SHAP & LIME are imported lazily because they are heavy
try:
    import shap
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_OK = True
except Exception:
    _LIME_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier as _RFC
    _RF_TYPE = _RFC
except Exception:
    _RF_TYPE = None

try:
    from xgboost import XGBClassifier as _XGBC
    _XGB_TYPE = _XGBC
except Exception:
    _XGB_TYPE = None


def _is_tree_model(model) -> bool:
    """Return True if model is a native tree model (RF or XGBoost)."""
    if _RF_TYPE is not None and isinstance(model, _RF_TYPE):
        return True
    if _XGB_TYPE is not None and isinstance(model, _XGB_TYPE):
        return True
    return False


def _shap_target(model):
    """Unwrap a wrapper model (e.g. Meta-Ensemble) to the inner tree used for
    SHAP. Falls back to the model itself if no explainer_model() is exposed."""
    if hasattr(model, "explainer_model"):
        try:
            return model.explainer_model()
        except Exception:
            return model
    return model


# ===========================================================================
# Background data (used by both SHAP and LIME)
# ===========================================================================
def _background_sample(preprocessor, X_train_df: pd.DataFrame,
                       max_samples: int = 100) -> np.ndarray:
    """Transform a sample of the training data for use as background."""
    n = min(max_samples, len(X_train_df))
    sample = X_train_df.sample(n=n, random_state=42) if n > 0 else X_train_df
    return preprocessor.transform(sample)


# ===========================================================================
# SHAP
# ===========================================================================
def shap_explain(model, X_row: np.ndarray, background: np.ndarray,
                 feature_names: List[str], class_index: int = 0,
                 class_names: List[str] | None = None) -> Dict:
    """
    Produce SHAP explanation for a single prediction.

    Uses TreeExplainer for RF/XGBoost (and for the Meta-Ensemble's inner RF,
    exposed via explainer_model()); falls back to KernelExplainer otherwise.

    Returns a dict with:
        - 'shap_values'    : 1-D array of SHAP values for the predicted class
        - 'base_value'     : expected value of the predicted class
        - 'feature_values' : 1-D array of feature values
        - 'feature_names'  : list of feature names
    """
    if not _SHAP_OK:
        raise RuntimeError("SHAP is not installed.")

    n_features = len(feature_names)
    X_row_2d = np.asarray(X_row).reshape(1, -1)

    # Unwrap to the inner tree for wrapper models (e.g. Meta-Ensemble).
    explain_target = _shap_target(model)

    if _is_tree_model(explain_target):
        # --- TreeExplainer (fast path for RF and XGBoost) ---
        explainer = shap.TreeExplainer(explain_target)
        shap_values = explainer.shap_values(X_row_2d)
        base_value = explainer.expected_value

        # RF returns list[(1, n_features)] one per class
        # XGBoost multiclass may return (1, n_features, n_classes) or list
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[class_index]).reshape(-1)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.asarray(base_value).reshape(-1)[class_index]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # shape (1, n_features, n_classes) or (n_classes, 1, n_features)
                if arr.shape[1] == n_features:
                    sv = arr[0, :, class_index]
                else:
                    sv = arr[class_index, 0, :]
            elif arr.ndim == 2:
                sv = arr[0]
            else:
                sv = arr.reshape(-1)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.asarray(base_value).reshape(-1)[class_index]
    else:
        # --- KernelExplainer (fallback for non-tree models) ---
        f = lambda data: explain_target.predict_proba(data)
        bg = shap.sample(background, min(50, len(background)), random_state=42)
        explainer = shap.KernelExplainer(f, bg)
        shap_values = explainer.shap_values(X_row_2d, nsamples=100, silent=True)
        base_value = explainer.expected_value

        # Possible shapes: list[(1,n_features)], (1,n_feat,n_cls), (n_cls,1,n_feat)
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[class_index]).reshape(-1)
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                if arr.shape == (1, n_features, len(class_names or [])) or \
                   (arr.shape[0] == 1 and arr.shape[1] == n_features):
                    sv = arr[0, :, class_index]
                elif arr.shape[-1] == n_features:
                    sv = arr[class_index, 0, :]
                else:
                    sv = arr[0, :, class_index]
            elif arr.ndim == 2:
                sv = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
            else:
                sv = arr.reshape(-1)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[class_index]

    sv = np.asarray(sv).reshape(-1)
    if sv.shape[0] != n_features:
        sv = np.resize(sv, n_features)

    return {
        "shap_values":    np.asarray(sv).astype(float),
        "base_value":     float(base_value),
        "feature_values": np.asarray(X_row).reshape(-1),
        "feature_names":  feature_names,
        "class_index":    class_index,
        "class_name":     class_names[class_index] if class_names else str(class_index),
    }


def global_feature_importance(model, X_background: np.ndarray,
                              feature_names: List[str],
                              max_samples: int = 80) -> pd.DataFrame:
    """Mean |SHAP| per feature across a background sample."""
    explain_target = _shap_target(model)

    if not _SHAP_OK:
        if hasattr(explain_target, "feature_importances_"):
            fi = explain_target.feature_importances_
            return (pd.DataFrame({"feature": feature_names, "importance": fi})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        return pd.DataFrame(columns=["feature", "importance"])

    bg = X_background[:min(max_samples, len(X_background))]

    if _is_tree_model(explain_target):
        explainer = shap.TreeExplainer(explain_target)
        sv = explainer.shap_values(bg)
    else:
        f = lambda data: explain_target.predict_proba(data)
        explainer = shap.KernelExplainer(f, shap.sample(bg, min(30, len(bg)), random_state=42))
        sv = explainer.shap_values(bg, nsamples=50, silent=True)

    if isinstance(sv, list):
        all_abs = np.mean([np.abs(np.asarray(v)) for v in sv], axis=0)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            n_features = len(feature_names)
            if arr.shape[1] == n_features:
                all_abs = np.abs(arr).mean(axis=2)
            else:
                all_abs = np.abs(arr).mean(axis=1)
        else:
            all_abs = np.abs(arr)

    mean_abs = np.asarray(all_abs).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


# ===========================================================================
# LIME
# ===========================================================================
def lime_explain(model, X_row: np.ndarray, background: np.ndarray,
                 feature_names: List[str], class_names: List[str],
                 class_index: int = 0, num_features: int = 12) -> Dict:
    """LIME explanation of one prediction (runs against the full model)."""
    if not _LIME_OK:
        raise RuntimeError("LIME is not installed.")
    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(
        data_row=np.asarray(X_row).reshape(-1),
        predict_fn=model.predict_proba,
        num_features=num_features,
        labels=[class_index],
    )
    pairs = exp.as_list(label=class_index)
    return {
        "pairs":      pairs,
        "class_name": class_names[class_index],
        "score":      exp.score,
        "intercept":  exp.intercept[class_index],
    }