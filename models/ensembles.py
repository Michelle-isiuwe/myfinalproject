"""
models/ensembles.py
===================
Meta-Ensemble Pipeline — the single ensemble strategy used by this project.

Architecture (matches the v2 script that produced ~80%):
  - Leakage-free OOF base probabilities for RF and XGB
  - Final base RF/XGB trained on the fully resampled training set
  - LightGBM + LR stacked on the [RF | XGB] OOF meta-features
  - Final meta-LR trained on: RF | XGB | SoftVote | LGBM_OOF | LR_OOF

Resampling (SMOTE+Tomek) is owned by this class and applied *inside* each
OOF fold, so the caller must pass UN-resampled training data.

sklearn-style interface:
    fit(X, y) -> self
    predict(X) -> np.ndarray
    predict_proba(X) -> np.ndarray of shape (n_samples, n_classes)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from config import META_LR_PARAMS, N_FOLDS, RANDOM_STATE
from models.base_models import make_lgbm, make_lr, make_rf, make_xgb


# ---------------------------------------------------------------------------
# Meta-Ensemble Pipeline
# ---------------------------------------------------------------------------
class MetaEnsemblePipeline(BaseEstimator, ClassifierMixin):
    """OOF-stacked meta-ensemble over Random Forest and XGBoost."""

    def __init__(self, resampling: str = "SMOTE+Tomek", n_folds: int = N_FOLDS):
        self.resampling = resampling
        self.n_folds = n_folds

    # -- resampling helpers -------------------------------------------------
    def _make_sampler(self, y):
        """SMOTE+Tomek with k_neighbors shrunk to the smallest class."""
        if self.resampling in (None, "None (Original)"):
            return None
        _, counts = np.unique(y, return_counts=True)
        k = max(1, min(5, int(counts.min()) - 1))
        return SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=k),
        )

    def _resample(self, X, y):
        sampler = self._make_sampler(y)
        if sampler is None:
            return X, y
        return sampler.fit_resample(X, y)

    def _safe_splits(self, y):
        _, counts = np.unique(y, return_counts=True)
        return max(2, min(self.n_folds, int(counts.min())))

    # -- probability alignment ---------------------------------------------
    def _align_proba(self, clf, proba):
        """Pad/reorder a proba matrix to match self.classes_ column order."""
        if list(clf.classes_) == list(self.classes_):
            return proba
        out = np.zeros((proba.shape[0], self.n_classes_))
        cls_list = list(self.classes_)
        for i, cls in enumerate(clf.classes_):
            out[:, cls_list.index(cls)] = proba[:, i]
        return out

    def _oof_proba(self, model_fn, X, y):
        """Leakage-free out-of-fold probabilities; resample inside each fold."""
        oof = np.zeros((len(y), self.n_classes_))
        skf = StratifiedKFold(
            n_splits=self._safe_splits(y), shuffle=True, random_state=RANDOM_STATE
        )
        for tr_idx, va_idx in skf.split(X, y):
            Xf, yf = self._resample(X[tr_idx], y[tr_idx])
            clf = model_fn()
            clf.fit(Xf, yf)
            oof[va_idx] = self._align_proba(clf, clf.predict_proba(X[va_idx]))
        return oof

    # -- fit / predict ------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Step 1: full-set resample for the final base learners
        X_bal, y_bal = self._resample(X, y)

        # Step 2: leakage-free OOF base probabilities
        O_rf = self._oof_proba(make_rf, X, y)
        O_xgb = self._oof_proba(make_xgb, X, y)

        # Step 3: base learners on the full resampled set
        self.rf_ = make_rf().fit(X_bal, y_bal)
        self.xgb_ = make_xgb().fit(X_bal, y_bal)

        # Step 4: stacked meta-features [RF | XGB]
        F_tr = np.hstack([O_rf, O_xgb])
        self.stack_lgbm_ = make_lgbm().fit(F_tr, y)
        self.stack_lr_ = make_lr().fit(F_tr, y)

        skf = StratifiedKFold(
            n_splits=self._safe_splits(y), shuffle=True, random_state=RANDOM_STATE
        )
        lgbm_oof = cross_val_predict(
            make_lgbm(), F_tr, y, cv=skf, method="predict_proba", n_jobs=-1
        )
        lr_oof = cross_val_predict(
            make_lr(), F_tr, y, cv=skf, method="predict_proba", n_jobs=-1
        )

        # Step 5: final meta-LR on the rich feature matrix
        soft_tr = (O_rf + O_xgb) / 2.0
        meta_tr = np.hstack([O_rf, O_xgb, soft_tr, lgbm_oof, lr_oof])
        self.meta_lr_ = LogisticRegression(**META_LR_PARAMS).fit(meta_tr, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        P_rf = self._align_proba(self.rf_, self.rf_.predict_proba(X))
        P_xgb = self._align_proba(self.xgb_, self.xgb_.predict_proba(X))
        F_te = np.hstack([P_rf, P_xgb])
        soft_te = (P_rf + P_xgb) / 2.0
        meta_te = np.hstack([
            P_rf, P_xgb, soft_te,
            self.stack_lgbm_.predict_proba(F_te),
            self.stack_lr_.predict_proba(F_te),
        ])
        return self.meta_lr_.predict_proba(meta_te)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def explainer_model(self):
        """Inner tree model for SHAP. The meta-ensemble is a stacked blend;
        for tractable, stable explanations we expose the Random Forest base
        learner — the strongest single contributor. Its SHAP values are a
        faithful proxy for feature direction on the final prediction."""
        return self.rf_


# ---------------------------------------------------------------------------
# Factory (kept so existing callers of build_ensemble still work)
# ---------------------------------------------------------------------------
ENSEMBLE_FACTORY = {
    "Meta-Ensemble Pipeline": MetaEnsemblePipeline,
}


def build_ensemble(name: str, **kwargs):
    """Instantiate an ensemble by its display name."""
    if name not in ENSEMBLE_FACTORY:
        raise ValueError(f"Unknown ensemble strategy: {name}")
    return ENSEMBLE_FACTORY[name](**kwargs)