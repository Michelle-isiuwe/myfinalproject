"""
models/base_models.py
=====================
Factory helpers for the base learners used by the Meta-Ensemble
(Random Forest, XGBoost, LightGBM, Logistic Regression).
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import LGBM_PARAMS, LR_PARAMS, RF_PARAMS, XGB_PARAMS


def make_rf(**overrides) -> RandomForestClassifier:
    return RandomForestClassifier(**{**RF_PARAMS, **overrides})


def make_xgb(**overrides) -> XGBClassifier:
    return XGBClassifier(**{**XGB_PARAMS, **overrides})


def make_lgbm(**overrides) -> LGBMClassifier:
    return LGBMClassifier(**{**LGBM_PARAMS, **overrides})


def make_lr(**overrides) -> LogisticRegression:
    return LogisticRegression(**{**LR_PARAMS, **overrides})