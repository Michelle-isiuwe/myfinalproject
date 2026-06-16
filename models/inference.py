"""
models/inference.py
===================
Loads saved artefacts and produces predictions + probabilities for new data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLS, MODELS_DIR, CGPA_BAND_DISPLAY

# ---------------------------------------------------------------------------
# Abstention threshold — if the top-class probability is below this value
# the system abstains from making a definitive prediction and instead shows
# an explanation of why it is uncertain.
# ---------------------------------------------------------------------------
ABSTENTION_THRESHOLD = 0.70


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


class ModelRegistry:
    """Lightweight cache + loader for trained artefacts."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._cache: Dict[str, object] = {}

    # ---- existence / metadata ----
    def is_trained(self) -> bool:
        return (self.models_dir / "preprocessor.joblib").exists() and \
               (self.models_dir / "label_encoder.joblib").exists()

    def training_meta(self) -> dict | None:
        path = self.models_dir / "training_meta.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def available_models(self) -> List[str]:
        if not self.models_dir.exists():
            return []
        return [
            p.stem for p in self.models_dir.glob("*.joblib")
            if p.stem not in ("preprocessor", "label_encoder")
        ]

    # ---- loading ----
    def get_preprocessor(self):
        if "_preproc" not in self._cache:
            self._cache["_preproc"] = joblib.load(self.models_dir / "preprocessor.joblib")
        return self._cache["_preproc"]

    def get_label_encoder(self):
        if "_le" not in self._cache:
            self._cache["_le"] = joblib.load(self.models_dir / "label_encoder.joblib")
        return self._cache["_le"]

    def get_model(self, name: str):
        slug = _slug(name)
        if slug not in self._cache:
            path = self.models_dir / f"{slug}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Model {name!r} not found. Please contact your administrator "
                    f"to train the model."
                )
            self._cache[slug] = joblib.load(path)
        return self._cache[slug]

    def clear_cache(self):
        self._cache.clear()


# ===========================================================================
# Prediction helpers
# ===========================================================================
def _ensure_dataframe(payload) -> pd.DataFrame:
    """Accept dict or DataFrame; return a DataFrame with the right columns."""
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, pd.DataFrame):
        df = payload.copy()
    else:
        raise ValueError("Payload must be dict or DataFrame.")
    # Ensure every expected column exists
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = None
    return df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Abstention helpers
# ---------------------------------------------------------------------------
def _compute_abstention_reasons(proba: np.ndarray, classes: list) -> list:
    """Analyse a probability vector and return human-readable reasons for
    uncertainty when the system cannot make a confident prediction.

    Returns a list of dicts, each with 'title' and 'detail' keys.
    """
    reasons = []
    sorted_proba = np.sort(proba)[::-1]
    top1, top2 = sorted_proba[0], sorted_proba[1] if len(sorted_proba) > 1 else 0.0

    # 1. Low top probability
    reasons.append({
        "title": "Low overall confidence",
        "detail": (
            f"The highest probability for any performance band is only "
            f"{top1*100:.0f}%. The system requires at least "
            f"{ABSTENTION_THRESHOLD*100:.0f}% confidence to make a "
            f"definitive prediction."
        ),
    })

    # 2. Close competition between top classes
    gap = top1 - top2
    if gap < 0.15:
        top1_cls = classes[int(np.argmax(proba))]
        top1_label = CGPA_BAND_DISPLAY.get(top1_cls, top1_cls)
        sorted_idx = np.argsort(proba)[::-1]
        top2_cls = classes[sorted_idx[1]]
        top2_label = CGPA_BAND_DISPLAY.get(top2_cls, top2_cls)
        reasons.append({
            "title": "Close competition between bands",
            "detail": (
                f"The top two predicted bands are very close: "
                f"**{top1_label}** ({top1*100:.0f}%) and "
                f"**{top2_label}** ({top2*100:.0f}%). "
                f"The system cannot reliably distinguish between them."
            ),
        })

    # 3. Spread across many classes
    above_10 = int(np.sum(proba >= 0.10))
    if above_10 >= 3:
        reasons.append({
            "title": "Probability spread across multiple bands",
            "detail": (
                f"The student's data has meaningful probability "
                f"(≥10%) spread across {above_10} different bands, "
                f"indicating mixed signals in the input data."
            ),
        })

    # 4. Entropy measure (simplified)
    entropy = -np.sum(proba * np.log2(proba + 1e-12))
    max_entropy = np.log2(len(classes))
    if entropy > 0.7 * max_entropy:
        reasons.append({
            "title": "High uncertainty in the data",
            "detail": (
                "The student's responses produce an unusually uncertain "
                "prediction. This may happen when the combination of study "
                "habits, lifestyle factors, and demographic data does not "
                "closely match common patterns in the training data."
            ),
        })

    return reasons


def predict_single(payload: Dict, model_name: str,
                   registry: ModelRegistry | None = None) -> Dict:
    """Predict CGPA band + probabilities + confidence for a single record.

    When confidence falls below ABSTENTION_THRESHOLD (0.70), the result
    includes ``abstained=True`` with a list of human-readable reasons
    explaining the uncertainty.
    """
    registry = registry or ModelRegistry()
    preproc = registry.get_preprocessor()
    le = registry.get_label_encoder()
    model = registry.get_model(model_name)

    df = _ensure_dataframe(payload)
    X = preproc.transform(df)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]
    classes = list(le.classes_)
    confidence = float(proba[pred_idx])

    abstained = confidence < ABSTENTION_THRESHOLD
    abstention_reasons = (
        _compute_abstention_reasons(proba, classes) if abstained else []
    )

    return {
        "predicted_band": pred_label,
        "confidence": confidence,
        "probabilities": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "X_transformed": X,             # used by SHAP / LIME
        "raw_input": df.iloc[0].to_dict(),
        "abstained": abstained,
        "abstention_reasons": abstention_reasons,
    }


def predict_bulk(df: pd.DataFrame, model_name: str,
                 registry: ModelRegistry | None = None) -> pd.DataFrame:
    """Predict for many rows. Returns df + predicted_band + confidence."""
    registry = registry or ModelRegistry()
    preproc = registry.get_preprocessor()
    le = registry.get_label_encoder()
    model = registry.get_model(model_name)

    rows = _ensure_dataframe(df)
    X = preproc.transform(rows)
    probas = model.predict_proba(X)
    pred_idx = np.argmax(probas, axis=1)
    out = df.copy().reset_index(drop=True)
    out["predicted_band"] = le.inverse_transform(pred_idx)
    out["confidence"] = probas[np.arange(len(probas)), pred_idx]
    for i, cls in enumerate(le.classes_):
        out[f"prob_{cls}"] = probas[:, i]
    return out