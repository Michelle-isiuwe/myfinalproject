"""
utils/reports.py
================
Generates downloadable prediction reports (text/CSV) without needing a
PDF library — keeping Streamlit Cloud dependencies light.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict

import pandas as pd

from config import CGPA_BAND_DISPLAY


def build_prediction_report(payload: Dict, prediction: Dict,
                            shap_top: pd.DataFrame | None = None,
                            lime_pairs: list | None = None,
                            model_name: str = "",
                            educator: str = "") -> str:
    """A readable single-prediction report (markdown style, downloadable as .txt).

    Accepts a prediction dict from either source:
      - predict_single() result, keyed 'predicted_band'
      - a database row, keyed 'predicted_risk'
    """
    # Resolve the predicted band under whichever key is present.
    predicted = (prediction.get("predicted_band")
                 or prediction.get("predicted_risk")
                 or "—")
    predicted_label = CGPA_BAND_DISPLAY.get(predicted, predicted)

    lines = []
    lines.append("=" * 64)
    lines.append("  STUDENT ACADEMIC PERFORMANCE — PREDICTION REPORT")
    lines.append("=" * 64)
    lines.append(f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model used:  {model_name}")
    if educator:
        lines.append(f"Educator:    {educator}")
    lines.append("")

    lines.append("-- PREDICTION ----------------------------------------------------")
    lines.append(f"  Predicted band : {predicted_label}")
    lines.append(f"  Confidence     : {prediction['confidence']*100:.2f}%")
    lines.append("  Class probabilities:")
    for cls, p in prediction["probabilities"].items():
        cls_label = CGPA_BAND_DISPLAY.get(cls, cls)
        lines.append(f"    - {cls_label:<30s} {p*100:>6.2f}%")
    lines.append("")

    lines.append("-- INPUT FEATURES ------------------------------------------------")
    for k, v in payload.items():
        lines.append(f"  • {k}:")
        lines.append(f"      {v}")
    lines.append("")

    if shap_top is not None and len(shap_top) > 0:
        lines.append("-- TOP SHAP CONTRIBUTIONS ---------------------------------------")
        for _, row in shap_top.iterrows():
            sign = "+" if row["shap_value"] >= 0 else "−"
            lines.append(f"  {sign} {row['feature']:<55s} {row['shap_value']:+.4f}")
        lines.append("")

    if lime_pairs:
        lines.append("-- LIME LOCAL EXPLANATION ---------------------------------------")
        for feat, weight in lime_pairs:
            sign = "+" if weight >= 0 else "−"
            lines.append(f"  {sign} {feat:<55s} {weight:+.4f}")
        lines.append("")

    lines.append("=" * 64)
    lines.append("  Note: This prediction is decision-support, not a substitute for")
    lines.append("  professional academic counselling. Use alongside other inputs.")
    lines.append("=" * 64)
    return "\n".join(lines)


def predictions_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame of predictions to CSV bytes for download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")