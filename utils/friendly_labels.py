"""
utils/friendly_labels.py
=========================
Human-readable translations of numeric model outputs.

Every function returns a dict with at least:
    text   – plain-English phrase
    colour – hex colour for UI badges / text
    emoji  – single emoji for quick scanning

These are consumed by all UI pages so that non-STEM educators (and
students) see comprehensible language instead of raw floats.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import FEATURE_ALIASES


# ---------------------------------------------------------------------------
# Confidence → friendly text
# ---------------------------------------------------------------------------
def friendly_confidence(confidence: float) -> Dict[str, str]:
    """Map a 0-1 confidence value to a readable label.

    Returns dict with keys: text, detail, colour, emoji.
    """
    pct = confidence * 100
    if confidence >= 0.85:
        return {
            "text": "Very high confidence",
            "detail": f"The system is very sure about this prediction ({pct:.0f}%).",
            "colour": "#10B981",
            "emoji": "🟢",
        }
    if confidence >= 0.70:
        return {
            "text": "High confidence",
            "detail": f"The system is fairly confident in this prediction ({pct:.0f}%).",
            "colour": "#3B82F6",
            "emoji": "🔵",
        }
    if confidence >= 0.50:
        return {
            "text": "Moderate confidence",
            "detail": (
                f"The system is only moderately sure ({pct:.0f}%). "
                "Take this prediction as a rough guide, not a certainty."
            ),
            "colour": "#F59E0B",
            "emoji": "🟡",
        }
    return {
        "text": "Low confidence",
        "detail": (
            f"The system has low confidence ({pct:.0f}%). "
            "The prediction should be treated with caution."
        ),
        "colour": "#EF4444",
        "emoji": "🔴",
    }


# ---------------------------------------------------------------------------
# Probability → friendly text
# ---------------------------------------------------------------------------
def friendly_probability(prob: float) -> Dict[str, str]:
    """Describe a single-class probability in plain English."""
    pct = prob * 100
    if prob >= 0.60:
        return {"text": "Strong likelihood", "colour": "#10B981", "emoji": "🟢"}
    if prob >= 0.30:
        return {"text": "Some chance", "colour": "#F59E0B", "emoji": "🟡"}
    if prob >= 0.10:
        return {"text": "Unlikely", "colour": "#94A3B8", "emoji": "⚪"}
    return {"text": "Very unlikely", "colour": "#CBD5E1", "emoji": "⚪"}


# ---------------------------------------------------------------------------
# SHAP impact → friendly text
# ---------------------------------------------------------------------------
def friendly_shap_impact(shap_value: float) -> Dict[str, str]:
    """Translate a SHAP value into a human-readable impact label."""
    mag = abs(shap_value)
    if shap_value >= 0:
        if mag >= 0.10:
            return {"text": "Strongly helped this prediction", "colour": "#10B981", "emoji": "⬆️"}
        if mag >= 0.03:
            return {"text": "Slightly helped this prediction", "colour": "#6EE7B7", "emoji": "↗️"}
        return {"text": "Very small positive effect", "colour": "#94A3B8", "emoji": "➡️"}
    else:
        if mag >= 0.10:
            return {"text": "Strongly pushed against this prediction", "colour": "#EF4444", "emoji": "⬇️"}
        if mag >= 0.03:
            return {"text": "Slightly pushed against this prediction", "colour": "#FCA5A5", "emoji": "↘️"}
        return {"text": "Very small negative effect", "colour": "#94A3B8", "emoji": "➡️"}


# ---------------------------------------------------------------------------
# LIME weight → friendly text
# ---------------------------------------------------------------------------
def friendly_lime_impact(weight: float) -> Dict[str, str]:
    """Translate a LIME weight into a human-readable impact label."""
    mag = abs(weight)
    if weight >= 0:
        if mag >= 0.10:
            return {"text": "Strong positive contributor", "colour": "#10B981", "emoji": "⬆️"}
        if mag >= 0.03:
            return {"text": "Mild positive contributor", "colour": "#6EE7B7", "emoji": "↗️"}
        return {"text": "Negligible positive effect", "colour": "#94A3B8", "emoji": "➡️"}
    else:
        if mag >= 0.10:
            return {"text": "Strong negative contributor", "colour": "#EF4444", "emoji": "⬇️"}
        if mag >= 0.03:
            return {"text": "Mild negative contributor", "colour": "#FCA5A5", "emoji": "↘️"}
        return {"text": "Negligible negative effect", "colour": "#94A3B8", "emoji": "➡️"}


# ---------------------------------------------------------------------------
# Feature name → short readable alias
# ---------------------------------------------------------------------------
def readable_feature(raw_name: str) -> str:
    """Return the short UI alias for a feature, or a cleaned version."""
    if raw_name in FEATURE_ALIASES:
        return FEATURE_ALIASES[raw_name]
    # Handle one-hot encoded names: try trimming the suffix
    if "_" in raw_name:
        base = raw_name.rsplit("_", 1)[0]
        if base in FEATURE_ALIASES:
            return FEATURE_ALIASES[base]
    # Fallback: truncate and clean
    return raw_name[:60]


# ---------------------------------------------------------------------------
# LIME narrative generator
# ---------------------------------------------------------------------------
def build_lime_narrative(
    lime_pairs: List[Tuple[str, float]],
    predicted_label: str,
    max_items: int = 8,
) -> List[Dict[str, str]]:
    """Build human-readable narrative sentences from LIME feature-weight pairs.

    Each item in the returned list has keys:
        feature   – readable feature name
        direction – "helped" or "hurt"
        impact    – friendly impact dict
        sentence  – full natural-language sentence
    """
    narratives = []
    sorted_pairs = sorted(lime_pairs, key=lambda p: abs(p[1]), reverse=True)

    for feat_condition, weight in sorted_pairs[:max_items]:
        impact = friendly_lime_impact(weight)
        feat_name = readable_feature(feat_condition)

        if weight >= 0:
            direction = "pushed the prediction **towards**"
        else:
            direction = "pushed the prediction **away from**"

        sentence = (
            f"{impact['emoji']} **{feat_name}** — "
            f"This factor {direction} **{predicted_label}**. "
            f"({impact['text']}.)"
        )
        narratives.append({
            "feature": feat_name,
            "condition": feat_condition,
            "direction": "helped" if weight >= 0 else "hurt",
            "impact": impact,
            "sentence": sentence,
        })
    return narratives


# ---------------------------------------------------------------------------
# SHAP narrative generator
# ---------------------------------------------------------------------------
def build_shap_narrative(
    shap_df,
    predicted_label: str,
    max_items: int = 8,
) -> List[Dict[str, str]]:
    """Build human-readable narrative sentences from a SHAP DataFrame.

    Expects columns 'feature' and 'shap_value'.

    Each item in the returned list has keys:
        feature   – readable feature name
        direction – "helped" or "hurt"
        impact    – friendly impact dict
        sentence  – full natural-language sentence
    """
    narratives = []
    df = shap_df.copy()
    df["abs"] = df["shap_value"].abs()
    top = df.sort_values("abs", ascending=False).head(max_items)

    for _, row in top.iterrows():
        sv = row["shap_value"]
        impact = friendly_shap_impact(sv)
        feat_name = readable_feature(str(row["feature"]))

        if sv >= 0:
            direction = "pushed the prediction **towards**"
        else:
            direction = "pushed the prediction **away from**"

        sentence = (
            f"{impact['emoji']} **{feat_name}** — "
            f"This factor {direction} **{predicted_label}**. "
            f"({impact['text']}.)"
        )
        narratives.append({
            "feature": feat_name,
            "raw_feature": str(row["feature"]),
            "direction": "helped" if sv >= 0 else "hurt",
            "impact": impact,
            "sentence": sentence,
        })
    return narratives
