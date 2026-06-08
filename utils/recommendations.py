"""
utils/recommendations.py
========================
Rule-based study recommendations derived from SHAP feature attributions.

For each ACTIONABLE feature that pushed the prediction toward a lower band
(negative SHAP value for the predicted class), we surface a vetted, fixed
piece of advice. Recommendations are fully transparent: each maps to a
specific feature the model flagged, and only to factors the student can change.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

# Map the model's raw feature column → student-facing advice.
# Only ACTIONABLE features appear here; fixed traits (gender, age, level,
# field, devices, internet) are intentionally excluded.
ACTIONABLE_ADVICE: Dict[str, str] = {
    "How many hours do you study per day on average?":
        "Try to increase your daily study time gradually — even an extra 30 "
        "minutes of focused study per day adds up over a semester.",
    "What is your average class attendance rate?":
        "Aim to attend more classes consistently. Regular attendance is one "
        "of the strongest predictors of stronger results.",
    "How many hours do you sleep daily?":
        "Prioritise consistent, adequate sleep. Poor sleep weakens "
        "concentration and memory, which affects performance.",
    "How often do you follow your study timetable?":
        "Follow your study timetable more closely. A consistent routine helps "
        "you cover material steadily instead of cramming.",
    "Do you have a personal study timetable?":
        "Consider creating a personal study timetable to structure your week "
        "and protect dedicated study time.",
    "How often do you prepare for tests/exams in advance?":
        "Start preparing for tests and exams earlier. Spreading revision over "
        "time is far more effective than last-minute study.",
    "How would you rate your concentration during study?":
        "Work on improving focus during study — try shorter, distraction-free "
        "study blocks and take regular short breaks.",
    "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?":
        "Make more use of online learning tools to reinforce difficult topics "
        "and get alternative explanations.",
    "Do you engage in group study":
        "Consider joining or forming a study group. Explaining topics to peers "
        "deepens your own understanding.",
    "How often do you feel academically stressed?":
        "Manage academic stress actively — break work into smaller tasks, and "
        "reach out to support services if stress feels overwhelming.",
}


def _base_feature(shap_feature_name: str) -> str:
    """Strip one-hot suffixes so a transformed column maps back to its
    original survey question. Ordinal features keep their full name; one-hot
    columns look like 'Question_Category' — we take the part before the last
    underscore only if the full name isn't itself an actionable key."""
    if shap_feature_name in ACTIONABLE_ADVICE:
        return shap_feature_name
    # Try trimming a one-hot suffix (everything after the last underscore)
    if "_" in shap_feature_name:
        trimmed = shap_feature_name.rsplit("_", 1)[0]
        if trimmed in ACTIONABLE_ADVICE:
            return trimmed
    return shap_feature_name


def recommendations_from_shap(shap_df: pd.DataFrame,
                              max_items: int = 4) -> List[str]:
    """
    Build a list of advice strings from a SHAP table.

    Parameters
    ----------
    shap_df : DataFrame with columns 'feature' and 'shap_value' (for the
              predicted class). Negative shap_value = pushed toward a LOWER
              band, i.e. a factor worth improving.
    max_items : cap on how many recommendations to return.

    Returns
    -------
    list[str] — vetted advice, most impactful first, de-duplicated.
    """
    if shap_df is None or shap_df.empty:
        return []

    # Negative SHAP = hurt the prediction → candidate for advice.
    hurt = shap_df[shap_df["shap_value"] < 0].copy()
    if hurt.empty:
        return []

    # Rank by magnitude (most negative first)
    hurt["mag"] = hurt["shap_value"].abs()
    hurt = hurt.sort_values("mag", ascending=False)

    seen = set()
    out: List[str] = []
    for _, row in hurt.iterrows():
        base = _base_feature(str(row["feature"]))
        advice = ACTIONABLE_ADVICE.get(base)
        if advice and advice not in seen:
            seen.add(advice)
            out.append(advice)
        if len(out) >= max_items:
            break
    return out