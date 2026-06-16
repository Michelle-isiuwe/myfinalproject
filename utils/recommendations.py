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


# ---------------------------------------------------------------------------
# Short display names for actionable features (UI-facing)
# ---------------------------------------------------------------------------
ACTIONABLE_SHORT_NAMES: Dict[str, str] = {
    "How many hours do you study per day on average?": "Daily Study Hours",
    "What is your average class attendance rate?": "Class Attendance",
    "How many hours do you sleep daily?": "Sleep Hours",
    "How often do you follow your study timetable?": "Timetable Adherence",
    "Do you have a personal study timetable?": "Study Timetable",
    "How often do you prepare for tests/exams in advance?": "Test Preparation",
    "How would you rate your concentration during study?": "Study Concentration",
    "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?": "Online Tools Usage",
    "Do you engage in group study": "Group Study",
    "How often do you feel academically stressed?": "Stress Management",
    "Do you usually complete and submit assignments on time?": "Assignment Submission",
    "How actively do you participate in class?": "Class Participation",
    "Do you attend tutorials/practical sessions regularly?": "Tutorial Attendance",
    "How well do you understand what is taught in class?": "Class Understanding",
    "How conducive is your study environment?": "Study Environment",
    "How would you rate your financial situation?": "Financial Situation",
}


def _severity(mag: float) -> Dict[str, str]:
    """Map SHAP magnitude to a severity level for UI display."""
    if mag >= 0.08:
        return {"level": "critical", "label": "Needs attention", "emoji": "🔴", "colour": "#EF4444"}
    if mag >= 0.03:
        return {"level": "warning", "label": "Could improve", "emoji": "🟡", "colour": "#F59E0B"}
    return {"level": "minor", "label": "Minor factor", "emoji": "🟠", "colour": "#FB923C"}


def detailed_recommendations_from_shap(
    shap_df: pd.DataFrame,
    payload: Dict | None = None,
    max_items: int = 5,
) -> List[Dict]:
    """Build structured improvement recommendations from SHAP values.

    Parameters
    ----------
    shap_df : DataFrame with columns 'feature' and 'shap_value'.
    payload : optional dict of the student's raw input answers.
    max_items : cap on recommendations returned.

    Returns
    -------
    list[dict] — each with keys:
        feature_name  – short display name
        question      – original survey question
        current_value – the student's answer (if payload supplied)
        severity      – dict with level, label, emoji, colour
        advice        – actionable advice text
        shap_value    – raw SHAP value (for reference)
    """
    if shap_df is None or shap_df.empty:
        return []

    hurt = shap_df[shap_df["shap_value"] < 0].copy()
    if hurt.empty:
        return []

    hurt["mag"] = hurt["shap_value"].abs()
    hurt = hurt.sort_values("mag", ascending=False)

    seen_questions: set = set()
    results: List[Dict] = []

    for _, row in hurt.iterrows():
        raw_feat = str(row["feature"])
        base = _base_feature(raw_feat)
        advice = ACTIONABLE_ADVICE.get(base)
        if not advice or base in seen_questions:
            continue
        seen_questions.add(base)

        short_name = ACTIONABLE_SHORT_NAMES.get(base, base[:40])
        current_val = (payload or {}).get(base, "—")
        sev = _severity(row["mag"])

        results.append({
            "feature_name": short_name,
            "question": base,
            "current_value": current_val,
            "severity": sev,
            "advice": advice,
            "shap_value": float(row["shap_value"]),
        })
        if len(results) >= max_items:
            break

    return results


def feature_status_cards(
    shap_df: pd.DataFrame,
    payload: Dict | None = None,
    max_items: int = 6,
) -> List[Dict]:
    """Return a mixed list of positive AND negative feature-status cards.

    Useful for educators to see a quick at-a-glance summary:
    green = contributing well, yellow/red = area to work on.

    Each item has keys: feature_name, status (good/warning/critical),
    current_value, label, emoji, colour.
    """
    if shap_df is None or shap_df.empty:
        return []

    df = shap_df.copy()
    df["mag"] = df["shap_value"].abs()
    df = df.sort_values("mag", ascending=False)

    seen: set = set()
    cards: List[Dict] = []

    for _, row in df.iterrows():
        raw_feat = str(row["feature"])
        base = _base_feature(raw_feat)
        if base in seen or base not in ACTIONABLE_ADVICE:
            continue
        seen.add(base)

        short_name = ACTIONABLE_SHORT_NAMES.get(base, base[:40])
        current_val = (payload or {}).get(base, "—")
        sv = row["shap_value"]

        if sv >= 0.03:
            status = {"status": "good", "label": "Good", "emoji": "🟢", "colour": "#10B981"}
        elif sv >= 0:
            status = {"status": "ok", "label": "Okay", "emoji": "🔵", "colour": "#3B82F6"}
        elif sv > -0.03:
            status = {"status": "minor", "label": "Minor concern", "emoji": "🟡", "colour": "#F59E0B"}
        elif sv > -0.08:
            status = {"status": "warning", "label": "Could improve", "emoji": "🟡", "colour": "#F59E0B"}
        else:
            status = {"status": "critical", "label": "Needs attention", "emoji": "🔴", "colour": "#EF4444"}

        cards.append({
            "feature_name": short_name,
            "question": base,
            "current_value": current_val,
            **status,
        })
        if len(cards) >= max_items:
            break

    return cards