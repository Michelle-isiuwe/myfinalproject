"""
app/pages/student_prediction.py
================================
Student prediction form.

- 6 demographic/profile fields are pulled from the student's registered
  profile and never asked again.
- 8 academic fields are pre-filled from the student's last prediction
  (editable each time).
- 10 lifestyle fields start fresh on every visit.

Combines all 24 into the model payload, runs the prediction, and stores
the result in session state for the My Results page.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from app.auth import current_user, require_login
from app.feedback_alerts import render_unread_feedback_banner
from config import (
    CGPA_BAND_DISPLAY, NOMINAL_OPTIONS, ORDINAL_ORDERINGS, RISK_COLORS,
)
from database import get_db
from models.inference import ABSTENTION_THRESHOLD, ModelRegistry, predict_single
from utils.friendly_labels import friendly_confidence, friendly_probability
from utils.preprocessing import load_raw

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------
# (feature column, UI label, optional help text)
ACADEMIC_FIELDS = [
    (
        "Do you usually complete and submit assignments on time?",
        "How often do you submit assignments on time?",
        "How consistently you meet coursework deadlines.",
    ),
    (
        "How often do you prepare for tests/exams in advance?",
        "How often do you prepare for tests in advance?",
        None,
    ),
    (
        "How many hours do you study per day on average?",
        "Average study hours per day",
        None,
    ),
    (
        "Do you have a personal study timetable?",
        "Do you use a personal study timetable?",
        "Choose Yes if you plan your study time in a weekly timetable.",
    ),
    (
        "How often do you follow your study timetable?",
        "How often do you stick to your study timetable?",
        "If you do not use a timetable, select Never.",
    ),
    (
        "What is your average class attendance rate?",
        "Average class attendance",
        None,
    ),
    (
        "How actively do you participate in class?",
        "How actively you participate in class",
        None,
    ),
    (
        "Do you attend tutorials/practical sessions regularly?",
        "How often you attend tutorials / practicals",
        None,
    ),
]

LIFESTYLE_FIELDS = [
    (
        "How would you rate your concentration during study?",
        "Concentration during study",
        None,
    ),
    (
        "How well do you understand what is taught in class?",
        "Understanding of what is taught in class",
        None,
    ),
    (
        "Do you engage in group study",
        "How often you take part in group study",
        None,
    ),
    (
        "Do you have a part-time job? If yes, how many hours do you work per week?",
        "Part-time job (hours per week)",
        None,
    ),
    (
        "How would you rate your financial situation?",
        "Your financial situation",
        None,
    ),
    (
        "How conducive is your study environment?",
        "How suitable your study environment is",
        None,
    ),
    (
        "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?",
        "How often you use online learning tools",
        None,
    ),
    (
        "How many hours do you spend on social media daily?",
        "Social media use per day",
        None,
    ),
    (
        "How many hours do you sleep daily?",
        "Sleep per night",
        None,
    ),
    (
        "How often do you feel academically stressed?",
        "How often you feel academically stressed",
        None,
    ),
]

PROFILE_TO_FEATURE = {
    "gender":          "Gender",
    "age_range":       "Age Range",
    "level_of_study":  "Level of Study",
    "field_of_study":  "Field of Study",
    "internet_access": "Do you have regular access to the internet?",
    "devices_used":    "What devices do you use for studying?",
}


def _options_for(col: str) -> list:
    if col in ORDINAL_ORDERINGS:
        return ORDINAL_ORDERINGS[col]
    if col in NOMINAL_OPTIONS:
        return NOMINAL_OPTIONS[col]
    df = load_raw()
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []


def _selectbox_indexed(
    label: str,
    col: str,
    last_inputs: dict | None,
    key: str,
    help: str | None = None,
) -> str:
    opts = _options_for(col)
    if not opts:
        return st.text_input(label, key=key, help=help)
    last_val = (last_inputs or {}).get(col)
    idx = opts.index(last_val) if last_val in opts else 0
    return st.selectbox(label, opts, index=idx, key=key, help=help)


def render():
    require_login()
    user = current_user()
    db   = get_db()
    registry = ModelRegistry()

    st.title("🔮 My Prediction")
    st.caption("Submit your study habits and lifestyle details to get your performance prediction.")

    render_unread_feedback_banner(user["user_id"])

    if not registry.is_trained():
        st.error("⚠️ No trained models available. Please contact your educator or admin.")
        return

    # --- load student profile and last academic inputs ---------------------
    profile = db.get_student_profile(user["user_id"])
    last_inputs = db.get_last_academic_inputs(user["user_id"])

    if not profile:
        st.warning(
            "⚠️ Your profile is incomplete. Please contact an admin or re-register "
            "to set your demographic information."
        )
        return

    meta = registry.training_meta() or {}

    # The deployed model is the Meta-Ensemble Pipeline (the only trained model).
    available = registry.available_models()
    _slug = lambda n: n.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    model_choice = "Meta-Ensemble Pipeline" if _slug("Meta-Ensemble Pipeline") in available else ""
    if not model_choice:
        st.error("No trained model found. Please retrain from the admin/retrain step.")
        return

    # Show profile summary
    with st.expander("📋 Your registered profile (read-only)", expanded=False):
        cols = st.columns(3)
        cols[0].markdown(f"**Gender:** {profile.get('gender', '—')}")
        cols[0].markdown(f"**Age Range:** {profile.get('age_range', '—')}")
        cols[1].markdown(f"**Level of Study:** {profile.get('level_of_study', '—')}")
        cols[1].markdown(f"**Field of Study:** {profile.get('field_of_study', '—')}")
        cols[2].markdown(f"**Internet Access:** {profile.get('internet_access', '—')}")
        cols[2].markdown(f"**Devices Used:** {profile.get('devices_used', '—')}")

    st.divider()

    with st.form("student_prediction_form", clear_on_submit=False):
        # ---- Academic fields (pre-fillable) ---------------------------------
        st.markdown("### 📚 Academic Habits")
        st.caption("These are pre-filled from your last submission — update if anything has changed.")
        ac = st.columns(2)
        academic_values = {}
        for i, (col, label, hint) in enumerate(ACADEMIC_FIELDS):
            with ac[i % 2]:
                academic_values[col] = _selectbox_indexed(
                    label, col, last_inputs, f"acad_{i}", help=hint
                )

        # ---- Lifestyle fields (fresh each time) ----------------------------
        st.markdown("### 🌙 Lifestyle & Context")
        st.caption("These reflect your current situation — answer honestly for best results.")
        lc = st.columns(2)
        lifestyle_values = {}
        for i, (col, label, hint) in enumerate(LIFESTYLE_FIELDS):
            opts = _options_for(col)
            with lc[i % 2]:
                if opts:
                    lifestyle_values[col] = st.selectbox(
                        label, opts, key=f"life_{i}", help=hint
                    )
                else:
                    lifestyle_values[col] = st.text_input(
                        label, key=f"life_{i}", help=hint
                    )

        submitted = st.form_submit_button(
            "🔍 Get My Prediction", type="primary", use_container_width=True
        )

    if submitted:
        # Build full 24-feature payload
        payload = {
            "Gender":           profile.get("gender", ""),
            "Age Range":        profile.get("age_range", ""),
            "Level of Study":   profile.get("level_of_study", ""),
            "Field of Study":   profile.get("field_of_study", ""),
            "Do you have regular access to the internet?": profile.get("internet_access", ""),
            "What devices do you use for studying?":       profile.get("devices_used", ""),
            **academic_values,
            **lifestyle_values,
        }

        with st.spinner("Running prediction…"):
            try:
                result = predict_single(payload, model_choice, registry)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # Persist to DB. (db.log_prediction's keyword is `predicted_risk`; we
        # feed it the renamed result value `predicted_band`.)
        try:
            pred_id = db.log_prediction(
                user_id=user["user_id"],
                model_used=model_choice,
                resampling=meta.get("resampling", "—"),
                predicted_risk=result["predicted_band"],
                confidence=result["confidence"],
                probs=result["probabilities"],
                input_payload=payload,
            )
        except Exception as e:
            st.warning(f"Prediction made but could not be saved: {e}")
            pred_id = None

        # Stash for My Results page
        st.session_state["last_prediction"] = {
            "id":        pred_id,
            "payload":   payload,
            "result":    {k: v for k, v in result.items() if k != "X_transformed"},
            "model":     model_choice,
            "timestamp": datetime.now().isoformat(),
        }

        # Immediate result display
        band       = result["predicted_band"]
        band_label = CGPA_BAND_DISPLAY.get(band, band)
        confidence = result["confidence"]
        probs      = result["probabilities"]
        conf_info  = friendly_confidence(confidence)
        abstained  = result.get("abstained", False)

        # ---- ABSTENTION HANDLING ----
        if abstained:
            with st.container(border=True):
                st.caption("Best estimate")
                st.markdown(f"## **{band_label}**")
                st.markdown(
                    f"⚠️ **Low confidence — only {confidence*100:.0f}% sure "
                    f"({ABSTENTION_THRESHOLD*100:.0f}% needed for a reliable prediction)**"
                )

            fig = go.Figure()
            for cls, p in probs.items():
                fig.add_trace(go.Bar(
                    y=[CGPA_BAND_DISPLAY.get(cls, cls)], x=[p * 100],
                    orientation="h",
                    text=f"{p*100:.1f}%",
                    textposition="outside",
                    marker_color=RISK_COLORS.get(cls, "#94A3B8"),
                    name=cls,
                ))
            fig.update_layout(
                xaxis=dict(range=[0, 100], title="Likelihood (%)"),
                height=240, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=80, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            reasons = result.get("abstention_reasons", [])
            if reasons:
                with st.expander("Why am I unsure?", expanded=False):
                    for r in reasons:
                        st.markdown(f"- **{r['title']}**: {r['detail']}")

            st.info(
                "Double-check your answers are accurate and up to date, then try again."
            )
        else:
            # ---- NORMAL RESULT ----
            st.success("Prediction complete! Visit **My Results** for the full explanation.")
            with st.container(border=True):
                r1, r2 = st.columns(2)
                r1.metric("Predicted Performance", band_label)
                r2.markdown(
                    f"**Confidence**\n\n"
                    f"{conf_info['emoji']} **{conf_info['text']}** ({confidence*100:.0f}%)"
                )
                r2.caption(conf_info["detail"])

                fig = go.Figure()
                for cls, p in probs.items():
                    prob_info = friendly_probability(p)
                    fig.add_trace(go.Bar(
                        y=[CGPA_BAND_DISPLAY.get(cls, cls)], x=[p * 100],
                        orientation="h",
                        text=f"{p*100:.1f}% — {prob_info['text']}",
                        textposition="auto",
                        marker_color=RISK_COLORS.get(cls, "#94A3B8"), name=cls,
                    ))
                fig.update_layout(
                    title="How likely is each performance band?",
                    xaxis=dict(range=[0, 100], title="Likelihood (%)"),
                    height=240, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)