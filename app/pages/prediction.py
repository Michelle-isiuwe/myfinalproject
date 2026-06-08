"""
app/pages/prediction.py
=======================
The main (educator) prediction interface.

Provides:
  • Single-student manual entry with friendly Streamlit widgets that
    expose only the survey's defined response options (no free text).
  • Bulk CSV upload for batch prediction.
  • Saving each prediction to the SQLite database so it appears in the
    history page.

The deployed model is the Meta-Ensemble Pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.auth import current_user, require_login
from config import (
    CGPA_BAND_DISPLAY, ENSEMBLE_STRATEGIES, FEATURE_COLS, NOMINAL_COLS,
    ORDINAL_ORDERINGS, RISK_CLASSES, RISK_COLORS,
)
from database import get_db
from models.inference import ModelRegistry, predict_bulk, predict_single
from utils.preprocessing import load_raw
from utils.reports import build_prediction_report, predictions_to_csv_bytes


# ---------------------------------------------------------------------------
# Helper: build the manual-entry form using the survey's actual response sets
# ---------------------------------------------------------------------------
def _options_for(col: str) -> list[str]:
    if col in ORDINAL_ORDERINGS:
        return ORDINAL_ORDERINGS[col]
    # Pull observed categories from the dataset
    df = load_raw()
    if col in df.columns:
        return [v for v in pd.Series(df[col]).dropna().unique().tolist()]
    return []


def _model_choices(registry: ModelRegistry) -> list[str]:
    available = set(registry.available_models())
    # Pretty display order — only the Meta-Ensemble is deployed now.
    desired = ["Random Forest", "XGBoost"] + ENSEMBLE_STRATEGIES

    def _slug(name): return (name.lower().replace(" ", "_").replace("(", "")
                              .replace(")", "").replace("-", "_"))
    return [m for m in desired if _slug(m) in available]


def render():
    require_login()
    user = current_user()
    db = get_db()
    registry = ModelRegistry()

    st.title("🔮 CGPA Band Prediction")
    st.caption("Predict a student's CGPA band (Below 1.50 – 4.50–5.00) from study habits and context.")

    if not registry.is_trained():
        st.error("⚠️ No trained models found. Please ask an admin to train the model first.")
        return

    available = _model_choices(registry)
    if not available:
        st.error("No trained model found on disk.")
        return
    meta = registry.training_meta() or {}

    # ---- top selector row -------------------------------------------------
    with st.container(border=True):
        cols = st.columns([2, 1])
        with cols[0]:
            # Default to the Meta-Ensemble if present, else the first available.
            default_idx = (available.index("Meta-Ensemble Pipeline")
                           if "Meta-Ensemble Pipeline" in available else 0)
            model_choice = st.selectbox(
                "Model used for this prediction",
                options=available,
                index=default_idx,
                help="The deployed model is the Meta-Ensemble Pipeline.",
            )
        with cols[1]:
            st.metric("Trained with resampling", meta.get("resampling", "—"))

    mode_tab, bulk_tab = st.tabs(["🧍 Single student", "📂 Bulk CSV upload"])

    # ============= Single-student form =====================================
    with mode_tab:
        with st.form("single_form", clear_on_submit=False):
            st.markdown("### Demographics")
            d_cols = st.columns(2)
            with d_cols[0]:
                gender = st.selectbox("Gender", _options_for("Gender"))
                age = st.selectbox("Age range", _options_for("Age Range"))
                level = st.selectbox("Level of study", _options_for("Level of Study"))
            with d_cols[1]:
                field = st.selectbox("Field of study", _options_for("Field of Study"))
                internet = st.selectbox("Regular internet access?", _options_for("Do you have regular access to the internet?"))
                devices = st.selectbox("Devices used for studying", _options_for("What devices do you use for studying?"))

            st.markdown("### Study habits")
            sh = st.columns(2)
            with sh[0]:
                hours_per_day = st.selectbox("Hours of study per day", _options_for("How many hours do you study per day on average?"))
                has_timetable = st.selectbox("Has personal study timetable?", _options_for("Do you have a personal study timetable?"))
                follows_timetable = st.selectbox("Follows study timetable", _options_for("How often do you follow your study timetable?"))
                concentration = st.selectbox("Concentration during study", _options_for("How would you rate your concentration during study?"))
            with sh[1]:
                group_study = st.selectbox("Engages in group study", _options_for("Do you engage in group study"))
                online_tools = st.selectbox("Uses online learning tools", _options_for("How often do you use online learning tools (e.g., YouTube, Coursera, AI)?"))
                test_prep = st.selectbox("Prepares for tests in advance", _options_for("How often do you prepare for tests/exams in advance?"))
                assign_subm = st.selectbox("Submits assignments on time", _options_for("Do you usually complete and submit assignments on time?"))

            st.markdown("### Engagement in class")
            ec = st.columns(2)
            with ec[0]:
                attendance = st.selectbox("Class attendance", _options_for("What is your average class attendance rate?"))
                participation = st.selectbox("Class participation", _options_for("How actively do you participate in class?"))
            with ec[1]:
                understanding = st.selectbox("Understanding of class content", _options_for("How well do you understand what is taught in class?"))
                tutorials = st.selectbox("Attends tutorials/practicals", _options_for("Do you attend tutorials/practical sessions regularly?"))

            st.markdown("### Lifestyle & context")
            lc = st.columns(2)
            with lc[0]:
                job = st.selectbox("Part-time job?", _options_for("Do you have a part-time job? If yes, how many hours do you work per week?"))
                finance = st.selectbox("Financial situation", _options_for("How would you rate your financial situation?"))
                environment = st.selectbox("Study environment", _options_for("How conducive is your study environment?"))
            with lc[1]:
                social = st.selectbox("Social media hours / day", _options_for("How many hours do you spend on social media daily?"))
                sleep = st.selectbox("Sleep hours / day", _options_for("How many hours do you sleep daily?"))
                stress = st.selectbox("Academic stress frequency", _options_for("How often do you feel academically stressed?"))

            st.divider()
            student_name = st.text_input("Student name (optional, for record-keeping)")
            student_id = st.text_input("Matric / Student ID (optional)")
            submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

        if submitted:
            payload = {
                "Gender": gender, "Age Range": age, "Level of Study": level,
                "Field of Study": field,
                "Do you usually complete and submit assignments on time?": assign_subm,
                "How often do you prepare for tests/exams in advance?": test_prep,
                "How many hours do you study per day on average?": hours_per_day,
                "Do you have a personal study timetable?": has_timetable,
                "How would you rate your concentration during study?": concentration,
                "How often do you follow your study timetable?": follows_timetable,
                "How well do you understand what is taught in class?": understanding,
                "Do you engage in group study": group_study,
                "What is your average class attendance rate?": attendance,
                "How actively do you participate in class?": participation,
                "Do you attend tutorials/practical sessions regularly?": tutorials,
                "Do you have a part-time job? If yes, how many hours do you work per week?": job,
                "How would you rate your financial situation?": finance,
                "How conducive is your study environment?": environment,
                "Do you have regular access to the internet?": internet,
                "What devices do you use for studying?": devices,
                "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?": online_tools,
                "How many hours do you spend on social media daily?": social,
                "How many hours do you sleep daily?": sleep,
                "How often do you feel academically stressed?": stress,
            }

            with st.spinner("Running prediction..."):
                try:
                    result = predict_single(payload, model_choice, registry)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            # Persist (DB keyword is `predicted_risk`; value is the band string)
            try:
                pred_id = db.log_prediction(
                    user_id=user["user_id"],
                    model_used=model_choice,
                    resampling=meta.get("resampling", "—"),
                    predicted_risk=result["predicted_band"],
                    confidence=result["confidence"],
                    probs=result["probabilities"],
                    input_payload={**payload,
                                   "_student_name": student_name,
                                   "_student_id": student_id},
                )
            except Exception as e:
                st.warning(f"Saved locally but failed to log to DB: {e}")
                pred_id = None

            # Stash so the Explainability page can show this prediction
            st.session_state["last_prediction"] = {
                "id":        pred_id,
                "payload":   payload,
                "result":    {k: v for k, v in result.items() if k != "X_transformed"},
                "model":     model_choice,
                "timestamp": datetime.now().isoformat(),
            }

            _render_single_result(payload, result, model_choice, user)

    # ============= Bulk CSV upload ==========================================
    with bulk_tab:
        st.write("Upload a CSV containing one row per student. The columns must match the survey questions (full names).")
        with st.expander("📝 Show expected column names"):
            st.code("\n".join(FEATURE_COLS), language=None)

        file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
        if file is not None:
            try:
                bulk_df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return
            missing = [c for c in FEATURE_COLS if c not in bulk_df.columns]
            if missing:
                st.warning(f"Missing {len(missing)} expected columns. Predictions will use defaults for those.")
                with st.expander("Missing columns"):
                    st.write(missing)

            st.write("Preview:")
            st.dataframe(bulk_df.head(10), use_container_width=True)

            if st.button("Run bulk prediction", type="primary"):
                with st.spinner(f"Predicting {len(bulk_df)} records..."):
                    out = predict_bulk(bulk_df, model_choice, registry)
                st.success(f"Predicted {len(out)} records.")
                st.dataframe(out.head(50), use_container_width=True)

                # Distribution chart
                dist = out["predicted_band"].value_counts().reindex(
                    RISK_CLASSES).fillna(0).astype(int).reset_index()
                dist.columns = ["band", "count"]
                dist["label"] = dist["band"].map(lambda b: CGPA_BAND_DISPLAY.get(b, b))
                import plotly.express as px
                fig = px.bar(dist, x="label", y="count", color="band",
                             color_discrete_map=RISK_COLORS, text="count",
                             title="Predicted CGPA band distribution (bulk)")
                fig.update_layout(showlegend=False, height=320,
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=predictions_to_csv_bytes(out),
                    file_name=f"bulk_predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )


# ---------------------------------------------------------------------------
# Result rendering helper
# ---------------------------------------------------------------------------
def _render_single_result(payload, result, model_name, user):
    band = result["predicted_band"]
    band_label = CGPA_BAND_DISPLAY.get(band, band)
    confidence = result["confidence"]
    probs = result["probabilities"]

    with st.container(border=True):
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.metric("Predicted band", band_label, delta=None)
        with cols[1]:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        with cols[2]:
            st.metric("Model", model_name)

        # Probability gauges
        fig = go.Figure()
        for cls, p in probs.items():
            fig.add_trace(go.Bar(
                y=[CGPA_BAND_DISPLAY.get(cls, cls)], x=[p * 100], orientation="h",
                text=f"{p*100:.1f}%", textposition="auto",
                marker_color=RISK_COLORS.get(cls, "#94A3B8"), name=cls,
            ))
        fig.update_layout(
            title="Class probabilities",
            xaxis=dict(range=[0, 100], title="probability (%)"),
            height=260, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"➡️ Visit the **Explainability** page to see *why* this student was predicted as **{band_label}**."
        )

        # Download report (text)
        report = build_prediction_report(
            payload, result, model_name=model_name,
            educator=user.get("full_name") or user["username"],
        )
        st.download_button(
            "⬇️ Download prediction report (.txt)",
            data=report.encode("utf-8"),
            file_name=f"prediction_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
        )