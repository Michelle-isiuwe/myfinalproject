"""
app/pages/dashboard.py
======================
Educator dashboard.

Shows:
  • System status (is a model trained?)
  • Overview of all registered students with their latest predicted CGPA band
  • Band distribution across all student predictions
  • Recent prediction activity
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.auth import current_user, require_login
from config import CGPA_BAND_DISPLAY, RISK_CLASSES, RISK_COLORS
from database import get_db
from models.inference import ModelRegistry

# Lowest CGPA bands — students who may benefit from additional support.
# (Pass and Third Class.) Keep in sync with educator_students.SUPPORT_BANDS.
SUPPORT_BANDS = ["Below 1.50", "1.50 – 2.49"]


def render():
    require_login()
    user = current_user()
    db = get_db()
    registry = ModelRegistry()

    st.title("📊 Educator Dashboard")
    st.caption(f"Welcome back, **{user.get('full_name') or user['username']}**.")

    # ---- system status ---------------------------------------------------
    st.metric(
        "System status",
        "🟢 Ready" if registry.is_trained() else "🔴 Not trained",
    )

    if not registry.is_trained():
        st.warning(
            "No trained model is available. The model must be trained before "
            "the system can make predictions."
        )

    st.divider()

    # ---- student overview ------------------------------------------------
    st.subheader("Registered Students")
    students_df = db.list_students_with_latest_prediction()

    if students_df.empty:
        st.info("No students have registered yet.")
    else:
        total_students = len(students_df)
        predicted = students_df["predicted_risk"].notna().sum()
        need_support = students_df["predicted_risk"].isin(SUPPORT_BANDS).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Students", total_students)
        m2.metric("Predictions Made", int(predicted))
        m3.metric("May Need Support", int(need_support), delta=None)

        # Band distribution chart
        if predicted > 0:
            dist = (
                students_df["predicted_risk"]
                .dropna()
                .value_counts()
                .reindex(RISK_CLASSES)
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            dist.columns = ["band", "count"]
            dist["label"] = dist["band"].map(lambda b: CGPA_BAND_DISPLAY.get(b, b))
            fig = px.bar(
                dist, x="label", y="count", color="band",
                color_discrete_map=RISK_COLORS, text="count",
                title="CGPA band distribution across all students",
            )
            fig.update_layout(
                showlegend=False, height=280,
                margin=dict(t=40, l=10, r=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Student table
        show_cols = [c for c in
                     ["full_name", "username", "level_of_study", "field_of_study",
                      "predicted_risk", "confidence", "predicted_at"]
                     if c in students_df.columns]
        display = students_df[show_cols].copy()
        if "predicted_risk" in display.columns:
            display["predicted_risk"] = display["predicted_risk"].apply(
                lambda b: CGPA_BAND_DISPLAY.get(b, b) if pd.notna(b) else "—"
            )
            display = display.rename(columns={"predicted_risk": "predicted_band"})
        if "confidence" in display.columns:
            display["confidence"] = display["confidence"].apply(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ---- recent predictions (all students) --------------------------------
    st.subheader("Recent Prediction Activity")
    preds = db.get_predictions(user_id=None, limit=10)
    if preds.empty:
        st.info("No predictions on record yet.")
    else:
        view = preds[[
            "created_at", "username", "predicted_risk", "confidence",
        ]].copy()
        view["predicted_risk"] = view["predicted_risk"].apply(
            lambda b: CGPA_BAND_DISPLAY.get(b, b))
        view = view.rename(columns={"predicted_risk": "predicted_band"})
        view["confidence"] = (view["confidence"] * 100).round(1).astype(str) + " %"
        st.dataframe(view, use_container_width=True, hide_index=True)