"""
app/pages/student_history.py
=============================
Student's own prediction history page.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.auth import current_user, require_login
from config import CGPA_BAND_DISPLAY, RISK_CLASSES, RISK_COLORS
from database import get_db


def render():
    require_login()
    user = current_user()
    db   = get_db()

    st.title("📜 My History")
    st.caption("All your previous predictions.")

    df = db.get_predictions(user_id=user["user_id"], limit=500)

    if df.empty:
        st.info("You haven't made any predictions yet. Go to **My Prediction** to get started.")
        return

    # ---- summary ---------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", len(df))
    latest_band = df.iloc[0]["predicted_risk"]
    c2.metric("Latest Result", CGPA_BAND_DISPLAY.get(latest_band, latest_band))
    c3.metric("Latest Confidence", f"{df.iloc[0]['confidence']*100:.1f}%")

    # ---- distribution chart ----------------------------------------------
    dist = (
        df["predicted_risk"]
        .value_counts()
        .reindex(RISK_CLASSES)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    dist.columns = ["band", "count"]
    dist["label"] = dist["band"].map(lambda b: CGPA_BAND_DISPLAY.get(b, b))
    fig = px.bar(dist, x="label", y="count", color="band", text="count",
                 color_discrete_map=RISK_COLORS, title="Your prediction history")
    fig.update_layout(showlegend=False, height=260,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- table -----------------------------------------------------------
    show = df[["prediction_id", "created_at", "predicted_risk", "confidence"]].copy()
    show["predicted_risk"] = show["predicted_risk"].map(
        lambda b: CGPA_BAND_DISPLAY.get(b, b))
    show = show.rename(columns={"predicted_risk": "predicted_band"})
    show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + " %"
    st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- row detail ------------------------------------------------------
    st.divider()
    st.subheader("Prediction detail")
    idx = st.selectbox("Select prediction", options=df["prediction_id"].tolist())
    row = df[df["prediction_id"] == idx].iloc[0]
    d1, d2 = st.columns(2)
    with d1:
        band = row["predicted_risk"]
        st.markdown(f"**Result**: {CGPA_BAND_DISPLAY.get(band, band)}")
        st.markdown(f"**Confidence**: {row['confidence']*100:.2f}%")
        st.markdown(f"**Date**: {row['created_at']}")
    with d2:
        try:
            payload = json.loads(row["input_payload"])
            payload = {k: v for k, v in payload.items() if not k.startswith("_")}
            st.dataframe(
                pd.Series(payload, name="value").rename_axis("field").reset_index(),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.write(row["input_payload"])