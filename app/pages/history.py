"""
app/pages/history.py
====================
Browses the prediction-history table from the database.

Educators see all predictions; students see only their own.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.auth import current_user, require_login
from config import RISK_CLASSES, RISK_COLORS
from database import get_db
from utils.reports import predictions_to_csv_bytes


def render():
    require_login()
    user = current_user()
    db = get_db()

    st.title("📚 Prediction History")
    st.caption("Browse predictions saved to the database.")

    show_all = False
    if user["role"] in ("admin", "educator"):
        show_all = st.toggle("Show predictions from **all** users", value=True)

    df = db.get_predictions(user_id=None if show_all else user["user_id"], limit=2000)

    if df.empty:
        st.info("No predictions on record yet.")
        return

    # ---- filters --------------------------------------------------------
    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            risk_filter = st.multiselect(
                "Risk class",
                options=RISK_CLASSES,
                default=RISK_CLASSES,
            )
        with f2:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, step=0.05)

    fdf = df[
        df["predicted_risk"].isin(risk_filter)
        & (df["confidence"] >= min_conf)
    ].copy()

    # ---- summary --------------------------------------------------------
    col_counts = st.columns(1 + len(RISK_CLASSES))
    col_counts[0].metric("Total", len(fdf))
    for i, cls in enumerate(RISK_CLASSES):
        col_counts[i + 1].metric(cls, int((fdf["predicted_risk"] == cls).sum()))

    # ---- chart ---------------------------------------------------------
    if not fdf.empty:
        dist = (
            fdf["predicted_risk"]
            .value_counts()
            .reindex(RISK_CLASSES)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        dist.columns = ["risk", "count"]
        fig = px.bar(dist, x="risk", y="count", color="risk", text="count",
                     color_discrete_map=RISK_COLORS)
        fig.update_layout(showlegend=False, height=280,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- table ---------------------------------------------------------
    base_cols = ["prediction_id", "created_at", "username",
                 "predicted_risk", "confidence"]
    prob_cols = [c for c in fdf.columns if c.startswith("prob_")]
    show_cols = [c for c in base_cols + prob_cols if c in fdf.columns]
    show = fdf[show_cols].copy()
    show["confidence"] = (show["confidence"] * 100).round(2)
    for c in prob_cols:
        show[c] = (show[c] * 100).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- row details ---------------------------------------------------
    st.divider()
    st.subheader("Row details")
    if not fdf.empty:
        idx = st.selectbox("Prediction ID", options=fdf["prediction_id"].tolist())
        row = fdf[fdf["prediction_id"] == idx].iloc[0]
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**Risk**: {row['predicted_risk']}")
            st.markdown(f"**Confidence**: {row['confidence']*100:.2f}%")
            if "username" in row:
                st.markdown(f"**By**: {row['username']}  •  **At**: {row['created_at']}")
        with cc2:
            try:
                payload = json.loads(row["input_payload"])
                payload = {k: v for k, v in payload.items() if not k.startswith("_")}
                st.dataframe(
                    pd.Series(payload, name="value").rename_axis("field").reset_index(),
                    use_container_width=True, hide_index=True,
                )
            except Exception:
                st.write(row["input_payload"])

    # ---- export ---------------------------------------------------------
    st.divider()
    st.download_button(
        "⬇️ Download filtered history as CSV",
        data=predictions_to_csv_bytes(fdf),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    # ---- destructive (admin only) ---------------------------------------
    if user["role"] == "admin":
        with st.expander("🗑️ Admin actions"):
            if st.button("Clear all prediction history",
                         key="clear_pred_history", type="secondary"):
                n = db.clear_predictions()
                st.success(f"Removed {n} prediction(s).")
                st.rerun()
