"""
app/pages/explainability.py
===========================
SHAP + LIME visualisations for:
  1. The most recent prediction (local explanation).
  2. The chosen model overall (global feature importance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.auth import require_login
from config import CGPA_BAND_DISPLAY, ENSEMBLE_STRATEGIES
from models.inference import ModelRegistry, predict_single
from utils.explainability import (
    global_feature_importance, lime_explain, shap_explain,
)
from utils.preprocessing import load_and_prepare


def _slug(name): return (name.lower().replace(" ", "_").replace("(", "")
                          .replace(")", "").replace("-", "_"))


def render():
    require_login()
    registry = ModelRegistry()

    st.title("🧠 Explainability")
    st.caption("Inspect *why* the model made a particular prediction (SHAP, LIME) and which features matter overall.")

    if not registry.is_trained():
        st.error("No trained models found. Train the model first.")
        return

    avail = registry.available_models()
    available_pretty = [n for n in (["Random Forest", "XGBoost"] + ENSEMBLE_STRATEGIES)
                        if _slug(n) in avail]

    # ---- shared background data ------------------------------------------
    @st.cache_data(show_spinner=False)
    def _cached_background():
        X, y, _ = load_and_prepare()
        preproc = registry.get_preprocessor()
        bg = preproc.transform(X.sample(min(80, len(X)), random_state=42))
        return bg, list(preproc.get_feature_names_out())

    background, feature_names = _cached_background()
    le = registry.get_label_encoder()
    class_names = list(le.classes_)

    tab_local, tab_global = st.tabs(["🔍 Local explanation (last prediction)",
                                     "🌍 Global feature importance"])

    # =================== LOCAL =============================================
    with tab_local:
        last = st.session_state.get("last_prediction")
        if not last:
            st.info("Make a prediction first on the **Prediction** page. The explanation will appear here.")
            return

        last_band = last["result"]["predicted_band"]
        last_label = CGPA_BAND_DISPLAY.get(last_band, last_band)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Last predicted band", last_label)
            c2.metric("Confidence", f"{last['result']['confidence']*100:.2f}%")
            c3.metric("Model", last["model"])

        # Recompute X_transformed (it isn't kept across reruns)
        model = registry.get_model(last["model"])
        res = predict_single(last["payload"], last["model"], registry)
        X_row = res["X_transformed"][0]
        pred_band = res["predicted_band"]
        pred_label = CGPA_BAND_DISPLAY.get(pred_band, pred_band)
        class_idx = class_names.index(pred_band)

        method = st.radio("Explanation method", ["SHAP", "LIME", "Both"],
                          horizontal=True, key="explain_method")

        # ---- SHAP -----------------------------------------------------
        if method in ("SHAP", "Both"):
            st.subheader("SHAP local explanation")
            with st.spinner("Computing SHAP values…"):
                try:
                    shap_res = shap_explain(model, X_row, background,
                                            feature_names, class_idx, class_names)
                    df = pd.DataFrame({
                        "feature": shap_res["feature_names"],
                        "shap_value": shap_res["shap_values"],
                    })
                    df["abs"] = df["shap_value"].abs()
                    top = df.sort_values("abs", ascending=False).head(15).iloc[::-1]
                    top["direction"] = np.where(top["shap_value"] >= 0, "Pushes towards", "Pushes away from")

                    fig = px.bar(
                        top, x="shap_value", y="feature", orientation="h",
                        color="shap_value", color_continuous_scale="RdBu_r",
                        title=f"Top 15 contributors towards predicting {pred_label!r}",
                    )
                    fig.update_layout(
                        height=520, coloraxis_showscale=True,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show full SHAP table"):
                        st.dataframe(df.sort_values("abs", ascending=False)
                                     .drop(columns="abs"), use_container_width=True)
                    # Persist for the report
                    st.session_state["last_explanation"] = {
                        "shap_top": top[["feature", "shap_value"]],
                    }
                except Exception as e:
                    st.error(f"SHAP failed: {e}")

        # ---- LIME -----------------------------------------------------
        if method in ("LIME", "Both"):
            st.subheader("LIME local explanation")
            with st.spinner("Computing LIME explanation…"):
                try:
                    lime_res = lime_explain(model, X_row, background, feature_names,
                                            class_names, class_idx, num_features=12)
                    rows = pd.DataFrame(lime_res["pairs"], columns=["feature", "weight"])
                    rows = rows.iloc[::-1]
                    lime_label = CGPA_BAND_DISPLAY.get(lime_res["class_name"], lime_res["class_name"])
                    fig = px.bar(
                        rows, x="weight", y="feature", orientation="h",
                        color="weight", color_continuous_scale="RdBu_r",
                        title=f"LIME contributions towards class {lime_label!r}",
                    )
                    fig.update_layout(
                        height=480, plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show LIME pairs"):
                        st.dataframe(pd.DataFrame(lime_res["pairs"],
                                                  columns=["feature", "weight"]),
                                     use_container_width=True)
                    st.caption(f"LIME local fidelity score: {lime_res['score']:.3f}")
                    if "last_explanation" in st.session_state:
                        st.session_state["last_explanation"]["lime_pairs"] = lime_res["pairs"]
                    else:
                        st.session_state["last_explanation"] = {"lime_pairs": lime_res["pairs"]}
                except Exception as e:
                    st.error(f"LIME failed: {e}")

    # =================== GLOBAL ===========================================
    with tab_global:
        gmodel_choice = st.selectbox("Model to inspect", options=available_pretty,
                                     key="global_model_select")
        if st.button("Compute global importance", key="run_global"):
            with st.spinner("Computing global SHAP importance (this can take ~30s)…"):
                try:
                    gmodel = registry.get_model(gmodel_choice)
                    gfi = global_feature_importance(gmodel, background,
                                                    feature_names, max_samples=60)
                    top = gfi.head(20).iloc[::-1]
                    fig = px.bar(top, x="importance", y="feature", orientation="h",
                                 color="importance", color_continuous_scale="Viridis",
                                 title=f"Top 20 features — {gmodel_choice}")
                    fig.update_layout(
                        height=560, plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show full global importance table"):
                        st.dataframe(gfi, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not compute global importance: {e}")