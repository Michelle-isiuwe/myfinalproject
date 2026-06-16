"""
app/pages/student_results.py
=============================
Student's My Results page.

Shows the most recent prediction with SHAP and LIME explanations, plus
rule-based suggestions for improvement, so the student can understand
and act on what drove their predicted outcome.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.auth import current_user, require_login
from config import CGPA_BAND_DISPLAY, RISK_COLORS
from database import get_db
from models.inference import ABSTENTION_THRESHOLD, ModelRegistry, predict_single
from utils.explainability import lime_explain, shap_explain
from utils.friendly_labels import (
    friendly_confidence, friendly_probability,
    build_lime_narrative, build_shap_narrative,
)
from utils.recommendations import (
    recommendations_from_shap,
    detailed_recommendations_from_shap,
)


def _load_latest_prediction(db, user_id: int) -> dict | None:
    """Prefer the latest saved prediction; fall back to this session."""
    view = db.get_latest_prediction_view(user_id)
    if view:
        return view
    return st.session_state.get("last_prediction")


def _render_educator_feedback(db, student_user_id: int, *, mark_read: bool) -> None:
    feedback = db.get_educator_feedback_for_student(student_user_id)
    if feedback.empty:
        return
    if mark_read:
        db.mark_feedback_read(student_user_id)
    st.subheader("💬 From your educator")
    st.caption("Personal advice from your lecturer — separate from the automated tips below.")
    for _, row in feedback.iterrows():
        name = row["educator_name"] or row["educator_username"]
        with st.container(border=True):
            st.markdown(f"**{name}** · {row['created_at']}")
            if pd.notna(row.get("prediction_id")) and pd.notna(row.get("prediction_at")):
                band = row.get("predicted_risk")
                band_label = CGPA_BAND_DISPLAY.get(band, band) if band else "—"
                st.caption(
                    f"Re: your prediction on {row['prediction_at']} ({band_label})"
                )
            st.markdown(row["message"])
    st.divider()


def render():
    require_login()
    user = current_user()
    db = get_db()
    registry = ModelRegistry()

    st.title("📈 My Results")
    st.caption("Your latest prediction result and what drove it.")

    _render_educator_feedback(db, user["user_id"], mark_read=True)

    if not registry.is_trained():
        st.error("No trained models found. Please contact your educator.")
        return

    last = _load_latest_prediction(db, user["user_id"])
    if not last:
        st.info(
            "You haven't made a prediction yet. Go to the **My Prediction** page "
            "and submit the form first."
        )
        return

    result = last["result"]
    band       = result["predicted_band"]
    band_label = CGPA_BAND_DISPLAY.get(band, band)
    confidence = result["confidence"]
    probs      = result["probabilities"]

    if last.get("timestamp"):
        st.caption(f"Based on your prediction from {last['timestamp']}.")

    # ---- result summary ---------------------------------------------------
    conf_info = friendly_confidence(confidence)
    abstained = result.get("abstained", False)

    if abstained:
        st.warning(
            "⚠️ **The system was not confident enough to give a definitive "
            "prediction for this submission.**"
        )
        with st.container(border=True):
            st.markdown(
                f"The system needs at least **{ABSTENTION_THRESHOLD*100:.0f}%** "
                f"confidence. Your result was only **{confidence*100:.0f}%** certain."
            )
            st.markdown(f"**Best guess:** {band_label} — but treat this with caution.")

            reasons = result.get("abstention_reasons", [])
            if reasons:
                st.markdown("#### Why is this uncertain?")
                for r in reasons:
                    st.markdown(f"- **{r['title']}**: {r['detail']}")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Predicted Performance", band_label)
        c2.markdown(
            f"**Confidence**\n\n"
            f"{conf_info['emoji']} **{conf_info['text']}** ({confidence*100:.0f}%)"
        )
        c2.caption(conf_info['detail'])

        dist = pd.DataFrame([{"Class": CGPA_BAND_DISPLAY.get(k, k),
                              "Probability (%)": round(v * 100, 1),
                              "Meaning": friendly_probability(v)["text"]}
                              for k, v in probs.items()])
        fig = px.bar(dist, x="Probability (%)", y="Class", orientation="h",
                     text=dist.apply(lambda r: f"{r['Probability (%)']}% — {r['Meaning']}", axis=1),
                     title="How likely is each performance band?")
        fig.update_layout(
            height=240, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- shared explainability setup -------------------------------------
    @st.cache_data(show_spinner=False)
    def _cached_background():
        from utils.preprocessing import load_and_prepare
        X, _, _ = load_and_prepare()
        preproc = registry.get_preprocessor()
        bg = preproc.transform(X.sample(min(80, len(X)), random_state=42))
        return bg, list(preproc.get_feature_names_out())

    background, feature_names = _cached_background()
    le          = registry.get_label_encoder()
    class_names = list(le.classes_)

    # Recompute X_transformed from stored payload
    res      = predict_single(last["payload"], last["model"], registry)
    X_row    = res["X_transformed"][0]
    class_idx = class_names.index(res["predicted_band"])
    model     = registry.get_model(last["model"])

    method = st.radio("Explanation method", ["SHAP", "LIME", "Both"],
                      horizontal=True, key="student_explain_method")

    # ---- SHAP -------------------------------------------------------------
    if method in ("SHAP", "Both"):
        st.subheader("🔍 What drove this prediction?")
        st.caption("Shows which of your study habits helped or hurt your predicted performance.")
        with st.spinner("Computing SHAP values…"):
            try:
                shap_res = shap_explain(model, X_row, background,
                                        feature_names, class_idx, class_names)
                df = pd.DataFrame({
                    "feature":    shap_res["feature_names"],
                    "shap_value": shap_res["shap_values"],
                })
                df["abs"] = df["shap_value"].abs()
                top = df.sort_values("abs", ascending=False).head(15).iloc[::-1]
                fig = px.bar(
                    top, x="shap_value", y="feature", orientation="h",
                    color="shap_value", color_continuous_scale="RdBu_r",
                    title=f"Top factors for predicting '{band_label}'",
                )
                fig.update_layout(
                    height=520, coloraxis_showscale=True,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- Human-readable SHAP narrative ----
                narratives = build_shap_narrative(df, band_label, max_items=6)
                if narratives:
                    st.markdown("#### 📖 What this means")
                    st.caption(
                        "Each factor below explains how one of your habits "
                        "influenced the prediction, in plain language."
                    )
                    for n in narratives:
                        st.markdown(n["sentence"])

                with st.expander("Show full SHAP table"):
                    st.dataframe(
                        df.sort_values("abs", ascending=False).drop(columns="abs"),
                        use_container_width=True,
                    )

                # ---- Areas to improve ----
                recs = detailed_recommendations_from_shap(
                    df[["feature", "shap_value"]],
                    payload=last.get("payload"),
                    max_items=4,
                )
                if recs:
                    st.subheader("📌 Areas to Improve")
                    st.caption(
                        "Based on the factors that lowered your prediction "
                        "— and that you can act on."
                    )
                    for rec in recs:
                        sev = rec["severity"]
                        with st.container(border=True):
                            st.markdown(
                                f"{sev['emoji']} **{rec['feature_name']}** — "
                                f"*{sev['label']}*"
                            )
                            st.markdown(f"**You answered:** {rec['current_value']}")
                            st.markdown(f"💡 {rec['advice']}")
                else:
                    st.success(
                        "✅ No specific improvement areas stood out — keep up "
                        "your current habits."
                    )
            except Exception as e:
                st.error(f"SHAP failed: {e}")

    # ---- LIME -------------------------------------------------------------
    if method in ("LIME", "Both"):
        st.subheader("🔍 LIME Explanation")
        st.caption(
            "An alternative way of showing which factors drove this "
            "prediction and whether each one helped or hurt."
        )
        with st.spinner("Computing LIME explanation…"):
            try:
                lime_res = lime_explain(model, X_row, background, feature_names,
                                        class_names, class_idx, num_features=12)
                rows = pd.DataFrame(lime_res["pairs"], columns=["feature", "weight"])
                rows = rows.iloc[::-1]
                lime_label = CGPA_BAND_DISPLAY.get(
                    lime_res["class_name"], lime_res["class_name"]
                )
                fig = px.bar(
                    rows, x="weight", y="feature", orientation="h",
                    color="weight", color_continuous_scale="RdBu_r",
                    title=f"LIME contributions for '{lime_label}'",
                )
                fig.update_layout(
                    height=480, plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- LIME narrative ----
                narratives = build_lime_narrative(
                    lime_res["pairs"], lime_label, max_items=8
                )
                if narratives:
                    st.markdown("#### 📖 What this means")
                    st.caption(
                        "Each line below explains one factor and whether "
                        "it helped or hurt your prediction."
                    )
                    for n in narratives:
                        st.markdown(n["sentence"])

                with st.expander("LIME feature weights"):
                    st.dataframe(
                        pd.DataFrame(lime_res["pairs"], columns=["feature", "weight"]),
                        use_container_width=True,
                    )
                st.caption(f"Local fidelity score: {lime_res['score']:.3f}")
            except Exception as e:
                st.error(f"LIME failed: {e}")
