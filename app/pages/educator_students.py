"""
app/pages/educator_students.py
================================
Educator student management page.

Tabs:
  1. All Students — browse/select any registered student + view their
     latest prediction and full input breakdown.
  2. Students Who May Need Support — filtered to the lowest CGPA bands.
  3. Student Detail — SHAP and LIME for the selected student's prediction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.auth import current_user, require_login
from config import CGPA_BAND_DISPLAY, RISK_CLASSES, RISK_COLORS
from database import get_db
from models.inference import ModelRegistry, predict_single
from utils.explainability import lime_explain, shap_explain

# Lowest CGPA bands — students who may benefit from additional support.
# (Pass and Third Class.) Add "2.50 – 3.49" to include Second Class Lower.
SUPPORT_BANDS = ["Below 1.50", "1.50 – 2.49"]


def _submit_feedback(
    db,
    educator: dict,
    student_user_id: int,
    message: str,
    link_to_latest: bool,
) -> None:
    if educator.get("role") not in ("educator", "admin"):
        st.error("Only educators can send suggestions.")
        return
    latest = db.get_latest_student_prediction(student_user_id)
    pred_id = (
        int(latest["prediction_id"])
        if link_to_latest and latest is not None
        else None
    )
    db.add_educator_feedback(
        student_user_id=student_user_id,
        educator_user_id=educator["user_id"],
        message=message,
        prediction_id=pred_id,
    )


def _render_feedback_form(
    db,
    educator: dict,
    student_user_id: int,
    *,
    form_key: str,
    latest: dict | None,
) -> None:
    with st.form(form_key, clear_on_submit=True):
        advice = st.text_area(
            "Your advice",
            height=100,
            placeholder="e.g. Please attend tutorials regularly and revise earlier before tests.",
        )
        link_latest = st.checkbox(
            "Link to their latest prediction",
            value=latest is not None,
            disabled=latest is None,
            help="The student will see which prediction this refers to.",
        )
        if st.form_submit_button("Send suggestion", type="primary"):
            if not advice.strip():
                st.error("Please enter a message before sending.")
            else:
                _submit_feedback(db, educator, student_user_id, advice, link_latest)
                st.success("Suggestion sent to the student.")
                st.rerun()


def _render_feedback_history(db, student_user_id: int) -> None:
    past = db.get_educator_feedback_for_student(student_user_id)
    if past.empty:
        st.caption("No suggestions sent yet.")
        return
    for _, row in past.iterrows():
        name = row["educator_name"] or row["educator_username"]
        with st.container(border=True):
            st.markdown(f"**{name}** · {row['created_at']}")
            if pd.notna(row.get("prediction_id")) and pd.notna(row.get("prediction_at")):
                band = row.get("predicted_risk")
                band_label = CGPA_BAND_DISPLAY.get(band, band) if band else "—"
                st.caption(
                    f"Linked to prediction on {row['prediction_at']} ({band_label})"
                )
            st.markdown(row["message"])


def render():
    require_login()
    db       = get_db()
    registry = ModelRegistry()

    st.title("👥 Students")
    st.caption("Browse registered students, view their predictions, and inspect explanations.")

    students_df = db.list_students_with_latest_prediction()

    tab_all, tab_risk, tab_detail = st.tabs([
        "📋 All Students",
        "🤝 May Need Support",
        "🔍 Student Detail",
    ])

    # ====================================================================
    # Tab 1 — All Students
    # ====================================================================
    with tab_all:
        if students_df.empty:
            st.info("No students have registered yet.")
        else:
            # Search filter
            search = st.text_input("Search by name or username", key="search_students")
            view = students_df.copy()
            if search:
                mask = (
                    view["full_name"].str.contains(search, case=False, na=False)
                    | view["username"].str.contains(search, case=False, na=False)
                )
                view = view[mask]

            # Display table (prettify the band column if present)
            show_cols = [c for c in
                         ["full_name", "username", "level_of_study", "field_of_study",
                          "predicted_risk", "confidence", "predicted_at"]
                         if c in view.columns]
            display = view[show_cols].copy()
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
            st.subheader("Select a Student")
            if not view.empty:
                opts = view.apply(
                    lambda r: f"{r['full_name'] or r['username']} ({r['username']})", axis=1
                ).tolist()
                chosen_label = st.selectbox("Student", opts, key="select_student")
                chosen_idx   = opts.index(chosen_label)
                chosen_row   = view.iloc[chosen_idx]
                if st.button("View Student Detail", type="primary"):
                    st.session_state["educator_selected_student_id"] = int(chosen_row["user_id"])
                    st.session_state["educator_selected_student_name"] = (
                        chosen_row.get("full_name") or chosen_row["username"]
                    )
                    st.info("Switch to the **Student Detail** tab to see SHAP and LIME.")

    # ====================================================================
    # Tab 2 — Students Who May Need Support
    # ====================================================================
    with tab_risk:
        if students_df.empty:
            st.info("No students registered yet.")
        else:
            need_support = students_df[
                students_df["predicted_risk"].isin(SUPPORT_BANDS)
            ].copy()
            st.metric("Students Who May Need Support", len(need_support))
            if need_support.empty:
                st.success("No students currently fall into the lower CGPA bands.")
            else:
                if "predicted_risk" in need_support.columns:
                    need_support["predicted_risk"] = need_support["predicted_risk"].apply(
                        lambda b: CGPA_BAND_DISPLAY.get(b, b)
                    )
                    need_support = need_support.rename(
                        columns={"predicted_risk": "predicted_band"})
                show_cols = [c for c in
                             ["full_name", "username", "level_of_study", "field_of_study",
                              "predicted_band", "confidence", "predicted_at"]
                             if c in need_support.columns]
                display = need_support[show_cols].copy()
                if "confidence" in display.columns:
                    display["confidence"] = display["confidence"].apply(
                        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"
                    )
                st.dataframe(display, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Send a quick suggestion")
                st.caption(
                    "Reach out to a student in a lower band without opening Student Detail."
                )
                educator = current_user()
                support_opts = need_support.apply(
                    lambda r: f"{r['full_name'] or r['username']} ({r['username']})",
                    axis=1,
                ).tolist()
                pick = st.selectbox("Student", support_opts, key="support_feedback_student")
                pick_row = need_support.iloc[support_opts.index(pick)]
                pick_id = int(pick_row["user_id"])
                pick_latest = db.get_latest_student_prediction(pick_id)
                _render_feedback_form(
                    db, educator, pick_id,
                    form_key="support_quick_feedback",
                    latest=pick_latest,
                )

            st.divider()
            st.subheader("CGPA Band Distribution")
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
            fig = px.bar(dist, x="label", y="count", color="band", text="count",
                         color_discrete_map=RISK_COLORS)
            fig.update_layout(showlegend=False, height=280,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # Tab 3 — Student Detail
    # ====================================================================
    with tab_detail:
        sel_id = st.session_state.get("educator_selected_student_id")
        sel_name = st.session_state.get("educator_selected_student_name", "")

        if sel_id is None:
            st.info("Select a student from the **All Students** tab first.")
            return

        st.subheader(f"Detail: {sel_name}")

        educator = current_user()

        # Load profile
        profile = db.get_student_profile(sel_id)
        if profile:
            with st.expander("Student Profile", expanded=False):
                pc = st.columns(3)
                pc[0].markdown(f"**Gender:** {profile.get('gender', '—')}")
                pc[0].markdown(f"**Age Range:** {profile.get('age_range', '—')}")
                pc[1].markdown(f"**Level:** {profile.get('level_of_study', '—')}")
                pc[1].markdown(f"**Field:** {profile.get('field_of_study', '—')}")
                pc[2].markdown(f"**Internet:** {profile.get('internet_access', '—')}")
                pc[2].markdown(f"**Devices:** {profile.get('devices_used', '—')}")

        latest = db.get_latest_student_prediction(sel_id)

        st.divider()
        st.subheader("Suggestions for this student")
        _render_feedback_form(
            db, educator, sel_id,
            form_key="educator_feedback_form",
            latest=latest,
        )

        st.markdown("**Previous suggestions**")
        _render_feedback_history(db, sel_id)

        if not latest:
            st.info(
                f"**{sel_name}** has not made a prediction yet. You can still send "
                "suggestions above — they will appear on the student's **My Results** page."
            )
            return

        st.divider()
        st.subheader("Latest prediction")

        # Prediction summary (latest is a DB row → 'predicted_risk' column)
        band = latest["predicted_risk"]
        band_label = CGPA_BAND_DISPLAY.get(band, band)
        with st.container(border=True):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Predicted Performance", band_label)
            mc2.metric("Confidence", f"{latest['confidence']*100:.1f}%")
            mc3.metric("Model", latest["model_used"])

        # Input breakdown
        st.subheader("Input Features")
        try:
            payload = json.loads(latest["input_payload"])
            payload_clean = {k: v for k, v in payload.items() if not k.startswith("_")}
            st.dataframe(
                pd.Series(payload_clean, name="value").rename_axis("field").reset_index(),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.write(latest["input_payload"])

        # All predictions for this student
        st.divider()
        st.subheader("Prediction History")
        hist_df = db.get_student_predictions(sel_id)
        if not hist_df.empty:
            show = hist_df[["prediction_id", "created_at", "model_used",
                            "predicted_risk", "confidence"]].copy()
            show["predicted_risk"] = show["predicted_risk"].apply(
                lambda b: CGPA_BAND_DISPLAY.get(b, b))
            show = show.rename(columns={"predicted_risk": "predicted_band"})
            show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + " %"
            st.dataframe(show, use_container_width=True, hide_index=True)

        # SHAP / LIME
        if not registry.is_trained():
            st.warning("Model not trained. Cannot compute explanations.")
            return

        st.divider()
        st.subheader("SHAP & LIME Explanations")

        # Reconstruct X_row from stored payload
        try:
            payload = json.loads(latest["input_payload"])
            model_name = latest["model_used"]
            res   = predict_single(payload, model_name, registry)
            X_row = res["X_transformed"][0]
        except Exception as e:
            st.error(f"Could not reconstruct prediction input: {e}")
            return

        @st.cache_data(show_spinner=False)
        def _bg():
            from utils.preprocessing import load_and_prepare
            X, _, _ = load_and_prepare()
            preproc = registry.get_preprocessor()
            bg = preproc.transform(X.sample(min(80, len(X)), random_state=42))
            return bg, list(preproc.get_feature_names_out())

        background, feature_names = _bg()
        le          = registry.get_label_encoder()
        class_names = list(le.classes_)
        pred_band   = res["predicted_band"]
        pred_label  = CGPA_BAND_DISPLAY.get(pred_band, pred_band)
        class_idx   = class_names.index(pred_band)
        model       = registry.get_model(model_name)

        method = st.radio("Method", ["SHAP", "LIME", "Both"],
                          horizontal=True, key="edu_explain_method")

        if method in ("SHAP", "Both"):
            st.markdown("**SHAP local explanation**")
            with st.spinner("Computing SHAP…"):
                try:
                    shap_res = shap_explain(model, X_row, background,
                                            feature_names, class_idx, class_names)
                    df_shap = pd.DataFrame({
                        "feature":    shap_res["feature_names"],
                        "shap_value": shap_res["shap_values"],
                    })
                    df_shap["abs"] = df_shap["shap_value"].abs()
                    top = df_shap.sort_values("abs", ascending=False).head(15).iloc[::-1]
                    fig = px.bar(
                        top, x="shap_value", y="feature", orientation="h",
                        color="shap_value", color_continuous_scale="RdBu_r",
                        title=f"Top factors for '{pred_label}'",
                    )
                    fig.update_layout(
                        height=520, coloraxis_showscale=True,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Full SHAP table"):
                        st.dataframe(
                            df_shap.sort_values("abs", ascending=False).drop(columns="abs"),
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"SHAP failed: {e}")

        if method in ("LIME", "Both"):
            st.markdown("**LIME local explanation**")
            with st.spinner("Computing LIME…"):
                try:
                    lime_res = lime_explain(model, X_row, background, feature_names,
                                            class_names, class_idx, num_features=12)
                    rows = pd.DataFrame(lime_res["pairs"], columns=["feature", "weight"]).iloc[::-1]
                    lime_label = CGPA_BAND_DISPLAY.get(lime_res["class_name"], lime_res["class_name"])
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
                    with st.expander("LIME pairs"):
                        st.dataframe(
                            pd.DataFrame(lime_res["pairs"], columns=["feature", "weight"]),
                            use_container_width=True,
                        )
                    st.caption(f"Local fidelity: {lime_res['score']:.3f}")
                except Exception as e:
                    st.error(f"LIME failed: {e}")