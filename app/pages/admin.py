"""
app/pages/admin.py
==================
Administration page (admin-only).

Tabs:
  1. User Management       — list / create / delete student and educator accounts
  2. Database Statistics   — aggregate counts and selection history
  3. Data Management       — list saved models, view metadata, danger zone
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.auth import require_role
from config import CGPA_BAND_DISPLAY, MODELS_DIR
from database import get_db


def render() -> None:
    """Render the admin page."""
    require_role("admin")

    st.title("⚙️ Manage Accounts")
    st.caption("Create, view, and delete student and educator accounts.")

    tab1, tab2, tab3 = st.tabs([
        "👥 User Management",
        "📊 Database Statistics",
        "🗄️ Data Management",
    ])

    with tab1:
        _render_user_tab()

    with tab2:
        _render_stats_tab()

    with tab3:
        _render_data_tab()


# ===========================================================================
# Tabs
# ===========================================================================
def _render_user_tab() -> None:
    st.subheader("Registered Users")
    db = get_db()
    users_df = db.list_users()

    if users_df.empty:
        st.info("No users registered yet.")
    else:
        display_cols = [c for c in
                        ["user_id", "username", "full_name", "email",
                         "role", "created_at", "last_login"]
                        if c in users_df.columns]
        st.dataframe(
            users_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Create Account")
    with st.form("create_user_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input(
                "Password (≥ 6 characters)", type="password"
            )
        with c2:
            new_fullname = st.text_input("Full name")
            new_role = st.selectbox("Role", ["student", "educator"])

        submitted = st.form_submit_button(
            "➕ Create Account", use_container_width=True
        )
        if submitted:
            if not new_username or not new_password or not new_email:
                st.error("Username, email and password are required.")
            else:
                ok, msg = db.create_user(
                    username=new_username,
                    full_name=new_fullname or new_username,
                    email=new_email,
                    password=new_password,
                    role=new_role,
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")
    st.subheader("Reset Password")
    st.caption(
        "Set a new password for a user. The old password cannot be recovered."
    )
    if users_df.empty:
        st.caption("No users to reset.")
    else:
        with st.form("reset_password_form", clear_on_submit=True):
            reset_labels = users_df.apply(
                lambda r: (
                    f"{r['username']} — {r.get('full_name') or r['username']} "
                    f"({r['role']})"
                ),
                axis=1,
            ).tolist()
            reset_pick = st.selectbox("User", reset_labels, key="reset_password_user")
            reset_row = users_df.iloc[reset_labels.index(reset_pick)]
            rp1, rp2 = st.columns(2)
            with rp1:
                reset_pass = st.text_input(
                    "New password (≥ 6 characters)", type="password"
                )
            with rp2:
                reset_pass2 = st.text_input(
                    "Confirm new password", type="password"
                )
            if st.form_submit_button("🔑 Reset password", use_container_width=True):
                if not reset_pass or not reset_pass2:
                    st.error("Enter and confirm the new password.")
                elif reset_pass != reset_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.reset_user_password(
                        int(reset_row["user_id"]), reset_pass
                    )
                    if ok:
                        st.success(
                            f"{msg} User **{reset_row['username']}** can sign in "
                            "with the new password."
                        )
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown("---")
    st.subheader("Delete Account")
    if not users_df.empty:
        current_username = st.session_state["user"]["username"]
        deletable = users_df[users_df["username"] != current_username]
        if deletable.empty:
            st.caption("No other users available to delete.")
        else:
            target_user = st.selectbox(
                "Select user to delete",
                options=deletable["username"].tolist(),
            )
            target_row = deletable[deletable["username"] == target_user].iloc[0]
            confirm = st.checkbox(
                f"I confirm deletion of '{target_user}'", value=False,
            )
            if st.button("🗑️ Delete Account", disabled=not confirm):
                db.delete_user(int(target_row["user_id"]))
                st.success(f"Account '{target_user}' deleted.")
                st.rerun()


def _render_stats_tab() -> None:
    st.subheader("Database Statistics")
    db = get_db()
    stats = db.get_statistics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users",             stats.get("total_users", 0))
    c2.metric("Predictions",       stats.get("total_predictions", 0))
    c3.metric("Students",          stats.get("total_students", 0))
    c4.metric("Saved Predictions", stats.get("total_saved_predictions", 0))

    st.markdown("---")
    st.subheader("Predictions by CGPA Band")
    risk_breakdown = stats.get("risk_breakdown", {})
    if risk_breakdown:
        band_df = pd.DataFrame(
            [{"CGPA Band": CGPA_BAND_DISPLAY.get(k, k), "Count": v}
             for k, v in risk_breakdown.items()]
        )
        st.bar_chart(band_df.set_index("CGPA Band"))
    else:
        st.caption("No predictions recorded yet.")

    st.markdown("---")
    st.subheader("Model Selection History")
    sel_history = db.get_model_selection_history()
    if sel_history.empty:
        st.caption("No model selections committed yet.")
    else:
        st.dataframe(sel_history, use_container_width=True, hide_index=True)


def _render_data_tab() -> None:
    st.subheader("Saved Models")
    model_files = sorted(Path(MODELS_DIR).glob("*.joblib"))
    if model_files:
        files_data = [
            {"Model File": f.name,
             "Size (KB)": round(f.stat().st_size / 1024, 1)}
            for f in model_files
        ]
        st.dataframe(
            pd.DataFrame(files_data),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No trained models found.")

    meta_path = MODELS_DIR / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as fp:
            meta = json.load(fp)
        with st.expander("Training metadata"):
            st.json(meta)

    st.markdown("---")
    st.subheader("⚠️ Danger Zone")
    st.warning("These actions are irreversible.")

    col_x, col_y = st.columns(2)
    with col_x:
        if st.checkbox("I want to clear all prediction history"):
            if st.button("🗑️ Clear All Predictions", type="secondary"):
                n = get_db().clear_predictions()
                st.success(f"Cleared {n} prediction records.")
                st.rerun()

    with col_y:
        if st.checkbox("I want to clear all saved (bookmarked) predictions"):
            if st.button("🗑️ Clear Saved Predictions", type="secondary"):
                n = get_db().clear_saved_predictions()
                st.success(f"Cleared {n} saved predictions.")
                st.rerun()