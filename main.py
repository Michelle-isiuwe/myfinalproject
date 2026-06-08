"""
main.py
=======
EduPredict — Streamlit entry point.

Role-based navigation:
  student  → Prediction, My Results, My History, Account
  educator → Dashboard, Students, Prediction History, Account
  admin    → Manage Accounts, Account

Run with::

    streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EduPredict — Explainable Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.auth import current_user, is_authenticated, logout          # noqa: E402
from app.pages import (                                              # noqa: E402
    admin,
    dashboard,
    history,
    login,
)
from app.pages import educator_students                               # noqa: E402
from app.pages import student_prediction                              # noqa: E402
from app.pages import student_results                                 # noqa: E402
from app.pages import student_history                                 # noqa: E402
from app.feedback_alerts import render_unread_feedback_banner                 # noqa: E402
from app.theme import apply_theme, hide_streamlit_branding, theme_toggle  # noqa: E402

apply_theme()
hide_streamlit_branding()


# ── Page wrappers ────────────────────────────────────────────────────────
def page_login():            login.render()
def page_dashboard():        dashboard.render()
def page_educator_students():educator_students.render()
def page_history():          history.render()
def page_admin():            admin.render()
def page_student_prediction():student_prediction.render()
def page_student_results():  student_results.render()
def page_student_history():  student_history.render()


# ── Sidebar ──────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🎓 EduPredict")
        st.caption("Explainable Student Performance Prediction")
        st.divider()

        if is_authenticated():
            user = current_user()
            st.markdown("**Signed in as**")
            st.markdown(f"👤 `{user['username']}`")
            st.markdown(f"🛡️ Role: **{user['role'].title()}**")
            if user.get("full_name"):
                st.caption(user["full_name"])
            if user.get("role") == "student":
                render_unread_feedback_banner(user["user_id"])
            st.divider()

        theme_toggle()

        st.divider()
        if is_authenticated():
            if st.button("🚪 Log out", use_container_width=True):
                logout()
                st.rerun()


# ── Navigation ───────────────────────────────────────────────────────────
def build_navigation():
    if not is_authenticated():
        return st.navigation([
            st.Page(page_login, title="Sign in", icon="🔐", default=True),
        ])

    user = current_user()
    role = user.get("role")

    if role == "student":
        return st.navigation({
            "My Learning": [
                st.Page(page_student_prediction, title="My Prediction", icon="🔮", default=True),
                st.Page(page_student_results,    title="My Results",    icon="📈"),
                st.Page(page_student_history,    title="My History",    icon="📜"),
                st.Page(page_login,              title="Profile",       icon="👤"),
            ]
        })

    if role == "educator":
        return st.navigation({
            "Workspace": [
                st.Page(page_dashboard,          title="Dashboard",         icon="📊", default=True),
                st.Page(page_educator_students,  title="Students",          icon="👥"),
                st.Page(page_history,            title="Prediction History",icon="📚"),
                st.Page(page_login,              title="Profile",           icon="👤"),
            ]
        })

    if role == "admin":
        return st.navigation({
            "Administration": [
                st.Page(page_admin,  title="Manage Accounts", icon="⚙️", default=True),
                st.Page(page_login,  title="Profile",         icon="👤"),
            ]
        })

    # Fallback for unknown roles
    return st.navigation([
        st.Page(page_login, title="Sign in", icon="🔐", default=True),
    ])


def main() -> None:
    render_sidebar()
    nav = build_navigation()
    nav.run()


if __name__ == "__main__":
    main()
