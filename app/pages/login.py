"""
app/pages/login.py
==================
Login + student registration page.

Students self-register and provide 6 profile fields (stored once).
Educator/admin accounts are created by an admin only.
"""

import streamlit as st

from app.auth import current_user, is_authenticated, login, logout
from config import ORDINAL_ORDERINGS, NOMINAL_COLS
from database import get_db
from utils.preprocessing import load_raw


def _options_for(col: str) -> list:
    if col in ORDINAL_ORDERINGS:
        return ORDINAL_ORDERINGS[col]
    df = load_raw()
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []


def render():
    if is_authenticated():
        user = current_user()
        st.title("👤 Profile")
        st.success(f"Signed in as **{user['username']}** (role: *{user['role']}*).")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Username", user["username"])
            st.metric("Role", user["role"].title())
        with col2:
            st.metric("Email", user.get("email") or "—")
            st.metric("Last login", str(user.get("last_login") or "—"))
        if st.button("Log out", type="primary"):
            logout()
            st.rerun()
        return

    st.title("🔐 Welcome to EduPredict")
    st.caption("Explainable student academic performance prediction.")

    with st.container(border=True):
        st.markdown(
            """
            **What you can do here**
            - **Students** — register, submit study habits, and get an explainable CGPA band prediction with tips to improve.
            - **Educators** — sign in with an account created by your admin; review students, send personal advice, and see who may need support.
            - **Administrators** — manage educator and student accounts.

            Predictions are **indicative only** — they support learning, not final grades.
            """
        )

    tab_register, tab_login = st.tabs(["Student Registration", "Sign in"])

    # ---- Student Registration ----------------------------------------------
    with tab_register:
        with st.container(border=True):
            st.subheader("Create a student account")
            st.caption("These profile details are saved once and never asked again.")

            c1, c2 = st.columns(2)
            with c1:
                r_name  = st.text_input("Full name", key="reg_name")
                r_user  = st.text_input("Username", key="reg_user")
                r_email = st.text_input("Email", key="reg_email")
                r_pass  = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
                r_pass2 = st.text_input("Confirm password", type="password", key="reg_pass2")
            with c2:
                r_gender   = st.selectbox("Gender", _options_for("Gender"), key="reg_gender")
                r_age      = st.selectbox("Age Range", _options_for("Age Range"), key="reg_age")
                r_level    = st.selectbox("Level of Study", _options_for("Level of Study"), key="reg_level")
                r_field    = st.selectbox("Field of Study", _options_for("Field of Study"), key="reg_field")
                r_internet = st.selectbox(
                    "Internet Access",
                    _options_for("Do you have regular access to the internet?"),
                    key="reg_internet",
                )
                r_devices  = st.selectbox(
                    "Devices Used for Studying",
                    _options_for("What devices do you use for studying?"),
                    key="reg_devices",
                )

            if st.button("Register", type="primary", key="register_btn"):
                if not r_user or not r_pass:
                    st.error("Username and password are required.")
                elif r_pass != r_pass2:
                    st.error("Passwords do not match.")
                else:
                    db = get_db()
                    ok, msg = db.create_user(
                        r_user.strip(), r_name.strip(), r_email.strip(), r_pass, "student"
                    )
                    if ok:
                        new_user = db.authenticate(r_user.strip(), r_pass)
                        if not new_user:
                            st.error(
                                "Account was created but sign-in failed. "
                                "Please use the Sign in tab."
                            )
                        else:
                            db.create_student_profile(
                                user_id=new_user["user_id"],
                                gender=r_gender,
                                age_range=r_age,
                                level_of_study=r_level,
                                field_of_study=r_field,
                                internet_access=r_internet,
                                devices_used=r_devices,
                            )
                            logged_in, welcome = login(
                                r_user.strip(), r_pass, new_account=True
                            )
                            if logged_in:
                                st.success(welcome)
                                st.rerun()
                            else:
                                st.error(
                                    "Account was created but sign-in failed. "
                                    "Please use the Sign in tab."
                                )
                    else:
                        st.error(msg)

    # ---- Sign in -----------------------------------------------------------
    with tab_login:
        with st.container(border=True):
            st.subheader("Existing user")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            cols = st.columns([1, 2])
            with cols[0]:
                clicked = st.button("Sign in", type="primary", use_container_width=True)
            if clicked:
                ok, msg = login(username.strip(), password)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
