"""
app/auth.py
===========
Helpers around session_state for login/logout and role gating.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from database import get_db


def is_authenticated() -> bool:
    return st.session_state.get("user") is not None


def current_user() -> Optional[dict]:
    return st.session_state.get("user")


def login(username: str, password: str, *, new_account: bool = False) -> tuple[bool, str]:
    db = get_db()
    user = db.authenticate(username, password)
    if user is None:
        return False, "Invalid username or password."
    # don't keep the password hash in session_state
    user.pop("password_hash", None)
    st.session_state["user"] = user
    name = user.get("full_name") or user["username"]
    if new_account:
        return True, f"Welcome, {name}! Your account is ready."
    return True, f"Welcome back, {name}!"


def logout():
    for key in ("user", "last_prediction", "last_explanation"):
        st.session_state.pop(key, None)


def require_login():
    """Stop the page early if the user is not logged in."""
    if not is_authenticated():
        st.warning("🔒 You must log in to access this page.")
        st.stop()


def require_role(role: str):
    """Stop the page early if the user does not have the required role."""
    require_login()
    user = current_user()
    if user.get("role") != role:
        st.error(f"⛔ This page requires the **{role}** role.")
        st.stop()
