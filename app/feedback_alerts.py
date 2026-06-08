"""
app/feedback_alerts.py
======================
Unread educator-feedback alerts for students.
"""

from __future__ import annotations

import streamlit as st

from database import get_db


def render_unread_feedback_banner(student_user_id: int) -> None:
    """Show a notice when the student has unread educator messages."""
    n = get_db().count_unread_feedback(student_user_id)
    if n <= 0:
        return
    label = "message" if n == 1 else "messages"
    st.info(
        f"💬 You have **{n}** new {label} from your educator. "
        "Open **My Results** to read them."
    )
