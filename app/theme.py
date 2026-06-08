"""
app/theme.py
============
Streamlit-native theme handling. We use st.session_state + the built-in
theming API for colours; minimal CSS hides default Streamlit chrome/footer.

Streamlit exposes ``st._config.set_option`` for runtime theme adjustments;
we apply the user's choice on every page load via ``apply_theme()``.
"""

from __future__ import annotations

import streamlit as st


LIGHT = {
    "theme.base":                    "light",
    "theme.primaryColor":            "#4F46E5",
    "theme.backgroundColor":         "#FFFFFF",
    "theme.secondaryBackgroundColor":"#F3F4F6",
    "theme.textColor":               "#111827",
}

DARK = {
    "theme.base":                    "dark",
    "theme.primaryColor":            "#818CF8",
    "theme.backgroundColor":         "#0F172A",
    "theme.secondaryBackgroundColor":"#1E293B",
    "theme.textColor":               "#F8FAFC",
}


def apply_theme():
    """Apply the currently selected theme to Streamlit's runtime config."""
    mode = st.session_state.get("theme_mode", "Light")
    palette = DARK if mode == "Dark" else LIGHT
    for key, value in palette.items():
        try:
            st._config.set_option(key, value)
        except Exception:
            pass


def hide_streamlit_branding() -> None:
    """Hide the default Streamlit footer and deploy button."""
    st.markdown(
        """
        <style>
        footer[data-testid="stFooter"],
        footer {visibility: hidden; height: 0;}
        .stDeployButton {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def theme_toggle(location=None):
    """Render the theme toggle (sidebar by default)."""
    container = location if location is not None else st.sidebar
    current = st.session_state.get("theme_mode", "Light")
    new_mode = container.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if current == "Light" else 1,
        horizontal=True,
        key="theme_toggle_radio",
    )
    if new_mode != current:
        st.session_state["theme_mode"] = new_mode
        apply_theme()
        st.rerun()
