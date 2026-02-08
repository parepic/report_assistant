from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Report Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if hasattr(st, "switch_page"):
    st.switch_page("pages/landing.py")
else:
    from pages import landing as landing_page

    landing_page.main()
