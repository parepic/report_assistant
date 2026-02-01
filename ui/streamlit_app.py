from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add repo root to sys.path so report_assistant can be imported
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

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
