from __future__ import annotations

from typing import Any

import streamlit as st


DEFAULT_STATE: dict[str, Any] = {
    "ui_phase": "landing",
    "selected_reports": [],
    "chat_messages": [],
    "compare_to_last_year_clicked": False,
    "compare_to_last_year_result": None,
    "compare_to_last_year_rendered": False,
    "browse_risk_factors_clicked": False,
    "browse_risk_factors_result": None,
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value