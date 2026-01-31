from __future__ import annotations

from typing import Any

import streamlit as st


DEFAULT_STATE: dict[str, Any] = {
    "ui_phase": "landing",
    "selected_doc_id": None,
    "selected_entry": None,
    "chat_messages": [],
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
