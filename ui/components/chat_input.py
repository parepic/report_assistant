from __future__ import annotations

import streamlit as st


def get_user_input() -> str | None:
    return st.chat_input("Ask a question about this report…")
