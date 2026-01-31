from __future__ import annotations

import streamlit as st


def render_chat_history(messages: list[dict]) -> None:
    if not messages:
        st.markdown("<div class='ra-empty'>Ask a question about this report to begin.</div>", unsafe_allow_html=True)
        return

    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
